import torch
import datetime
from doctr.models import detection
import numpy as np
from PIL import Image
# from torchvision.transforms import Normalize
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pypdfium2 as pdfium
from typing import Any
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from collections import OrderedDict
from doctr.utils.visualization import visualize_page
from datetime import date
import cv2
import os
import json
import re
import shutil
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage


aliases = {
    'model1': '/home2/sreevatsa/Robust-word-detector-for-Indic-Documents/weights/db_resnet50.pt',
    'model2': '/home2/sreevatsa/models/final/Class_balanced finetune for all_layers_epoch11.pt',
    'model3': '/home2/sreevatsa/models/final/Random_sampling-finetune for all layers (Backbone unfreezed)_epoch22.pt',
    'model4': '/home2/sreevatsa/models/final/Random_sampling-finetune for last layers (Backbone freezed)_v2_epoch9.pt'
}

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DocTR training script for text detection (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    #changed
    # parser.add_argument("easy_train_path", type=str,default="/scratch/abhaynew/newfolder/train/Easy", help="path to training data folder")
    # parser.add_argument("medium_train_path", type=str,default="/scratch/abhaynew/newfolder/train/Medium", help="path to training data folder")
    # parser.add_argument("hard_train_path", type=str,default="/scratch/abhaynew/newfolder/train/Hard", help="path to training data folder")


    # parser.add_argument("val_path", type=str,default="/scratch/abhaynew/newfolder/val", help="path to validation data folder")
    # parser.add_argument("test_path", type=str,default="/scratch/abhaynew/newfolder/test", help="path to test data folder")
    parser.add_argument("--ImageFile", type=str, default=None, help="Input file to get text detections")
    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train the model on")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size for training")
    parser.add_argument("--device", default=None, type=int, help="device")
    parser.add_argument("--input_size", type=int, default=1024, help="model input size, H = W")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for the optimizer (Adam)")
    parser.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("-j", "--workers", type=int, default=0, help="number of workers used for dataloading")
    parser.add_argument("--resume", type=str, default=None, choices=aliases.keys(), metavar='choice',help="Path to your checkpoint")
    # parser.add_argument("--resume", type=str, default=None, choices=aliases.keys(), metavar='choice',help="Path to your checkpoint")
    # parser.add_argument("--resume", type=str, default=None, help="Path to your checkpoint")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Run the validation loop")
    parser.add_argument(
        "--freeze-backbone", dest="freeze_backbone", action="store_true", help="freeze model backbone for fine-tuning"
    )
    parser.add_argument(
        "--show-samples", dest="show_samples", action="store_true", help="Display unormalized training samples"
    )
    parser.add_argument("--wb", dest="wb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--push-to-hub", dest="push_to_hub", action="store_true", help="Push to Huggingface Hub")
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Load pretrained parameters before starting the training",
    )
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    parser.add_argument('--padding',default=0,type=int,
                        help = 'amount of padding to bounding boxes (in pixels)')                    
    args = parser.parse_args()

    return args

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

selection = args.resume
selected_model = aliases[args.resume]


if isinstance(selected_model, str):
    predictor = ocr_predictor(pretrained=True).to(device)
    state_dict = torch.load(selected_model)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    predictor.det_predictor.model.load_state_dict(new_state_dict)
else:
    predictor = ocr_predictor(pretrained=True).to(device)

today = date.today()
d=today.strftime("%d%m%y")

current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%H%M%S")

def doctr_predictions(directory): 
    doc = DocumentFile.from_images(directory)
    result = predictor(doc)
    dic = result.export()
    
    page_dims = [page['dimensions'] for page in dic['pages']]
    
    regions = []
    abs_coords = []
    
    regions = [[word for block in page['blocks'] for line in block['lines'] for word in line['words']] for page in dic['pages']]
    abs_coords = [
    [[int(round(word['geometry'][0][0] * dims[1])), 
      int(round(word['geometry'][0][1] * dims[0])), 
      int(round(word['geometry'][1][0] * dims[1])), 
      int(round(word['geometry'][1][1] * dims[0]))] for word in words]
    for words, dims in zip(regions, page_dims)
    ]

#     pred = torch.Tensor(abs_coords[0])
    return (abs_coords,page_dims,regions)

# image_file = '/home2/sreevatsa/test_images/242.jpg'
image_file = args.ImageFile
pred,a,r = doctr_predictions(image_file)    
image = cv2.imread(image_file)
for w in pred[0]:
    cv2.rectangle(image,(w[0], w[1]),(w[2], w[3]),(0,0,255),1)   
cv2.imwrite('/home2/sreevatsa/output_test_doctrv2_{}_{}.png'.format(d,formatted_time), image)
print('Image saved!!!!')

