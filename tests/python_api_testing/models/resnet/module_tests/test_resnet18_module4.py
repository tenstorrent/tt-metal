from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from torch_resnet import _make_layer, Bottleneck
from torch_resnet import *

import torch
import torch.nn as nn
from torch import Tensor
import torchvision
from torchvision import models
from torchvision import transforms

from libs import tt_lib as ttl
from common import ImageNet
from typing import Type, Union, Optional, Callable

from loguru import logger

from imagenet import prep_ImageNet
from tqdm import tqdm

from utility_functions import comp_allclose_and_pcc, comp_pcc
batch_size=1

def test_resnet18_module4():
    with torch.no_grad():
        # torch.manual_seed(1234)
        # Initialize the device
        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        ttl.device.InitializeDevice(device)
        host = ttl.device.GetHost()

        torch_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        torch_resnet.eval()
        state_dict = torch_resnet.state_dict()
        torch_module4 = torch_resnet.layer4

        layer4 = _make_layer(BasicBlock, 512, 2, name="layer4", stride=2, dilate=False, state_dict=state_dict)
        layer4.eval()

        dataloader = prep_ImageNet(batch_size=batch_size)
        for i, (images, targets, _, _, _) in enumerate(tqdm(dataloader)):
            image = images
            break

        transformed_input = torch_resnet.conv1(image)
        transformed_input = torch_resnet.bn1(transformed_input)
        transformed_input = torch_resnet.relu(transformed_input)
        transformed_input = torch_resnet.maxpool(transformed_input)
        transformed_input = torch_resnet.layer1(transformed_input)
        transformed_input = torch_resnet.layer2(transformed_input)
        input = torch_resnet.layer3(transformed_input)



        torch_output = torch_module4(input)
        tt_output = layer4(input)

        passing, info = comp_allclose_and_pcc(torch_output, tt_output)

        logger.info(f"{passing}, {info}")
        assert passing
