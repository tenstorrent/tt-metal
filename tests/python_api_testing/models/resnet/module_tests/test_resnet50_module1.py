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

import pytest

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

@pytest.mark.parametrize("fuse_ops", [False, True], ids=['Not Fused', "Ops Fused"])
def test_resnet50_module1(fuse_ops):

    with torch.no_grad():
        # torch.manual_seed(1234)
        # Initialize the device
        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        ttl.device.InitializeDevice(device)
        host = ttl.device.GetHost()

        torch_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        torch_resnet.eval()
        state_dict = torch_resnet.state_dict()
        torch_module1 = torch_resnet.layer1

        layer1 = _make_layer(Bottleneck, 64, 3, name="layer1", state_dict=state_dict)
        layer1.eval()
        dataloader = prep_ImageNet(batch_size=batch_size)
        for i, (images, targets, _, _, _) in enumerate(tqdm(dataloader)):
            image = images
            break

        transformed_input = torch_resnet.conv1(image)
        transformed_input = torch_resnet.bn1(transformed_input)
        transformed_input = torch_resnet.relu(transformed_input)
        input = torch_resnet.maxpool(transformed_input)

        if fuse_ops:
            modules_to_fuse = [['0.conv1', '0.bn1', '0.relu1'], ['0.conv2', '0.bn2', '0.relu2'], ['0.conv3', '0.bn3']]
            modules_to_fuse.extend([['1.conv1', '1.bn1', '1.relu1'], ['1.conv2', '1.bn2', '1.relu2'], ['1.conv3', '1.bn3']])
            modules_to_fuse.extend([['2.conv1', '2.bn1', '2.relu1'], ['2.conv2', '2.bn2', '2.relu2'], ['2.conv3', '2.bn3']])
            modules_to_fuse.extend([['0.downsample.0', '0.downsample.1']])
            layer1 = torch.ao.quantization.fuse_modules(layer1, modules_to_fuse)

        torch_output = torch_module1(input)
        tt_output = layer1(input)

        passing, info = comp_allclose_and_pcc(torch_output, tt_output)

        logger.info(f"{passing}, {info}")
        assert passing
