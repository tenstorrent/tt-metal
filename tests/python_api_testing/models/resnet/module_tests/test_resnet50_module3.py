from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from typing import Type, Union, Optional, Callable
from loguru import logger

import torch
from torchvision import models, transforms
import pytest

import tt_lib as ttl
from torch_resnet import _make_layer, Bottleneck
from tt_models.utility_functions import comp_allclose_and_pcc, comp_pcc

batch_size=1

@pytest.mark.parametrize("fuse_ops", [False, True], ids=['Not Fused', "Ops Fused"])
def test_resnet50_module3(fuse_ops, imagenet_sample_input):
    image = imagenet_sample_input
    with torch.no_grad():

        torch_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        torch_resnet.eval()
        state_dict = torch_resnet.state_dict()
        torch_module = torch_resnet.layer3

        layer3 = _make_layer(Bottleneck, 256, 6, name="layer3", stride=2, dilate=False, state_dict=state_dict)
        layer3.eval()

        transformed_input = torch_resnet.conv1(image)
        transformed_input = torch_resnet.bn1(transformed_input)
        transformed_input = torch_resnet.relu(transformed_input)
        transformed_input = torch_resnet.maxpool(transformed_input)
        transformed_input = torch_resnet.layer1(transformed_input)
        input = torch_resnet.layer2(transformed_input)

        if fuse_ops:
            modules_to_fuse = [['0.conv1', '0.bn1', '0.relu1'], ['0.conv2', '0.bn2', '0.relu2'], ['0.conv3', '0.bn3']]
            modules_to_fuse.extend([['1.conv1', '1.bn1', '1.relu1'], ['1.conv2', '1.bn2', '1.relu2'], ['1.conv3', '1.bn3']])
            modules_to_fuse.extend([['2.conv1', '2.bn1', '2.relu1'], ['2.conv2', '2.bn2', '2.relu2'], ['2.conv3', '2.bn3']])
            modules_to_fuse.extend([['3.conv1', '3.bn1', '3.relu1'], ['3.conv2', '3.bn2', '3.relu2'], ['3.conv3', '3.bn3']])
            modules_to_fuse.extend([['4.conv1', '4.bn1', '4.relu1'], ['4.conv2', '4.bn2', '4.relu2'], ['4.conv3', '4.bn3']])
            modules_to_fuse.extend([['5.conv1', '5.bn1', '5.relu1'], ['5.conv2', '5.bn2', '5.relu2'], ['5.conv3', '5.bn3']])
            modules_to_fuse.extend([['0.downsample.0', '0.downsample.1']])
            layer3 = torch.ao.quantization.fuse_modules(layer3, modules_to_fuse)

        torch_output = torch_module(input)
        tt_output = layer3(input)

        passing, info = comp_pcc(torch_output, tt_output)
        # we cannot use comp_allclose_and_pcc because the values are close, rtol ends up being nan.logger.info(f"{passing}, {info}")
        logger.info(f"{passing}, {info}")

        assert passing
