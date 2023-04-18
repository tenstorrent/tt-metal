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
from tqdm import tqdm

from imagenet import prep_ImageNet
from libs import tt_lib as ttl
from torch_resnet import _make_layer, Bottleneck
from torch_resnet import *
from utility_functions import comp_allclose_and_pcc, comp_pcc


batch_size=1

@pytest.mark.parametrize("fuse_ops", [False, True], ids=['Not Fused', "Ops Fused"])
def test_resnet18_module2(fuse_ops):
    with torch.no_grad():
        # Initialize the device
        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        ttl.device.InitializeDevice(device)
        host = ttl.device.GetHost()

        torch_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        torch_resnet.eval()
        state_dict = torch_resnet.state_dict()
        torch_module2 = torch_resnet.layer2

        layer2 = _make_layer(BasicBlock, 128, 2, name="layer2", stride=2, dilate=False, state_dict=state_dict)
        layer2.eval()

        if fuse_ops:
            modules_to_fuse = [['0.conv1', '0.bn1', '0.relu1'], ['0.conv2', '0.bn2']]
            modules_to_fuse.extend([['1.conv1', '1.bn1', '1.relu1'], ['1.conv2', '1.bn2']])
            modules_to_fuse.extend([['0.downsample.0', '0.downsample.1']])
            layer2 = torch.ao.quantization.fuse_modules(layer2, modules_to_fuse)

        dataloader = prep_ImageNet(batch_size=batch_size)
        for i, (images, targets, _, _, _) in enumerate(tqdm(dataloader)):
            image = images
            break

        transformed_input = torch_resnet.conv1(image)
        transformed_input = torch_resnet.bn1(transformed_input)
        transformed_input = torch_resnet.relu(transformed_input)
        transformed_input = torch_resnet.maxpool(transformed_input)
        input = torch_resnet.layer1(transformed_input)

        torch_output = torch_module2(input)
        tt_output = layer2(input)

        passing, info = comp_pcc(torch_output, tt_output)
        # we cannot use comp_allclose_and_pcc because the values are close, rtol ends up being nan.logger.info(f"{passing}, {info}")

        logger.info(f"{passing}, {info}")
        assert passing
