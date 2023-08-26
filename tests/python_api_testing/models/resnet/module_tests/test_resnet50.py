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
from torch_resnet import _make_layer, ResNet, Bottleneck
from tt_models.utility_functions import comp_allclose_and_pcc, comp_pcc


batch_size=1

@pytest.mark.parametrize("fuse_ops", [False, True], ids=['Not Fused', "Ops Fused"])
def test_resnet50(fuse_ops, imagenet_sample_input):
    image = imagenet_sample_input
    with torch.no_grad():

        torch_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        torch_resnet.eval()
        state_dict = torch_resnet.state_dict()

        layers = [3, 4, 6, 3]
        tt_resnet = ResNet(Bottleneck, layers, state_dict=state_dict)
        tt_resnet.eval()

        if fuse_ops:
            modules_to_fuse = [['conv1', 'bn1']]
            modules_to_fuse.extend([['layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu1'], ['layer1.0.conv2', 'layer1.0.bn2', 'layer1.0.relu2'], ['layer1.0.conv3', 'layer1.0.bn3']])
            modules_to_fuse.extend([['layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu1'], ['layer1.1.conv2', 'layer1.1.bn2', 'layer1.1.relu2'], ['layer1.1.conv3', 'layer1.1.bn3']])
            modules_to_fuse.extend([['layer1.2.conv1', 'layer1.2.bn1', 'layer1.2.relu1'], ['layer1.2.conv2', 'layer1.2.bn2', 'layer1.2.relu2'], ['layer1.2.conv3', 'layer1.2.bn3']])
            modules_to_fuse.extend([['layer1.0.downsample.0', 'layer1.0.downsample.1']])

            modules_to_fuse.extend([['layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.relu1'], ['layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.relu2'], ['layer2.0.conv3', 'layer2.0.bn3']])
            modules_to_fuse.extend([['layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.relu1'], ['layer2.1.conv2', 'layer2.1.bn2', 'layer2.1.relu2'], ['layer2.1.conv3', 'layer2.1.bn3']])
            modules_to_fuse.extend([['layer2.2.conv1', 'layer2.2.bn1', 'layer2.2.relu1'], ['layer2.2.conv2', 'layer2.2.bn2', 'layer2.2.relu2'], ['layer2.2.conv3', 'layer2.2.bn3']])
            modules_to_fuse.extend([['layer2.3.conv1', 'layer2.3.bn1', 'layer2.3.relu1'], ['layer2.3.conv2', 'layer2.3.bn2', 'layer2.3.relu2'], ['layer2.3.conv3', 'layer2.3.bn3']])
            modules_to_fuse.extend([['layer2.0.downsample.0', 'layer2.0.downsample.1']])

            modules_to_fuse.extend([['layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.relu1'], ['layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.relu2'], ['layer3.0.conv3', 'layer3.0.bn3']])
            modules_to_fuse.extend([['layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.relu1'], ['layer3.1.conv2', 'layer3.1.bn2', 'layer3.1.relu2'], ['layer3.1.conv3', 'layer3.1.bn3']])
            modules_to_fuse.extend([['layer3.2.conv1', 'layer3.2.bn1', 'layer3.2.relu1'], ['layer3.2.conv2', 'layer3.2.bn2', 'layer3.2.relu2'], ['layer3.2.conv3', 'layer3.2.bn3']])
            modules_to_fuse.extend([['layer3.3.conv1', 'layer3.3.bn1', 'layer3.3.relu1'], ['layer3.3.conv2', 'layer3.3.bn2', 'layer3.3.relu2'], ['layer3.3.conv3', 'layer3.3.bn3']])
            modules_to_fuse.extend([['layer3.4.conv1', 'layer3.4.bn1', 'layer3.4.relu1'], ['layer3.4.conv2', 'layer3.4.bn2', 'layer3.4.relu2'], ['layer3.4.conv3', 'layer3.4.bn3']])
            modules_to_fuse.extend([['layer3.5.conv1', 'layer3.5.bn1', 'layer3.5.relu1'], ['layer3.5.conv2', 'layer3.5.bn2', 'layer3.5.relu2'], ['layer3.5.conv3', 'layer3.5.bn3']])
            modules_to_fuse.extend([['layer3.0.downsample.0', 'layer3.0.downsample.1']])

            modules_to_fuse.extend([['layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.relu1'], ['layer4.0.conv2', 'layer4.0.bn2', 'layer4.0.relu2'], ['layer4.0.conv3', 'layer4.0.bn3']])
            modules_to_fuse.extend([['layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.relu1'], ['layer4.1.conv2', 'layer4.1.bn2', 'layer4.1.relu2'], ['layer4.1.conv3', 'layer4.1.bn3']])
            modules_to_fuse.extend([['layer4.2.conv1', 'layer4.2.bn1', 'layer4.2.relu1'], ['layer4.2.conv2', 'layer4.2.bn2', 'layer4.2.relu2'], ['layer4.2.conv3', 'layer4.2.bn3']])
            modules_to_fuse.extend([['layer4.0.downsample.0', 'layer4.0.downsample.1']])



            tt_resnet = torch.ao.quantization.fuse_modules(tt_resnet, modules_to_fuse)

        input = image

        torch_output = torch_resnet(input)
        tt_output = tt_resnet(input)

        passing, info = comp_pcc(torch_output, tt_output)
        # passing, info = comp_allclose_and_pcc(torch_output, tt_output)
        # we cannot use comp_allclose_and_pcc because the values are close, rtol ends up being nan.logger.info(f"{passing}, {info}")

        logger.info(f"{passing}, {info}")
        assert passing
