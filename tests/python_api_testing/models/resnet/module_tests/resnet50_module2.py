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

from utility_functions import comp_allclose_and_pcc, comp_pcc
batch_size=1

def test_resnet50_module1():
    # inputs
    layer2_input = (1, 64, 64, 64)
    input_shape = layer2_input

    with torch.no_grad():
        # torch.manual_seed(1234)
        # Initialize the device
        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        ttl.device.InitializeDevice(device)
        host = ttl.device.GetHost()

        torch_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        torch_resnet.eval()
        state_dict = torch_resnet.state_dict()
        torch_module = torch_resnet.layer2

        layer2 = _make_layer(Bottleneck, 128, 4, name="layer2", stride=2, dilate=False, state_dict=state_dict)


        input = torch.randn(input_shape)

        torch_output = torch_module(input)
        tt_output = layer2(input)

        print(comp_allclose_and_pcc(torch_output, tt_output), "outputs")


test_resnet50_module1()
