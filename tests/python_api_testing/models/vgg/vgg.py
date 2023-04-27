from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from utils import *

from libs import tt_lib as ttl
from python_api_testing.fused_ops.linear import Linear as TtLinear
from libs.tt_lib.utils import pad_weight
from typing import List, Union, Optional, Dict, cast

num_classes = 1000

class TtVGG(nn.Module):
    def __init__(
        self, features: List, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5, device=None, host=None, state_dict=None, base_address=""
    ) -> None:
        super().__init__()
        assert init_weights==False, "we are loading weights, not initializing them"
        self.device = device
        self.host = host
        self.state_dict = state_dict
        self.base_address = base_address

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))


        linear1_weight = pad_weight(state_dict[f"classifier.0.weight"])
        linear1_weight = ttl.tensor.Tensor(linear1_weight.reshape(-1).tolist(), linear1_weight.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        linear1_bias = pad_weight(state_dict[f"classifier.0.bias"])
        linear1_bias = ttl.tensor.Tensor(linear1_bias.reshape(-1).tolist(), linear1_bias.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()


        linear2_weight = pad_weight(state_dict[f"classifier.3.weight"])
        linear2_weight = ttl.tensor.Tensor(linear2_weight.reshape(-1).tolist(), linear2_weight.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        linear2_bias = pad_weight(state_dict[f"classifier.3.bias"])
        linear2_bias = ttl.tensor.Tensor(linear2_bias.reshape(-1).tolist(), linear2_bias.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        linear3_weight = pad_weight(state_dict[f"classifier.6.weight"])
        linear3_weight = ttl.tensor.Tensor(linear3_weight.reshape(-1).tolist(), linear3_weight.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        linear3_bias = pad_weight(state_dict[f"classifier.6.bias"])
        linear3_bias = ttl.tensor.Tensor(linear3_bias.reshape(-1).tolist(), linear3_bias.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        linear1 = TtLinear(512 * 7 * 7, 4096, linear1_weight, linear1_bias, self.device)

        linear2 = TtLinear(4096, 4096, linear2_weight, linear2_bias, self.device)

        linear3 = TtLinear(4096, 1024, linear3_weight, linear3_bias, self.device) # output features are num_classes=1000
        self.classifier = [
            linear1,
            ttl.tensor.relu,
            linear2,
            ttl.tensor.relu,
            linear3,
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        for layer in self.features:
            if layer is ttl.tensor.relu:
                # pad
                x, initial_shape = pad_by_zero(x, self.device)
                x = layer(x)
                x = unpad_from_zero(x, initial_shape, self.host)
                # unpad
            else:
                x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1).unsqueeze(1).unsqueeze(1)
        # pad
        x, initial_shape = pad_by_zero(x, self.device)
        for layer in self.classifier:
            x = layer(x)

        desired_shape = [batch_size, 1, 1, num_classes]
        x = unpad_from_zero(x, desired_shape, self.host)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, state_dict=None, base_address="features") -> nn.Sequential:
    layers: List = []
    in_channels = 3

    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            if batch_norm:
                assert False, "we do not support batchnorm"
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:

                layers += [conv2d, ttl.tensor.relu]
                ind = len(layers) - 2
                conv2d.weight = nn.Parameter(state_dict[f"{base_address}.{ind}.weight"])
                conv2d.bias = nn.Parameter(state_dict[f"{base_address}.{ind}.bias"])
            in_channels = v

    return layers


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def vgg16(device, host, state_dict, base_address="") -> TtVGG:

    model = TtVGG(make_layers(cfgs["D"], batch_norm=False, state_dict=state_dict), init_weights=False, device=device, host=host, state_dict=state_dict)
    return model
