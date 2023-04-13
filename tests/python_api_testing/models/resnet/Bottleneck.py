import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

import torch
import torch.nn as nn

from typing import Optional, Callable

from utils import conv3x3, conv1x1, pad_by_zero, unpad_from_zero

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        device = None,
        host = None,
        state_dict = None,
        base_address = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, state_dict=state_dict, base_address=f"{base_address}.conv1")
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, state_dict=state_dict, base_address=f"{base_address}.conv2")
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, state_dict=state_dict, base_address=f"{base_address}.conv3")
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = ttl.tensor.relu
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = tt2torch_tensor(x, self.host)
        out = self.conv1(x)
        out = self.bn1(out)
        out, initial_shape = pad_by_zero(out, self.device)

        out = self.relu(out)
        out = unpad_from_zero(out, initial_shape)

        out = self.conv2(out)
        out = self.bn2(out)

        out, initial_shape = pad_by_zero(out, self.device)
        out = self.relu(out)
        out = unpad_from_zero(out, initial_shape)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out, initial_shape = pad_by_zero(out, self.device)
        out = ttl.tensor.add(out, identity)

        out = self.relu(out)
        out = unpad_from_zero(out, initial_shape)
        return out
