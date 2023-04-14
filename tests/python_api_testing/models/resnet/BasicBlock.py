import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from libs.tt_lib.utils import (
    _nearest_32 as nearest_32,
    pad_activation,
    pad_weight,
    tilize,
    tilize_to_list,
    untilize,
    print_diff_argmax,
    tt2torch,
    tt2torch_rm,
    roundup,
    roundup32,
    float_to_bits,
    divup,
    channels_last,
    convert_weights_2d_matrix
)

from utils import conv3x3, conv1x1
import torch.nn as nn
import torch

from utility_functions import tt2torch_tensor, torch2tt_tensor
from libs import tt_lib as ttl
from python_api_testing.fused_ops.linear import Linear as TtLinear
from python_api_testing.fused_ops.softmax import softmax as TtSoftmax

from typing import Optional, Callable
from utils import pad_by_zero, unpad_from_zero

class BasicBlock(nn.Module):
    expansion: int = 1

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
        self.device = device
        self.host = host
        self.base_address = base_address
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, state_dict=state_dict, base_address=f"{base_address}.conv1")
        self.bn1 = norm_layer(planes)
        self.bn1.weight = nn.Parameter(state_dict[f"{self.base_address}.bn1.weight"])
        self.bn1.bias = nn.Parameter(state_dict[f"{self.base_address}.bn1.bias"])
        self.bn1.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn1.running_mean"])
        self.bn1.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn1.running_var"])
        self.bn1.num_batches_tracked = nn.Parameter(state_dict[f"{self.base_address}.bn1.num_batches_tracked"], requires_grad=False)
        self.bn1.eval()

        self.relu = ttl.tensor.relu
        self.conv2 = conv3x3(planes, planes, state_dict=state_dict, base_address=f"{base_address}.conv2")
        self.bn2 = norm_layer(planes)

        self.bn2.weight = nn.Parameter(state_dict[f"{self.base_address}.bn2.weight"])
        self.bn2.bias = nn.Parameter(state_dict[f"{self.base_address}.bn2.bias"])
        self.bn2.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn2.running_mean"])
        self.bn2.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn2.running_var"])
        self.bn2.num_batches_tracked = nn.Parameter(state_dict[f"{self.base_address}.bn2.num_batches_tracked"], requires_grad=False)
        self.bn2.eval()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        # pad
        out, initial_shape = pad_by_zero(out, self.device)

        out = self.relu(out)
        # unpad
        out = unpad_from_zero(out, initial_shape, self.host)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        identity, identity_initial_shape = pad_by_zero(identity, self.device)
        out, out_initial_shape = pad_by_zero(out, self.device)

        out = ttl.tensor.add(out, identity)

        out = self.relu(out)
        out = unpad_from_zero(out, out_initial_shape, self.host)

        return out
