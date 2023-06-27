from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../")
sys.path.append(f"{f}/../..")

import torch
import torch.nn as nn
import torch.nn.functional as F

from utility_functions_new import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

import tt_lib
from tt_lib.fallback_ops import fallback_ops


class TtEffectiveSEModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 1,
        bias=None,
        dilation=1,
        add_maxpool=False,
        state_dict=None,
        device=None,
        host=None,
        base_address=None,
        **_,
    ):
        super(TtEffectiveSEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.device = device
        self.host = host
        self.base_address = base_address

        conv_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.fc.weight"], self.device, put_on_device=False
        )
        conv_bias = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.fc.bias"], self.device, put_on_device=False
        )
        bias = None
        self.fc = fallback_ops.Conv2d(
            weights=conv_weight,
            biases=conv_bias,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode="zeros",
        )

        self.activation = torch.nn.Hardsigmoid()

    def forward(self, input: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        out = tt_to_torch_tensor(input, self.host)
        out = out.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            out = 0.5 * out + 0.5 * input.amax((2, 3), keepdim=True)
        out = torch_to_tt_tensor_rm(out, self.host)
        out = self.fc(out)
        out = tt_to_torch_tensor(out, self.host)
        out = self.activation(out)
        out = torch_to_tt_tensor_rm(out, self.host)
        out = tt_lib.tensor.bcast(
            input, out, tt_lib.tensor.BcastOpMath.MUL, tt_lib.tensor.BcastOpDim.HW
        )
        return out
