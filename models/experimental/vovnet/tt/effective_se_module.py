# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)
import ttnn
from tt_lib import fallback_ops


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
        base_address=None,
        **_,
    ):
        super(TtEffectiveSEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.device = device
        self.base_address = base_address

        conv_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.fc.weight"], self.device, put_on_device=False)
        conv_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.fc.bias"], self.device, put_on_device=False)

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

        self.activation = ttnn.hardsigmoid

    def forward(self, input: ttnn.Tensor) -> ttnn.Tensor:
        out = tt_to_torch_tensor(input)
        out = out.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            out = 0.5 * out + 0.5 * input.amax((2, 3), keepdim=True)
        out = torch_to_tt_tensor_rm(out, self.device, put_on_device=False)
        out = self.fc(out)
        out = self.activation(out)
        out = ttnn.multiply(input, out)
        return out
