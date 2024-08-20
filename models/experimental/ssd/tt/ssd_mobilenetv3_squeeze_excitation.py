# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import tt_lib.fallback_ops as fallback_ops
from models.utility_functions import (
    torch_to_tt_tensor_rm,
)


class TtSqueezeExcitation(torch.nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
        fc_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        state_dict=None,
        base_address="",
        device=None,
    ) -> None:
        super().__init__()
        self.device = device

        self.avgpool = fallback_ops.AdaptiveAvgPool2d(1)

        weight_fc1 = torch_to_tt_tensor_rm(state_dict[f"{base_address}.fc1.weight"], device, put_on_device=False)
        bias_fc1 = torch_to_tt_tensor_rm(state_dict[f"{base_address}.fc1.bias"], device, put_on_device=False)
        self.fc1 = fallback_ops.Conv2d(
            weights=weight_fc1,
            biases=bias_fc1,
            in_channels=in_channels,
            out_channels=fc_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        weight_fc2 = torch_to_tt_tensor_rm(state_dict[f"{base_address}.fc2.weight"], device, put_on_device=False)
        bias_fc2 = torch_to_tt_tensor_rm(state_dict[f"{base_address}.fc2.bias"], device, put_on_device=False)
        self.fc2 = fallback_ops.Conv2d(
            weights=weight_fc2,
            biases=bias_fc2,
            in_channels=fc_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        self.activation = ttnn.relu
        self.scale_activation = ttnn.hardsigmoid

    def forward(self, input: ttnn.Tensor) -> ttnn.Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        final_out = ttnn.multiply(input, scale)
        return final_out
