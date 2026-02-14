# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

import ttnn

from .conv_block import ConvBlock
from .max_pool3d import max_pool3d


class Encoder:
    def __init__(
        self,
        device,
        is_bottleneck: bool,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        num_groups: int = 8,
        kernel_size: int = 3,
        scale_factor: int = 2,
    ):
        self.conv_block_1 = ConvBlock(device, in_channels, hid_channels, num_groups, kernel_size)
        self.conv_block_2 = ConvBlock(device, hid_channels, out_channels, num_groups, kernel_size)
        self.scale_factor = scale_factor
        self.is_bottleneck = is_bottleneck

    def load_state_dict(
        self,
        device,
        params_dict: dict[str, torch.Tensor],
        module_prefix: Optional[str] = None,
    ):
        conv1_prefix = f"{module_prefix}.conv_block_1" if module_prefix else "conv_block_1"
        conv2_prefix = f"{module_prefix}.conv_block_2" if module_prefix else "conv_block_2"
        self.conv_block_1.load_state_dict(device, params_dict, conv1_prefix)
        self.conv_block_2.load_state_dict(device, params_dict, conv2_prefix)

    def __call__(self, x, device) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        x0 = self.conv_block_1(x, device)
        ttnn.deallocate(x)
        x1 = self.conv_block_2(x0, device)
        ttnn.deallocate(x0)
        if self.is_bottleneck:
            return x1
        x_pooled = max_pool3d(x1, kernel_size=self.scale_factor)
        return x_pooled, x1
