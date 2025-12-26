# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

import ttnn

from .conv_block import ConvBlock
from .max_pool3d import max_pool3d


class Encoder:
    def __init__(self, device, in_channels: int, hid_channels: int, out_channels: int, kernel_size: int = 3):
        self.conv_block_1 = ConvBlock(device, in_channels, hid_channels, kernel_size)
        self.conv_block_2 = ConvBlock(device, hid_channels, out_channels, kernel_size)

    def init_params(self, device, params_dict: dict[str, torch.Tensor], module_prefix: Optional[str] = None):
        conv1_prefix = f"{module_prefix}.conv_block_1" if module_prefix else "conv_block_1"
        conv2_prefix = f"{module_prefix}.conv_block_2" if module_prefix else "conv_block_2"
        self.conv_block_1.init_params(device, params_dict, conv1_prefix)
        self.conv_block_2.init_params(device, params_dict, conv2_prefix)

    def __call__(self, x0) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        x1 = self.conv_block_1(x0)
        ttnn.deallocate(x0)
        x2 = self.conv_block_2(x1)
        ttnn.deallocate(x1)
        x_pooled = max_pool3d(x2, kernel_size=2, stride=2, padding=0)
        return x_pooled, x2
