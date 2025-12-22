# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

import ttnn

from .conv_block import ConvBlock
from .upsample3d import upsample3d


class Decoder:
    def __init__(self, device, in_channels: int, out_channels: int, kernel_size: int = 3):
        self.conv_block_1 = ConvBlock(device, in_channels, out_channels, kernel_size)
        self.conv_block_2 = ConvBlock(device, out_channels, out_channels, kernel_size)

    def init_params(self, device, params_dict: dict[str, torch.Tensor], module_prefix: Optional[str] = None):
        conv1_prefix = f"{module_prefix}.conv_block_1" if module_prefix else "conv_block_1"
        conv2_prefix = f"{module_prefix}.conv_block_2" if module_prefix else "conv_block_2"
        self.conv_block_1.init_params(device, params_dict, conv1_prefix)
        self.conv_block_2.init_params(device, params_dict, conv2_prefix)

    def __call__(self, x0, skip_connection) -> ttnn.Tensor:
        x1 = upsample3d(x0, scale_factor=2)
        ttnn.deallocate(x0)
        # concatenate along channel dimension
        x2 = ttnn.concat([x1, skip_connection], dim=4)
        ttnn.deallocate(x1)
        x3 = self.conv_block_1(x2)
        ttnn.deallocate(x2)
        x4 = self.conv_block_2(x3)
        ttnn.deallocate(x3)
        return x4
