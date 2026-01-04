# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

import ttnn

from .conv_block import ConvBlock
from .upsample3d import upsample3d


class Decoder:
    def __init__(self, device, in_channels: int, out_channels: int, num_groups: int = 8, kernel_size: int = 3):
        self.conv_block_1 = ConvBlock(device, in_channels, out_channels, num_groups, kernel_size)
        self.conv_block_2 = ConvBlock(device, out_channels, out_channels, num_groups, kernel_size)

    def load_state_dict(self, device, params_dict: dict[str, torch.Tensor], module_prefix: Optional[str] = None):
        conv1_prefix = f"{module_prefix}.conv_block_1" if module_prefix else "conv_block_1"
        conv2_prefix = f"{module_prefix}.conv_block_2" if module_prefix else "conv_block_2"
        self.conv_block_1.load_state_dict(device, params_dict, conv1_prefix)
        self.conv_block_2.load_state_dict(device, params_dict, conv2_prefix)

    def __call__(self, x, skip_connection, device) -> ttnn.Tensor:
        x0 = upsample3d(x, scale_factor=2)
        ttnn.deallocate(x)
        # concatenate along channel dimension
        x1 = ttnn.concat([skip_connection, x0], dim=4)
        ttnn.deallocate(skip_connection)
        ttnn.deallocate(x0)
        x2 = self.conv_block_1(x1, device)
        ttnn.deallocate(x1)
        x3 = self.conv_block_2(x2, device)
        ttnn.deallocate(x2)
        return x3
