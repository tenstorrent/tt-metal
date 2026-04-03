# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

import ttnn

from .conv3d import Conv3D
from .group_norm3d import GroupNorm3D


class ConvBlock:
    def __init__(self, device, in_channels: int, out_channels: int, num_groups: int = 8, kernel_size: int = 3):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_channels_padding = (32 - in_channels % 32) % 32
        self.out_channels_padding = (32 - out_channels % 32) % 32
        self.conv = Conv3D(device, in_channels, out_channels, kernel_size)
        if in_channels == 1:
            num_groups = 1
        self.norm = GroupNorm3D(device, in_channels, num_groups=num_groups)

    def load_state_dict(self, device, params_dict: dict[str, torch.Tensor], module_prefix: Optional[str] = None):
        conv_prefix = f"{module_prefix}.conv" if module_prefix else "conv"
        norm_prefix = f"{module_prefix}.norm" if module_prefix else "norm"
        self.conv.load_state_dict(device, params_dict, conv_prefix)
        self.norm.load_state_dict(device, params_dict, norm_prefix)

    def __call__(self, x, device) -> ttnn.Tensor:
        if x.layout != ttnn.ROW_MAJOR_LAYOUT:
            x0 = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(x)
        else:
            x0 = x
        N, D, H, W, C = x0.shape
        if self.in_channels_padding > 0:
            x1 = ttnn.reshape(x0, (N * D * H, W, C))
            x2 = ttnn.pad(x1, [(0, 0), (0, 0), (0, self.in_channels_padding)], 0)
            ttnn.deallocate(x1)
            x2 = ttnn.reshape(x2, (N, D, H, W, C + self.in_channels_padding))
        else:
            x2 = x0
        x3 = self.norm(x2, device)
        ttnn.deallocate(x2)
        x4 = self.conv(x3)
        ttnn.deallocate(x3)
        if self.out_channels_padding > 0:
            x5 = ttnn.reshape(x4, (N * D * H, W, self.out_channels + self.out_channels_padding))
            x6 = ttnn.slice(x5, [0, 0, 0], [N * D * H, W, self.out_channels])
            ttnn.deallocate(x5)
            x6 = ttnn.reshape(x6, (N, D, H, W, self.out_channels))
        else:
            x6 = x4

        N, D, H, W, C = x6.shape
        x7 = ttnn.reshape(x6, (N * D * H, W, C))
        x8 = ttnn.to_layout(x7, ttnn.TILE_LAYOUT)
        ttnn.deallocate(x7)
        ttnn.relu(x8, output_tensor=x8)
        return ttnn.reshape(x8, (N, D, H, W, self.out_channels))
