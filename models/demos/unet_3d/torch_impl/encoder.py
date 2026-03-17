# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from .conv_block import ConvBlockTch


class EncoderTch(nn.Module):
    def __init__(
        self, is_bottleneck, in_channels, hid_channels, out_channels, num_groups=8, kernel_size=3, scale_factor=2
    ):
        super(EncoderTch, self).__init__()
        self.conv_block_1 = ConvBlockTch(in_channels, hid_channels, kernel_size, num_groups)
        self.conv_block_2 = ConvBlockTch(hid_channels, out_channels, kernel_size, num_groups)
        self.pool = nn.MaxPool3d(kernel_size=scale_factor, stride=scale_factor)
        self.is_bottleneck = is_bottleneck

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        if self.is_bottleneck:
            return x
        x_pooled = self.pool(x)
        return x_pooled, x
