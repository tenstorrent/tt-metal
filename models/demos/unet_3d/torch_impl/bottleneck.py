# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from .conv_block import ConvBlockTch


class BottleneckTch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(BottleneckTch, self).__init__()
        self.conv_block_1 = ConvBlockTch(in_channels, out_channels, kernel_size, padding)
        self.conv_block_2 = ConvBlockTch(out_channels, out_channels, kernel_size, padding)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x
