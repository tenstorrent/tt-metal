# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from .conv_block import ConvBlockTch


class EncoderTch(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, kernel_size=3, padding=1):
        super(EncoderTch, self).__init__()
        self.conv_block_1 = ConvBlockTch(in_channels, hid_channels, kernel_size, padding)
        self.conv_block_2 = ConvBlockTch(hid_channels, out_channels, kernel_size, padding)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x_pooled = self.pool(x)
        return x_pooled, x
