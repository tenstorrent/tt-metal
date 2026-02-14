# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from .conv_block import ConvBlockTch


class DecoderTch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8):
        super(DecoderTch, self).__init__()
        self.conv_block_1 = ConvBlockTch(in_channels, out_channels, kernel_size, num_groups)
        self.conv_block_2 = ConvBlockTch(out_channels, out_channels, kernel_size, num_groups)

    def forward(self, x, skip_connection):
        size_to_match = skip_connection.shape[2:]  # D, H, W
        x = torch.nn.functional.interpolate(x, size=size_to_match, mode="nearest")
        x = torch.cat((skip_connection, x), dim=1)  # Concatenate along channel dimension
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x
