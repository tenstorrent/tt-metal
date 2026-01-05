# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


class ConvBlockTch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8):
        super(ConvBlockTch, self).__init__()
        # padding to keep same size
        padding = (kernel_size - 1) // 2
        if in_channels == 1:
            num_groups = 1
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        # set norm weight based on randn (since it defaults 1 for weight and 0 for bias)
        self.norm.weight.data = torch.randn(in_channels)
        self.norm.bias.data = torch.randn(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.relu(x)
        return x
