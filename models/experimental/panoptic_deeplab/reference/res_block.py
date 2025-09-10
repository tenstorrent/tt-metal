# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch


class ResModel(torch.nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1, 1, bias=False), nn.BatchNorm2d(in_channels // 8), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(intermediate_channels, intermediate_channels, 5, 1, 2, 1, intermediate_channels, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(intermediate_channels, out_channels, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()
        )

    def forward(self, x, res2):
        out = nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
        res2 = self.conv1(res2)
        out = torch.cat((res2, out), dim=1)
        out = self.conv2(out)
        out = self.conv3(out)
        return out
