# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch


class HeadModel(torch.nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super().__init__()

        if out_channels == 1:  # instance center head
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, 1), nn.BatchNorm2d(in_channels), nn.ReLU()
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels, intermediate_channels, 3, 1, 1, 1),
                nn.BatchNorm2d(intermediate_channels),
                nn.ReLU(),
            )
        else:  # instance offset head
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 5, 1, 2, 1, in_channels), nn.BatchNorm2d(in_channels), nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels, intermediate_channels, 1, 1), nn.BatchNorm2d(intermediate_channels), nn.ReLU()
            )
        self.conv3 = nn.Sequential(nn.Conv2d(intermediate_channels, out_channels, 1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = nn.functional.interpolate(out, scale_factor=4, mode="bilinear")
        return out
