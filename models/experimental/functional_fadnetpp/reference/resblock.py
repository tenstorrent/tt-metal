# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(ResBlock, self).__init__()
        self.sc = False
        self.resblock_1_conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.resblock_1_bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.resblock_2_conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.resblock_2_bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.sc = True
            self.shortcut_c = nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride)
            self.shortcut_b = nn.BatchNorm2d(n_out)
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.sc:
            residual = self.shortcut_c(x)
            residual = self.shortcut_b(residual)
        out = self.resblock_1_conv1(x)
        out = self.resblock_1_bn1(out)
        out = self.relu(out)
        out = self.resblock_2_conv2(out)
        out = self.resblock_2_bn2(out)

        out += residual
        out = self.relu(out)
        return out
