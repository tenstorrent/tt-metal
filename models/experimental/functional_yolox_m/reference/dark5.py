# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
from models.experimental.functional_yolox_m.reference.bottleneck_block import BottleNeckBlock


class Dark5(nn.Module):
    def __init__(self):
        super(Dark5, self).__init__()

        # Initial Conv layer
        self.c1 = nn.Conv2d(384, 768, kernel_size=3, stride=2, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(768, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        # CSP Layer components
        self.c2 = nn.Conv2d(768, 384, kernel_size=1, stride=1, bias=False)
        self.b2 = nn.BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.p1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        self.p2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
        self.p3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)

        self.c3 = nn.Conv2d(1536, 768, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm2d(768, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c4 = nn.Conv2d(768, 384, kernel_size=1, stride=1, bias=False)
        self.b4 = nn.BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c5 = nn.Conv2d(768, 384, kernel_size=1, stride=1, bias=False)
        self.b5 = nn.BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c6 = nn.Conv2d(768, 768, kernel_size=1, stride=1, bias=False)
        self.b6 = nn.BatchNorm2d(768, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.bblock = BottleNeckBlock(384, 2, False)
        # Activation function
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor):
        # Initial Conv layer
        x1 = self.c1(x)
        x1_b = self.b1(x1)
        x1_m = self.silu(x1_b)

        # CSP Layer
        x2 = self.c2(x1_m)
        x2_b = self.b2(x2)
        x2_m = self.silu(x2_b)

        p1_m = self.p1(x2_m)
        p2_m = self.p2(x2_m)
        p3_m = self.p3(x2_m)

        conc1 = torch.cat([x2_m] + [p1_m, p2_m, p3_m], dim=1)

        x3 = self.c3(conc1)
        x3_b = self.b3(x3)
        x3_m = self.silu(x3_b)

        x4 = self.c4(x3_m)
        x4_b = self.b4(x4)
        x4_m = self.silu(x4_b)

        x5 = self.c5(x3_m)
        x5_b = self.b5(x5)
        x5_m = self.silu(x5_b)

        bblock_out = self.bblock(x4_m)

        conc2 = torch.cat((bblock_out, x5_m), dim=1)
        x6 = self.c6(conc2)
        x6_b = self.b6(x6)
        x6_m = self.silu(x6_b)

        return x6_m
