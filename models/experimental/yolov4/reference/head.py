# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        # left side of graph
        # in_chan, out_chan, kernel, stride,
        output_ch = 255

        self.c1 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(256)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.c2 = nn.Conv2d(256, output_ch, 1, 1, 0, bias=True)

        # R -4
        self.c3 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.b3 = nn.BatchNorm2d(256)

        # R -1 -16
        self.c4 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b4 = nn.BatchNorm2d(256)

        self.c5 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.b5 = nn.BatchNorm2d(512)

        self.c6 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b6 = nn.BatchNorm2d(256)

        self.c7 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.b7 = nn.BatchNorm2d(512)

        self.c8 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b8 = nn.BatchNorm2d(256)

        self.c9 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.b9 = nn.BatchNorm2d(512)

        self.c10 = nn.Conv2d(512, output_ch, 1, 1, 0, bias=True)

        # R -4
        self.c11 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.b11 = nn.BatchNorm2d(512)

        self.c12 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.b12 = nn.BatchNorm2d(512)

        self.c13 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.b13 = nn.BatchNorm2d(1024)

        self.c14 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.b14 = nn.BatchNorm2d(512)

        self.c15 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.b15 = nn.BatchNorm2d(1024)

        self.c16 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.b16 = nn.BatchNorm2d(512)

        self.c17 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.b17 = nn.BatchNorm2d(1024)

        self.c18 = nn.Conv2d(1024, output_ch, 1, 1, 0, bias=True)

    def forward(self, input1, input2, input3):
        x1 = self.c1(input1)
        x1 = self.b1(x1)
        x1 = self.relu(x1)

        x2 = self.c2(x1)

        x3 = self.c3(input1)
        x3 = self.b3(x3)
        x3 = self.relu(x3)

        # R -1 -16
        outfromNeck1 = input3
        x3 = torch.cat([x3, outfromNeck1], dim=1)

        x4 = self.c4(x3)
        x4 = self.b4(x4)
        x4 = self.relu(x4)

        x5 = self.c5(x4)
        x5 = self.b5(x5)
        x5 = self.relu(x5)

        x6 = self.c6(x5)
        x6 = self.b6(x6)
        x6 = self.relu(x6)

        x7 = self.c7(x6)
        x7 = self.b7(x7)
        x7 = self.relu(x7)

        x8 = self.c8(x7)
        x8 = self.b8(x8)
        x8 = self.relu(x8)

        x9 = self.c9(x8)
        x9 = self.b9(x9)
        x9 = self.relu(x9)

        x10 = self.c10(x9)

        # R -4
        x11 = self.c11(x8)
        x11 = self.b11(x11)
        x11 = self.relu(x11)

        # R -1 -37
        outfromNeck2 = input2
        x11 = torch.cat([x11, outfromNeck2], dim=1)

        x12 = self.c12(x11)
        x12 = self.b12(x12)
        x12 = self.relu(x12)

        x13 = self.c13(x12)
        x13 = self.b13(x13)
        x13 = self.relu(x13)

        x14 = self.c14(x13)
        x14 = self.b14(x14)
        x14 = self.relu(x14)

        x15 = self.c15(x14)
        x15 = self.b15(x15)
        x15 = self.relu(x15)

        x16 = self.c16(x15)
        x16 = self.b16(x16)
        x16 = self.relu(x16)

        x17 = self.c17(x16)
        x17 = self.b17(x17)
        x17 = self.relu(x17)

        x18 = self.c18(x17)
        return x2, x10, x18
