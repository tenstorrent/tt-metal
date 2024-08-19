# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn


class BlockOne(nn.Module):
    def __init__(self, groups=1):
        super(BlockOne, self).__init__()
        # Contracting Path
        self.c1 = nn.Conv2d(4 * groups, 16 * groups, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(16 * groups, momentum=1)
        self.r1 = nn.ReLU(inplace=True)
        self.c1_2 = nn.Conv2d(16 * groups, 16 * groups, kernel_size=3, padding=1)
        self.b1_2 = nn.BatchNorm2d(16 * groups, momentum=1)
        self.r1_2 = nn.ReLU(inplace=True)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Contracting Path
        c1 = self.c1(x)
        b1 = self.b1(c1)
        r1 = self.r1(b1)
        c1_2 = self.c1_2(r1)
        b1_2 = self.b1_2(c1_2)
        r1_2 = self.r1_2(b1_2)
        p1 = self.p1(r1_2)

        return p1


class BlockTwo(nn.Module):
    def __init__(self, groups=1):
        super(BlockTwo, self).__init__()
        self.c2 = nn.Conv2d(16 * groups, 16 * groups, kernel_size=3, padding=1)
        self.b2 = nn.BatchNorm2d(16 * groups, momentum=1)
        self.r2 = nn.ReLU(inplace=True)
        self.c2_2 = nn.Conv2d(16 * groups, 16 * groups, kernel_size=3, padding=1)
        self.b2_2 = nn.BatchNorm2d(16 * groups, momentum=1)
        self.r2_2 = nn.ReLU(inplace=True)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Contracting Path
        c2 = self.c2(x)
        b2 = self.b2(c2)
        r2 = self.r2(b2)
        c2_2 = self.c2_2(r2)
        b2_2 = self.b2_2(c2_2)
        r2_2 = self.r2_2(b2_2)
        p2 = self.p2(r2_2)

        return p2


class BlockThree(nn.Module):
    def __init__(self, groups=1):
        super(BlockThree, self).__init__()
        self.c3 = nn.Conv2d(16 * groups, 32 * groups, kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(32 * groups, momentum=1)
        self.r3 = nn.ReLU(inplace=True)
        self.c3_2 = nn.Conv2d(32 * groups, 32 * groups, kernel_size=3, padding=1)
        self.b3_2 = nn.BatchNorm2d(32 * groups, momentum=1)
        self.r3_2 = nn.ReLU(inplace=True)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Contracting Path
        c3 = self.c3(x)
        b3 = self.b3(c3)
        r3 = self.r3(b3)
        c3_2 = self.c3_2(r3)
        b3_2 = self.b3_2(c3_2)
        r3_2 = self.r3_2(b3_2)
        p3 = self.p3(r3_2)

        return p3


class BlockFour(nn.Module):
    def __init__(self, groups=1):
        super(BlockFour, self).__init__()
        self.c4 = nn.Conv2d(32 * groups, 32 * groups, kernel_size=3, padding=1)
        self.b4 = nn.BatchNorm2d(32 * groups, momentum=1)
        self.r4 = nn.ReLU(inplace=True)
        self.c4_2 = nn.Conv2d(32 * groups, 32 * groups, kernel_size=3, padding=1)
        self.b4_2 = nn.BatchNorm2d(32 * groups, momentum=1)
        self.r4_2 = nn.ReLU(inplace=True)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Contracting Path
        c4 = self.c4(x)
        b4 = self.b4(c4)
        r4 = self.r4(b4)
        c4_2 = self.c4_2(r4)
        b4_2 = self.b4_2(c4_2)
        r4_2 = self.r4_2(b4_2)
        p4 = self.p4(r4_2)

        return p4


class BlockBNC(nn.Module):
    def __init__(self, groups=1):
        super(BlockBNC, self).__init__()
        self.bnc = nn.Conv2d(32 * groups, 64 * groups, kernel_size=3, padding=1)
        self.bnb = nn.BatchNorm2d(64 * groups, momentum=1)
        self.bnr = nn.ReLU(inplace=True)
        self.bnc_2 = nn.Conv2d(64 * groups, 64 * groups, kernel_size=3, padding=1)
        self.bnb_2 = nn.BatchNorm2d(64 * groups, momentum=1)
        self.bnr_2 = nn.ReLU(inplace=True)
        self.u4 = nn.Upsample(scale_factor=(2, 2), mode="nearest")

    def forward(self, x):
        # Contracting Path
        bnc = self.bnc(x)
        bnb = self.bnb(bnc)
        bnr = self.bnr(bnb)
        bnc_2 = self.bnc_2(bnr)
        bnb_2 = self.bnb_2(bnc_2)
        bnr_2 = self.bnr_2(bnb_2)
        u4 = self.u4(bnr_2)
        conc1 = torch.cat([u4, r4_2], dim=1)

        return conc1


class BlockFive(nn.Module):
    def __init__(self, groups=1):
        super(BlockFive, self).__init__()
        self.c5 = nn.Conv2d(96 * groups, 32 * groups, kernel_size=3, padding=1)
        self.b5 = nn.BatchNorm2d(32 * groups, momentum=1)
        self.r5 = nn.ReLU(inplace=True)
        self.c5_2 = nn.Conv2d(32 * groups, 32 * groups, kernel_size=3, padding=1)
        self.b5_2 = nn.BatchNorm2d(32 * groups, momentum=1)
        self.r5_2 = nn.ReLU(inplace=True)
        self.c5_3 = nn.Conv2d(32 * groups, 32 * groups, kernel_size=3, padding=1)
        self.b5_3 = nn.BatchNorm2d(32 * groups, momentum=1)
        self.r5_3 = nn.ReLU(inplace=True)
        self.u3 = nn.Upsample(scale_factor=(2, 2), mode="nearest")

    def forward(self, x):
        # Contracting Path
        c5 = self.c5(x)
        b5 = self.b5(c5)
        r5 = self.r5(b5)
        c5_2 = self.c5_2(r5)
        b5_2 = self.b5_2(c5_2)
        r5_2 = self.r5_2(b5_2)
        c5_3 = self.c5_3(r5_2)
        b5_3 = self.b5_3(c5_3)
        r5_3 = self.r5_3(b5_3)
        u3 = self.u3(r5_3)
        conc2 = torch.cat([u3, r3_2], dim=1)

        return conc2


class BlockSix(nn.Module):
    def __init__(self, groups=1):
        super(BlockSix, self).__init__()
        self.c6 = nn.Conv2d(64 * groups, 32 * groups, kernel_size=3, padding=1)
        self.b6 = nn.BatchNorm2d(32 * groups, momentum=1)
        self.r6 = nn.ReLU(inplace=True)
        self.c6_2 = nn.Conv2d(32 * groups, 32 * groups, kernel_size=3, padding=1)
        self.b6_2 = nn.BatchNorm2d(32 * groups, momentum=1)
        self.r6_2 = nn.ReLU(inplace=True)
        self.c6_3 = nn.Conv2d(32 * groups, 32 * groups, kernel_size=3, padding=1)
        self.b6_3 = nn.BatchNorm2d(32 * groups, momentum=1)
        self.r6_3 = nn.ReLU(inplace=True)
        self.u2 = nn.Upsample(scale_factor=(2, 2), mode="nearest")

    def forward(self, x):
        # Contracting Path
        c6 = self.c6(x)
        b6 = self.b6(c6)
        r6 = self.r6(b6)
        c6_2 = self.c6_2(r6)
        b6_2 = self.b6_2(c6_2)
        r6_2 = self.r6_2(b6_2)
        c6_3 = self.c6_3(r6_2)
        b6_3 = self.b6_3(c6_3)
        r6_3 = self.r6_3(b6_3)
        u2 = self.u2(r6_3)
        conc3 = torch.cat([u2, r2_2], dim=1)

        return conc3


class BlockSeven(nn.Module):
    def __init__(self, groups=1):
        super(BlockSeven, self).__init__()
        self.c7 = nn.Conv2d(48 * groups, 16 * groups, kernel_size=3, padding=1)
        self.b7 = nn.BatchNorm2d(16 * groups, momentum=1)
        self.r7 = nn.ReLU(inplace=True)
        self.c7_2 = nn.Conv2d(16 * groups, 16 * groups, kernel_size=3, padding=1)
        self.b7_2 = nn.BatchNorm2d(16 * groups, momentum=1)
        self.r7_2 = nn.ReLU(inplace=True)
        self.c7_3 = nn.Conv2d(16 * groups, 16 * groups, kernel_size=3, padding=1)
        self.b7_3 = nn.BatchNorm2d(16 * groups, momentum=1)
        self.r7_3 = nn.ReLU(inplace=True)
        self.u1 = nn.Upsample(scale_factor=(2, 2), mode="nearest")

    def forward(self, x):
        # Contracting Path
        c7 = self.c7(x)
        b7 = self.b7(c7)
        r7 = self.r7(b7)
        c7_2 = self.c7_2(r7)
        b7_2 = self.b7_2(c7_2)
        r7_2 = self.r7_2(b7_2)
        c7_3 = self.c7_3(r7_2)
        b7_3 = self.b7_3(c7_3)
        r7_3 = self.r7_3(b7_3)
        u1 = self.u1(r7_3)
        conc4 = torch.cat([u1, r1_2], dim=1)

        return conc4


class BlockEight(nn.Module):
    def __init__(self, groups=1):
        super(BlockEight, self).__init__()
        self.c8 = nn.Conv2d(32 * groups, 16 * groups, kernel_size=3, padding=1)
        self.b8 = nn.BatchNorm2d(16 * groups, momentum=1)
        self.r8 = nn.ReLU(inplace=True)
        self.c8_2 = nn.Conv2d(16 * groups, 16 * groups, kernel_size=3, padding=1)
        self.b8_2 = nn.BatchNorm2d(16 * groups, momentum=1)
        self.r8_2 = nn.ReLU(inplace=True)
        self.c8_3 = nn.Conv2d(16 * groups, 16 * groups, kernel_size=3, padding=1)
        self.b8_3 = nn.BatchNorm2d(16 * groups, momentum=1)
        self.r8_3 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Contracting Path
        c8 = self.c8(x)
        b8 = self.b8(c8)
        r8 = self.r8(b8)
        c8_2 = self.c8_2(r8)
        b8_2 = self.b8_2(c8_2)
        r8_2 = self.r8_2(b8_2)
        c8_3 = self.c8_3(r8_2)
        b8_3 = self.b8_3(c8_3)
        r8_3 = self.r8_3(b8_3)

        return r8_3


class BlockOutput(nn.Module):
    def __init__(self, groups=1):
        super(BlockOutput, self).__init__()
        self.output_layer = nn.Conv2d(16 * groups, 1 * groups, kernel_size=1)

    def forward(self, x):
        # Contracting Path
        output = self.output_layer(x)

        return output
