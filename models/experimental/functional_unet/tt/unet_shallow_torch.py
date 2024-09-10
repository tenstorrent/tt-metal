# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn

from loguru import logger


# Unet shallow torch implementation
class UNet(nn.Module):
    def __init__(self, groups=1):
        super(UNet, self).__init__()
        # Contracting Path
        self.c1 = nn.Conv2d(4 * groups, 16 * groups, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(16 * groups, momentum=1)
        self.r1 = nn.ReLU(inplace=True)
        self.c1_2 = nn.Conv2d(16 * groups, 16 * groups, kernel_size=3, padding=1)
        self.b1_2 = nn.BatchNorm2d(16 * groups, momentum=1)
        self.r1_2 = nn.ReLU(inplace=True)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c2 = nn.Conv2d(16 * groups, 16 * groups, kernel_size=3, padding=1)
        self.b2 = nn.BatchNorm2d(16 * groups, momentum=1)
        self.r2 = nn.ReLU(inplace=True)
        self.c2_2 = nn.Conv2d(16 * groups, 16 * groups, kernel_size=3, padding=1)
        self.b2_2 = nn.BatchNorm2d(16 * groups, momentum=1)
        self.r2_2 = nn.ReLU(inplace=True)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c3 = nn.Conv2d(16 * groups, 32 * groups, kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(32 * groups, momentum=1)
        self.r3 = nn.ReLU(inplace=True)
        self.c3_2 = nn.Conv2d(32 * groups, 32 * groups, kernel_size=3, padding=1)
        self.b3_2 = nn.BatchNorm2d(32 * groups, momentum=1)
        self.r3_2 = nn.ReLU(inplace=True)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c4 = nn.Conv2d(32 * groups, 32 * groups, kernel_size=3, padding=1)
        self.b4 = nn.BatchNorm2d(32 * groups, momentum=1)
        self.r4 = nn.ReLU(inplace=True)
        self.c4_2 = nn.Conv2d(32 * groups, 32 * groups, kernel_size=3, padding=1)
        self.b4_2 = nn.BatchNorm2d(32 * groups, momentum=1)
        self.r4_2 = nn.ReLU(inplace=True)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bnc = nn.Conv2d(32 * groups, 64 * groups, kernel_size=3, padding=1)
        self.bnb = nn.BatchNorm2d(64 * groups, momentum=1)
        self.bnr = nn.ReLU(inplace=True)
        self.bnc_2 = nn.Conv2d(64 * groups, 64 * groups, kernel_size=3, padding=1)
        self.bnb_2 = nn.BatchNorm2d(64 * groups, momentum=1)
        self.bnr_2 = nn.ReLU(inplace=True)
        self.u4 = nn.Upsample(scale_factor=(2, 2), mode="nearest")

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

        self.c8 = nn.Conv2d(32 * groups, 16 * groups, kernel_size=3, padding=1)
        self.b8 = nn.BatchNorm2d(16 * groups, momentum=1)
        self.r8 = nn.ReLU(inplace=True)
        self.c8_2 = nn.Conv2d(16 * groups, 16 * groups, kernel_size=3, padding=1)
        self.b8_2 = nn.BatchNorm2d(16 * groups, momentum=1)
        self.r8_2 = nn.ReLU(inplace=True)
        self.c8_3 = nn.Conv2d(16 * groups, 16 * groups, kernel_size=3, padding=1)
        self.b8_3 = nn.BatchNorm2d(16 * groups, momentum=1)
        self.r8_3 = nn.ReLU(inplace=True)

        # Output layer
        self.output_layer = nn.Conv2d(16 * groups, 1 * groups, kernel_size=1)

    def downblock1(self, x):
        c1 = self.c1(x)
        b1 = self.b1(c1)
        r1 = self.r1(b1)
        c1_2 = self.c1_2(r1)
        b1_2 = self.b1_2(c1_2)
        r1_2 = self.r1_2(b1_2)
        p1 = self.p1(r1_2)
        return p1, r1_2

    def downblock2(self, p1):
        c2 = self.c2(p1)
        b2 = self.b2(c2)
        r2 = self.r2(b2)
        c2_2 = self.c2_2(r2)
        b2_2 = self.b2_2(c2_2)
        r2_2 = self.r2_2(b2_2)
        p2 = self.p2(r2_2)
        return p2, r2_2

    def downblock3(self, p2):
        c3 = self.c3(p2)
        b3 = self.b3(c3)
        r3 = self.r3(b3)
        c3_2 = self.c3_2(r3)
        b3_2 = self.b3_2(c3_2)
        r3_2 = self.r3_2(b3_2)
        p3 = self.p3(r3_2)
        return p3, r3_2

    def downblock4(self, p3):
        c4 = self.c4(p3)
        b4 = self.b4(c4)
        r4 = self.r4(b4)
        c4_2 = self.c4_2(r4)
        b4_2 = self.b4_2(c4_2)
        r4_2 = self.r4_2(b4_2)
        p4 = self.p4(r4_2)
        return p4, r4_2

    def upblock1(self, bnr_2, r4_2):
        u4 = self.u4(bnr_2)
        conc1 = torch.cat([u4, r4_2], dim=1)

        c5 = self.c5(conc1)
        b5 = self.b5(c5)
        r5 = self.r5(b5)
        c5_2 = self.c5_2(r5)
        b5_2 = self.b5_2(c5_2)
        r5_2 = self.r5_2(b5_2)
        c5_3 = self.c5_3(r5_2)
        b5_3 = self.b5_3(c5_3)
        r5_3 = self.r5_3(b5_3)
        return r5_3

    def upblock2(self, r5_3, r3_2):
        u3 = self.u3(r5_3)
        conc2 = torch.cat([u3, r3_2], dim=1)

        c6 = self.c6(conc2)
        b6 = self.b6(c6)
        r6 = self.r6(b6)
        c6_2 = self.c6_2(r6)
        b6_2 = self.b6_2(c6_2)
        r6_2 = self.r6_2(b6_2)
        c6_3 = self.c6_3(r6_2)
        b6_3 = self.b6_3(c6_3)
        r6_3 = self.r6_3(b6_3)
        return r6_3

    def upblock3(self, r6_3, r2_2):
        u2 = self.u2(r6_3)
        conc3 = torch.cat([u2, r2_2], dim=1)

        c7 = self.c7(conc3)
        b7 = self.b7(c7)
        r7 = self.r7(b7)
        c7_2 = self.c7_2(r7)
        b7_2 = self.b7_2(c7_2)
        r7_2 = self.r7_2(b7_2)
        c7_3 = self.c7_3(r7_2)
        b7_3 = self.b7_3(c7_3)
        r7_3 = self.r7_3(b7_3)
        return r7_3

    def upblock4(self, r7_3, r1_2):
        u1 = self.u1(r7_3)
        conc4 = torch.cat([u1, r1_2], dim=1)

        c8 = self.c8(conc4)
        b8 = self.b8(c8)
        r8 = self.r8(b8)
        c8_2 = self.c8_2(r8)
        b8_2 = self.b8_2(c8_2)
        r8_2 = self.r8_2(b8_2)
        c8_3 = self.c8_3(r8_2)
        b8_3 = self.b8_3(c8_3)
        r8_3 = self.r8_3(b8_3)
        return r8_3

    def bottleneck(self, p4):
        bnc = self.bnc(p4)
        bnb = self.bnb(bnc)
        bnr = self.bnr(bnb)
        bnc_2 = self.bnc_2(bnr)
        bnb_2 = self.bnb_2(bnc_2)
        bnr_2 = self.bnr_2(bnb_2)
        return bnr_2

    def forward(self, x):
        p1, r1_2 = self.downblock1(x)
        p2, r2_2 = self.downblock2(p1)
        p3, r3_2 = self.downblock3(p2)
        p4, r4_2 = self.downblock4(p3)

        bnr_2 = self.bottleneck(p4)

        r5_3 = self.upblock1(bnr_2, r4_2)
        r6_3 = self.upblock2(r5_3, r3_2)
        r7_3 = self.upblock3(r6_3, r2_2)
        r8_3 = self.upblock4(r7_3, r1_2)

        output = self.output_layer(r8_3)

        return output

    @staticmethod
    def from_random_weights(groups=1):
        model = UNet(groups=groups)
        model.eval()
        return model
