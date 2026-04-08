# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental.blazepose.demo.blazebase import BlazeLandmark, BlazeBlock


class BlazePoseLandmark(BlazeLandmark):
    """The hand landmark model from MediaPipe."""

    def __init__(self):
        super(BlazePoseLandmark, self).__init__()

        # size of ROIs used for input
        self.resolution = 256

        self._define_layers()

    def _define_layers(self):
        self.backbone1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=24,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            BlazeBlock(24, 24, 3),
            BlazeBlock(24, 24, 3),
        )

        self.backbone2 = nn.Sequential(
            BlazeBlock(24, 48, 3, 2),
            BlazeBlock(48, 48, 3),
            BlazeBlock(48, 48, 3),
            BlazeBlock(48, 48, 3),
        )

        self.backbone3 = nn.Sequential(
            BlazeBlock(48, 96, 3, 2),
            BlazeBlock(96, 96, 3),
            BlazeBlock(96, 96, 3),
            BlazeBlock(96, 96, 3),
            BlazeBlock(96, 96, 3),
        )

        self.backbone4 = nn.Sequential(
            BlazeBlock(96, 192, 3, 2),
            BlazeBlock(192, 192, 3),
            BlazeBlock(192, 192, 3),
            BlazeBlock(192, 192, 3),
            BlazeBlock(192, 192, 3),
            BlazeBlock(192, 192, 3),
        )

        self.backbone5 = nn.Sequential(
            BlazeBlock(192, 288, 3, 2),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(288, 288, 3, 1, 1, groups=288, bias=True),
            nn.Conv2d(288, 48, 1, bias=True),
            nn.ReLU(True),
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(192, 192, 3, 1, 1, groups=192, bias=True),
            nn.Conv2d(192, 48, 1, bias=True),
            nn.ReLU(True),
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(96, 96, 3, 1, 1, groups=96, bias=True),
            nn.Conv2d(96, 48, 1, bias=True),
            nn.ReLU(True),
        )

        self.up4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, 1, 1, groups=48, bias=True),
            nn.Conv2d(48, 48, 1, bias=True),
            nn.ReLU(True),
        )

        self.block1 = nn.Sequential(
            BlazeBlock(48, 96, 3, 2),
            BlazeBlock(96, 96, 3),
            BlazeBlock(96, 96, 3),
            BlazeBlock(96, 96, 3),
            BlazeBlock(96, 96, 3),
        )

        self.up5 = nn.Sequential(
            nn.Conv2d(96, 96, 3, 1, 1, groups=96, bias=True),
            nn.Conv2d(96, 96, 1, bias=True),
            nn.ReLU(True),
        )

        self.block2 = nn.Sequential(
            BlazeBlock(96, 192, 3, 2),
            BlazeBlock(192, 192, 3),
            BlazeBlock(192, 192, 3),
            BlazeBlock(192, 192, 3),
            BlazeBlock(192, 192, 3),
            BlazeBlock(192, 192, 3),
        )

        self.up6 = nn.Sequential(
            nn.Conv2d(192, 192, 3, 1, 1, groups=192, bias=True),
            nn.Conv2d(192, 192, 1, bias=True),
            nn.ReLU(True),
        )

        self.block3 = nn.Sequential(
            BlazeBlock(192, 288, 3, 2),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
        )

        self.up7 = nn.Sequential(
            nn.Conv2d(288, 288, 3, 1, 1, groups=288, bias=True),
            nn.Conv2d(288, 288, 1, bias=True),
            nn.ReLU(True),
        )

        self.block4 = nn.Sequential(
            BlazeBlock(288, 288, 3, 2),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3, 2),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
            BlazeBlock(288, 288, 3),
        )

        self.up8 = nn.Sequential(
            nn.Conv2d(48, 48, 3, 1, 1, groups=48, bias=True),
            nn.Conv2d(48, 8, 1, bias=True),
            nn.ReLU(True),
        )

        self.up9 = nn.Sequential(
            nn.Conv2d(24, 24, 3, 1, 1, groups=24, bias=True),
            nn.Conv2d(24, 8, 1, bias=True),
            nn.ReLU(True),
        )

        self.block5 = BlazeBlock(288, 288, 3)

        self.block6 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1, groups=8, bias=True),
            nn.Conv2d(8, 8, 1, bias=True),
            nn.ReLU(True),
        )

        self.flag = nn.Conv2d(288, 1, 2, bias=True)
        self.segmentation = nn.Conv2d(8, 1, 3, padding=1, bias=True)
        self.landmarks = nn.Conv2d(288, 124, 2, bias=True)

    def forward(self, x):
        batch = x.shape[0]
        if batch == 0:
            return (
                torch.zeros((0,)),
                torch.zeros((0, 31, 4)),
                torch.zeros((0, 128, 128)),
            )

        x = F.pad(x, (0, 1, 0, 1), "constant", 0)

        x = self.backbone1(x)
        y = self.backbone2(x)
        z = self.backbone3(y)
        w = self.backbone4(z)
        v = self.backbone5(w)

        w1 = self.up2(w) + F.interpolate(self.up1(v), scale_factor=2, mode="bilinear")
        z1 = self.up3(z) + F.interpolate(w1, scale_factor=2, mode="bilinear")
        y1 = self.up4(y) + F.interpolate(z1, scale_factor=2, mode="bilinear")

        seg = self.up9(x) + F.interpolate(self.up8(y1), scale_factor=2, mode="bilinear")
        seg = self.segmentation(self.block6(seg)).squeeze(1)

        out = self.block1(y1) + self.up5(z)
        out = self.block2(out) + self.up6(w)
        out = self.block3(out) + self.up7(v)
        out = self.block4(out)
        out = self.block5(out)
        flag = self.flag(out).view(-1).sigmoid()
        landmarks = self.landmarks(out).view(batch, 31, 4) / 256

        return flag, landmarks, seg
