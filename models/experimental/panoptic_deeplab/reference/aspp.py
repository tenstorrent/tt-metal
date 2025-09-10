# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch


class PanopticDeeplabASPPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ASPP_0_Conv = nn.Sequential(nn.Conv2d(2048, 256, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.ASPP_1_Depthwise = nn.Sequential(
            nn.Conv2d(2048, 2048, 3, 1, 6, 6, 2048, bias=False), nn.BatchNorm2d(2048), nn.ReLU()
        )
        self.ASPP_1_pointwise = nn.Sequential(nn.Conv2d(2048, 256, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())

        self.ASPP_2_Depthwise = nn.Sequential(
            nn.Conv2d(2048, 2048, 3, 1, 12, 12, 2048, bias=False), nn.BatchNorm2d(2048), nn.ReLU()
        )
        self.ASPP_2_pointwise = nn.Sequential(nn.Conv2d(2048, 256, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())

        self.ASPP_3_Depthwise = nn.Sequential(
            nn.Conv2d(2048, 2048, 3, 1, 18, 18, 2048, bias=False), nn.BatchNorm2d(2048), nn.ReLU()
        )
        self.ASPP_3_pointwise = nn.Sequential(nn.Conv2d(2048, 256, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())

        self.ASPP_4_avg_pool = torch.nn.AvgPool2d((32, 64), stride=1, count_include_pad=True)
        self.ASPP_4_Conv_1 = nn.Sequential(
            nn.Conv2d(2048, 256, 1, 1),
            nn.ReLU(),
        )
        self.ASPP_project = nn.Sequential(nn.Conv2d(1280, 256, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())

    def forward(self, x):
        t0 = self.ASPP_0_Conv(x)
        t1 = self.ASPP_1_Depthwise(x)
        t2 = self.ASPP_2_Depthwise(x)
        t3 = self.ASPP_3_Depthwise(x)
        t4 = self.ASPP_4_avg_pool(x)

        t4 = self.ASPP_4_Conv_1(t4)
        t4 = nn.functional.interpolate(t4, (32, 64), mode="bilinear")

        t1 = self.ASPP_1_pointwise(t1)
        t2 = self.ASPP_2_pointwise(t2)
        t3 = self.ASPP_3_pointwise(t3)

        y = torch.cat((t0, t1, t2, t3, t4), dim=1)
        y = self.ASPP_project(y)

        return y
