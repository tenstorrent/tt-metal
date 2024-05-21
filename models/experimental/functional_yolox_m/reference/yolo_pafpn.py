# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch.nn as nn
import torch
from typing import List
from models.experimental.functional_yolox_m.reference.csp_darknet import CSPDarknet
from models.experimental.functional_yolox_m.reference.bottleneck_block import BottleNeckBlock


class YOLOPAFPN(nn.Module):
    def __init__(self):
        super(YOLOPAFPN, self).__init__()

        self.backbone = CSPDarknet()
        self.u = nn.Upsample(scale_factor=2, mode="nearest")
        self.c1 = nn.Conv2d(768, 384, kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.silu = nn.SiLU(inplace=True)

        # C3_p4
        self.c2 = nn.Conv2d(768, 192, kernel_size=1, stride=1, bias=False)
        self.b2 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c3 = nn.Conv2d(768, 192, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c4 = nn.Conv2d(384, 384, kernel_size=1, stride=1, bias=False)
        self.b4 = nn.BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.bblock1 = BottleNeckBlock(192, 2, False)

        # reduce_conv1
        self.c5 = nn.Conv2d(384, 192, kernel_size=1, stride=1, bias=False)
        self.b5 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        # C3_p3
        self.c6 = nn.Conv2d(384, 96, kernel_size=1, stride=1, bias=False)
        self.b6 = nn.BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c7 = nn.Conv2d(384, 96, kernel_size=1, stride=1, bias=False)
        self.b7 = nn.BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c8 = nn.Conv2d(192, 192, kernel_size=1, stride=1, bias=False)
        self.b8 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.bblock2 = BottleNeckBlock(96, 2, False)

        # bu_conv2
        self.c9 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1, bias=False)
        self.b9 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        # C3_n3
        self.c10 = nn.Conv2d(384, 192, kernel_size=1, stride=1, bias=False)
        self.b10 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c11 = nn.Conv2d(384, 192, kernel_size=1, stride=1, bias=False)
        self.b11 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c12 = nn.Conv2d(384, 384, kernel_size=1, stride=1, bias=False)
        self.b12 = nn.BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.bblock3 = BottleNeckBlock(192, 2, False)

        # bu_conv1
        self.c13 = nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1, bias=False)
        self.b13 = nn.BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        # C3_n4
        self.c14 = nn.Conv2d(768, 384, kernel_size=1, stride=1, bias=False)
        self.b14 = nn.BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c15 = nn.Conv2d(768, 384, kernel_size=1, stride=1, bias=False)
        self.b15 = nn.BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c16 = nn.Conv2d(768, 768, kernel_size=1, stride=1, bias=False)
        self.b16 = nn.BatchNorm2d(768, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.bblock4 = BottleNeckBlock(384, 2, False)

        # Activation function
        self.silu = nn.SiLU(inplace=True)

        self.in_features = ("dark3", "dark4", "dark5")

    def forward(self, input: List[torch.Tensor]):
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]

        [d2, d1, d0] = features
        x1 = self.c1(d0)
        x1_b = self.b1(x1)
        x1_m = self.silu(x1_b)
        fpn_out0 = x1_m

        f_out01 = self.u(fpn_out0)
        f_out0 = torch.cat([f_out01, d1], 1)
        # c3_p4
        x2 = self.c2(f_out0)
        x2_b = self.b2(x2)
        x2_m = self.silu(x2_b)

        x3 = self.c3(f_out0)
        x3_b = self.b3(x3)
        x3_m = self.silu(x3_b)

        bblock1_out = self.bblock1(x2_m)
        conc1 = torch.cat((bblock1_out, x3_m), dim=1)

        x4 = self.c4(conc1)
        x4_b = self.b4(x4)
        x4_m = self.silu(x4_b)

        # reduce_conv1
        x5 = self.c5(x4_m)
        x5_b = self.b5(x5)
        x5_m = self.silu(x5_b)
        fpn_out1 = x5_m

        f_out1 = self.u(fpn_out1)
        f_out1 = torch.cat([f_out1, d2], 1)

        # C3_p3
        x6 = self.c6(f_out1)
        x6_b = self.b6(x6)
        x6_m = self.silu(x6_b)

        x7 = self.c7(f_out1)
        x7_b = self.b7(x7)
        x7_m = self.silu(x7_b)

        bblock2_out = self.bblock2(x6_m)
        conc2 = torch.cat((bblock2_out, x7_m), dim=1)

        x8 = self.c8(conc2)
        x8_b = self.b8(x8)
        x8_m = self.silu(x8_b)
        pan_out2 = x8_m
        # bu_conv2
        x9 = self.c9(x8_m)
        x9_b = self.b9(x9)
        x9_m = self.silu(x9_b)
        p_out1 = x9_m

        p_out1 = torch.cat([p_out1, fpn_out1], 1)

        # C3_n3
        x10 = self.c10(p_out1)
        x10_b = self.b10(x10)
        x10_m = self.silu(x10_b)

        x11 = self.c11(p_out1)
        x11_b = self.b11(x11)
        x11_m = self.silu(x11_b)

        bblock3_out = self.bblock3(x10_m)
        conc3 = torch.cat((bblock3_out, x11_m), dim=1)

        x12 = self.c12(conc3)
        x12_b = self.b12(x12)
        x12_m = self.silu(x12_b)
        pan_out1 = x12_m

        # bu_conv1
        x13 = self.c13(x12_m)
        x13_b = self.b13(x13)
        x13_m = self.silu(x13_b)
        p_out0 = x13_m

        p_out0 = torch.cat([p_out0, fpn_out0], 1)

        # C3_n4
        x14 = self.c14(p_out0)
        x14_b = self.b14(x14)
        x14_m = self.silu(x14_b)

        x15 = self.c15(p_out0)
        x15_b = self.b15(x15)
        x15_m = self.silu(x15_b)

        bblock4_out = self.bblock4(x14_m)
        conc4 = torch.cat((bblock4_out, x15_m), dim=1)

        x16 = self.c16(conc4)
        x16_b = self.b16(x16)
        x16_m = self.silu(x16_b)
        pan_out0 = x16_m
        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
