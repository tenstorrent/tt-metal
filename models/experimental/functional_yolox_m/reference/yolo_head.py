# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from typing import List
import torch


class YOLOXHead(nn.Module):
    def __init__(self):
        super(YOLOXHead, self).__init__()

        # cls_convs
        self.c1 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.silu = nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.c2 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c3 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.b3 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c4 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.b4 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c5 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.b5 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c6 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.b6 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        # reg_convs
        self.c7 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.b7 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c8 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.b8 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c9 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.b9 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c10 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.b10 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c11 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.b11 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c12 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.b12 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        # cls_preds
        self.c13 = nn.Conv2d(192, 80, kernel_size=1, stride=1, bias=True)
        self.c14 = nn.Conv2d(192, 80, kernel_size=1, stride=1, bias=True)
        self.c15 = nn.Conv2d(192, 80, kernel_size=1, stride=1, bias=True)

        # reg_preds
        self.c16 = nn.Conv2d(192, 4, kernel_size=1, stride=1, bias=True)
        self.c17 = nn.Conv2d(192, 4, kernel_size=1, stride=1, bias=True)
        self.c18 = nn.Conv2d(192, 4, kernel_size=1, stride=1, bias=True)

        # obj_preds
        self.c19 = nn.Conv2d(192, 1, kernel_size=1, stride=1, bias=True)
        self.c20 = nn.Conv2d(192, 1, kernel_size=1, stride=1, bias=True)
        self.c21 = nn.Conv2d(192, 1, kernel_size=1, stride=1, bias=True)

        # stems
        self.c22 = nn.Conv2d(192, 192, kernel_size=1, stride=1, bias=False)
        self.b22 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c23 = nn.Conv2d(384, 192, kernel_size=1, stride=1, bias=False)
        self.b23 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

        self.c24 = nn.Conv2d(768, 192, kernel_size=1, stride=1, bias=False)
        self.b24 = nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

    def forward(self, x: List[torch.Tensor]):
        outputs = []

        # output 1 ops
        x22 = self.c22(x[0])
        x22_b = self.b22(x22)
        x22_m = self.silu(x22_b)

        cls_x = x22_m
        reg_x = x22_m

        x1 = self.c1(cls_x)
        x1_b = self.b1(x1)
        x1_m = self.silu(x1_b)

        x2 = self.c2(x1_m)
        x2_b = self.b2(x2)
        x2_m = self.silu(x2_b)

        cls_feat = x2_m
        cls_output = self.c13(cls_feat)
        cls_output = self.sigmoid(cls_output)

        x7 = self.c7(reg_x)
        x7_b = self.b7(x7)
        x7_m = self.silu(x7_b)

        x8 = self.c8(x7_m)
        x8_b = self.b8(x8)
        x8_m = self.silu(x8_b)

        reg_feat = x8_m
        reg_output = self.c16(reg_feat)

        obj_output = self.c19(reg_feat)
        obj_output = self.sigmoid(obj_output)

        output = torch.cat([reg_output, obj_output, cls_output], 1)
        outputs.append(output)
        # output 2 ops
        x23 = self.c23(x[1])
        x23_b = self.b23(x23)
        x23_m = self.silu(x23_b)

        cls_x = x23_m
        reg_x = x23_m

        x3 = self.c3(cls_x)
        x3_b = self.b3(x3)
        x3_m = self.silu(x3_b)

        x4 = self.c4(x3_m)
        x4_b = self.b4(x4)
        x4_m = self.silu(x4_b)

        cls_feat = x4_m
        cls_output = self.c14(cls_feat)
        cls_output = self.sigmoid(cls_output)

        x9 = self.c9(reg_x)
        x9_b = self.b9(x9)
        x9_m = self.silu(x9_b)

        x10 = self.c10(x9_m)
        x10_b = self.b10(x10)
        x10_m = self.silu(x10_b)

        reg_feat = x10_m
        reg_output = self.c17(reg_feat)
        obj_output = self.c20(reg_feat)
        obj_output = self.sigmoid(obj_output)

        output = torch.cat([reg_output, obj_output, cls_output], 1)
        outputs.append(output)

        # output 3 ops
        x24 = self.c24(x[2])
        x24_b = self.b24(x24)
        x24_m = self.silu(x24_b)

        cls_x = x24_m
        reg_x = x24_m

        x5 = self.c5(cls_x)
        x5_b = self.b5(x5)
        x5_m = self.silu(x5_b)

        x6 = self.c6(x5_m)
        x6_b = self.b6(x6)
        x6_m = self.silu(x6_b)

        cls_feat = x6_m
        cls_output = self.c15(cls_feat)
        cls_output = self.sigmoid(cls_output)

        x11 = self.c11(reg_x)
        x11_b = self.b11(x11)
        x11_m = self.silu(x11_b)

        x12 = self.c12(x11_m)
        x12_b = self.b12(x12)
        x12_m = self.silu(x12_b)

        reg_feat = x12_m
        reg_output = self.c18(reg_feat)
        obj_output = self.c21(reg_feat)
        obj_output = self.sigmoid(obj_output)

        output = torch.cat([reg_output, obj_output, cls_output], 1)
        outputs.append(output)

        return outputs
