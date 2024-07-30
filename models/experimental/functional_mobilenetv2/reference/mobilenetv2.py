# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn


class Mobilenetv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.b1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6(inplace=True)

        self.c2 = nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False)
        self.b2 = nn.BatchNorm2d(32)

        self.c3 = nn.Conv2d(32, 16, 1, 1, bias=False)
        self.b3 = nn.BatchNorm2d(16)

        self.c4 = nn.Conv2d(16, 96, 1, 1, bias=False)
        self.b4 = nn.BatchNorm2d(96)

        self.c5 = nn.Conv2d(96, 96, 3, 2, 1, groups=96, bias=False)
        self.b5 = nn.BatchNorm2d(96)

        self.c6 = nn.Conv2d(96, 24, 1, 1, bias=False)
        self.b6 = nn.BatchNorm2d(24)

        self.c7 = nn.Conv2d(24, 144, 1, 1, bias=False)
        self.b7 = nn.BatchNorm2d(144)

        self.c8 = nn.Conv2d(144, 144, 3, 1, 1, groups=144, bias=False)
        self.b8 = nn.BatchNorm2d(144)

        self.c9 = nn.Conv2d(144, 24, 1, 1, bias=False)
        self.b9 = nn.BatchNorm2d(24)

        self.c10 = nn.Conv2d(24, 144, 1, 1, bias=False)
        self.b10 = nn.BatchNorm2d(144)

        self.c11 = nn.Conv2d(144, 144, 3, 2, 1, groups=144, bias=False)
        self.b11 = nn.BatchNorm2d(144)

        self.c12 = nn.Conv2d(144, 32, 1, 1, bias=False)
        self.b12 = nn.BatchNorm2d(32)

        self.c13 = nn.Conv2d(32, 192, 1, 1, bias=False)
        self.b13 = nn.BatchNorm2d(192)

        self.c14 = nn.Conv2d(192, 192, 3, 1, 1, groups=192, bias=False)
        self.b14 = nn.BatchNorm2d(192)

        self.c15 = nn.Conv2d(192, 32, 1, 1, bias=False)
        self.b15 = nn.BatchNorm2d(32)

        self.c16 = nn.Conv2d(32, 192, 1, 1, bias=False)
        self.b16 = nn.BatchNorm2d(192)

        self.c17 = nn.Conv2d(192, 192, 3, 1, 1, groups=192, bias=False)
        self.b17 = nn.BatchNorm2d(192)

        self.c18 = nn.Conv2d(192, 32, 1, 1, bias=False)
        self.b18 = nn.BatchNorm2d(32)

        self.c19 = nn.Conv2d(32, 192, 1, 1, bias=False)
        self.b19 = nn.BatchNorm2d(192)

        self.c20 = nn.Conv2d(192, 192, 3, 2, 1, groups=192, bias=False)
        self.b20 = nn.BatchNorm2d(192)

        self.c21 = nn.Conv2d(192, 64, 1, 1, bias=False)
        self.b21 = nn.BatchNorm2d(64)

        self.c22 = nn.Conv2d(64, 384, 1, 1, bias=False)
        self.b22 = nn.BatchNorm2d(384)

        self.c23 = nn.Conv2d(384, 384, 3, 1, 1, groups=384, bias=False)
        self.b23 = nn.BatchNorm2d(384)

        self.c24 = nn.Conv2d(384, 64, 1, 1, bias=False)
        self.b24 = nn.BatchNorm2d(64)

        self.c25 = nn.Conv2d(64, 384, 1, 1, bias=False)
        self.b25 = nn.BatchNorm2d(384)

        self.c26 = nn.Conv2d(384, 384, 3, 1, 1, groups=384, bias=False)
        self.b26 = nn.BatchNorm2d(384)

        self.c27 = nn.Conv2d(384, 64, 1, 1, bias=False)
        self.b27 = nn.BatchNorm2d(64)

        self.c28 = nn.Conv2d(64, 384, 1, 1, bias=False)
        self.b28 = nn.BatchNorm2d(384)

        self.c29 = nn.Conv2d(384, 384, 3, 1, 1, groups=384, bias=False)
        self.b29 = nn.BatchNorm2d(384)

        self.c30 = nn.Conv2d(384, 64, 1, 1, bias=False)
        self.b30 = nn.BatchNorm2d(64)

        self.c31 = nn.Conv2d(64, 384, 1, 1, bias=False)
        self.b31 = nn.BatchNorm2d(384)

        self.c32 = nn.Conv2d(384, 384, 3, 1, 1, groups=384, bias=False)
        self.b32 = nn.BatchNorm2d(384)

        self.c33 = nn.Conv2d(384, 96, 1, 1, bias=False)
        self.b33 = nn.BatchNorm2d(96)

        self.c34 = nn.Conv2d(96, 576, 1, 1, bias=False)
        self.b34 = nn.BatchNorm2d(576)

        self.c35 = nn.Conv2d(576, 576, 3, 1, 1, groups=576, bias=False)
        self.b35 = nn.BatchNorm2d(576)

        self.c36 = nn.Conv2d(576, 96, 1, 1, bias=False)
        self.b36 = nn.BatchNorm2d(96)

        self.c37 = nn.Conv2d(96, 576, 1, 1, bias=False)
        self.b37 = nn.BatchNorm2d(576)

        self.c38 = nn.Conv2d(576, 576, 3, 1, 1, groups=576, bias=False)
        self.b38 = nn.BatchNorm2d(576)

        self.c39 = nn.Conv2d(576, 96, 1, 1, bias=False)
        self.b39 = nn.BatchNorm2d(96)

        self.c40 = nn.Conv2d(96, 576, 1, 1, bias=False)
        self.b40 = nn.BatchNorm2d(576)

        self.c41 = nn.Conv2d(576, 576, 3, 2, 1, groups=576, bias=False)
        self.b41 = nn.BatchNorm2d(576)

        self.c42 = nn.Conv2d(576, 160, 1, 1, bias=False)
        self.b42 = nn.BatchNorm2d(160)

        self.c43 = nn.Conv2d(160, 960, 1, 1, bias=False)
        self.b43 = nn.BatchNorm2d(960)

        self.c44 = nn.Conv2d(960, 960, 3, 1, 1, groups=960, bias=False)
        self.b44 = nn.BatchNorm2d(960)

        self.c45 = nn.Conv2d(960, 160, 1, 1, bias=False)
        self.b45 = nn.BatchNorm2d(160)

        self.c46 = nn.Conv2d(160, 960, 1, 1, bias=False)
        self.b46 = nn.BatchNorm2d(960)

        self.c47 = nn.Conv2d(960, 960, 3, 1, 1, groups=960, bias=False)
        self.b47 = nn.BatchNorm2d(960)

        self.c48 = nn.Conv2d(960, 160, 1, 1, bias=False)
        self.b48 = nn.BatchNorm2d(160)

        self.c49 = nn.Conv2d(160, 960, 1, 1, bias=False)
        self.b49 = nn.BatchNorm2d(960)

        self.c50 = nn.Conv2d(960, 960, 3, 1, 1, groups=960, bias=False)
        self.b50 = nn.BatchNorm2d(960)

        self.c51 = nn.Conv2d(960, 320, 1, 1, bias=False)
        self.b51 = nn.BatchNorm2d(320)

        self.c52 = nn.Conv2d(320, 1280, 1, 1, bias=False)
        self.b52 = nn.BatchNorm2d(1280)

        self.l1 = nn.Linear(in_features=1280, out_features=1000)

    def forward(self, input: torch.Tensor):
        x1 = self.c1(input)
        x1_b = self.b1(x1)
        x1_m = self.relu(x1_b)

        x2 = self.c2(x1_m)
        x2_b = self.b2(x2)
        x2_m = self.relu(x2_b)

        x3 = self.c3(x2_m)
        x3_b = self.b3(x3)

        x4 = self.c4(x3_b)
        x4_b = self.b4(x4)
        x4_m = self.relu(x4_b)

        x5 = self.c5(x4_m)
        x5_b = self.b5(x5)
        x5_m = self.relu(x5_b)

        x6 = self.c6(x5_m)
        x6_b = self.b6(x6)

        x7 = self.c7(x6_b)
        x7_b = self.b7(x7)
        x7_m = self.relu(x7_b)

        x8 = self.c8(x7_m)
        x8_b = self.b8(x8)
        x8_m = self.relu(x8_b)

        x9 = self.c9(x8_m)
        x9_b = self.b9(x9)

        a1 = x9_b + x6_b

        x10 = self.c10(a1)
        x10_b = self.b10(x10)
        x10_m = self.relu(x10_b)

        x11 = self.c11(x10_m)
        x11_b = self.b11(x11)
        x11_m = self.relu(x11_b)

        x12 = self.c12(x11_m)
        x12_b = self.b12(x12)

        x13 = self.c13(x12_b)
        x13_b = self.b13(x13)
        x13_m = self.relu(x13_b)

        x14 = self.c14(x13_m)
        x14_b = self.b14(x14)
        x14_m = self.relu(x14_b)

        x15 = self.c15(x14_m)
        x15_b = self.b15(x15)

        a2 = x15_b + x12_b

        x16 = self.c16(a2)
        x16_b = self.b16(x16)
        x16_m = self.relu(x16_b)

        x17 = self.c17(x16_m)
        x17_b = self.b17(x17)
        x17_m = self.relu(x17_b)

        x18 = self.c18(x17_m)
        x18_b = self.b18(x18)

        a3 = a2 + x18_b

        x19 = self.c19(a3)
        x19_b = self.b19(x19)
        x19_m = self.relu(x19_b)

        x20 = self.c20(x19_m)
        x20_b = self.b20(x20)
        x20_m = self.relu(x20_b)

        x21 = self.c21(x20_m)
        x21_b = self.b21(x21)

        x22 = self.c22(x21_b)
        x22_b = self.b22(x22)
        x22_m = self.relu(x22_b)

        x23 = self.c23(x22_m)
        x23_b = self.b23(x23)
        x23_m = self.relu(x23_b)

        x24 = self.c24(x23_m)
        x24_b = self.b24(x24)

        a4 = x21_b + x24_b

        x25 = self.c25(a4)
        x25_b = self.b25(x25)
        x25_m = self.relu(x25_b)

        x26 = self.c26(x25_m)
        x26_b = self.b26(x26)
        x26_m = self.relu(x26_b)

        x27 = self.c27(x26_m)
        x27_b = self.b27(x27)

        a5 = a4 + x27_b

        x28 = self.c28(a5)
        x28_b = self.b28(x28)
        x28_m = self.relu(x28_b)

        x29 = self.c29(x28_m)
        x29_b = self.b29(x29)
        x29_m = self.relu(x29_b)

        x30 = self.c30(x29_m)
        x30_b = self.b30(x30)

        a6 = a5 + x30_b

        x31 = self.c31(a6)
        x31_b = self.b31(x31)
        x31_m = self.relu(x31_b)

        x32 = self.c32(x31_m)
        x32_b = self.b32(x32)
        x32_m = self.relu(x32_b)

        x33 = self.c33(x32_m)
        x33_b = self.b33(x33)

        x34 = self.c34(x33_b)
        x34_b = self.b34(x34)
        x34_m = self.relu(x34_b)

        x35 = self.c35(x34_m)
        x35_b = self.b35(x35)
        x35_m = self.relu(x35_b)

        x36 = self.c36(x35_m)
        x36_b = self.b36(x36)

        a7 = x33_b + x36_b

        x37 = self.c37(a7)
        x37_b = self.b37(x37)
        x37_m = self.relu(x37_b)

        x38 = self.c38(x37_m)
        x38_b = self.b38(x38)
        x38_m = self.relu(x38_b)

        x39 = self.c39(x38_m)
        x39_b = self.b39(x39)

        a8 = a7 + x39_b

        x40 = self.c40(a8)
        x40_b = self.b40(x40)
        x40_m = self.relu(x40_b)

        x41 = self.c41(x40_m)
        x41_b = self.b41(x41)
        x41_m = self.relu(x41_b)

        x42 = self.c42(x41_m)
        x42_b = self.b42(x42)

        x43 = self.c43(x42_b)
        x43_b = self.b43(x43)
        x43_m = self.relu(x43_b)

        x44 = self.c44(x43_m)
        x44_b = self.b44(x44)
        x44_m = self.relu(x44_b)
        print(x44_m.shape)
        x45 = self.c45(x44_m)
        x45_b = self.b45(x45)
        print(x45_b.shape)
        a9 = x45_b + x42_b

        x46 = self.c46(a9)
        x46_b = self.b46(x46)
        x46_m = self.relu(x46_b)

        x47 = self.c47(x46_m)
        x47_b = self.b47(x47)
        x47_m = self.relu(x47_b)

        x48 = self.c48(x47_m)
        x48_b = self.b48(x48)

        a10 = a9 + x48_b

        x49 = self.c49(a10)
        x49_b = self.b49(x49)
        x49_m = self.relu(x49_b)

        x50 = self.c50(x49_m)
        x50_b = self.b50(x50)
        x50_m = self.relu(x50_b)

        x51 = self.c51(x50_m)
        x51_b = self.b51(x51)

        x52 = self.c52(x51_b)
        x52_b = self.b52(x52)
        x52_m = self.relu(x52_b)
        x = nn.functional.adaptive_avg_pool2d(x52_m, (1, 1))
        x = torch.flatten(x, 1)
        return x
