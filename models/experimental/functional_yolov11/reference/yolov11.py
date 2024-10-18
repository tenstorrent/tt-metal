# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=1, stride=1, pad=0, dilation=1, groups=1, enable_act=True):
        super().__init__()
        self.enable_act = enable_act
        if enable_act:
            self.conv = nn.Conv2d(
                in_channel,
                out_channel,
                kernel,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.act = nn.SiLU(inplace=True)
        else:
            self.conv = nn.Conv2d(
                in_channel,
                out_channel,
                kernel,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=groups,
                bias=False,
            )

    def forward(self, x):
        if self.enable_act:
            x = self.conv(x)
            x = self.act(x)
        else:
            x = self.conv(x)
        return x


class Bottleneck(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel=[1, 1], stride=[1, 1], pad=[0, 0], dilation=[1, 1], groups=[1, 1]
    ):
        super().__init__(in_channel, out_channel, kernel, stride=stride, padding=pad, dilation=dilation, groups=groups)
        self.cv1 = Conv(in_channel, out_channel, kernel, stride=stride, padding=pad, dilation=dilation, groups=groups)
        self.cv2 = Conv(in_channel, out_channel, kernel, stride=stride, padding=pad, dilation=dilation, groups=groups)

    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        return x


class SPPF(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel=[1, 1],
        stride=[1, 1],
        pad=[0, 0],
        dilation=[1, 1],
        groups=[1, 1],
        m_kernel=5,
        padding=2,
    ):
        super().__init__(in_channel, out_channel, kernel, stride=stride, padding=pad, dilation=dilation, groups=groups)
        self.cv1 = Conv(in_channel, out_channel, kernel, stride=stride, padding=pad, dilation=dilation, groups=groups)
        self.cv2 = Conv(in_channel, out_channel, kernel, stride=stride, padding=pad, dilation=dilation, groups=groups)
        self.m = nn.Maxpool2d(kernel_size=m_kernel, stride=1, padding=padding)

    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.m(x)
        return x


class C3K(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, pad, dilation, groups):
        self.cv1 = Conv(
            in_channel[0],
            out_channel[0],
            kernel[0],
            stride=stride[0],
            padding=pad[0],
            dilation=dilation[0],
            groups=groups[0],
        )
        self.cv2 = Conv(
            in_channel[1],
            out_channel[1],
            kernel[1],
            stride=stride[1],
            padding=pad[1],
            dilation=dilation[1],
            groups=groups[1],
        )
        self.cv3 = Conv(
            in_channel[2],
            out_channel[2],
            kernel[2],
            stride=stride[2],
            padding=pad[2],
            dilation=dilation[2],
            groups=groups[2],
        )
        self.k1 = Bottleneck(
            in_channel[3:5],
            out_channel[3:5],
            kernel[3:5],
            stride=stride[3:5],
            padding=pad[3:5],
            dilation=dilation[3:5],
            groups=groups[3:5],
        )
        self.k2 = Bottleneck(
            in_channel[5:7],
            out_channel[5:7],
            kernel[5:7],
            stride=stride[5:7],
            padding=pad[5:7],
            dilation=dilation[5:7],
            groups=groups[5:7],
        )

    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.cv3(x)
        x = self.k1(x)
        x = self.k2(x)


class C3k2(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, pad, dilation, groups, is_bk_enabled=False):
        self.is_bk_enabled = is_bk_enabled
        if is_bk_enabled:
            self.cv1 = Conv(
                in_channel[0],
                out_channel[0],
                kernel[0],
                stride=stride[0],
                padding=pad[0],
                dilation=dilation[0],
                groups=groups[0],
            )
            self.cv2 = Conv(
                in_channel[1],
                out_channel[1],
                kernel[1],
                stride=stride[1],
                padding=pad[1],
                dilation=dilation[1],
                groups=groups[1],
            )
            self.k = Bottleneck(
                in_channel[2:4],
                out_channel[2:4],
                kernel[2:4],
                stride=stride[2:4],
                padding=pad[2:4],
                dilation=dilation[2:4],
                groups=groups[2:4],
            )
        else:
            self.cv1 = Conv(
                in_channel[0],
                out_channel[0],
                kernel[0],
                stride=stride[0],
                padding=pad[0],
                dilation=dilation[0],
                groups=groups[0],
            )
            self.cv2 = Conv(
                in_channel[1],
                out_channel[1],
                kernel[1],
                stride=stride[1],
                padding=pad[1],
                dilation=dilation[1],
                groups=groups[1],
            )
            self.c3k = C3K(
                in_channel[2:9], out_channel[2:9], kernel[2:9], stride[2:9], pad[2:9], dilation[2:9], groups[2:9]
            )

    def forward(self, x):
        if self.is_bk_enabled:
            x = self.cv1(x)
            x = self.cv2(x)
            x = self.k1(x)
        else:
            x = self.cv1(x)
            x = self.cv2(x)
            x = self.c3k(x)


class Attention(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, pad, dilation, groups):
        super().__init__()
        self.qkv = Conv(
            in_channel[0],
            out_channel[0],
            kernel[0],
            stride=stride[0],
            padding=pad[0],
            dilation=dilation[0],
            groups=groups[0],
            enable_act=False,
        )
        self.proj = Conv(
            in_channel[1],
            out_channel[1],
            kernel[1],
            stride=stride[1],
            padding=pad[1],
            dilation=dilation[1],
            groups=groups[1],
            enable_act=False,
        )
        self.pe = Conv(
            in_channel[2],
            out_channel[2],
            kernel[2],
            stride=stride[2],
            padding=pad[2],
            dilation=dilation[2],
            groups=groups[2],
            enable_act=False,
        )

    def forward(self, x):
        x = self.qkv(x)
        x = self.proj(x)
        x = self.pe(x)
        return x


class PSABlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, pad, dilation, groups):
        self.attn = Attention(
            in_channel[0:3], out_channel[0:3], kernel[0:3], stride[0:3], pad[0:3], dilation[0:3], groups[0:3]
        )
        self.ffn_conv1 = Conv(
            in_channel[3],
            out_channel[3],
            kernel[3],
            stride=stride[3],
            padding=pad[3],
            dilation=dilation[3],
            groups=groups[3],
        )
        self.ffn_conv2 = Conv(
            in_channel[4],
            out_channel[4],
            kernel[4],
            stride=stride[4],
            padding=pad[4],
            dilation=dilation[4],
            groups=groups[4],
            enable_act=False,
        )

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn_conv1(x)
        x = self.ffn_conv2(x)
        return x


class C2PSA(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, pad, dilation, groups):
        self.cv1 = Conv(
            in_channel[0],
            out_channel[0],
            kernel[0],
            stride=stride[0],
            padding=pad[0],
            dilation=dilation[0],
            groups=groups[0],
        )
        self.cv2 = Conv(
            in_channel[1],
            out_channel[1],
            kernel[1],
            stride=stride[1],
            padding=pad[1],
            dilation=dilation[1],
            groups=groups[1],
        )
        self.psablock = PSABlock(
            in_channel[2:7],
            out_channel[2:7],
            kernel[2:7],
            stride=stride[2:7],
            padding=pad[2:7],
            dilation=dilation[2:7],
            groups=groups[2:7],
        )

    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.psablock(x)


class YoloV11(nn.Module):
    def __init__(self):
        super().__init__()
