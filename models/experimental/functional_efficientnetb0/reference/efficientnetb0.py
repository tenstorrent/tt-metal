# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchview import draw_graph


class Efficientnetb0(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv_stem = Conv2dDynamicSamePadding(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self._bn0 = nn.BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        self._blocks0 = MBConvBlock([32, 32, 8, 32], [32, 8, 32, 16], [3, 1], is_depthwise_first=True)
        self._blocks1 = MBConvBlock([16, 96, 96, 4, 96], [96, 96, 4, 96, 24], [3, 2])
        self._blocks2 = MBConvBlock([24, 144, 144, 6, 144], [144, 144, 6, 144, 24], [3, 1])
        self._blocks3 = MBConvBlock([24, 144, 144, 6, 144], [144, 144, 6, 144, 40], [5, 2])
        self._blocks4 = MBConvBlock([40, 240, 240, 10, 240], [240, 240, 10, 240, 40], [5, 1])
        self._blocks5 = MBConvBlock([40, 240, 240, 10, 240], [240, 240, 10, 240, 80], [3, 2])
        self._blocks6 = MBConvBlock([80, 480, 480, 20, 480], [480, 480, 20, 480, 80], [3, 1])
        self._blocks7 = MBConvBlock([80, 480, 480, 20, 480], [480, 480, 20, 480, 80], [3, 1])
        self._blocks8 = MBConvBlock([80, 480, 480, 20, 480], [480, 480, 20, 480, 112], [5, 1])
        self._blocks9 = MBConvBlock([112, 672, 672, 28, 672], [672, 672, 28, 672, 112], [5, 1])
        self._blocks10 = MBConvBlock([112, 672, 672, 28, 672], [672, 672, 28, 672, 112], [5, 1])
        self._blocks11 = MBConvBlock([112, 672, 672, 28, 672], [672, 672, 28, 672, 192], [5, 2])
        self._blocks12 = MBConvBlock([192, 1152, 1152, 48, 1152], [1152, 1152, 48, 1152, 192], [5, 1])
        self._blocks13 = MBConvBlock([192, 1152, 1152, 48, 1152], [1152, 1152, 48, 1152, 192], [5, 1])
        self._blocks14 = MBConvBlock([192, 1152, 1152, 48, 1152], [1152, 1152, 48, 1152, 192], [5, 1])
        self._blocks15 = MBConvBlock([192, 1152, 1152, 48, 1152], [1152, 1152, 48, 1152, 320], [3, 1])

        self._conv_head = Conv2dDynamicSamePadding(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self._avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self._bn1 = nn.BatchNorm2d(
            1280, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True
        )
        self._fc = nn.Linear(in_features=1280, out_features=1000, bias=True)

    def forward(self, x):
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = x * torch.sigmoid(x)
        x = self._blocks0(x)
        x_1 = self._blocks1(x)
        x = self._blocks2(x_1)
        x = x + x_1
        x_3 = self._blocks3(x)
        x = self._blocks4(x_3)
        x = x + x_3
        x_5 = self._blocks5(x)
        x = self._blocks6(x_5)
        x_7_in = x + x_5
        x = self._blocks7(x_7_in)
        x = x_7_in + x
        x_8 = self._blocks8(x)
        x = self._blocks9(x_8)
        x_10_in = x + x_8
        x = self._blocks10(x_10_in)
        x = x + x_10_in
        x_11 = self._blocks11(x)
        x = self._blocks12(x_11)
        x_13_in = x + x_11
        x = self._blocks13(x_13_in)
        x_14_in = x + x_13_in
        x = self._blocks14(x)
        x = x_14_in + x
        x = self._blocks15(x)
        x = self._conv_head(x)
        x = self._bn1(x)
        x = x * torch.sigmoid(x)
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self._fc(x)

        return x


class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
    The padding is operated in forward function by calculating dynamically.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), dilation=(1, 1), groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        # self.conv = nn.Conv2d(
        #     in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups, bias=bias
        # )

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class MBConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, params, is_depthwise_first=False):
        super().__init__()
        self.is_depthwise_first = is_depthwise_first
        self._avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        if is_depthwise_first:
            self._depthwise_conv = Conv2dDynamicSamePadding(
                in_channel[0],
                out_channel[0],
                kernel_size=(params[0], params[0]),
                stride=(params[1], params[1]),
                groups=out_channel[0],
                bias=False,
            )
            self._bn1 = nn.BatchNorm2d(
                out_channel[0], eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True
            )
            self._se_reduce = Conv2dDynamicSamePadding(in_channel[1], out_channel[1], kernel_size=(1, 1), stride=(1, 1))
            self._se_expand = Conv2dDynamicSamePadding(in_channel[2], out_channel[2], kernel_size=(1, 1), stride=(1, 1))
            self._project_conv = Conv2dDynamicSamePadding(
                in_channel[3], out_channel[3], kernel_size=(1, 1), stride=(1, 1), bias=False
            )
            self._bn2 = nn.BatchNorm2d(
                out_channel[3], eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True
            )

        else:
            self._expand_conv = Conv2dDynamicSamePadding(
                in_channel[0], out_channel[0], kernel_size=(1, 1), stride=(1, 1), bias=False
            )
            self._bn0 = nn.BatchNorm2d(
                out_channel[0], eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True
            )
            self._depthwise_conv = Conv2dDynamicSamePadding(
                in_channel[1],
                out_channel[1],
                kernel_size=(params[0], params[0]),
                stride=(params[1], params[1]),
                groups=out_channel[1],
                bias=False,
            )
            self._bn1 = nn.BatchNorm2d(
                out_channel[1], eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True
            )
            self._se_reduce = Conv2dDynamicSamePadding(in_channel[2], out_channel[2], kernel_size=(1, 1), stride=(1, 1))
            self._se_expand = Conv2dDynamicSamePadding(in_channel[3], out_channel[3], kernel_size=(1, 1), stride=(1, 1))
            self._project_conv = Conv2dDynamicSamePadding(
                in_channel[4], out_channel[4], kernel_size=(1, 1), stride=(1, 1), bias=False
            )
            self._bn2 = nn.BatchNorm2d(
                out_channel[4], eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True
            )

    def forward(self, x):
        if not self.is_depthwise_first:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = x * torch.sigmoid(x)
        print(x.shape)
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        torch.save(x, "torch_out.pt")
        x = x * torch.sigmoid(x)
        mul1 = x
        x = self._avg_pooling(x)
        x = self._se_reduce(x)
        x = x * torch.sigmoid(x)
        x = self._se_expand(x)
        x = torch.sigmoid(x)
        x = x * mul1
        x = self._project_conv(x)
        x = self._bn2(x)

        return x
