# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

##########################################################################
# Adapted from BEVFormer (https://github.com/fundamentalvision/BEVFormer).
# Original work Copyright (c) OpenMMLab.
# Modified by Zhiqi Li.
# Licensed under the Apache License, Version 2.0.
##########################################################################

import torch.nn as nn


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, act_cfg=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.with_bias = bias
        self.with_activation = act_cfg is not None

        if self.with_activation:
            self.activate = nn.ReLU(inplace=True)
        else:
            self.activate = None

        if not bias:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activate is not None:
            x = self.activate(x)
        return x
