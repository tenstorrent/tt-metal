# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.vadv2.tt.common import TtConv2D


class TtConvModule:
    def __init__(self, conv_args, conv_pth, device=None):
        self.device = device
        self.conv = TtConv2D(conv_args.conv, conv_pth.conv, device=self.device, dealloc_act=True)

    def __call__(self, x):
        x = self.conv(x)
        return x[0]


class TtFPN:
    def __init__(self, conv_args, conv_pth, device):
        self.device = device
        self.lateral_convs = TtConvModule(conv_args.lateral_convs, conv_pth.lateral_convs, device=device)
        self.fpn_convs = TtConvModule(conv_args.fpn_convs, conv_pth.fpn_convs, device=device)

    def __call__(self, inputs):
        # Build laterals
        laterals = self.lateral_convs(inputs[0])
        # Apply FPN convs
        outs = self.fpn_convs(laterals)
        ttnn.deallocate(laterals)

        return tuple(outs)
