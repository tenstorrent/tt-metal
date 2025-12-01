# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.vadv2.tt.common import TtConv2D

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class TtConvModule:
    def __init__(self, conv_args, conv_pth, device=None):
        self.device = device
        self.conv = TtConv2D(conv_args.conv, conv_pth.conv, device=self.device, dealloc_act=True)

    def __call__(self, x):
        if use_signpost:
            signpost(header="TtConvModule_call_start")
        x = self.conv(x)
        if use_signpost:
            signpost(header="TtConvModule_call_end")
        return x[0]


class TtFPN:
    def __init__(self, conv_args, conv_pth, device):
        self.device = device
        self.lateral_convs = TtConvModule(conv_args.lateral_convs, conv_pth.lateral_convs, device=device)
        self.fpn_convs = TtConvModule(conv_args.fpn_convs, conv_pth.fpn_convs, device=device)

    def __call__(self, inputs):
        if use_signpost:
            signpost(header="TtFPN_call_start")
        # Build laterals
        laterals = self.lateral_convs(inputs[0])
        # Apply FPN convs
        outs = self.fpn_convs(laterals)
        ttnn.deallocate(laterals)
        if use_signpost:
            signpost(header="TtFPN_call_end")
        return tuple(outs)
