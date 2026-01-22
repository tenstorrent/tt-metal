# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import ttnn

from models.tt_cnn.tt.builder import TtConv2d, Conv2dConfiguration


class TtConvModule:
    def __init__(self, config: Conv2dConfiguration, device: ttnn.Device):
        self.device = device
        self.conv = TtConv2d(config, device)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.conv(x)


class TtFPN:
    def __init__(
        self,
        lateral_conv_config: Conv2dConfiguration,
        fpn_conv_config: Conv2dConfiguration,
        device: ttnn.Device,
    ):
        self.device = device
        self.lateral_conv_config = lateral_conv_config
        self.fpn_conv_config = fpn_conv_config
        self.lateral_convs = TtConvModule(lateral_conv_config, device=device)
        self.fpn_convs = TtConvModule(fpn_conv_config, device=device)

    def __call__(self, inputs: list) -> Tuple[ttnn.Tensor, ...]:
        laterals = self.lateral_convs(inputs[0])
        laterals = ttnn.to_memory_config(laterals, ttnn.DRAM_MEMORY_CONFIG)

        outs = self.fpn_convs(laterals)
        ttnn.deallocate(laterals)

        return (outs,)
