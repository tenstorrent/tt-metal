# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_yolov10.tt.common import Conv
from models.experimental.functional_yolov10.tt.cib import TtnnCIB


class TtnnC2fCIB:
    def __init__(self, shortcut=True, n=3, device=None, parameters=None, conv_pt=None):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.shortcut = shortcut

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
            auto_shard=True,
        )
        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt.cv2,
            auto_shard=True,
        )

        self.m = [
            TtnnCIB(
                shortcut=self.shortcut,
                device=self.device,
                parameters=self.parameters[2],
                conv_pt=self.conv_pt.m[_],
            )
            for _ in range(n)
        ]

    def __call__(self, x):
        x = self.cv1(x)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x1 = x[:, :, :, : x.shape[-1] // 2]
        x2 = x[:, :, :, x.shape[-1] // 2 : x.shape[-1]]
        y = [ttnn.to_layout(x1, ttnn.TILE_LAYOUT), ttnn.to_layout(x2, ttnn.TILE_LAYOUT)]

        for m in self.m:
            out = m(y[-1])

            y.append(ttnn.to_layout(out, ttnn.TILE_LAYOUT))

        out = ttnn.concat(y, -1, memory_config=ttnn.L1_MEMORY_CONFIG)

        x = self.cv2(out)

        return x
