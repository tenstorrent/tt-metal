# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_yolov10.tt.bottleneck import TtnnBottleNeck
from models.experimental.functional_yolov10.tt.common import Conv


class TtnnC2f:
    def __init__(self, shortcut=True, n=3, device=None, parameters=None, conv_pt=None):
        self.shortcut = shortcut
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
        )
        self.cv2 = Conv(device, parameters.cv2, self.conv_pt.cv2, auto_shard=True)

        self.m = [
            TtnnBottleNeck(self.shortcut, device=self.device, parameters=self.parameters[_], conv_pt=self.conv_pt.m[_])
            for _ in range(n)
        ]

    def __call__(self, x):
        x = self.cv1(x)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x1 = x[:, :, :, : x.shape[-1] // 2]
        x2 = x[:, :, :, x.shape[-1] // 2 : x.shape[-1]]
        x1 = ttnn.from_device(x1)
        x1 = ttnn.to_layout(x1, ttnn.TILE_LAYOUT)
        x1 = ttnn.to_dtype(x1, dtype=ttnn.bfloat8_b)
        x1 = ttnn.to_device(x1, self.device)

        x2 = ttnn.from_device(x2)
        x2 = ttnn.to_layout(x2, ttnn.TILE_LAYOUT)
        x2 = ttnn.to_dtype(x2, dtype=ttnn.bfloat8_b)
        x2 = ttnn.to_device(x2, self.device)
        y = [x1, x2]

        for m in self.m:
            out = m(y[-1])
            y.append(ttnn.to_layout(out, ttnn.TILE_LAYOUT))

        out = ttnn.concat(y, -1)

        out = self.cv2(out)

        return out
