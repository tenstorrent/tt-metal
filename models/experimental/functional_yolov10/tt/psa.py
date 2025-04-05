# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_yolov10.tt.attention import TtnnAttention
from models.experimental.functional_yolov10.tt.common import Conv


class TtnnPSA:
    def __init__(self, device=None, parameters=None, conv_pt=None):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
        )

        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt.cv2,
        )

        self.attn = TtnnAttention(
            dim=320,
            num_heads=5,
            attn_ratio=0.5,
            device=self.device,
            parameters=self.parameters.attn,
            conv_pt=self.conv_pt.attn,
        )

        self.ffn_0 = Conv(
            device,
            parameters.ffn[0],
            self.conv_pt.ffn[0],
        )

        self.ffn_1 = Conv(
            device,
            parameters.ffn[1],
            self.conv_pt.ffn[1],
            enable_identity=True,
        )

    def __call__(self, x):
        x = self.cv1(x)
        a = x[:, :, :, : x.shape[-1] // 2]
        b = x[:, :, :, x.shape[-1] // 2 : x.shape[-1]]
        out = self.attn(b)

        b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)
        b = b + out

        out = self.ffn_0(b)
        out = self.ffn_1(out)

        b = b + out

        out = ttnn.concat([a, b], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        out = self.cv2(out)
        ttnn.deallocate(a)
        ttnn.deallocate(b)

        return out
