# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.yolov12x.tt.aattn import AAttn
from models.experimental.yolov12x.tt.common import Yolov12x_Conv2D


class ABlock:
    def __init__(self, device, parameter, conv_pt, dim=384, num_heads=12, mlp_ratio=1.2, area=1, is_bk_enabled=False):
        self.area = area
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        all_head_dim = self.head_dim * self.num_heads
        self.scale = self.head_dim**-0.5

        self.attn = AAttn(
            device, parameter.attn, conv_pt.attn, dim=dim, num_heads=num_heads, area=area, is_bk_enabled=False
        )
        mlp_hidden_dim = dim * mlp_ratio
        self.mlp_0 = Yolov12x_Conv2D(
            conv=parameter.mlp[0].conv, conv_pth=conv_pt.mlp[0].conv, device=device, activation="silu"
        )
        self.mlp_1 = Yolov12x_Conv2D(
            conv=parameter.mlp[1].conv, conv_pth=conv_pt.mlp[1].conv, device=device, activation="silu"
        )

    def __call__(self, x):
        print(f" x memroy config: {x.memory_config()}")
        print(f" slef.attn(x): {self.attn(x).memory_config()}")

        x = x + self.attn(x)
        x_0 = self.mlp_0(x)

        x = x + self.mlp_1(x_0)
        return x
