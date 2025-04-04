# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.yolov12x.tt.aattn import TtnnAattn
from models.experimental.yolov12x.tt.common import TtYOLOv12xConv2D


class TtnnABlock:
    def __init__(self, device, parameter, conv_pt, dim=384, num_heads=12, mlp_ratio=1.2, area=1, is_bk_enabled=False):
        self.area = area
        self.device = device
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        all_head_dim = self.head_dim * self.num_heads
        self.scale = self.head_dim**-0.5

        self.attn = TtnnAattn(
            device, parameter.attn, conv_pt.attn, dim=dim, num_heads=num_heads, area=area, is_bk_enabled=False
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_0 = TtYOLOv12xConv2D(
            conv=parameter.mlp[0].conv,
            conv_pth=conv_pt.mlp[0].conv,
            device=device,
            activation="silu",
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )
        self.mlp_1 = TtYOLOv12xConv2D(
            conv=parameter.mlp[1].conv,
            conv_pth=conv_pt.mlp[1].conv,
            device=device,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )

    def __call__(self, x, i=0, j=0):
        x = x + self.attn(x, i=i, j=j)
        x_0 = self.mlp_0(x)
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x_1 = ttnn.sharded_to_interleaved(self.mlp_1(x_0), ttnn.L1_MEMORY_CONFIG)
        x = x + x_1
        return x
