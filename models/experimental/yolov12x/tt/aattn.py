# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.yolov12x.tt.common import TtYOLOv12xConv2D


class TtnnAattn:
    def __init__(self, device, parameter, conv_pt, dim=384, num_heads=8, area=1, is_bk_enabled=False):
        self.area = area
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        all_head_dim = self.head_dim * self.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = TtYOLOv12xConv2D(conv=parameter.qkv.conv, conv_pth=conv_pt.qkv.conv, device=device)
        self.proj = TtYOLOv12xConv2D(
            conv=parameter.proj.conv,
            conv_pth=conv_pt.proj.conv,
            device=device,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )
        self.pe = TtYOLOv12xConv2D(
            conv=parameter.pe.conv,
            conv_pth=conv_pt.pe.conv,
            device=device,
            config_override={"act_block_h": 32},
            use_1d_systolic_array=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )

    def __call__(self, x, i=0, j=0):
        batch_size, qkv_height, qkv_width, qkv_chan = x.shape
        qkv_n = qkv_height * qkv_width
        qkv = self.qkv(x)
        if qkv.is_sharded():
            qkv = ttnn.sharded_to_interleaved(qkv, ttnn.L1_MEMORY_CONFIG)
        qkv = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)

        if self.area > 1:
            qkv = ttnn.reshape(qkv, (1, batch_size * self.area, qkv_chan * 3, qkv_n // self.area))
            _, batch_size, _, qkv_n = qkv.shape

        # Using ttnn.reshape instead of view as "The last dimension can not change in view"
        qkv = ttnn.reshape(qkv, (batch_size, qkv_n, self.num_heads, self.head_dim * 3))
        qkv = ttnn.permute(qkv, (0, 2, 3, 1))
        q, k, v = ttnn.split(qkv, qkv.shape[2] // 3, 2)
        ttnn.deallocate(qkv)

        q = ttnn.permute(q, (0, 1, 3, 2))

        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
        attn = ttnn.matmul(q, k)

        ttnn.deallocate(q)
        ttnn.deallocate(k)

        attn = ttnn.multiply(attn, self.scale)
        attn = ttnn.softmax(attn, dim=-1)
        attn = ttnn.permute(attn, (0, 1, 3, 2))

        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
        attn = ttnn.to_layout(attn, ttnn.TILE_LAYOUT)
        x = ttnn.matmul(v, attn, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn)

        x = ttnn.permute(x, (0, 3, 1, 2))
        v = ttnn.permute(v, (0, 3, 1, 2))

        if self.area > 1:
            x = ttnn.reshape(x, (1, batch_size // self.area, qkv_n * self.area, qkv_chan))
            v = ttnn.reshape(v, (1, batch_size // self.area, qkv_n * self.area, qkv_chan))
            batch_size, qkv_n, _, _ = x.shape

        x = ttnn.reshape(x, (batch_size, qkv_height, qkv_width, qkv_chan))
        v = ttnn.reshape(v, (batch_size, qkv_height, qkv_width, qkv_chan))
        y = self.pe(v)
        y = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
        x = x + ttnn.reshape(y, x.shape)
        ttnn.deallocate(v)

        x = self.proj(x)

        return x
