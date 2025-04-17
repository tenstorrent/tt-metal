# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
from models.experimental.yolov10.tt.common import Conv


class TtnnAttention:
    def __init__(self, dim, num_heads=8, attn_ratio=0.5, device=None, parameters=None, conv_pt=None):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2

        self.qkv = Conv(
            device,
            parameters.qkv,
            self.conv_pt.qkv,
            enable_identity=True,
        )
        self.proj = Conv(
            device,
            parameters.proj,
            self.conv_pt.proj,
            enable_identity=True,
        )

        self.pe = Conv(
            device,
            parameters.pe,
            self.conv_pt.pe,
            enable_identity=True,
        )

    def __call__(self, input_tensor):
        B, C, H, W = (
            1,
            input_tensor.shape[-1],
            int(math.sqrt(input_tensor.shape[2])),
            int(math.sqrt(input_tensor.shape[2])),
        )
        N = H * W

        qkv = self.qkv(input_tensor)
        qkv = ttnn.permute(qkv, (0, 3, 1, 2))
        qkv = ttnn.reshape(qkv, (1, qkv.shape[1], int(math.sqrt(qkv.shape[-1])), int(math.sqrt(qkv.shape[-1]))))
        qkv = ttnn.reshape(qkv, (B, self.num_heads, self.key_dim * 2 + self.head_dim, N))

        qkv = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)

        q, k, v = qkv[:, :, :32, :], qkv[:, :, 32:64, :], qkv[:, :, 64:, :]

        q = ttnn.permute(q, (0, 1, 3, 2))

        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)

        attn = ttnn.matmul(q, k)
        attn = attn * self.scale

        attn = ttnn.softmax(attn, dim=-1)

        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
        attn = ttnn.permute(attn, (0, 1, 3, 2))
        attn = ttnn.matmul(v, attn)
        attn = ttnn.reshape(attn, (B, C, H, W))

        v = ttnn.reshape(v, (B, C, H, W))

        v = ttnn.permute(v, (0, 2, 3, 1))
        v = ttnn.reshape(v, (1, 1, v.shape[0] * v.shape[1] * v.shape[2], v.shape[3]))

        v = self.pe(v)

        attn = ttnn.permute(attn, (0, 2, 3, 1))
        attn = ttnn.reshape(attn, (1, 1, attn.shape[0] * attn.shape[1] * attn.shape[2], attn.shape[3]))

        x = attn + v

        output = self.proj(x)

        return output
