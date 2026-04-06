# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11s.tt.common import TtnnConv, deallocate_tensors


class TtnnAttention:
    def __init__(self, device, parameter, conv_pt):
        self.qkv = TtnnConv(device, parameter.qkv, conv_pt.qkv, enable_act=False)
        self.proj = TtnnConv(device, parameter.proj, conv_pt.proj, enable_act=False)
        self.pe = TtnnConv(device, parameter.pe, conv_pt.pe, enable_act=False)
        dim = conv_pt.qkv.conv.weight.shape[1]
        self.head_dim = 64
        self.num_heads = dim // self.head_dim
        self.key_dim = self.head_dim // 2
        # self.key_dim = int(self.head_dim * 0.5)
        self.scale = self.key_dim**-0.5

    def __call__(self, device, x, batch_size=1):
        _l1 = ttnn.L1_MEMORY_CONFIG
        qkv = self.qkv(device, x, to_interleaved=True)
        qkv = ttnn.permute(qkv, (0, 1, 3, 2))
        qkv = ttnn.reshape(
            qkv,
            (batch_size, self.num_heads, self.key_dim * 2 + self.head_dim, qkv.shape[-1]),
            memory_config=_l1,
        )

        q, k, v = ttnn.split(
            qkv,
            [self.key_dim, self.key_dim, self.head_dim],
            dim=2,
            memory_config=_l1,
        )
        attn = ttnn.matmul(q, k, transpose_a=True, memory_config=_l1)
        attn = ttnn.multiply(attn, self.scale)
        attn = ttnn.softmax_in_place(attn, dim=-1, numeric_stable=True)
        x1 = ttnn.matmul(v, attn, transpose_b=True, memory_config=_l1)
        x1 = ttnn.reshape(
            x1,
            (1, 1, x1.shape[0] * x1.shape[1] * x1.shape[2], x1.shape[3]),
            memory_config=_l1,
        )
        x1 = ttnn.permute(x1, (0, 1, 3, 2))
        v = ttnn.reshape(
            v,
            (1, 1, v.shape[0] * v.shape[1] * v.shape[2], v.shape[3]),
            memory_config=_l1,
        )
        v = ttnn.permute(v, (0, 1, 3, 2))
        x2 = self.pe(device=device, x=v)
        x = ttnn.add(x1, x2, memory_config=x1.memory_config())
        x = self.proj(device=device, x=x)
        deallocate_tensors(x1, qkv, attn, q, k, v, x2)

        return x
