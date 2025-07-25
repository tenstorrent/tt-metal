# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11.tt.common import TtnnConv, deallocate_tensors


class TtnnAttention:
    def __init__(self, device, parameter, conv_pt):
        self.qkv = TtnnConv(device, parameter.qkv, conv_pt.qkv, enable_act=False)
        self.proj = TtnnConv(device, parameter.proj, conv_pt.proj, enable_act=False)
        self.pe = TtnnConv(device, parameter.pe, conv_pt.pe, enable_act=False)
        self.num_heads = 2
        self.key_dim = 32
        self.head_dim = 64
        self.scale = self.key_dim**-0.5

    def __call__(self, device, x, batch_size=1):
        qkv = self.qkv(device, x)
        qkv = ttnn.sharded_to_interleaved(qkv, memory_config=ttnn.L1_MEMORY_CONFIG)
        qkv = ttnn.permute(qkv, (0, 3, 1, 2))
        qkv = ttnn.reshape(qkv, (batch_size, self.num_heads, self.key_dim * 2 + self.head_dim, qkv.shape[-1]))
        q, k, v = (
            qkv[:, :, : self.key_dim, :],
            qkv[:, :, self.key_dim : self.head_dim, :],
            qkv[:, :, self.head_dim :, :],
        )
        q_permuted = ttnn.permute(q, (0, 1, 3, 2))
        attn = ttnn.matmul(q_permuted, k, memory_config=ttnn.L1_MEMORY_CONFIG)
        attn = ttnn.multiply(attn, self.scale)
        attn = ttnn.softmax(attn, dim=-1)
        attn = ttnn.permute(attn, (0, 1, 3, 2))
        x1 = ttnn.matmul(v, attn, memory_config=ttnn.L1_MEMORY_CONFIG)
        x1 = ttnn.reshape(x1, (1, 1, (x1.shape[0] * x1.shape[1] * x1.shape[2]), x1.shape[3]))
        x1 = ttnn.permute(x1, (0, 1, 3, 2))
        v = ttnn.reshape(v, (1, 1, (v.shape[0] * v.shape[1] * v.shape[2]), v.shape[3]))
        v = ttnn.permute(v, (0, 1, 3, 2))
        x2 = self.pe(device=device, x=v)
        x = ttnn.add(x1, x2, memory_config=x2.memory_config())
        x = self.proj(device=device, x=x)

        deallocate_tensors(x1, qkv, q_permuted, attn, q, k, v, x2)

        return x
