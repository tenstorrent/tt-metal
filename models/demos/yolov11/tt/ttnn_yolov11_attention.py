# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11.tt.common import TtnnConv, deallocate_tensors


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape,x.padded_shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class TtnnAttention:
    def __init__(self, device, parameter, conv_pt):
        self.qkv = TtnnConv(device, parameter.qkv, conv_pt.qkv, enable_act=False)
        self.proj = TtnnConv(device, parameter.proj, conv_pt.proj, enable_act=False)
        self.pe = TtnnConv(device, parameter.pe, conv_pt.pe, enable_act=False, reshard=True)
        self.num_heads = 2
        self.key_dim = 32
        self.head_dim = 64
        self.scale = self.key_dim**-0.5

    def __call__(self, device, x, batch_size=1):
        p(x, "qkv in")
        qkv = self.qkv(device, x, output_rm_needed=False)
        p(qkv, "qkv out")
        qkv = ttnn.sharded_to_interleaved(qkv, memory_config=ttnn.L1_MEMORY_CONFIG)
        qkv = ttnn.permute(qkv, (0, 3, 1, 2))
        qkv = ttnn.reshape(qkv, (batch_size, self.num_heads, self.key_dim * 2 + self.head_dim, qkv.shape[-1]))
        p(qkv, "qkv after reshape")
        q, k, v = (
            qkv[:, :, : self.key_dim, :],
            qkv[:, :, self.key_dim : self.head_dim, :],
            qkv[:, :, self.head_dim :, :],
        )
        p(q, "q out")
        p(k, "k out")
        p(q, "v out")
        q_permuted = ttnn.permute(q, (0, 1, 3, 2))
        attn = ttnn.matmul(q_permuted, k, memory_config=ttnn.L1_MEMORY_CONFIG)
        attn = ttnn.multiply(attn, self.scale)
        attn = ttnn.softmax_in_place(attn, dim=-1, numeric_stable=False)
        attn = ttnn.permute(attn, (0, 1, 3, 2))
        x1 = ttnn.matmul(v, attn, memory_config=ttnn.L1_MEMORY_CONFIG)
        p(x1, "x1 outt")
        x1 = ttnn.reshape(x1, (1, 1, (x1.shape[0] * x1.shape[1] * x1.shape[2]), x1.shape[3]))
        x1 = ttnn.permute(x1, (0, 1, 3, 2))
        v = ttnn.reshape(v, (1, 1, (v.shape[0] * v.shape[1] * v.shape[2]), v.shape[3]))
        v = ttnn.permute(v, (0, 1, 3, 2))
        p(v, "v out")
        x2 = self.pe(device=device, x=v, output_rm_needed=False)
        p(x2, "x2 out")
        # x2 = ttnn.pad(x2, ((0, 0), (0, 0), (0, 16), (0, 0)), value=0.0)
        p(x2, "x2 out after pad")
        print("outt is", x2.memory_config())
        # if(x1.shape[2] > x2.shape[2]):
        #     p(x1,"before slice x2 is")
        #     # x1 = ttnn.to_layout(x1,ttnn.ROW_MAJOR_LAYOUT)
        #     x1 = x1[:,:,:x2.shape[2],:]
        #     p(x1,"after slice x2 is")
        x = ttnn.add(x1, x2, memory_config=x1.memory_config())
        p(x, "after add")
        x = self.proj(device=device, x=x, output_rm_needed=False)
        p(x, "after proj")
        deallocate_tensors(x1, qkv, q_permuted, attn, q, k, v, x2)

        return x
