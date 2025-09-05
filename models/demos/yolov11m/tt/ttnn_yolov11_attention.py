# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11m.tt.common import TtnnConv, deallocate_tensors


class TtnnAttention:
    def __init__(self, device, parameter, conv_pt):
        self.qkv = TtnnConv(device, parameter.qkv, conv_pt.qkv, enable_act=False)
        self.proj = TtnnConv(device, parameter.proj, conv_pt.proj, enable_act=False)
        self.pe = TtnnConv(device, parameter.pe, conv_pt.pe, enable_act=False)
        self.num_heads = 8
        self.key_dim = 16
        self.head_dim = 32
        self.scale = self.key_dim**-0.5

    def __call__(self, device, x, batch_size=1):
        print(f"DEBUG Attention: Input x shape = {x.shape}")
        print(f"DEBUG Attention: Input x elements = {x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]}")
        
        qkv = self.qkv(device, x)
        print(f"DEBUG Attention: After QKV conv shape = {qkv.shape}")
        print(f"DEBUG Attention: After QKV conv elements = {qkv.shape[0] * qkv.shape[1] * qkv.shape[2] * qkv.shape[3]}")
        
        qkv = ttnn.sharded_to_interleaved(qkv, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"DEBUG Attention: After sharded_to_interleaved shape = {qkv.shape}")
        
        qkv = ttnn.permute(qkv, (0, 3, 1, 2))
        print(f"DEBUG Attention: After permute (0,3,1,2) shape = {qkv.shape}")
        print(f"DEBUG Attention: After permute elements = {qkv.shape[0] * qkv.shape[1] * qkv.shape[2] * qkv.shape[3]}")
        print(f"DEBUG Attention: qkv.shape[-1] = {qkv.shape[-1]}")
        
        target_shape = (batch_size, self.num_heads, self.key_dim * 2 + self.head_dim, qkv.shape[-1])
        target_elements = target_shape[0] * target_shape[1] * target_shape[2] * target_shape[3]
        print(f"DEBUG Attention: Target reshape = {target_shape}")
        print(f"DEBUG Attention: Target elements = {target_elements}")
        print(f"DEBUG Attention: num_heads={self.num_heads}, key_dim={self.key_dim}, head_dim={self.head_dim}")
        
        # Show what reference implementation expects
        B, C, H, W = x.shape
        N = H * W
        expected_shape = (B, self.num_heads, self.key_dim * 2 + self.head_dim, N)
        expected_elements = expected_shape[0] * expected_shape[1] * expected_shape[2] * expected_shape[3]
        print(f"DEBUG Attention: Reference expects B={B}, C={C}, H={H}, W={W}, N={N}")
        print(f"DEBUG Attention: Reference expects shape = {expected_shape}")
        print(f"DEBUG Attention: Reference expects elements = {expected_elements}")
        
        qkv = ttnn.reshape(qkv, target_shape)
        print(f"DEBUG Attention: After reshape shape = {qkv.shape}")
        
        q, k, v = (
            qkv[:, :, : self.key_dim, :],
            qkv[:, :, self.key_dim : self.head_dim, :],
            qkv[:, :, self.head_dim :, :],
        )
        print(f"DEBUG Attention: Q shape = {q.shape}")
        print(f"DEBUG Attention: K shape = {k.shape}")
        print(f"DEBUG Attention: V shape = {v.shape}")
        print(f"DEBUG Attention: K slice indices = {self.key_dim} : {self.head_dim} (should be {self.key_dim} : {self.key_dim + self.key_dim})")
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
