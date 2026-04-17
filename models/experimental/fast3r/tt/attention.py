"""tt-nn attention for Fast3R encoder/decoder blocks.

This iteration implements self-attention WITHOUT RoPE (decoder path uses no RoPE
anyway). RoPE for the encoder is a follow-up change so we can isolate bugs.

Layout conventions:
- activation tensor: (1, 1, N, C) in TILE_LAYOUT
- qkv_weight stored transposed as (1, 1, C, 3C)
- qkv_bias stored as (1, 1, 1, 3C)
"""
from __future__ import annotations

import torch
import ttnn

from .mlp import TtMlp, to_device_bias, to_device_weight


class TtAttention:
    CORE_GRID = TtMlp.CORE_GRID
    COMPUTE = TtMlp.COMPUTE
    SDPA_PROG = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
        q_chunk_size=128,
        k_chunk_size=1024,
        exp_approx_mode=True,
    )

    def __init__(
        self,
        device,
        num_heads: int,
        qkv_w: torch.Tensor,
        qkv_b: torch.Tensor,
        proj_w: torch.Tensor,
        proj_b: torch.Tensor,
    ):
        self.num_heads = num_heads
        self.qkv_w = to_device_weight(device, qkv_w)
        self.qkv_b = to_device_bias(device, qkv_b)
        self.proj_w = to_device_weight(device, proj_w)
        self.proj_b = to_device_bias(device, proj_b)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        qkv = ttnn.linear(
            x, self.qkv_w, bias=self.qkv_b,
            core_grid=self.CORE_GRID, compute_kernel_config=self.COMPUTE,
            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.num_heads, transpose_k_heads=False
        )
        ttnn.deallocate(qkv)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, program_config=self.SDPA_PROG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        out = ttnn.experimental.nlp_concat_heads(attn)
        ttnn.deallocate(attn)
        return ttnn.linear(
            out, self.proj_w, bias=self.proj_b,
            core_grid=self.CORE_GRID, compute_kernel_config=self.COMPUTE,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
