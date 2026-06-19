# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3-VL encoder layer.

Pre-norm transformer block, mirroring `MiniMaxM3VLVisionEncoderLayer.forward`:
    residual = x
    x = layer_norm1(x); x = self_attn(x, (cos, sin)); x = x + residual
    residual = x
    x = layer_norm2(x); x = mlp(x); x = x + residual
    return x

Both LayerNorms and the MLP are bias-carrying; attention is our
`M3VLAttention` (packed QKV with bias, rotate-half 3D RoPE, full SDPA).
There is no attention mask / cu_seqlens (the tower attends fully).
"""
from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimax_m3_vl.tt.attention import M3VLAttention
from models.demos.minimax_m3_vl.tt.layernorm import M3VLLayerNorm
from models.demos.minimax_m3_vl.tt.mlp import M3VLMLP


class M3VLBlock(LightweightModule):
    """One M3-VL transformer block (pre-norm attn + pre-norm MLP, both residual)."""

    def __init__(
        self,
        mesh_device,
        layer_norm1: M3VLLayerNorm,
        attention: M3VLAttention,
        layer_norm2: M3VLLayerNorm,
        mlp: M3VLMLP,
    ):
        super().__init__()
        self.device = mesh_device
        self.layer_norm1 = layer_norm1
        self.attention = attention
        self.layer_norm2 = layer_norm2
        self.mlp = mlp

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref_layer: torch.nn.Module,  # _Layer with layer_norm1/layer_norm2/self_attn/mlp
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        dtype=ttnn.bfloat16,
    ) -> "M3VLBlock":
        for attr in ("layer_norm1", "layer_norm2", "self_attn", "mlp"):
            assert hasattr(
                ref_layer, attr
            ), f"expected MiniMaxM3VLVisionEncoderLayer-like module with .{attr}, got {type(ref_layer).__name__}"
        return cls(
            mesh_device=mesh_device,
            layer_norm1=M3VLLayerNorm.from_torch(mesh_device, ref_layer.layer_norm1, dtype=dtype),
            attention=M3VLAttention.from_torch(
                mesh_device,
                ref_layer.self_attn,
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=dtype,
            ),
            layer_norm2=M3VLLayerNorm.from_torch(mesh_device, ref_layer.layer_norm2, dtype=dtype),
            mlp=M3VLMLP.from_torch(mesh_device, ref_layer.mlp, dtype=dtype),
        )

    def forward(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        memory_config: Optional["ttnn.MemoryConfig"] = None,
    ) -> ttnn.Tensor:
        # Sublayer 1: pre-norm attention with residual.
        residual = x
        normed = self.layer_norm1(x)
        attn_out = self.attention(normed, cos, sin)
        ttnn.deallocate(normed)
        x = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        # Sublayer 2: pre-norm MLP with residual.
        residual = x
        normed = self.layer_norm2(x)
        mlp_out = self.mlp(normed)
        ttnn.deallocate(normed)
        out = ttnn.add(
            residual,
            mlp_out,
            memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(residual)
        ttnn.deallocate(mlp_out)
        return out
