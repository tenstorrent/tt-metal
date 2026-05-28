# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MoonViT encoder layer.

Pre-norm transformer block, mirroring `MoonVitEncoderLayer.forward`:
    residual = x
    x = norm0(x); x = attn(x, cu_seqlens, rope_freqs_cis); x = x + residual
    residual = x
    x = norm1(x); x = mlp(x); x = x + residual
    return x

Both LayerNorms and the MLP are bias-carrying torch modules; the
attention uses our `MoonVisionAttention` which handles both QKV-packed
projection (with bias) and the cu_seqlens windowed SDPA.
"""
from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3.tt.moonvit.attention import MoonVisionAttention
from models.demos.deepseek_v3.tt.moonvit.layernorm import MoonVisionLayerNorm
from models.demos.deepseek_v3.tt.moonvit.mlp import MoonVisionMLP


class MoonVisionBlock(LightweightModule):
    """One MoonViT transformer block (pre-norm attn + pre-norm MLP, both residual)."""

    def __init__(
        self,
        mesh_device,
        norm0: MoonVisionLayerNorm,
        attention: MoonVisionAttention,
        norm1: MoonVisionLayerNorm,
        mlp: MoonVisionMLP,
    ):
        super().__init__()
        self.device = mesh_device
        self.norm0 = norm0
        self.attention = attention
        self.norm1 = norm1
        self.mlp = mlp

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref_layer: torch.nn.Module,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        dtype=ttnn.bfloat16,
    ) -> "MoonVisionBlock":
        """Construct from an HF MoonVitEncoderLayer reference module."""
        for attr in ("norm0", "norm1", "wqkv", "wo", "mlp"):
            assert hasattr(
                ref_layer, attr
            ), f"expected MoonVitEncoderLayer-like module with .{attr}, got {type(ref_layer).__name__}"

        norm0 = MoonVisionLayerNorm.from_torch(mesh_device, ref_layer.norm0, dtype=dtype)
        norm1 = MoonVisionLayerNorm.from_torch(mesh_device, ref_layer.norm1, dtype=dtype)
        attention = MoonVisionAttention.from_torch(
            mesh_device,
            ref_layer,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        mlp = MoonVisionMLP.from_torch(mesh_device, ref_layer.mlp, dtype=dtype)
        return cls(
            mesh_device=mesh_device,
            norm0=norm0,
            attention=attention,
            norm1=norm1,
            mlp=mlp,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        cu_seqlens: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        memory_config: Optional["ttnn.MemoryConfig"] = None,
    ) -> ttnn.Tensor:
        # Sublayer 1: pre-norm attention with residual.
        residual = x
        normed = self.norm0(x)
        attn_out = self.attention(normed, cu_seqlens, cos, sin)
        ttnn.deallocate(normed)
        x = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)
        # NOTE: `residual` aliased `x` on entry; ttnn.add output is a fresh tensor,
        # so the original input is no longer referenced — caller's tensor may be
        # deallocated by the caller, but we don't free here to avoid double-free.

        # Sublayer 2: pre-norm MLP with residual.
        residual = x
        normed = self.norm1(x)
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
