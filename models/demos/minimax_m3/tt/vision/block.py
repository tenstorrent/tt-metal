# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
One MiniMax-M3 vision encoder layer, as Hugging Face's MiniMaxM3VLVisionEncoderLayer computes it:

    x = x + attention(layer_norm1(x))
    x = x + mlp(layer_norm2(x))
"""
from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimax_m3.tt.vision.attention import M3VLAttention
from models.demos.minimax_m3.tt.vision.layernorm import M3VLLayerNorm
from models.demos.minimax_m3.tt.vision.mlp import M3VLMLP


class M3VLBlock(LightweightModule):
    """One vision layer: attention, then the MLP, each after a LayerNorm and with a residual."""

    def __init__(
        self,
        layer_norm1: M3VLLayerNorm,
        attention: M3VLAttention,
        layer_norm2: M3VLLayerNorm,
        mlp: M3VLMLP,
    ):
        super().__init__()
        self.layer_norm1 = layer_norm1
        self.attention = attention
        self.layer_norm2 = layer_norm2
        self.mlp = mlp

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref_layer: torch.nn.Module,  # Hugging Face's MiniMaxM3VLVisionEncoderLayer: layer_norm1/2, self_attn, mlp
        padded_head_dim: int,
        dtype=ttnn.bfloat16,
    ) -> "M3VLBlock":
        return cls(
            layer_norm1=M3VLLayerNorm.from_torch(mesh_device, ref_layer.layer_norm1, dtype=dtype),
            attention=M3VLAttention.from_torch(mesh_device, ref_layer.self_attn, padded_head_dim, dtype=dtype),
            layer_norm2=M3VLLayerNorm.from_torch(mesh_device, ref_layer.layer_norm2, dtype=dtype),
            mlp=M3VLMLP.from_torch(mesh_device, ref_layer.mlp, dtype=dtype),
        )

    def forward(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
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
        out = ttnn.add(residual, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(residual)
        ttnn.deallocate(mlp_out)
        return out
