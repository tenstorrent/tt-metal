# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One ViT block (HF ``Qwen3_5VisionBlock``): ``h = x + attn(norm1(x)); out = h + mlp(norm2(h))``."""

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.pplx_decider_v1_27b.tt.optimizations import VisionOptimizations
from models.demos.pplx_decider_v1_27b.tt.vision.attention import VisionAttention, VisionAttentionConfig
from models.demos.pplx_decider_v1_27b.tt.vision.config import PplxVisionArgs
from models.demos.pplx_decider_v1_27b.tt.vision.layernorm import VisionLayerNorm, VisionLayerNormConfig
from models.demos.pplx_decider_v1_27b.tt.vision.mlp import VisionMLP, VisionMLPConfig
from models.demos.pplx_decider_v1_27b.tt.vision.weights import VisionBlockWeights


@dataclass
class VisionBlockConfig:
    weights: VisionBlockWeights
    args: PplxVisionArgs
    optimizations: VisionOptimizations
    block_idx: int
    mesh_device: object | None = None


class VisionBlock(LightweightModule):
    def __init__(self, config: VisionBlockConfig):
        super().__init__()
        self.config = replace(config, mesh_device=config.mesh_device or config.optimizations.mesh_device)
        c, w = self.config, self.config.weights
        eps = c.args.layer_norm_eps
        self.norm1 = VisionLayerNorm.from_config(VisionLayerNormConfig(w.norm1, eps, c.optimizations, c.mesh_device))
        self.norm2 = VisionLayerNorm.from_config(VisionLayerNormConfig(w.norm2, eps, c.optimizations, c.mesh_device))
        self.attention = VisionAttention.from_config(
            VisionAttentionConfig(w.attention, c.args, c.optimizations, c.mesh_device)
        )
        self.mlp = VisionMLP.from_config(VisionMLPConfig(w.mlp, c.optimizations, c.mesh_device))

    @classmethod
    def from_config(cls, config: VisionBlockConfig) -> "VisionBlock":
        return cls(config)

    def load_device_weights(self) -> None:
        for module in (self.norm1, self.norm2, self.attention, self.mlp):
            module.load_device_weights()

    def forward(
        self, x: ttnn.Tensor, *, cos: ttnn.Tensor, sin: ttnn.Tensor, cu_window_seqlens: ttnn.Tensor
    ) -> ttnn.Tensor:
        """x: [1, 1, S, 1152] BF16 TILE; returns a new tensor (x is not deallocated)."""
        normed = self.norm1(x)
        attn = self.attention(normed, cos=cos, sin=sin, cu_window_seqlens=cu_window_seqlens)
        ttnn.deallocate(normed)
        h = ttnn.add(x, attn, memory_config=self.config.optimizations.output_memcfg)
        ttnn.deallocate(attn)
        normed = self.norm2(h)
        mlp = self.mlp(normed)
        ttnn.deallocate(normed)
        out = ttnn.add(h, mlp, memory_config=self.config.optimizations.output_memcfg)
        ttnn.deallocate(h)
        ttnn.deallocate(mlp)
        return out
