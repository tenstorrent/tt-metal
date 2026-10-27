# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VisionEmbedder (HF `EmbeddingGemma2MultimodalEmbedder` as `embed_vision`, modeling_embedding_gemma2.py:565-591).

    soft = RMSNorm_768(pooled, no scale, eps 1e-6) @ W_proj        W_proj 768 -> 512 (text hidden), no bias

Contract
    input   pooled [B, 1, S_out, 768] float32 TILE DRAM (VisionPooler output)
    output  [B, 1, S_out, 512] bf16 TILE DRAM; rows >= the image's soft-token count are not meaningful
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import compute_kernel_config_hifi4
from models.demos.blackhole.embedding_gemma2.tt.rms_norm import RMSNorm
from models.demos.blackhole.embedding_gemma2.tt.vision.config import VisionConfig
from models.demos.blackhole.embedding_gemma2.tt.weights import as_weight, linear_weight


class VisionEmbedder(LightweightModule):
    def __init__(self, config: VisionConfig, norm: RMSNorm, w_proj: ttnn.Tensor):
        super().__init__()
        self.config = config
        self.norm = norm
        self.w_proj = w_proj
        self.compute_kernel_config = compute_kernel_config_hifi4()

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: VisionConfig,
        device,
        *,
        cache_dir: Path | None = None,
        name: str = "embed_vision",
    ) -> "VisionEmbedder":
        """`state_dict` holds the HF `embed_vision.` sub-state."""
        w = state_dict["embedding_projection.weight"]
        if tuple(w.shape) != (config.text_hidden_size, config.hidden_size):
            raise ValueError(f"{name}.embedding_projection: unexpected shape {tuple(w.shape)}")
        w_proj = as_weight(linear_weight(w), device, cache_dir=cache_dir, name=f"{name}.embedding_projection.weight")
        return cls(config, RMSNorm(config.norm(with_scale=False), None), w_proj)

    def forward(self, pooled: ttnn.Tensor) -> ttnn.Tensor:
        dram = ttnn.DRAM_MEMORY_CONFIG
        n = self.norm(pooled)
        out = ttnn.linear(
            n, self.w_proj, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config, memory_config=dram
        )
        ttnn.deallocate(n)
        return out
