# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Audio output head: HF `Gemma4AudioModel.output_proj` (modeling_gemma4.py:1978) and `embed_audio`
(`EmbeddingGemma2MultimodalEmbedder`, modeling_embedding_gemma2.py:565-591).

    hidden = x @ W_out^T + b_out                      1024 -> 1536, with bias      (AudioOutputProj)
    soft   = RMSNorm_1536(hidden, no scale, eps 1e-6) @ W_embed^T     1536 -> 512, no bias  (AudioEmbedder)

Contract
    AudioOutputProj  [B, 1, S, 1024] bf16 TILE -> [B, 1, S, 1536] fp32 TILE (the audio tower output)
    AudioEmbedder    [B, 1, S, 1536] fp32 TILE -> [B, 1, S, 512] bf16 TILE (the audio soft tokens)
"""

from __future__ import annotations

from pathlib import Path

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.audio.config import AudioConfig
from models.demos.blackhole.embedding_gemma2.tt.audio.linear import audio_linear
from models.demos.blackhole.embedding_gemma2.tt.config import compute_kernel_config_hifi4
from models.demos.blackhole.embedding_gemma2.tt.rms_norm import RMSNorm
from models.demos.blackhole.embedding_gemma2.tt.weights import as_weight, linear_weight


class AudioOutputProj(LightweightModule):
    def __init__(self, weight: ttnn.Tensor, bias: ttnn.Tensor):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.compute_kernel_config = compute_kernel_config_hifi4()

    @classmethod
    def from_state_dict(
        cls, state_dict, config: AudioConfig, device, *, cache_dir: Path | None = None, name: str = "audio.output_proj"
    ) -> "AudioOutputProj":
        """`state_dict` holds the HF `audio_tower.output_proj.` sub-state."""
        w, b = state_dict["weight"], state_dict["bias"]
        if tuple(w.shape) != (config.output_proj_dims, config.hidden_size) or tuple(b.shape) != (
            config.output_proj_dims,
        ):
            raise ValueError(f"{name}: unexpected shapes {tuple(w.shape)}, {tuple(b.shape)}")
        weight = as_weight(linear_weight(w), device, cache_dir=cache_dir, name=f"{name}.weight")
        bias = as_weight(
            b.float().reshape(1, 1, 1, -1), device, cache_dir=cache_dir, name=f"{name}.bias", dtype=ttnn.float32
        )
        return cls(weight, bias)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return audio_linear(
            x,
            self.weight,
            bias=self.bias,
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


class AudioEmbedder(LightweightModule):
    def __init__(self, config: AudioConfig, norm: RMSNorm, w_proj: ttnn.Tensor):
        super().__init__()
        self.config = config
        self.norm = norm
        self.w_proj = w_proj
        self.compute_kernel_config = compute_kernel_config_hifi4()

    @classmethod
    def from_state_dict(
        cls, state_dict, config: AudioConfig, device, *, cache_dir: Path | None = None, name: str = "embed_audio"
    ) -> "AudioEmbedder":
        """`state_dict` holds the HF `embed_audio.` sub-state."""
        w = state_dict["embedding_projection.weight"]
        if tuple(w.shape) != (config.text_hidden_size, config.output_proj_dims):
            raise ValueError(f"{name}.embedding_projection: unexpected shape {tuple(w.shape)}")
        w_proj = as_weight(linear_weight(w), device, cache_dir=cache_dir, name=f"{name}.embedding_projection.weight")
        return cls(config, RMSNorm(config.norm(with_scale=False, dim=config.output_proj_dims), None), w_proj)

    def forward(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        n = self.norm(hidden)
        out = audio_linear(
            n,
            self.w_proj,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(n)
        return out
