# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ClippableLinear (HF `Gemma4ClippableLinear`, modeling_gemma4.py:176-202) with use_clipped_linears=True.

    y = clamp( clamp(x, in_min, in_max) @ W^T, out_min, out_max )          no bias

The four bounds are per-tensor scalars stored in the checkpoint (bf16). They are read once in
`from_state_dict` and folded into `ttnn.clamp` as python floats, so the forward has no extra inputs.

Contract
    input   x [B, 1, S, in_features] bf16 or fp32 TILE, DRAM interleaved
    output  [B, 1, S, out_features] TILE DRAM, dtype `out_dtype` (default bf16)
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.audio.config import ClipBounds
from models.demos.blackhole.embedding_gemma2.tt.audio.linear import audio_linear, hardtanh
from models.demos.blackhole.embedding_gemma2.tt.audio.weights import clip_bounds
from models.demos.blackhole.embedding_gemma2.tt.config import compute_kernel_config_hifi4
from models.demos.blackhole.embedding_gemma2.tt.weights import as_weight, linear_weight


class ClippableLinear(LightweightModule):
    def __init__(self, weight: ttnn.Tensor, bounds: ClipBounds, out_dtype=ttnn.bfloat16):
        super().__init__()
        self.weight = weight
        self.bounds = bounds
        self.out_dtype = out_dtype
        self.compute_kernel_config = compute_kernel_config_hifi4()

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        in_features: int,
        out_features: int,
        device,
        *,
        cache_dir: Path | None = None,
        name: str,
        out_dtype=ttnn.bfloat16,
    ) -> "ClippableLinear":
        """`state_dict` holds one clippable linear's sub-state (`weight`, `input_min`, ...)."""
        w = state_dict["weight"]
        if tuple(w.shape) != (out_features, in_features):
            raise ValueError(f"{name}: expected weight {(out_features, in_features)}, got {tuple(w.shape)}")
        weight = as_weight(linear_weight(w), device, cache_dir=cache_dir, name=f"{name}.weight")
        return cls(weight, clip_bounds(state_dict, name), out_dtype)

    def input_clamp(self) -> ttnn.UnaryWithParam:
        """The input clamp as a unary op, for callers that fuse it into the producer of x."""
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDTANH, self.bounds.input_min, self.bounds.input_max)

    def project(self, xc: ttnn.Tensor) -> ttnn.Tensor:
        """clamp_out(xc @ W) for an input already clamped to the input bounds."""
        b = self.bounds
        # output clamp fused into the matmul (HARDTANH on the fp32 destination before packing; the
        # bounds are bf16 values, so clamp-then-round == round-then-clamp)
        return audio_linear(
            xc,
            self.weight,
            dtype=self.out_dtype,
            compute_kernel_config=self.compute_kernel_config,
            activation=hardtanh(b.output_min, b.output_max),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        b = self.bounds
        xc = ttnn.clamp(x, min=b.input_min, max=b.input_max, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = self.project(xc)
        ttnn.deallocate(xc)
        return out
