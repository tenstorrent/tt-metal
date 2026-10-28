# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ClippableLinear (HF `Gemma4ClippableLinear`, modeling_gemma4.py:176-202) with use_clipped_linears=True.

    y = clamp( clamp(x, in_min, in_max) @ W^T, out_min, out_max )          no bias

The four bounds are per-tensor scalars stored in the checkpoint (bf16). They are read once in
`from_state_dict` and folded into `ttnn.clamp` as python floats, so the forward has no extra inputs.

Contract
    input   x [B, 1, S, in_features] bf16 or fp32 TILE, DRAM or L1 interleaved
    output  [B, 1, S, out_features] TILE, dtype `out_dtype` (default bf16), L1 interleaved when it fits
            (config.audio_act_memory_config) else DRAM. E5c: 2D-multicast program config
            (audio/linear.py) with the output clamp fused into the matmul as HARDTANH.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.embedding_gemma2.tt.audio.config import ClipBounds, audio_act_memory_config
from models.demos.embedding_gemma2.tt.audio.linear import audio_linear, hardtanh
from models.demos.embedding_gemma2.tt.audio.weights import clip_bounds
from models.demos.embedding_gemma2.tt.precision import MatmulPolicy
from models.demos.embedding_gemma2.tt.weights import as_weight, linear_weight

_E5_POLICY = MatmulPolicy("bf16", "HiFi4", True, "bf16")  # E3 / E5c: bf16 weights, HiFi4 + fp32 acc


class ClippableLinear(LightweightModule):
    def __init__(self, weight: ttnn.Tensor, bounds: ClipBounds, out_dtype=None, policy: MatmulPolicy | None = None):
        super().__init__()
        self.policy = policy or _E5_POLICY
        self.weight = weight
        self.bounds = bounds
        self.out_dtype = self.policy.out_ttnn_dtype if out_dtype is None else out_dtype
        self.compute_kernel_config = self.policy.compute_kernel_config

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
        out_dtype=None,
        policy: MatmulPolicy | None = None,
    ) -> "ClippableLinear":
        """`state_dict` holds one clippable linear's sub-state (`weight`, `input_min`, ...). `policy`:
        weight dtype, fidelity, fp32 accumulation and output dtype (a precision-policy audio group;
        None: bf16 weights, HiFi4 + fp32 acc, bf16 output). `out_dtype` overrides the policy's output
        dtype (AudioModel `act_dtype`)."""
        policy = policy or _E5_POLICY
        w = state_dict["weight"]
        if tuple(w.shape) != (out_features, in_features):
            raise ValueError(f"{name}: expected weight {(out_features, in_features)}, got {tuple(w.shape)}")
        weight = as_weight(
            linear_weight(w), device, cache_dir=cache_dir, name=f"{name}.weight", dtype=policy.ttnn_dtype
        )
        return cls(weight, clip_bounds(state_dict, name), out_dtype, policy)

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
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        b = self.bounds
        mc = audio_act_memory_config(x.device(), x.shape, x.dtype)
        xc = ttnn.clamp(x, min=b.input_min, max=b.input_max, memory_config=mc)
        out = self.project(xc)
        ttnn.deallocate(xc)
        return out
