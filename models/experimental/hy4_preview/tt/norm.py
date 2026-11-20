# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm on a hidden state that is split over the hidden axis (tenstorrent/tt-metal#65893).

Mirrors HF transformers' HYV4RMSNorm, which reference.attention.rms_norm reproduces:

    y = weight * x * rsqrt(mean(x^2) + eps)

It is the decoder layer's two pre-norms (``input_layernorm``, ``post_attention_layernorm``,
eps = rms_norm_eps = 1e-5) and the model's final norm.

Layout (tt/common.py MeshPlan). x is [1, 1, T, hidden/tp] per chip, the chip's own hidden slice,
the same on every chip that shares a hidden coordinate. The mean of squares runs over the whole hidden dimension, so each
chip sums the squares of its slice and ONE all-reduce over the hidden axis adds the tp partial
sums: a [1, 1, T, 1] tensor, the only cross-chip traffic of the norm. The divisor is the global
width, not the slice's. The scale and the weight, which is sharded like x, are then chip-local.

Precision. HF normalises in fp32 whatever the model dtype and so does this: an input in another
dtype is cast to fp32 first. The result is cast to ``output_dtype`` after the weight is applied.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.hy4_preview.tt.common import MeshPlan


class TtRMSNorm(LightweightModule):
    """``weight`` is the [hidden] vector of the checkpoint. ``output_dtype`` is the dtype of the
    result; None keeps the input's."""

    def __init__(self, plan: MeshPlan, weight: torch.Tensor, eps: float, output_dtype=None):
        self.plan = plan
        self.eps = float(eps)
        self.output_dtype = output_dtype
        self.hidden_size = weight.numel()
        assert weight.dim() == 1, f"weight is {tuple(weight.shape)}, expected [hidden]"
        assert self.hidden_size % plan.hidden_tp == 0, f"hidden {self.hidden_size} does not split {plan.hidden_tp} ways"
        self.weight = ttnn.from_torch(
            weight.float().contiguous().reshape(1, 1, 1, self.hidden_size),
            layout=ttnn.TILE_LAYOUT,
            device=plan.mesh_device,
            dtype=ttnn.float32,
            mesh_mapper=plan.shard_hidden(3),
        )

    def inverse_rms(self, x):
        """x: fp32 [1, 1, T, hidden/tp] -> rsqrt(mean(x^2) + eps), [1, 1, T, 1], the same on every chip."""
        squares = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
        squares = self.plan.all_reduce_hidden(squares)
        return ttnn.rsqrt(ttnn.add(ttnn.multiply(squares, 1.0 / self.hidden_size), self.eps))

    def forward(self, x):
        """[1, 1, T, hidden/tp] -> [1, 1, T, hidden/tp], split over the hidden axis like x."""
        assert x.shape[-1] * self.plan.hidden_tp == self.hidden_size, (tuple(x.shape), self.hidden_size)
        out_dtype = self.output_dtype or x.dtype
        xf = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
        y = ttnn.multiply(ttnn.multiply(xf, self.inverse_rms(xf)), self.weight)
        return y if out_dtype == ttnn.float32 else ttnn.typecast(y, out_dtype)
