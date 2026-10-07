# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet's per-head output norm on flat activations.

``ttml.ops.metal.gated_rmsnorm_fw`` / ``gated_rmsnorm_bw`` apply

    out = rmsnorm_per_head(x) * gamma * silu(gate)

directly on ``[B, 1, T, H*V]`` tensors, treating each contiguous ``V``-wide
slice of the last dim as one head. The composite path had to reshape ``x`` and
``gate`` to ``[B, 1, T*H, V]`` for the ttml RMSNorm (an 8x TILE-padded copy in
each direction, plus separate silu and mul passes); here nothing is reshaped
and the backward emits ``dx`` and ``dgate`` in one kernel.
"""

from __future__ import annotations

import ttnn

import ttml
from ttml.autograd import Function

__all__ = ["GatedRmsNorm", "gated_rmsnorm"]


def _as_bf16(x):
    return x if x.dtype == ttnn.bfloat16 else ttnn.typecast(x, ttnn.bfloat16)


class GatedRmsNorm(Function):
    @staticmethod
    def forward(ctx, x, gate, gamma, eps):
        x_val = _as_bf16(x.get_value())
        gate_val = _as_bf16(gate.get_value())
        gamma_val = _as_bf16(gamma.get_value())
        ctx.save_for_backward(x_val, gate_val, gamma_val)
        ctx.eps = eps
        ctx.need_dgamma = bool(gamma.get_requires_grad())
        return ttml.ops.metal.gated_rmsnorm_fw(x_val, gate_val, gamma_val, eps)

    @staticmethod
    def backward(ctx, grad_output):
        x_val, gate_val, gamma_val = ctx.saved_tensors
        ctx.save_for_backward()
        dx, dgate, dgamma = ttml.ops.metal.gated_rmsnorm_bw(
            x_val, gate_val, gamma_val, _as_bf16(grad_output), ctx.eps, ctx.need_dgamma
        )
        return dx, dgate, dgamma


def gated_rmsnorm(x, gate, gamma, eps=1e-6):
    """``rmsnorm_per_head(x) * gamma * silu(gate)`` without reshaping to the head axis.

    Args:
        x, gate: ``[B, 1, T, H*V]`` TILE tensors (``T`` a multiple of 32).
        gamma: ``[1, 1, 1, V]`` norm weight shared by all heads.
        eps: RMSNorm epsilon.

    Returns:
        ``[B, 1, T, H*V]`` TILE bf16.
    """
    return GatedRmsNorm.apply(x, gate, gamma, eps)
