# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Chunked gated delta rule on the fused ttnn device ops.

Forward is ``ttnn.transformer.chunk_gated_delta_rule`` and backward is
``ttnn.operations.gated_delta_net_backward``, glued into the ttml autograd
graph by a :class:`ttml.autograd.Function`.  This replaces the ttml-op
decomposition in :mod:`.delta_rule` (kept as the ``"composite"`` path), whose
backward is a graph of a few hundred small ops per layer.

The backward op recomputes everything it needs from the raw inputs, so the
forward's intermediates are not requested and nothing beyond the five inputs is
kept alive between the passes.

Layout
------
Both ops are token-major: ``q, k`` are ``[B, T, H, K]``, ``v`` is
``[B, T, H, V]`` and ``g, beta`` are ``[B, T, H]``.  ``g`` and ``beta`` cross
the autograd boundary as ``[B, 1, T, H]`` because ttml tensors are rank 4;
reshaping between the two keeps the tiled ``(T, H)`` face, so it moves no data.

Caller contract (shared with both ops): ``q`` and ``k`` must be L2-normalized
along ``K`` and already GVA-expanded to the value-head count, since the
backward has no grouped-head support.
"""

from __future__ import annotations

import ttnn
from ttnn.operations.gated_delta_net_backward import gated_delta_net_backward

from ttml.autograd import Function

__all__ = ["FusedChunkGatedDeltaRule", "fused_chunk_gated_delta_rule", "MAX_HEADS"]

# Both ops address a head's rows within a single tile row, so H <= 32.
MAX_HEADS = 32


def _as_bf16(x):
    return x if x.dtype == ttnn.bfloat16 else ttnn.typecast(x, ttnn.bfloat16)


class FusedChunkGatedDeltaRule(Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, chunk_size, scale):
        q_val, k_val, v_val = q.get_value(), k.get_value(), v.get_value()
        batch, seq, heads, _ = [int(d) for d in q_val.shape]
        if heads > MAX_HEADS:
            raise ValueError(
                f"fused delta rule supports at most {MAX_HEADS} heads per chip, got {heads}; "
                "use tensor parallelism to shard the heads, or delta_rule_impl='composite'"
            )
        g_val = ttnn.reshape(g.get_value(), [batch, seq, heads])
        beta_val = ttnn.reshape(beta.get_value(), [batch, seq, heads])

        out, _ = ttnn.transformer.chunk_gated_delta_rule(
            q_val, k_val, v_val, g_val, beta_val, scale=scale, chunk_size=chunk_size
        )
        # The op hands back o as [B, T, H, V] ROW_MAJOR fp32. Folding (T, H)
        # into one axis is free before tilizing and leaves no padded rows.
        val_dim = int(out.shape[3])
        out = ttnn.reshape(out, [batch, 1, seq * heads, val_dim])
        out = _as_bf16(ttnn.to_layout(out, ttnn.TILE_LAYOUT))

        ctx.inputs = (q_val, k_val, v_val, g_val, beta_val)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q_val, k_val, v_val, g_val, beta_val = ctx.inputs
        batch, seq, heads, _ = [int(d) for d in q_val.shape]
        grad_output = ttnn.reshape(_as_bf16(grad_output), list(v_val.shape))
        dq, dk, dv, dg, dbeta, _ = gated_delta_net_backward(
            q_val,
            k_val,
            v_val,
            g_val,
            beta_val,
            grad_output,
            chunk_size=ctx.chunk_size,
            scale=ctx.scale,
        )
        ctx.inputs = None
        gate_shape = [batch, 1, seq, heads]
        return dq, dk, dv, ttnn.reshape(dg, gate_shape), ttnn.reshape(dbeta, gate_shape)


def fused_chunk_gated_delta_rule(q, k, v, g, beta, chunk_size: int = 64, scale: float | None = None):
    """Gated delta rule over a full sequence on the fused device ops.

    Args:
        q, k: ``[B, T, H, K]``, L2-normalized along ``K`` and GVA-expanded to ``H``.
        v: ``[B, T, H, V]``.
        g: ``[B, 1, T, H]`` log-space decay gate (negative).
        beta: ``[B, 1, T, H]`` write strength in ``(0, 1)``.
        chunk_size: 32 or 64.
        scale: query scale, defaults to ``K ** -0.5``.

    Returns:
        ``[B, 1, T * H, V]`` output: one row per (token, head), token-major, so
        a per-head norm over ``V`` sees no padded rows.
    """
    return FusedChunkGatedDeltaRule.apply(q, k, v, g, beta, chunk_size, scale)
