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

Flat layout
-----------
:func:`fused_chunk_gated_delta_rule_flat` is the same pair of ops in their
rank-3 token-major mode: ``q, k`` are ``[B, 1, T, H_k * K]``, ``v`` is
``[B, 1, T, H_v * V]`` -- the projections' own layout -- and the output is
``[B, 1, T, H_v * V]``.  Nothing is reshaped to put the heads on their own
axis, which in TILE layout would pad the head count up to 32 (an 8x blow-up
for the 4 key heads a chip owns).  Both kernels address head ``h`` as a column
block of the flat row and map value head ``i`` to key head ``i // G`` while
reading, so the GVA repeat is free; its transpose -- summing the ``G`` copies
of each key head's ``dq``/``dk`` -- is one matmul with a constant 0/1 matrix.
The per-head L2 norm is :func:`flat_l2_norm`, whose row sums are likewise a
matmul with a block-diagonal ones matrix.
"""

from __future__ import annotations

import importlib

import numpy as np
import ttnn
from ttnn.operations.gated_delta_net_backward import gated_delta_net_backward

import ttml
from ttml.autograd import Function

from .autograd_ops import _const_tile

# The backward op's module keeps the last call's scratch tensors in a global
# debug hook (_LAST_SCRATCH), pinning ~2 GB of DRAM at seq_len=16384. The
# package re-exports the function under the module's name, so fetch the module
# itself from the import system.
_gdn_backward_module = importlib.import_module("ttnn.operations.gated_delta_net_backward.gated_delta_net_backward")

__all__ = [
    "FusedChunkGatedDeltaRule",
    "fused_chunk_gated_delta_rule",
    "FusedChunkGatedDeltaRuleFlat",
    "fused_chunk_gated_delta_rule_flat",
    "FlatL2Norm",
    "flat_l2_norm",
    "MAX_HEADS",
]

# Both ops address a head's rows within a single tile row, so H <= 32.
MAX_HEADS = 32


def _as_bf16(x):
    return x if x.dtype == ttnn.bfloat16 else ttnn.typecast(x, ttnn.bfloat16)


# ---------------------------------------------------------------------------
# Constant 0/1 matrices for the flat layout (device-resident, memoized,
# replicated across the mesh like every other constant in this package).
# ---------------------------------------------------------------------------


def _const_matrix(key, build_np):
    return _const_tile(
        ("qwen38.flat",) + key,
        lambda: ttml.autograd.Tensor.from_numpy(
            build_np().astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
        ).get_value(),
    )


def block_ones(width: int, group: int):
    """``[width, width]`` with ones on the ``group x group`` diagonal blocks.

    ``x @ block_ones`` puts each head's row sum in every one of that head's
    columns -- a per-head reduce that is already broadcast back, with no
    reshape of ``x``.
    """

    def build():
        heads = width // group
        return np.kron(np.eye(heads), np.ones((group, group)))

    return _const_matrix(("block_ones", width, group), build)


def gva_sum_matrix(num_k_heads: int, repeats: int, head_dim: int):
    """``[H_k * R * D, H_k * D]`` summing the ``R`` interleaved copies of each key head.

    Row ``(h * R + r) * D + d`` maps to column ``h * D + d``, i.e. value head
    ``i`` contributes to key head ``i // R`` -- the transpose of the GVA repeat.
    """

    def build():
        return np.kron(np.eye(num_k_heads), np.kron(np.ones((repeats, 1)), np.eye(head_dim)))

    return _const_matrix(("gva_sum", num_k_heads, repeats, head_dim), build)


# ---------------------------------------------------------------------------
# Per-head L2 norm on the flat layout
# ---------------------------------------------------------------------------


class FlatL2Norm(Function):
    """``y = x / sqrt(sum_head(x^2) + eps)`` per ``group``-wide head of a flat row."""

    @staticmethod
    def forward(ctx, x, ones, eps):
        x_val = x.get_value()
        ss = ttnn.matmul(ttnn.multiply(x_val, x_val), ones)
        r = ttnn.rsqrt(ttnn.add(ss, eps))
        y = ttnn.multiply(x_val, r)
        ctx.saved = (y, r, ones)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, r, ones = ctx.saved
        ctx.saved = None
        g = _as_bf16(grad_output)
        # dx = r * (g - y * sum_head(g * y))
        s = ttnn.matmul(ttnn.multiply(g, y), ones)
        return ttnn.multiply(r, ttnn.subtract(g, ttnn.multiply(y, s)))


def flat_l2_norm(x, group: int, eps: float = 1e-6):
    """L2-normalize every ``group``-wide head of ``[..., H * group]`` without reshaping."""
    width = int(x.shape()[-1])
    return FlatL2Norm.apply(x, block_ones(width, group), eps)


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
        _gdn_backward_module._LAST_SCRATCH = {}
        ctx.inputs = None
        gate_shape = [batch, 1, seq, heads]
        return dq, dk, dv, ttnn.reshape(dg, gate_shape), ttnn.reshape(dbeta, gate_shape)


class FusedChunkGatedDeltaRuleFlat(Function):
    """The fused ops in their rank-3 token-major (flat) mode; see the module docstring."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, gva_sum, chunk_size, scale, num_v_heads, key_dim):
        q_val, k_val, v_val = q.get_value(), k.get_value(), v.get_value()
        batch, _, seq, q_width = [int(d) for d in q_val.shape]
        v_width = int(v_val.shape[3])
        if num_v_heads > MAX_HEADS:
            raise ValueError(
                f"fused delta rule supports at most {MAX_HEADS} heads per chip, got {num_v_heads}; "
                "use tensor parallelism to shard the heads, or delta_rule_impl='composite'"
            )
        if seq % chunk_size:
            raise ValueError(
                f"flat fused delta rule needs seq_len ({seq}) to be a multiple of chunk_size ({chunk_size})"
            )
        # Dropping the unit dim keeps the tiled (T, width) face: no data moves.
        q3 = ttnn.reshape(q_val, [batch, seq, q_width])
        k3 = ttnn.reshape(k_val, [batch, seq, q_width])
        v3 = ttnn.reshape(v_val, [batch, seq, v_width])
        g3 = ttnn.reshape(g.get_value(), [batch, seq, num_v_heads])
        beta3 = ttnn.reshape(beta.get_value(), [batch, seq, num_v_heads])

        out, _ = ttnn.transformer.chunk_gated_delta_rule(q3, k3, v3, g3, beta3, scale=scale, chunk_size=chunk_size)
        # o comes back [B, T, H, V] ROW_MAJOR; folding (H, V) is free in row-major
        # and the tilize then pads nothing (T and H*V are tile multiples).
        out = ttnn.reshape(out, [batch, 1, seq, v_width])
        out = _as_bf16(ttnn.to_layout(out, ttnn.TILE_LAYOUT))

        ctx.inputs = (q3, k3, v3, g3, beta3, gva_sum)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.key_dim = key_dim
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q3, k3, v3, g3, beta3, gva_sum = ctx.inputs
        batch, seq, q_width = [int(d) for d in q3.shape]
        v_width = int(v3.shape[2])
        num_v_heads = int(g3.shape[2])
        do3 = ttnn.reshape(_as_bf16(grad_output), [batch, seq, v_width])
        dq, dk, dv, dg, dbeta, _ = gated_delta_net_backward(
            q3,
            k3,
            v3,
            g3,
            beta3,
            do3,
            chunk_size=ctx.chunk_size,
            scale=ctx.scale,
            key_head_dim=ctx.key_dim,
        )
        _gdn_backward_module._LAST_SCRATCH = {}
        ctx.inputs = None
        # dq/dk are per VALUE head [B, T, H_v*K]; sum the G copies of each key head.
        dq = ttnn.reshape(ttnn.matmul(dq, gva_sum), [batch, 1, seq, q_width])
        dk = ttnn.reshape(ttnn.matmul(dk, gva_sum), [batch, 1, seq, q_width])
        dv = ttnn.reshape(dv, [batch, 1, seq, v_width])
        gate_shape = [batch, 1, seq, num_v_heads]
        return dq, dk, dv, ttnn.reshape(dg, gate_shape), ttnn.reshape(dbeta, gate_shape)


def fused_chunk_gated_delta_rule_flat(
    q,
    k,
    v,
    g,
    beta,
    *,
    num_k_heads: int,
    num_v_heads: int,
    key_dim: int,
    chunk_size: int = 64,
    scale: float | None = None,
):
    """Gated delta rule on the fused ops, token-major with the heads folded into the row.

    Args:
        q, k: ``[B, 1, T, H_k * K]``, each ``K``-wide head L2-normalized (see
            :func:`flat_l2_norm`); NOT GVA-expanded -- the kernels map value
            head ``i`` to key head ``i // (H_v / H_k)`` themselves.
        v: ``[B, 1, T, H_v * V]``.
        g: ``[B, 1, T, H_v]`` log-space decay gate (negative).
        beta: ``[B, 1, T, H_v]`` write strength in ``(0, 1)``.
        chunk_size: 32 or 64; ``T`` must be a multiple of it.
        scale: query scale, defaults to ``K ** -0.5``.

    Returns:
        ``[B, 1, T, H_v * V]`` output in the projections' own layout.
    """
    if num_v_heads % num_k_heads:
        raise ValueError(f"value heads ({num_v_heads}) must be a multiple of key heads ({num_k_heads})")
    repeats = num_v_heads // num_k_heads
    if scale is None:
        scale = float(key_dim) ** -0.5
    gva_sum = gva_sum_matrix(num_k_heads, repeats, key_dim)
    return FusedChunkGatedDeltaRuleFlat.apply(q, k, v, g, beta, gva_sum, chunk_size, scale, num_v_heads, key_dim)


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
