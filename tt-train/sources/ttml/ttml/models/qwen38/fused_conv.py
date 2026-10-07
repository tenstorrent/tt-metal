# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet's depthwise causal conv1d + SiLU on fused device ops.

Forward is ``ttnn.experimental.kda.qkv_causal_conv1d_silu``, which convolves
the fused QKV projection and splits it into ``q | k | v`` in one pass. Backward
is two passes of ``ttml.ops.metal.depthwise_conv1d_k4``:

    du = dy * silu'(conv(x))        causal conv, fused with the SiLU derivative
    dx = sum_j tap_j * du[t + 3 - j]  anti-causal conv (the conv's transpose)

This replaces :func:`.gated_deltanet.causal_conv1d_silu`, which builds the conv
from ``shift_along_dim`` (an unaligned slice + concat per tap) and broadcast
multiplies, so each tap costs several full passes over the activation in both
directions.

The taps get no gradient: this path is only taken when they are frozen (LoRA).
"""

from __future__ import annotations

import ttnn

import ttml
from ttml.autograd import Function

__all__ = ["FusedCausalConv1dSilu", "fused_causal_conv1d_silu", "CHANNEL_CHUNK"]

# Channels per forward work item; must divide the per-chip Q+K+V width.
CHANNEL_CHUNK = 256

_zero_cache: dict = {}


def _zeros(key, shape, dtype):
    """Zero history / chronology tensors, allocated once per shape on the mesh."""
    cached = _zero_cache.get(key)
    if cached is None:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        cached = ttnn.zeros(shape, dtype, ttnn.ROW_MAJOR_LAYOUT, device)
        _zero_cache[key] = cached
    return cached


def _as_bf16(x):
    return x if x.dtype == ttnn.bfloat16 else ttnn.typecast(x, ttnn.bfloat16)


class FusedCausalConv1dSilu(Function):
    @staticmethod
    def forward(ctx, qkv, tap0, tap1, tap2, tap3, widths):
        q_width, k_width, v_width = widths
        x = qkv.get_value()
        _, _, seq, channels = [int(d) for d in x.shape]
        taps = [_as_bf16(t.get_value()) for t in (tap0, tap1, tap2, tap3)]

        # The forward op wants [1, T, C] ROW_MAJOR; dropping the leading 1 of a
        # row-major tensor moves no data.
        x_rm = ttnn.untilize(_as_bf16(x))
        history = _zeros(("history", channels), [1, 3, channels], ttnn.bfloat16)
        actual_start = _zeros(("actual_start",), [1], ttnn.uint32)
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            ttnn.reshape(x_rm, [1, seq, channels]),
            history,
            *taps,
            q_width,
            k_width,
            v_width,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=CHANNEL_CHUNK),
            actual_start=actual_start,
            predecessor_carry=history,
        )

        ctx.x_rm = x_rm
        ctx.taps = taps
        return tuple(ttnn.reshape(t, [1, 1, seq, int(t.shape[-1])]) for t in (q, k, v))

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        x_rm, taps = ctx.x_rm, ctx.taps
        ctx.x_rm = ctx.taps = None
        grad_y = ttnn.concat([_as_bf16(g) for g in (grad_q, grad_k, grad_v)], dim=-1)
        grad_u = ttml.ops.metal.depthwise_conv1d_k4(x_rm, *taps, anti_causal=False, silu_grad=grad_y)
        grad_x = ttml.ops.metal.depthwise_conv1d_k4(ttnn.untilize(grad_u), *taps, anti_causal=True)
        return grad_x, None, None, None, None


def fused_causal_conv1d_silu(qkv, taps, widths):
    """Causal depthwise conv1d (K=4) + SiLU over the fused QKV projection, split into q, k, v.

    Args:
        qkv: ``[1, 1, T, Q + K + V]`` TILE, ``T`` a multiple of 32.
        taps: four frozen ``[1, 1, 1, Q + K + V]`` taps; tap 3 multiplies the current token.
        widths: ``(Q, K, V)``, each a multiple of 32.

    Returns:
        ``q [1, 1, T, Q]``, ``k [1, 1, T, K]``, ``v [1, 1, T, V]``.
    """
    return FusedCausalConv1dSilu.apply(qkv, *taps, tuple(widths))
