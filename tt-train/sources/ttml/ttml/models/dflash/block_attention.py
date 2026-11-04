# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DFlash anchor-block attention on the square fused SDPA.

DFlash attention is rectangular. Q holds ``n_anchors * block_size`` synthetic block
slots, and KV is ``[packed context (T) | all block slots]``. Block slot ``j*B + k`` may
see context tokens of its anchor's document that precede the anchor (optionally
windowed) plus the slots of its own block. See speculators
``models/dflash/attention.py::create_anchor_block_mask_mod`` for the reference.

The fused tt-train SDPA needs ``q_len == kv_len`` and one ``(1, 1, S, S)`` mask, so
this runs it over ``X = [context ; blocks]`` (``S = T + n*B``). Context rows are dummy
zero queries that attend only to themselves, which keeps their softmax finite. Their
outputs are sliced away, so they contribute exactly zero gradient. Padded anchor
slots (``anchor_valid == 0``) are handled the same way. Callers must give them zero
loss weight.

Shapes are fixed per call: use a constant anchor budget (padded with
``anchor_valid == 0``) so ``n*B`` stays tile-aligned.
"""

from __future__ import annotations

import ttnn

import ttml
from ttml.models.deepseek.autograd_ops import autograd_concat, autograd_slice

from .host import TILE, build_dflash_square_mask  # noqa: F401  (re-exported, D1a API)


def dflash_square_attention(
    q_blk: ttml.autograd.Tensor,
    k_ctx: ttml.autograd.Tensor,
    v_ctx: ttml.autograd.Tensor,
    k_blk: ttml.autograd.Tensor,
    v_blk: ttml.autograd.Tensor,
    mask: ttml.autograd.Tensor,
) -> ttml.autograd.Tensor:
    """DFlash block attention through the fused square SDPA.

    Args:
        q_blk: ``[B, Hq, n*B, D]`` block-slot queries.
        k_ctx, v_ctx: ``[B, Hkv, T, D]`` context keys/values.
        k_blk, v_blk: ``[B, Hkv, n*B, D]`` block-slot keys/values.
        mask: ``[1, 1, T + n*B, T + n*B]`` 1/0 mask from :func:`build_dflash_square_mask`.

    Returns:
        ``[B, Hq, n*B, D]`` attention output for the block slots.
    """
    batch, q_heads, n_slots, head_dim = list(q_blk.get_value().shape)
    T = list(k_ctx.get_value().shape)[2]

    q_ctx_value = ttnn.zeros(
        [batch, q_heads, T, head_dim],
        dtype=q_blk.get_value().dtype,
        layout=ttnn.Layout.TILE,
        device=q_blk.get_value().device(),
    )
    q_ctx = ttml.autograd.create_tensor(q_ctx_value, requires_grad=False)

    query = autograd_concat([q_ctx, q_blk], dim=2)
    key = autograd_concat([k_ctx, k_blk], dim=2)
    value = autograd_concat([v_ctx, v_blk], dim=2)
    out = ttml.ops.attention.scaled_dot_product_attention(query, key, value, mask)
    return autograd_slice(out, [0, 0, T, 0], [batch, q_heads, T + n_slots, head_dim])
