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

import numpy as np
import ttnn

import ttml
from ttml.models.deepseek.autograd_ops import autograd_concat, autograd_slice

TILE = 32


def build_dflash_square_mask(
    document_ids: np.ndarray,
    anchor_positions: np.ndarray,
    anchor_valid: np.ndarray,
    block_size: int,
    sliding_window: int | None = None,
    sliding_window_non_causal: bool = False,
    sliding_window_per_query: bool = False,
) -> np.ndarray:
    """Build the bool ``[T + n*B, T + n*B]`` mask for :func:`dflash_square_attention`.

    Args:
        document_ids: ``[T]`` document index per packed position, ``-1`` for padding.
        anchor_positions: ``[n]`` anchor positions into the packed sequence.
        anchor_valid: ``[n]`` bool; padded anchors attend only to themselves.
        block_size: slots per anchor block.
        sliding_window, sliding_window_non_causal, sliding_window_per_query: same
            meaning as in speculators ``create_anchor_block_mask_mod``.

    Returns:
        ``mask[q, kv] == True`` where query row ``q`` of ``X`` may attend to key ``kv``.
    """
    document_ids = np.asarray(document_ids, dtype=np.int64)
    anchor_positions = np.asarray(anchor_positions, dtype=np.int64)
    anchor_valid = np.asarray(anchor_valid, dtype=bool)
    if document_ids.ndim != 1 or anchor_positions.ndim != 1 or anchor_valid.shape != anchor_positions.shape:
        raise ValueError("document_ids, anchor_positions and anchor_valid must be 1D (anchors of equal length)")

    T = document_ids.size
    nB = anchor_positions.size * block_size
    if T % TILE or nB % TILE:
        raise ValueError(
            f"context length ({T}) and anchor slots n*B ({nB}) must be multiples of {TILE}; "
            "pad the anchor budget with anchor_valid=0"
        )
    non_causal = sliding_window is None or sliding_window_non_causal

    S = T + nB
    mask = np.eye(S, dtype=bool)

    slot = np.arange(nB)
    slot_offset = slot % block_size  # [nB]
    slot_anchor = np.repeat(anchor_positions, block_size)  # [nB]
    slot_valid = np.repeat(anchor_valid, block_size)  # [nB]

    # Context keys: same document, before the anchor, inside the window.
    ctx_pos = np.arange(T)[None, :]  # [1, T]
    q_anchor = slot_anchor[:, None]  # [nB, 1]
    q_doc = document_ids[slot_anchor][:, None]
    ctx_allowed = (q_doc == document_ids[None, :]) & (q_doc != -1) & (ctx_pos < q_anchor)
    if sliding_window is not None:
        if sliding_window_per_query:
            query_position = q_anchor + slot_offset[:, None]
            ctx_allowed &= ctx_pos >= query_position - sliding_window + 1
        else:
            ctx_allowed &= ctx_pos >= q_anchor - sliding_window

    # Block keys: own block only (causal within the block unless non-causal).
    q_slot = slot[:, None]
    kv_slot = slot[None, :]
    blk_allowed = (q_slot // block_size) == (kv_slot // block_size)
    if not non_causal:
        blk_allowed &= kv_slot <= q_slot
    if sliding_window_per_query and sliding_window is not None:
        blk_allowed &= kv_slot >= q_slot - sliding_window + 1

    rows = T + slot[slot_valid]
    mask[rows, :T] = ctx_allowed[slot_valid]
    mask[rows, T:] = blk_allowed[slot_valid]
    return mask


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
