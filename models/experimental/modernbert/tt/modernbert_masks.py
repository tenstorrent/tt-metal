# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Additive attention masks, one per layer type.

Built once per sequence length and shared by every layer of that type: 8 full
layers and 14 sliding layers for ModernBERT-base.

Band geometry: config.local_attention is 128 and the band is symmetric +/-64,
total width 129. HF's attn.sliding_window attribute holds 65, an internal
half-representation that is not the band width.

The sliding window is carried by the mask rather than SDPA's
`sliding_window_size` (see modernbert_attention), so the two layer types differ
only in this tensor. Full attention needs no mask when nothing is padded.
"""

import torch

import ttnn
from models.experimental.modernbert.common import FULL_ATTENTION, SLIDING_ATTENTION
from models.experimental.modernbert.tt.model_config import ACTIVATIONS_DTYPE

# bfloat16's most negative value is around -3.39e38; use a large finite negative
# rather than -inf so that softmax cannot produce NaN on fully-masked rows.
MASK_NEG = -1e30

# Pass no mask on full-attention layers when nothing is padded, rather than an
# all-zero (B,1,S,S) tensor. Worth -1.8%, and it is why full-attention layers cost
# 93 us against 146 for sliding ones.
FULL_MASK_NONE = True

# Materialise the batch dimension rather than broadcasting (1,1,S,S). Broadcasting
# measured 0.9% slower: SDPA re-reads the mask per (batch, head) either way, so a
# smaller tensor moves the same bytes.
BROADCAST_BATCH = False


def build_masks(config, device, seq_len, attention_mask=None, batch_size=1, dtype=ACTIVATIONS_DTYPE):
    """Return {layer_type: ttnn additive mask}, both attention-shaped (B, 1, S, S).

    attention_mask: torch (B, S), 1 for real tokens and 0 for padding, or None.
    batch_size: used only when attention_mask is None.

    The masked value is finite, not -inf: see MASK_NEG. Batch is materialised: see
    BROADCAST_BATCH.
    """
    padded = attention_mask is not None and not bool(torch.all(attention_mask == 1))

    pad_row = None
    if padded:
        # (B, 1, 1, S) broadcast over query positions
        pad_row = torch.zeros(attention_mask.shape, dtype=torch.float32)
        pad_row = pad_row.masked_fill(attention_mask == 0, MASK_NEG)[:, None, None, :]

    batch = attention_mask.shape[0] if attention_mask is not None else batch_size

    half = config.local_attention // 2
    idx = torch.arange(seq_len)
    band = (idx[None, :] - idx[:, None]).abs() <= half
    sliding = torch.zeros(seq_len, seq_len, dtype=torch.float32).masked_fill(~band, MASK_NEG)[None, None]
    full = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float32)

    if pad_row is not None:
        # broadcasting with (B, 1, 1, S) already yields the batch dimension
        sliding = sliding + pad_row
        full = full + pad_row
    elif not BROADCAST_BATCH:
        sliding = sliding.expand(batch, 1, seq_len, seq_len).contiguous()
        full = full.expand(batch, 1, seq_len, seq_len).contiguous()

    full_mask = None
    if padded or not FULL_MASK_NONE:
        full_mask = ttnn.from_torch(full, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    return {
        SLIDING_ATTENTION: ttnn.from_torch(sliding, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device),
        FULL_ATTENTION: full_mask,
    }


def deallocate_masks(masks):
    for m in masks.values():
        if m is not None:
            ttnn.deallocate(m)
