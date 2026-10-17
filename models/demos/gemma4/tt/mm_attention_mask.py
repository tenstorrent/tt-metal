# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill attention masks with bidirectional image spans (Gemma 4 ``use_bidirectional_attention="vision"``).

HF treats every contiguous run of vision soft tokens (image or video) as one block: tokens of a block attend
to each other in both directions, on top of the causal (full layers) or causal-and-windowed (sliding layers)
text mask. The overlay is OR'ed in, so a block wider than the sliding window still attends to itself fully.

The builders here are host-side (torch) and mirror ``transformers.masking_utils``: ``vision_block_ids`` is
``get_block_sequence_ids_for_mask`` applied to the placeholder ids of a prompt, and ``build_mm_prefill_masks``
is ``create_causal_mask`` / ``create_sliding_window_causal_mask`` with ``block_sequence_ids``. The device
consumer (``attention/prefill.py``) hands them to ``ttnn.transformer.scaled_dot_product_attention`` as the
explicit ``attn_mask`` (which replaces both ``is_causal`` and ``sliding_window_size``).
"""
from __future__ import annotations

import torch

# Additive mask value for disallowed positions. Not -inf: a fully masked row produces NaN in the flash
# softmax; -1e9 underflows to exactly 0 after exp in bf16 and keeps the row finite.
MASK_NEG = -1e9


def vision_block_ids(input_ids: torch.Tensor, vision_token_ids) -> torch.Tensor:
    """Block id per position: contiguous runs of vision placeholder tokens get 0, 1, 2, ...; text gets -1.

    ``vision_token_ids`` is one id or an iterable of ids (image and video placeholders). Audio placeholders
    are NOT vision tokens (HF keeps them causal), so they must not be passed here.
    """
    ids = torch.as_tensor(input_ids).reshape(-1)
    if isinstance(vision_token_ids, int):
        vision_token_ids = [vision_token_ids]
    is_vision = torch.zeros_like(ids, dtype=torch.bool)
    for tok in vision_token_ids:
        is_vision |= ids == int(tok)
    is_prev_vision = torch.roll(is_vision, shifts=1, dims=0)
    is_prev_vision[0] = False
    new_starts = is_vision & ~is_prev_vision
    group_ids = torch.cumsum(new_starts.int(), dim=0) - 1
    return torch.where(is_vision, group_ids, torch.full_like(group_ids, -1))


def mm_allowed_matrix(block_ids: torch.Tensor, seq_len: int, sliding_window: int | None = None) -> torch.Tensor:
    """Boolean [seq_len, seq_len] ``allowed[q, k]``: causal (and within ``sliding_window`` when given) OR same block.

    ``block_ids`` covers the prompt; positions past it (padding) are text (-1) and stay causal.
    """
    blk = torch.full((seq_len,), -1, dtype=torch.long)
    n = min(int(block_ids.shape[0]), seq_len)
    blk[:n] = block_ids[:n].to(torch.long)
    q = torch.arange(seq_len).view(-1, 1)
    k = torch.arange(seq_len).view(1, -1)
    allowed = k <= q
    if sliding_window is not None and sliding_window > 0:
        # HF sliding_window_causal_mask_function: kv_idx > q_idx - sliding_window
        allowed &= k > (q - int(sliding_window))
    same_block = (blk.view(-1, 1) == blk.view(1, -1)) & (blk.view(-1, 1) >= 0)
    return allowed | same_block


def build_mm_prefill_masks(block_ids: torch.Tensor, seq_len: int, sliding_window: int) -> dict:
    """Additive bf16 masks ``[1, 1, seq_len, seq_len]`` per layer type: ``{"full_attention", "sliding_attention"}``.

    Returns None when the prompt has no vision block (plain causal attention is exact and cheaper).
    """
    if block_ids.numel() == 0 or int(block_ids.max()) < 0:
        return None

    def _additive(allowed):
        return torch.where(allowed, torch.zeros((), dtype=torch.float32), torch.full((), MASK_NEG)).to(torch.bfloat16)

    full = _additive(mm_allowed_matrix(block_ids, seq_len, None))
    sliding = _additive(mm_allowed_matrix(block_ids, seq_len, sliding_window))
    return {
        "full_attention": full.view(1, 1, seq_len, seq_len),
        "sliding_attention": sliding.view(1, 1, seq_len, seq_len),
    }


def vision_block_spans(block_ids: torch.Tensor):
    """``[(start, end), ...]`` half-open spans of the vision blocks in ``block_ids`` (a cache key for the masks)."""
    b = torch.as_tensor(block_ids).reshape(-1).tolist()
    spans, i, n = [], 0, len(b)
    while i < n:
        if b[i] >= 0:
            j = i
            while j < n and b[j] == b[i]:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return tuple(spans)
