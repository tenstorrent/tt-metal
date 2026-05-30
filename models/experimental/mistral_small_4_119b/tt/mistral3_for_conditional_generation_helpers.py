# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Helpers for splicing image embeddings into the text-embedding sequence on device.

Used by ``TtMistral3ForConditionalGenerationUnified`` to fuse vision-projector
outputs with the text-tokenizer embedding sequence. Pure ttnn slice + concat —
no host scratch tensors during inference.
"""

from __future__ import annotations

from typing import List, Tuple

import ttnn


def contiguous_runs(positions: List[int]) -> List[Tuple[int, int]]:
    """
    Group sorted positions into half-open contiguous runs ``[start, end)``.

    >>> contiguous_runs([3, 4, 5, 9, 10])
    [(3, 6), (9, 11)]
    """
    if not positions:
        return []
    runs: List[Tuple[int, int]] = []
    start = positions[0]
    prev = start
    for p in positions[1:]:
        if p == prev + 1:
            prev = p
            continue
        runs.append((start, prev + 1))
        start = p
        prev = p
    runs.append((start, prev + 1))
    return runs


def splice_embeddings(
    text_embeds: ttnn.Tensor,
    img_embeds: ttnn.Tensor,
    runs: List[Tuple[int, int]],
    seq_len: int,
    hidden_size: int,
) -> ttnn.Tensor:
    """
    Build the full input embedding tensor by replacing each image-slot run in
    ``text_embeds`` with the corresponding chunk of ``img_embeds``.

    Args:
        text_embeds: ttnn [1, 1, seq_len, hidden_size] — text embedding lookup.
        img_embeds:  ttnn [1, 1, num_image_tokens, hidden_size] — projector output.
        runs:        list of (start, end) half-open ranges in text_embeds that
                     correspond to image-token slots. Their total length must
                     equal ``num_image_tokens``.
        seq_len:     full sequence length (text_embeds.shape[-2]).
        hidden_size: HIDDEN_SIZE (text_embeds.shape[-1]).
    Returns:
        ttnn [1, 1, seq_len, hidden_size] with image runs spliced in.
    """
    segments: List[ttnn.Tensor] = []
    cursor = 0
    img_off = 0
    for start, end in runs:
        run_len = end - start
        if start > cursor:
            segments.append(ttnn.slice(text_embeds, [0, 0, cursor, 0], [1, 1, start, hidden_size]))
        segments.append(ttnn.slice(img_embeds, [0, 0, img_off, 0], [1, 1, img_off + run_len, hidden_size]))
        img_off += run_len
        cursor = end
    if cursor < seq_len:
        segments.append(ttnn.slice(text_embeds, [0, 0, cursor, 0], [1, 1, seq_len, hidden_size]))

    if len(segments) == 1:
        return segments[0]
    out = ttnn.concat(segments, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for s in segments:
        ttnn.deallocate(s)
    return out
