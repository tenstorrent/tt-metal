# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# On-device sequence scatter via concat (same pattern as tt/vision/inject.py).

from __future__ import annotations

from typing import Sequence

import ttnn

from models.experimental.hunyuan_image_3_0.tt.vision.inject import scatter_cond_vision_embeddings_multi


def mask_to_spans(mask_row: Sequence[bool] | Sequence[int]) -> list[slice]:
    """Contiguous True runs in a 1-D mask -> ``slice`` list."""
    spans: list[slice] = []
    start = None
    for i, v in enumerate(mask_row):
        if bool(v):
            if start is None:
                start = i
        elif start is not None:
            spans.append(slice(start, i))
            start = None
    if start is not None:
        spans.append(slice(start, len(mask_row)))
    return spans


def scatter_token_spans(
    hidden: ttnn.Tensor,
    spans: list[tuple[slice, ttnn.Tensor]],
) -> ttnn.Tensor:
    """Inject ``(slice, [B,n,H])`` token blocks into ``hidden`` ``[B,S,H]`` on device."""
    return scatter_cond_vision_embeddings_multi(hidden, spans)


def scatter_tokens_from_mask(
    hidden: ttnn.Tensor,
    token_embeds: ttnn.Tensor,
    mask_row,
    *,
    batch_row: int = 0,
) -> ttnn.Tensor:
    """Scatter ``token_embeds`` ``[1,n,H]`` or ``[B,n,H]`` at contiguous spans from ``mask_row``."""
    if hasattr(mask_row, "tolist"):
        row = mask_row[batch_row].tolist() if getattr(mask_row, "ndim", 1) > 1 else mask_row.tolist()
    else:
        row = list(mask_row)
    spans_idx = mask_to_spans(row)
    bsz, n_total, hidden = token_embeds.shape
    cursor = 0
    paired: list[tuple[slice, ttnn.Tensor]] = []
    for sl in spans_idx:
        n = sl.stop - sl.start
        if bsz == 1:
            chunk = token_embeds if n_total == n else ttnn.slice(token_embeds, [0, cursor, 0], [1, cursor + n, hidden])
        else:
            chunk = ttnn.slice(token_embeds, [batch_row, cursor, 0], [batch_row + 1, cursor + n, hidden])
        paired.append((sl, chunk))
        cursor += n
    out = scatter_token_spans(hidden, paired)
    for _, chunk in paired:
        if chunk is not token_embeds:
            ttnn.deallocate(chunk, force=False)
    return out
