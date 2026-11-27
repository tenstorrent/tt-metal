# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Top-k selection against a reference selection, shared by MoE routing (experts) and sparse attention (keys).

Selections are ``[rows, k]`` int64 tensors of ids; a negative id (or the uint32 0xFFFFFFFF no-key sentinel in
``got``) selects nothing. ``topk_hits`` computes which reference ids a selection holds, once per selection pair;
recall and missed keys are read from its mask.
"""

from __future__ import annotations

import torch


def topk_hits(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """``[rows, k_want]`` bool: True where ``want``'s entry is a valid (>= 0) id that occurs in the same row of
    ``got``. ``got``'s duplicates and no-key entries need no special case: a valid id equals neither."""
    valid = want >= 0
    if not got.shape[1]:
        return torch.zeros_like(valid)
    held = got.sort(dim=1).values
    at = torch.searchsorted(held, want).clamp_max_(held.shape[1] - 1)
    return (held.gather(1, at) == want) & valid


def topk_recall(hits: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Per row the fraction of ``want``'s valid (>= 0) ids that the selection holds (``hits = topk_hits(got,
    want)``), i.e. |got ∩ want| / |want| as sets (an id ``got`` holds twice counts once). Rows with no valid
    ``want`` entry read NaN."""
    return hits.sum(dim=1) / (want >= 0).sum(dim=1)


def missed_keys(hits: torch.Tensor, want: torch.Tensor, scores: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every ``want`` id that the selection misses (``hits = topk_hits(got, want)``), one entry per miss: its ``row``
    (index into the chunk), its ``rank`` in ``want``'s own selection ordered by ``scores`` (``want``'s score of each
    of its ids, larger first; 0 = best), that selection's size ``selected``, and its ``spread_share`` (score − cut) /
    (top − cut), the cut being the lowest selected score. A rank near ``selected`` or a share near 0 is a near-tie at
    the cut."""
    valid = want >= 0
    missed = ~hits & valid
    masked = scores.masked_fill(~valid, -float("inf"))
    rank = torch.empty_like(want)
    rank.scatter_(1, masked.argsort(dim=1, descending=True), torch.arange(want.shape[1]).expand_as(want).contiguous())
    top = masked.max(dim=1, keepdim=True).values
    cut = scores.masked_fill(~valid, float("inf")).min(dim=1, keepdim=True).values
    row = torch.arange(want.shape[0]).unsqueeze(1).expand_as(want)[missed]
    share = (scores[missed] - cut[row, 0]) / (top - cut).clamp_min(1e-30)[row, 0]  # (elementwise: on the misses only)
    return {
        "row": row,
        "rank": rank[missed],
        "selected": valid.sum(dim=1, keepdim=True).expand_as(want)[missed],
        "spread_share": share.float(),
    }
