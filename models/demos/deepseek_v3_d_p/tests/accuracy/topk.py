# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Top-k selection against a reference selection, shared by MoE routing (experts) and sparse attention (keys).

Selections are ``[rows, k]`` int64 tensors of ids; a negative id (or the uint32 0xFFFFFFFF no-key sentinel in
``got``) selects nothing.
"""

from __future__ import annotations

import torch


def topk_recall(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Per row the fraction of ``want``'s valid (>= 0) ids that occur in ``got``, i.e. |got ∩ want| / |want| as sets
    (an id ``got`` holds twice counts once). ``got``'s no-key entries match nothing. Rows with no valid ``want`` entry
    read NaN."""
    rows = torch.arange(want.shape[0]).unsqueeze(1) << 32
    valid = want >= 0
    return (torch.isin(rows + want, rows + got) & valid).sum(dim=1) / valid.sum(dim=1)


def missed_keys(got: torch.Tensor, want: torch.Tensor, scores: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every ``want`` id that ``got`` misses, one entry per miss: its ``row`` (index into the chunk), its ``rank`` in
    ``want``'s own selection ordered by ``scores`` (``want``'s score of each of its ids, larger first; 0 = best),
    that selection's size ``selected``, and its ``spread_share`` (score − cut) / (top − cut), the cut being the
    lowest selected score. A rank near ``selected`` or a share near 0 is a near-tie at the cut."""
    rows = torch.arange(want.shape[0]).unsqueeze(1) << 32
    valid = want >= 0
    missed = ~torch.isin(rows + want, rows + got) & valid
    masked = scores.masked_fill(~valid, -float("inf"))
    rank = torch.empty_like(want)
    rank.scatter_(1, masked.argsort(dim=1, descending=True), torch.arange(want.shape[1]).expand_as(want).contiguous())
    top = masked.max(dim=1, keepdim=True).values
    cut = scores.masked_fill(~valid, float("inf")).min(dim=1, keepdim=True).values
    share = (scores - cut) / (top - cut).clamp_min(1e-30)
    row = torch.arange(want.shape[0]).unsqueeze(1).expand_as(want)
    return {
        "row": row[missed],
        "rank": rank[missed],
        "selected": valid.sum(dim=1, keepdim=True).expand_as(want)[missed],
        "spread_share": share[missed].float(),
    }
