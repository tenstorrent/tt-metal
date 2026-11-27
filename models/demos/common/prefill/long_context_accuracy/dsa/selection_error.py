# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sparse attention (DSA) accuracy against a reference: what a different key selection costs.

Keys (latent, rope part, index keys) and the attention output are continuous products, scored by
``windowed_errors.errors`` (absolute and relative). Selection recall and misses come from ``topk_selection``. This
module adds the attention weight a selection difference moves, given the reference attention weights over each selection
(computing them is the model's). These are diagnostics: no bar is defined on them.
"""

from __future__ import annotations

import math

import torch


def missed_key_max(want_weights: torch.Tensor, missed: torch.Tensor) -> torch.Tensor:
    """Per query row (``[rows]``, the weights' dtype), the largest weight of one missed key in one head: the maximum
    of ``want_weights`` ``[rows, heads, k]`` (the reference attention softmax over the reference selection) over the
    heads and the keys ``missed`` ``[rows, k]`` marks (``~topk_selection.topk_hits & valid``); 0 on a row with no
    miss."""
    masked = want_weights.masked_fill(~missed.unsqueeze(1), -math.inf)
    return masked.amax(dim=(1, 2)).where(missed.any(dim=1), 0.0)
