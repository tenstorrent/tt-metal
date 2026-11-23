# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MoE accuracy against a reference: what a routing difference costs, per row.

The MoE output is a continuous product, scored by ``windowed.error_metrics`` (absolute and relative). Routing is
scored here. ``dropped`` is in the router's own weight units (the reference weight lost), so it is absolute already;
``recall`` and set equality are fractions. Router math (scores, bias, top-k, scaling) is the model's and stays with
the model. MoE is stateless: judge its error per window absolutely, not by growth over context.
"""

from __future__ import annotations

import torch

from models.demos.deepseek_v3_d_p.tests.accuracy.topk import missed_keys, topk_recall


def routing_rows(
    got: torch.Tensor,
    got_weights: torch.Tensor,
    want: torch.Tensor,
    want_weights: torch.Tensor,
    want_scores: torch.Tensor,
) -> dict:
    """Per row, a routing ``got`` (ids ``[rows, k]``, weights ``got_weights``) against the reference ``want`` /
    ``want_weights``:

    * ``recall``: the fraction of the reference's experts ``got`` also picks;
    * ``dropped``: the reference weight on the experts ``got`` misses;
    * ``weight_err``: the largest weight difference on an expert both pick (0 when none is shared);
    * ``misses``: every missed reference expert, ranked in the reference's selection by ``want_scores`` (its score
      of each of its experts, larger first; ``topk.missed_keys``)."""
    rows = torch.arange(want.shape[0]).unsqueeze(1) << 32
    missed = ~torch.isin(rows + want, rows + got)
    dropped = (want_weights.float() * missed).sum(dim=1)
    same = want.unsqueeze(2) == got.unsqueeze(1)  # [rows, k_want, k_got]
    diff = (want_weights.float().unsqueeze(2) - got_weights.float().unsqueeze(1)).abs()
    weight_err = (diff * same).amax(dim=(1, 2)) if want.shape[1] and got.shape[1] else torch.zeros(want.shape[0])
    return {
        "recall": topk_recall(got, want),
        "dropped": dropped,
        "weight_err": weight_err,
        "misses": missed_keys(got, want, want_scores),
    }
