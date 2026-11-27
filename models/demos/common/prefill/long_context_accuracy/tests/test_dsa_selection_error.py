# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSA selection error (largest missed-key weight) on small hand-computed values and against the per-row form it
replaced. CPU only."""

import torch

from models.demos.common.prefill.long_context_accuracy.dsa.selection_error import missed_key_max
from models.demos.common.prefill.long_context_accuracy.topk_selection import topk_hits

F64 = torch.float64


def test_missed_key_max_hand_case() -> None:
    """Row 0: two heads over keys (best first) with weights .6 .3 .1 and .2 .1 .7; the device misses the last key:
    the largest missed weight is head 1's .7. Row 1: no miss reads 0, though its weights are larger. Row 2: two
    misses, the larger in head 0 (.25); its -1 entry is not a key and carries no weight."""
    weights = torch.tensor(
        [
            [[0.6, 0.3, 0.1], [0.2, 0.1, 0.7]],
            [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]],
            [[0.25, 0.75, 0.0], [0.15, 0.85, 0.0]],
        ],
        dtype=F64,
    )
    missed = torch.tensor([[False, False, True], [False, False, False], [True, False, False]])
    assert missed_key_max(weights, missed).tolist() == [0.7, 0.0, 0.25]
    missed[2, 1] = True
    assert missed_key_max(weights, missed).tolist() == [0.7, 0.0, 0.85]
    assert missed_key_max(weights, missed).dtype == F64


def per_row_missed_key_max(p_ref: torch.Tensor, want: torch.Tensor, got: torch.Tensor) -> list[float]:
    """The per-row form ``score_rows`` used before (``selection_masses``'s ``missed_key_max``): each row's valid
    reference keys and weights, membership by ``torch.isin`` against the device's valid keys."""
    out = []
    for i in range(want.shape[0]):
        valid = want[i] >= 0
        w, missed = p_ref[i][:, valid], ~torch.isin(want[i][valid], got[i][got[i] >= 0])
        out.append(float(w[:, missed].max()) if missed.any() else 0.0)
    return out


def test_missed_key_max_equals_the_per_row_form() -> None:
    """Random softmax weights over selections with -1 padding and partial overlap: bit-equal to the per-row form."""
    g = torch.Generator().manual_seed(1)
    rows, heads, k = 16, 4, 12
    want = torch.stack([torch.randperm(40, generator=g)[:k] for _ in range(rows)])
    want[torch.rand(rows, k, generator=g) < 0.2] = -1
    got = torch.where(torch.rand(rows, k, generator=g) < 0.7, want, torch.randint(0, 40, (rows, k), generator=g))
    got[torch.rand(rows, k, generator=g) < 0.1] = -1
    p_ref = torch.randn(rows, heads, k, generator=g, dtype=F64).masked_fill((want < 0).unsqueeze(1), -torch.inf)
    p_ref = p_ref.softmax(-1)
    missed = ~topk_hits(got, want) & (want >= 0)
    expected = per_row_missed_key_max(p_ref, want, got)
    assert 0.0 in expected and any(v > 0 for v in expected)  # rows with and without a miss
    assert missed_key_max(p_ref, missed).tolist() == expected
