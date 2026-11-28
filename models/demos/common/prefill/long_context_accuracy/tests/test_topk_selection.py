# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Top-k selection (membership, recall, missed keys) on small hand-computed values. CPU only."""

import math

import pytest
import torch

from models.demos.common.prefill.long_context_accuracy.topk_selection import missed_keys, topk_hits, topk_recall

NO_KEY = 0xFFFFFFFF  # the device's uint32 no-key sentinel, read as int64


def recall_of(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    return topk_recall(topk_hits(got, want), want)


def test_topk_hits_by_hand() -> None:
    """Row 0: device duplicates (3 twice) and its -1 / no-key entries match nothing; reference 9 is past every
    device id and 0 before every one. Row 1: a reference -1 is never a hit, even where the device holds -1. Row 2:
    the device holds the reference's ids in another order."""
    want = torch.tensor([[3, 0, 9, 4], [-1, 2, -1, 5], [7, 8, 6, -1]])
    got = torch.tensor([[3, 3, -1, NO_KEY], [-1, 5, 2, 2], [6, 7, 8, -1]])
    assert topk_hits(got, want).tolist() == [
        [True, False, False, False],
        [False, True, False, True],
        [True, True, True, False],
    ]


@pytest.mark.parametrize(
    "got, want, hits",
    [
        (torch.zeros(2, 0, dtype=torch.long), torch.tensor([[1, -1], [0, 2]]), [[False, False], [False, False]]),
        (torch.tensor([[1, 2], [0, 2]]), torch.zeros(2, 0, dtype=torch.long), [[], []]),
        (torch.zeros(0, 3, dtype=torch.long), torch.zeros(0, 2, dtype=torch.long), []),
        (torch.tensor([[5]]), torch.tensor([[5, 4, 6]]), [[True, False, False]]),  # k_got = 1
        (torch.tensor([[-1, -1, -1]]), torch.tensor([[0, 1, -1]]), [[False, False, False]]),  # device selects nothing
    ],
    ids=["no-device-keys", "no-reference-keys", "no-rows", "one-device-key", "all-padding"],
)
def test_topk_hits_edge_shapes(got: torch.Tensor, want: torch.Tensor, hits: list) -> None:
    out = topk_hits(got, want)
    assert out.dtype == torch.bool and tuple(out.shape) == tuple(want.shape) and out.tolist() == hits


def test_topk_hits_equals_per_row_isin() -> None:
    """The membership the scorers used before (``torch.isin`` on row-offset ids, masked to valid reference ids), on
    selections with duplicates, -1 padding and no-key sentinels: equal entry by entry."""
    g = torch.Generator().manual_seed(0)
    rows, k = 64, 32
    want = torch.randint(0, 50, (rows, k), generator=g)
    want[torch.rand(rows, k, generator=g) < 0.2] = -1
    got = torch.randint(0, 50, (rows, k), generator=g)
    got[torch.rand(rows, k, generator=g) < 0.1] = -1
    got[torch.rand(rows, k, generator=g) < 0.1] = NO_KEY
    offset = torch.arange(rows).unsqueeze(1) << 32
    old = torch.isin(offset + want, offset + got) & (want >= 0)
    assert old.any() and (~old & (want >= 0)).any()  # both outcomes occur
    assert torch.equal(topk_hits(got, want), old)


def test_topk_recall_counts_valid_reference_ids_once() -> None:
    want = torch.tensor([[0, 1, 2, 3], [4, 5, -1, -1], [-1, -1, -1, -1]])
    got = torch.tensor([[3, 1, 9, 9], [4, NO_KEY, 6, 7], [1, 2, 3, 4]])
    recall = recall_of(got, want)
    assert recall[:2].tolist() == [0.5, 0.5] and math.isnan(recall[2])


def test_topk_recall_and_missed_keys_by_hand() -> None:
    """Row 0: reference keys 5 / 6 / 7 / 8 scored 4 / 3 / 2 / 1, device {5, 7, 9, 8}: recall 3/4, one miss, key 6, at
    rank 1 of 4 (top half). Row 1: two valid reference keys (and two -1),
    both found: recall 1, no miss. A device -1 (no key) matches nothing, and duplicates count once."""
    want = torch.tensor([[5, 6, 7, 8], [1, 0, -1, -1]])
    got = torch.tensor([[5, 7, 9, 8], [0, 1, -1, -1]])
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 5.0, 0.0, 0.0]])
    assert recall_of(got, want).tolist() == pytest.approx([3 / 4, 1.0])
    misses = missed_keys(topk_hits(got, want), want, scores)
    assert misses["row"].tolist() == [0] and misses["rank"].tolist() == [1] and misses["selected"].tolist() == [4]
    assert recall_of(torch.tensor([[-1, 0, 0]]), torch.tensor([[0, 1, 2]])).tolist() == pytest.approx([1 / 3])


def test_missed_keys_rank_in_the_reference_order() -> None:
    want = torch.tensor([[10, 11, 12, 13]])
    scores = torch.tensor([[0.9, 0.5, 0.7, 0.1]])  # order 10, 12, 11, 13: ranks 0, 2, 1, 3
    m = missed_keys(topk_hits(torch.tensor([[10, 12, 99, 98]]), want), want, scores)
    assert m["row"].tolist() == [0, 0] and m["rank"].tolist() == [2, 3] and m["selected"].tolist() == [4, 4]
