# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Top-k selection (recall, missed keys) on small hand-computed values. CPU only."""

import math

import pytest
import torch

from models.demos.common.prefill.long_context_accuracy.topk_selection import missed_keys, topk_recall


def test_topk_recall_counts_valid_reference_ids_once() -> None:
    want = torch.tensor([[0, 1, 2, 3], [4, 5, -1, -1], [-1, -1, -1, -1]])
    got = torch.tensor([[3, 1, 9, 9], [4, 0xFFFFFFFF, 6, 7], [1, 2, 3, 4]])
    recall = topk_recall(got, want)
    assert recall[:2].tolist() == [0.5, 0.5] and math.isnan(recall[2])


def test_topk_recall_and_missed_keys_by_hand() -> None:
    """Row 0: reference keys 5 / 6 / 7 / 8 scored 4 / 3 / 2 / 1, device {5, 7, 9, 8}: recall 3/4, one miss, key 6, at
    rank 1 of 4 (top half) and (3 - 1) / (4 - 1) = 2/3 of the spread. Row 1: two valid reference keys (and two -1),
    both found: recall 1, no miss. A device -1 (no key) matches nothing, and duplicates count once."""
    want = torch.tensor([[5, 6, 7, 8], [1, 0, -1, -1]])
    got = torch.tensor([[5, 7, 9, 8], [0, 1, -1, -1]])
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 5.0, 0.0, 0.0]])
    assert topk_recall(got, want).tolist() == pytest.approx([3 / 4, 1.0])
    misses = missed_keys(got, want, scores)
    assert misses["row"].tolist() == [0] and misses["rank"].tolist() == [1] and misses["selected"].tolist() == [4]
    assert misses["spread_share"].tolist() == pytest.approx([2 / 3])
    assert topk_recall(torch.tensor([[-1, 0, 0]]), torch.tensor([[0, 1, 2]])).tolist() == pytest.approx([1 / 3])


def test_missed_keys_rank_in_the_reference_order() -> None:
    want = torch.tensor([[10, 11, 12, 13]])
    scores = torch.tensor([[0.9, 0.5, 0.7, 0.1]])  # order 10, 12, 11, 13: ranks 0, 2, 1, 3
    m = missed_keys(torch.tensor([[10, 12, 99, 98]]), want, scores)
    assert m["row"].tolist() == [0, 0] and m["rank"].tolist() == [2, 3] and m["selected"].tolist() == [4, 4]
    assert m["spread_share"].tolist() == pytest.approx([0.5, 0.0])  # (0.5 − 0.1) / (0.9 − 0.1); the cut itself
