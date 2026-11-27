# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MoE routing error on small hand-computed values. CPU only."""

import pytest
import torch

from models.demos.common.prefill.long_context_accuracy.moe.routing_error import routing_rows


def test_routing_rows_hand_case() -> None:
    """Reference experts 0-3 with weights .4 .3 .2 .1 (best first); the device picks 0, 1, 2 and 7 instead of 3, with
    weight .25 on expert 1: recall 3/4, dropped .1 (expert 3), weight error .05 on expert 1, one miss at rank 3."""
    want, want_w = torch.tensor([[0, 1, 2, 3]]), torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    got, got_w = torch.tensor([[0, 1, 2, 7]]), torch.tensor([[0.4, 0.25, 0.2, 0.15]])
    order = -torch.arange(4.0).expand(1, 4)  # the reference's own order
    r = routing_rows(got, got_w, want, want_w, order)
    assert r["recall"].tolist() == [0.75] and r["dropped"].tolist() == pytest.approx([0.1])
    assert r["weight_err"].tolist() == pytest.approx([0.05])
    assert r["misses"]["rank"].tolist() == [3]
    same = routing_rows(want, want_w, want, want_w, order)
    assert (
        same["recall"].tolist() == [1.0] and same["dropped"].tolist() == [0.0] and same["weight_err"].tolist() == [0.0]
    )
    assert same["misses"]["row"].numel() == 0


def test_routing_rows_rank_misses_by_the_reference_scores() -> None:
    """Row 0: reference experts 3 / 5 / 7 / 1 at weights 0.4 / 0.3 / 0.2 / 0.1, scores order 3 > 5 > 7 > 1; the
    device picks {3, 5, 2, 1} at 0.5 / 0.25 / 0.15 / 0.1: recall 3/4, expert 7 missed at rank 2 of 4, dropped
    reference weight 0.2, largest weight difference on the shared picks |0.4 - 0.5| = 0.1. Row 1: the same set in
    another order: recall 1, nothing dropped."""
    want = torch.tensor([[3, 5, 7, 1], [0, 1, 2, 3]])
    want_w = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]])
    got = torch.tensor([[3, 5, 2, 1], [3, 2, 1, 0]])
    got_w = torch.tensor([[0.5, 0.25, 0.15, 0.1], [0.25, 0.25, 0.25, 0.25]])
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    r = routing_rows(got, got_w, want, want_w, scores)
    assert r["recall"].tolist() == pytest.approx([0.75, 1.0])
    assert r["dropped"].tolist() == pytest.approx([0.2, 0.0])
    assert r["weight_err"].tolist() == pytest.approx([0.1, 0.0])
    assert r["misses"]["row"].tolist() == [0] and r["misses"]["rank"].tolist() == [2]
