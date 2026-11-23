# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Family scorers (top-k, MoE routing, DSA selection masses, KDA state) on small hand-computed values. CPU only."""

import math

import pytest
import torch

from models.demos.deepseek_v3_d_p.tests.accuracy.dsa import error_split, selection_masses
from models.demos.deepseek_v3_d_p.tests.accuracy.kda import state_errors
from models.demos.deepseek_v3_d_p.tests.accuracy.moe import routing_rows
from models.demos.deepseek_v3_d_p.tests.accuracy.topk import missed_keys, topk_recall


def test_topk_recall_counts_valid_reference_ids_once() -> None:
    want = torch.tensor([[0, 1, 2, 3], [4, 5, -1, -1], [-1, -1, -1, -1]])
    got = torch.tensor([[3, 1, 9, 9], [4, 0xFFFFFFFF, 6, 7], [1, 2, 3, 4]])
    recall = topk_recall(got, want)
    assert recall[:2].tolist() == [0.5, 0.5] and math.isnan(recall[2])


def test_missed_keys_rank_in_the_reference_order() -> None:
    want = torch.tensor([[10, 11, 12, 13]])
    scores = torch.tensor([[0.9, 0.5, 0.7, 0.1]])  # order 10, 12, 11, 13: ranks 0, 2, 1, 3
    m = missed_keys(torch.tensor([[10, 12, 99, 98]]), want, scores)
    assert m["row"].tolist() == [0, 0] and m["rank"].tolist() == [2, 3] and m["selected"].tolist() == [4, 4]
    assert m["spread_share"].tolist() == pytest.approx([0.5, 0.0])  # (0.5 − 0.1) / (0.9 − 0.1); the cut itself


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


def test_selection_masses_hand_case() -> None:
    """Reference keys 5, 6, 7 (best first) with weights .6 .3 .1 in one head; the device picks 5, 6, 9 with weights
    .5 .3 .2: key 7 missed (mass .1, the bottom key too), key 9 an intruder (mass .2)."""
    m = selection_masses(
        torch.tensor([[0.6, 0.3, 0.1]]),
        torch.tensor([[0.5, 0.3, 0.2]]),
        torch.tensor([5, 6, 7]),
        torch.tensor([5, 6, 9]),
    )
    assert (m["missed"], m["intruders"]) == (1, 1)
    assert m["missed_mass"].tolist() == pytest.approx([0.1]) and m["bottom_mass"].tolist() == pytest.approx([0.1])
    assert m["intruder_mass"].tolist() == pytest.approx([0.2]) and m["missed_key_max"] == pytest.approx(0.1)
    none = selection_masses(torch.tensor([[1.0]]), torch.tensor([[1.0]]), torch.tensor([5]), torch.tensor([5]))
    assert none["missed_key_max"] == 0.0 and none["missed_mass"].tolist() == [0.0]


def test_error_split_hand_case() -> None:
    """want 0, the reference over the device's selection (1, 0), the device (1, 1): selection part (1, 0) and rest
    (0, 1) are orthogonal and share the squared error 2 equally."""
    s = error_split(torch.zeros(2), torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0]))
    assert s == {"selection_rest_cos": 0.0, "selection_share": 0.5, "rest_share": 0.5}


def test_state_errors_hand_case() -> None:
    """Two heads of 1x2 recurrent state: head 0 exact (3, 4), head 1 (0, 2) read as (0, 3): ‖e‖ 1 over ‖y‖ √29, RMS
    error √(1/4), RMS √(29/4), head rels 0 and 1/2. Conv: q exact, k off by 1 in (2, 0), v off by 1 in (4, 0)."""
    want = {"recurrent": torch.tensor([[[3.0, 4.0]], [[0.0, 2.0]]])}
    got = {"recurrent": torch.tensor([[[3.0, 4.0]], [[0.0, 3.0]]])}
    for c, (w, g) in {
        "q": ([1.0, 0.0], [1.0, 0.0]),
        "k": ([2.0, 0.0], [3.0, 0.0]),
        "v": ([4.0, 0.0], [5.0, 0.0]),
    }.items():
        want[f"{c}_convolution"], got[f"{c}_convolution"] = torch.tensor(w), torch.tensor(g)
    e = state_errors(got, want)
    assert e["recurrent_rel"] == pytest.approx(1 / math.sqrt(29))
    assert e["recurrent_rms_err"] == pytest.approx(0.5) and e["recurrent_rms_ref"] == pytest.approx(math.sqrt(29 / 4))
    assert e["recurrent_head_rel_max"] == pytest.approx(0.5) and e["recurrent_head_rel_median"] == 0.0
    assert e["conv_channel"] == "k" and e["conv_rel"] == pytest.approx(0.5)
    assert e["conv_rms_err"] == pytest.approx(math.sqrt(0.5)) and e["conv_rms_ref"] == pytest.approx(math.sqrt(2.0))
