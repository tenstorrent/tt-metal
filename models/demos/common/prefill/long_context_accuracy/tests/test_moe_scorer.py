# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host contracts of ``moe/scorer.MoeScorer``, with hand-computed values: a scored chunk reports the absolute and
relative error, recall, dropped weight, miss ranks and gain the hand case has, in the metric tiers its docstring names
(one tier per metric name); misses land in the window of their absolute row; ``check_chunk`` reports non-finite
output, a row routed to one expert twice and a chunk over its bars. CPU only."""

from __future__ import annotations

import math

import pytest
import torch

from models.demos.common.prefill.long_context_accuracy.acceptance_bars import AcceptanceBar, check
from models.demos.common.prefill.long_context_accuracy.block import METRIC_TIERS, BlockOutputs
from models.demos.common.prefill.long_context_accuracy.moe.scorer import MoeScorer

BARS = (
    AcceptanceBar("moe_row_nrmse_median", "max", 0.20),
    AcceptanceBar("recall_mean", "min", 0.98),
    AcceptanceBar("miss_top_half_fraction", "max", 0.01),
    AcceptanceBar("dropped_weight_mean", "max", 0.01),
)


def _scorer(window: int = 16) -> MoeScorer:
    return MoeScorer(output="out", expert_ids="ids", expert_weights="weights", window=window)


def _golden_rows(output, ids, weights) -> dict[str, torch.Tensor]:
    return {"out": torch.tensor(output), "ids": torch.tensor(ids, dtype=torch.int32), "weights": torch.tensor(weights)}


def _outputs(output, ids, weights) -> BlockOutputs:
    return BlockOutputs(
        torch.tensor(output, dtype=torch.bfloat16),
        {"expert_ids": torch.tensor(ids), "expert_weights": torch.tensor(weights, dtype=torch.bfloat16)},
    )


def _hand_case() -> tuple[BlockOutputs, dict[str, torch.Tensor]]:
    """3 rows, hidden 2, top-2. Output: rows 0 and 2 exact, row 1 off by 1 in one element (reference [0, 2]).
    Routing: row 0 the golden's set; row 1 misses golden rank 1 (weight 0.4); row 2 misses golden rank 0 (0.6)."""
    outputs = _outputs(
        [[3.0, 4.0], [0.0, 1.0], [6.0, 8.0]], [[0, 1], [2, 9], [9, 5]], [[0.6, 0.4], [0.6, 0.4], [0.5, 0.4]]
    )
    rows = _golden_rows([[3.0, 4.0], [0.0, 2.0], [6.0, 8.0]], [[0, 1], [2, 3], [4, 5]], [[0.6, 0.4]] * 3)
    return outputs, rows


def test_a_scored_run_reports_the_hand_case_in_its_tiers() -> None:
    scorer = _scorer()
    outputs, rows = _hand_case()
    scorer.score_chunk(0, outputs, rows)
    tiers = scorer.score_run({}, None)
    assert sorted(tiers) == list(METRIC_TIERS)
    span = {"start": 0, "end": 3, "rows": 3}
    names = [set().union(*(set(m) - set(span) for m in tiers[t])) for t in METRIC_TIERS]
    assert not (names[0] & names[1] or names[0] & names[2] or names[1] & names[2]), "a metric name in two tiers"
    assert all(set(span) <= set(m) for t in METRIC_TIERS for m in tiers[t])

    # tier 0: the run's NRMSE (one error of 1 against Σy² = 25 + 4 + 100) and recall
    assert tiers[0] == [
        {**span, "run_moe_nrmse": pytest.approx(math.sqrt(1 / 129)), "run_recall_mean": pytest.approx(2 / 3)}
    ]
    # tier 1: the run's other measures, then each window (here one partial window: the same values, unprefixed)
    run, *windows = tiers[1]
    assert len(windows) == 1 and windows[0]["start"] == 0 and windows[0]["end"] == 3
    # (reference row norms 5, 2, 10: floor 1% of the median, 0.05; row 1's NRMSE 1 / 2)
    want = {
        "moe_rms_err": pytest.approx(math.sqrt(1 / 6)),  # (one error of 1 over 6 elements)
        "moe_rms_ref": pytest.approx(math.sqrt(129 / 6)),
        "moe_row_nrmse_median": 0.0,
        "moe_row_nrmse_p99": pytest.approx(0.5 - 0.02 * 0.5),  # (between the sorted rows 0, 0, 0.5: 0.98 of the way)
        "moe_row_nrmse_max": pytest.approx(0.5),
        "dropped_weight_mean": pytest.approx((0.4 + 0.6) / 3),
        "dropped_weight_p99": pytest.approx(0.6 - 0.02 * 0.2),
        "misses": 2,
        "miss_top_half_fraction": 0.5,  # (rank 0 of 2 is the top half)
    }
    assert run == {**span, **{f"run_{k}": v for k, v in want.items()}}
    assert windows[0] == {
        **span,
        **want,
        "moe_nrmse": pytest.approx(math.sqrt(1 / 129)),
        "recall_mean": pytest.approx(2 / 3),
    }
    # tier 2: gain <x, y> / <y, y> = (25 + 2 + 100) / 129, the misses' golden ranks 1 and 0, the windows' trends
    (diag,) = tiers[2]
    assert diag["run_moe_gain"] == pytest.approx(127 / 129)
    assert diag["run_miss_rank_histogram"] == [1, 1]
    assert diag["recall_mean_trend"] == {
        "first": pytest.approx(2 / 3),
        "last": pytest.approx(2 / 3),
        "worst": pytest.approx(2 / 3),
        "worst_start": 0,
        "worst_over_first": 1.0,
    }
    assert sorted(k for k in diag if k.endswith("_trend")) == sorted(
        f"{m}_trend"
        for m in (
            "moe_rms_err",
            "moe_rms_ref",
            "moe_nrmse",
            "moe_row_nrmse_median",
            "recall_mean",
            "dropped_weight_mean",
        )
    )
    # the window bars fail on the window only; the run's measures carry other names
    failures = check([m for t in METRIC_TIERS for m in tiers[t]], BARS)
    assert [f.split()[0] for f in failures] == ["recall_mean", "miss_top_half_fraction", "dropped_weight_mean"]
    assert all(f.endswith("in [0, 3)") for f in failures)


def test_misses_land_in_the_window_of_their_absolute_row() -> None:
    """Two 2-row chunks; only the second (rows 2-3) misses an expert, so only window [2, 4) counts it."""
    scorer = _scorer(window=2)
    out = [[1.0, 1.0], [1.0, 1.0]]
    for start, ids in ((0, [[0, 1], [0, 1]]), (2, [[0, 1], [0, 7]])):
        scorer.score_chunk(
            start, _outputs(out, ids, [[0.5, 0.5]] * 2), _golden_rows(out, [[0, 1]] * 2, [[0.5, 0.5]] * 2)
        )
    run, *windows = scorer.score_run({}, None)[1]
    assert run["run_misses"] == 1
    assert [(w["start"], w["misses"]) for w in windows] == [(0, 0), (2, 1)]
    assert windows[0]["recall_mean"] == 1.0 and windows[1]["recall_mean"] == pytest.approx(0.75)
    assert check(windows[:1], BARS) == []


def test_check_chunk_reports_contract_breaks_and_a_chunk_over_its_bars() -> None:
    scorer = _scorer()
    good = [[1.0, 2.0], [3.0, 4.0]]
    rows = _golden_rows(good, [[0, 1], [2, 3]], [[0.5, 0.5]] * 2)
    exact = _outputs(good, [[1, 0], [3, 2]], [[0.5, 0.5]] * 2)  # (the golden's sets, in another order)
    scorer.score_chunk(0, exact, rows)
    assert scorer.check_chunk(0, exact, rows, BARS) == []

    nan = _outputs([[1.0, float("nan")], [3.0, 4.0]], [[0, 1], [2, 3]], [[0.5, 0.5]] * 2)
    scorer.score_chunk(2, nan, rows)
    assert scorer.check_chunk(2, nan, rows, BARS) == ["non-finite MoE output"]

    twice = _outputs(good, [[0, 1], [2, 2]], [[0.5, 0.5]] * 2)
    scorer.score_chunk(4, twice, rows)
    assert scorer.check_chunk(4, twice, rows, BARS) == ["a row routed to one expert twice"]

    # row 1 misses golden rank 0 (of 2): recall 0.75, dropped weight 0.25 / 2, its one miss in the top half
    missing = _outputs(good, [[0, 1], [3, 9]], [[0.5, 0.5]] * 2)
    scorer.score_chunk(6, missing, rows)
    failures = scorer.check_chunk(6, missing, rows, BARS)
    assert [f.split()[0] for f in failures] == ["recall_mean", "miss_top_half_fraction", "dropped_weight_mean"]
    assert all(f.endswith("in [6, 8)") for f in failures)
    assert scorer.check_chunk(6, missing, rows, ()) == []
