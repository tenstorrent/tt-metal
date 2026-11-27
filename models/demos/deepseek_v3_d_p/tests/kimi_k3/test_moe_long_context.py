# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host contracts of the K3.1 MoE long-context scoring, with hand-computed values: the routing readback places each
chip's owner rows; a scored chunk reports the absolute and relative error, recall, dropped weight and miss ranks the
hand case has, and the bars fail on it."""

from __future__ import annotations

import math

import pytest
import torch

from models.demos.common.prefill.long_context_accuracy.acceptance_bars import check
from models.demos.common.prefill.long_context_accuracy.windowed_errors import WindowedErrors
from models.demos.deepseek_v3_d_p.tests.kimi_k3.moe_long_context import BARS, append_chunk, measure, owner_positions


def test_owner_positions_hand_case() -> None:
    """6 real rows a SP row plus 2 padding rows, 2 row blocks of 4, TP 2: column 0 owns block rows 0-1 of each block
    (chunk rows 6 r + [0, 1, 4, 5]); column 1 owns 2-3 of each, but rows 6-7 are padding, so only [2, 3]."""
    owned, pos = owner_positions(1, 0, rows_per_sp_row=6, pad_rows=2, row_blocks=2, tp=2)
    assert owned.tolist() == [0, 1, 2, 3] and pos.tolist() == [6, 7, 10, 11]
    owned, pos = owner_positions(1, 1, rows_per_sp_row=6, pad_rows=2, row_blocks=2, tp=2)
    assert owned.tolist() == [0, 1] and pos.tolist() == [8, 9]


def test_a_scored_chunk_reports_the_hand_case() -> None:
    """3 rows, hidden 2, top-2. Output: rows 0 and 2 exact, row 1 off by 1 in one element (reference [0, 2]).
    Routing: row 0 the FP64 set; row 1 misses FP64 rank 1 (weight 0.4); row 2 misses FP64 rank 0 (weight 0.6)."""
    series = WindowedErrors()
    want_output = torch.tensor([[3.0, 4.0], [0.0, 2.0], [6.0, 8.0]])
    append_chunk(
        series,
        0,
        output=torch.tensor([[3.0, 4.0], [0.0, 1.0], [6.0, 8.0]], dtype=torch.bfloat16),
        ids=torch.tensor([[0, 1], [2, 9], [9, 5]]),
        weights=torch.tensor([[0.6, 0.4], [0.6, 0.4], [0.5, 0.4]], dtype=torch.bfloat16),
        want_output=want_output,
        want_ids=torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32),
        want_weights=torch.tensor([[0.6, 0.4], [0.6, 0.4], [0.6, 0.4]]),
    )
    # reference row norms 5, 2, 10: floor 1% of the median, 0.05
    w = series.whole(measure(0.05))
    assert w["moe_rms_err"] == pytest.approx(math.sqrt(1 / 6))  # one error of 1 over 6 elements
    assert w["moe_rms_ref"] == pytest.approx(math.sqrt((25 + 4 + 100) / 6))
    assert w["moe_nrmse"] == pytest.approx(math.sqrt(1 / 129))
    assert w["moe_row_nrmse_median"] == 0.0 and w["moe_row_nrmse_max"] == pytest.approx(0.5)
    assert w["recall_mean"] == pytest.approx(2 / 3) and w["set_equal_fraction"] == pytest.approx(1 / 3)
    assert w["dropped_weight_mean"] == pytest.approx((0.4 + 0.6) / 3)
    assert w["misses"] == 2 and w["miss_top_half_fraction"] == pytest.approx(0.5)  # (rank 0 of 2 is the top half)
    failures = check([w], BARS)
    assert [f.split()[0] for f in failures] == ["recall_mean", "miss_top_half_fraction", "dropped_weight_mean"]


def test_misses_land_in_the_window_of_their_absolute_row() -> None:
    """Two 2-row chunks; only the second (rows 2-3) misses an expert, so only window [2, 4) counts it."""
    series = WindowedErrors()
    out = torch.ones(2, 2)
    for start, ids in ((0, [[0, 1], [0, 1]]), (2, [[0, 1], [0, 7]])):
        append_chunk(
            series,
            start,
            output=out,
            ids=torch.tensor(ids),
            weights=torch.full((2, 2), 0.5),
            want_output=out,
            want_ids=torch.tensor([[0, 1], [0, 1]]),
            want_weights=torch.full((2, 2), 0.5),
        )
    windows = series.windows(2, measure(0.01))
    assert [w["misses"] for w in windows] == [0, 1]
    assert windows[0]["recall_mean"] == 1.0 and windows[1]["recall_mean"] == pytest.approx(0.75)
    assert check(windows[:1], BARS) == []
