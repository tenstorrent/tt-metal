# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The windowed scoring core on small hand-computed values. CPU only, no device."""

import math

import pytest
import torch

from models.demos.deepseek_v3_d_p.tests.accuracy.windowed import (
    Bar,
    GrowthBar,
    RowSeries,
    check,
    nrmse,
    pcc,
    row_nrmse,
    row_sums,
    trend,
)

# Golden rows of norm 5 (3, 4) or 0; the device is off by (0, 1) on rows 1 and 3, so those rows have error norm 1.
EXPECTED = torch.tensor([[3.0, 4.0], [3.0, 4.0], [0.0, 0.0], [3.0, 4.0]])
ACTUAL = torch.tensor([[3.0, 4.0], [3.0, 5.0], [0.0, 0.0], [3.0, 5.0]])


def test_row_sums_and_metrics_match_hand_values() -> None:
    sums = row_sums(EXPECTED, ACTUAL)
    assert sums.dtype == torch.float64
    # Row 1 (y 3, 4; x 3, 5): ȳ 3.5, x̄ 4, Σ(y − ȳ)² 0.5, Σ(x − x̄)² 2, Σ(x − x̄)(y − ȳ) 1, Σy² 25, Σe² 1, n 2.
    assert sums[1].tolist() == [3.5, 4.0, 0.5, 2.0, 1.0, 25.0, 1.0, 2.0]
    assert nrmse(sums) == pytest.approx(math.sqrt(2 / 75))  # Σe² 2 over Σy² 75
    assert nrmse(sums[:1]) == 0.0 and nrmse(sums[2:3]) == 0.0  # exact, also on the all-zero row
    assert nrmse(row_sums(torch.zeros(1, 2), torch.ones(1, 2))) == math.inf
    assert pcc(row_sums(torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[2.0, 4.0, 6.0]]))) == pytest.approx(1.0)
    assert math.isnan(pcc(row_sums(torch.ones(1, 3), torch.ones(1, 3))))
    # Per row: 0, 1/5, 0 (exact zero row), 1/5; a floor of 10 caps the denominator from below.
    assert row_nrmse(sums, floor=0.5).tolist() == pytest.approx([0.0, 0.2, 0.0, 0.2])
    assert row_nrmse(sums, floor=10.0).tolist() == pytest.approx([0.0, 0.1, 0.0, 0.1])


@pytest.mark.parametrize("offset", [0.0, 1e8])
def test_pcc_is_exact_on_data_with_a_large_mean(offset: float) -> None:
    """y = c + (1, 2, 3, 4), x = c + (1, 2, 3, 5): about their means 2.5 and 2.75, Σdy² 5, Σdx² 8.75 and Σdxdy 6.5,
    so PCC = 6.5 / √43.75 = 13 / (5√7) for any offset c. Split over two rows, so the row merge is exercised too.
    At c = 1e8 raw sums are ~4e16 and an uncentred formula cancels to noise."""
    expected = offset + torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    actual = offset + torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float64)
    assert pcc(row_sums(expected, actual)) == pytest.approx(13 / (5 * math.sqrt(7)), rel=1e-12)


def test_row_sums_flatten_trailing_dims_and_reject_mismatched_shapes(expect_error) -> None:
    sums = row_sums(EXPECTED.reshape(4, 1, 2), ACTUAL.reshape(4, 1, 2))
    assert torch.equal(sums, row_sums(EXPECTED, ACTUAL))
    with expect_error(ValueError, "matching"):
        row_sums(EXPECTED, ACTUAL[:3])


def test_windows_align_to_absolute_rows_whatever_the_chunking() -> None:
    sums = row_sums(EXPECTED, ACTUAL)
    one, two = RowSeries(), RowSeries()
    one.append(0, sums)
    for start in range(4):  # one-row chunks
        two.append(start, sums[start : start + 1])
    metrics = {"nrmse": nrmse}
    assert one.windows(3, metrics) == two.windows(3, metrics)
    windows = one.windows(3, metrics)
    assert [(w["start"], w["end"], w["rows"]) for w in windows] == [(0, 3, 3), (3, 4, 1)]  # last window partial
    assert windows[0]["nrmse"] == pytest.approx(math.sqrt(1 / 50)) and windows[1]["nrmse"] == pytest.approx(0.2)

    late = RowSeries(start=2)  # a series from row 2 still uses windows aligned to row 0
    late.append(2, sums[2:])
    assert [(w["start"], w["end"]) for w in late.windows(3, metrics)] == [(2, 3), (3, 4)]


def test_series_refuses_a_gap_or_overlap(expect_error) -> None:
    series = RowSeries()
    series.append(0, row_sums(EXPECTED[:2], ACTUAL[:2]))
    with expect_error(ValueError, "does not continue"):
        series.append(3, row_sums(EXPECTED[:1], ACTUAL[:1]))
    with expect_error(ValueError, "does not continue"):
        series.append(1, row_sums(EXPECTED[:1], ACTUAL[:1]))


WINDOWS = [
    {"start": 0, "end": 10, "e": 0.010, "p": 0.999},
    {"start": 10, "end": 20, "e": 0.012, "p": 0.998},
    {"start": 20, "end": 30, "e": 0.011},  # this window does not measure p
]


def test_trend_reports_first_last_and_worst() -> None:
    assert trend(WINDOWS, "e", "max") == {
        "first": 0.010,
        "last": 0.011,
        "worst": 0.012,
        "worst_start": 10,
        "worst_over_first": pytest.approx(1.2),
    }
    assert trend(WINDOWS, "p", "min")["worst"] == 0.998 and trend(WINDOWS, "p", "min")["last"] == 0.998


def test_bars_pass_at_the_value_and_name_each_breaking_window() -> None:
    assert check(WINDOWS, [Bar("e", "max", 0.012), Bar("p", "min", 0.998), GrowthBar("e", 1.2)]) == []
    failures = check(WINDOWS, [Bar("e", "max", 0.0115), Bar("p", "min", 0.9985), GrowthBar("e", 1.1)])
    assert failures == [
        "e 0.012 > 0.0115 in [10, 20)",
        "p 0.998 < 0.9985 in [10, 20)",
        "e grows 1.2x (> 1.1) from 0.01 to 0.012 at row 10",
    ]


def test_bars_fail_closed() -> None:
    nan = [{"start": 0, "end": 1, "e": math.nan}]
    assert check(nan, [Bar("e", "max", 1.0)]) == ["e nan is not finite in [0, 1)"]
    assert check(nan, [GrowthBar("e", 2.0)]) == ["e nan is not finite in [0, 1)"]
    assert check(WINDOWS, [Bar("missing", "max", 1.0)]) == ["missing: not measured by any window"]
    exact_first = [{"start": 0, "end": 1, "e": 0.0}, {"start": 1, "end": 2, "e": 1e-9}]
    assert check(exact_first, [GrowthBar("e", 1000.0)]) == ["e grows infx (> 1000.0) from 0 to 1e-09 at row 1"]
