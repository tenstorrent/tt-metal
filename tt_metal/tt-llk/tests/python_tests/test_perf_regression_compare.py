# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the perf-regression-check compare script (lives in .claude/scripts).

Run: pytest test_perf_regression_compare.py
"""

import pathlib
import sys

import pandas as pd

# The script lives in the skill's scripts dir, not on the test path; add it.
_SCRIPTS = pathlib.Path(__file__).parents[2] / ".claude" / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from perf_regression_compare import (  # noqa: E402
    DEFAULT_MIN_CYCLES,
    DEFAULT_THRESHOLD,
    compare_runs,
    render_report,
)

# Magnitudes are realistic cycle counts, not toy numbers: the verdict now depends
# on the absolute change as well as the relative one, so a fixture measured in
# tens of cycles would be filtered as jitter and test nothing.


def _csv(tmp_path, name, math):
    p = tmp_path / name
    pd.DataFrame(
        {
            "marker": ["INIT", "KERNEL"],
            "tile_cnt": [4, 4],  # a config column -> part of the point key
            "mean(MATH_ISOLATE)": [math, math + 100.0],
        }
    ).to_csv(p, index=False)
    return str(p)


def test_flags_regression_median_vs_median(tmp_path):
    baseline = [_csv(tmp_path, f"b{i}.csv", 1000.0) for i in range(3)]
    current = [_csv(tmp_path, f"c{i}.csv", 1200.0) for i in range(3)]  # +20%

    result = compare_runs(current, baseline, threshold=0.05)

    assert result["regressions"]
    assert abs(result["regressions"][0]["delta"] - 0.20) < 1e-9
    assert result["regressions"][0]["abs_delta"] == 200.0


def test_no_regression_within_threshold(tmp_path):
    baseline = [_csv(tmp_path, f"b{i}.csv", 1000.0) for i in range(3)]
    current = [_csv(tmp_path, f"c{i}.csv", 1020.0) for i in range(3)]  # +2%, under 5%

    result = compare_runs(current, baseline, threshold=0.05)

    assert not result["regressions"]
    assert result["records"]  # still compared


def test_faster_current_is_an_improvement_not_a_regression(tmp_path):
    # Comparing two arbitrary commits, the current side can be the faster one.
    baseline = [_csv(tmp_path, f"b{i}.csv", 1200.0) for i in range(3)]
    current = [_csv(tmp_path, f"c{i}.csv", 1000.0) for i in range(3)]  # -16.7%

    result = compare_runs(current, baseline, threshold=0.05)

    assert not result["regressions"]
    assert len(result["improvements"]) == 2  # both markers
    assert result["improvements"][0]["delta"] < -0.05


# --- the absolute-cycle clause -------------------------------------------------
# Measured on 5 runs of unchanged code: INIT points are a few hundred cycles and
# wobble by up to 25, which is a large percentage of a small number. Without the
# cycle floor those points fail the gate constantly.


def test_small_marker_over_threshold_but_under_cycle_floor_is_not_a_regression(
    tmp_path,
):
    # 350 -> 370 is +5.7%, but only 20 cycles: exactly the INIT jitter we measured.
    baseline = [_csv(tmp_path, "b.csv", 350.0)]
    current = [_csv(tmp_path, "c.csv", 370.0)]

    result = compare_runs(current, baseline, threshold=0.02, min_cycles=30)

    assert not result["regressions"]
    assert (
        result["noise_filtered"] == 2
    )  # both markers cleared %, neither cleared cycles


def test_regression_needs_both_clauses(tmp_path):
    # 1000 -> 1100 is +10% and +100 cycles: both clauses hold.
    baseline = [_csv(tmp_path, "b.csv", 1000.0)]
    current = [_csv(tmp_path, "c.csv", 1100.0)]

    result = compare_runs(current, baseline, threshold=0.02, min_cycles=30)

    assert len(result["regressions"]) == 2
    assert result["noise_filtered"] == 0


def test_large_cycle_move_under_threshold_is_not_a_regression(tmp_path):
    # 100000 -> 100500 is +500 cycles, far over the floor, but only +0.5%.
    # A cycles-only rule would fail here; the percentage clause saves it.
    baseline = [_csv(tmp_path, "b.csv", 100_000.0)]
    current = [_csv(tmp_path, "c.csv", 100_500.0)]

    result = compare_runs(current, baseline, threshold=0.02, min_cycles=30)

    assert not result["regressions"]
    assert result["noise_filtered"] == 0  # never cleared the percentage


def test_improvement_also_needs_the_cycle_floor(tmp_path):
    # A 20-cycle speedup on a small marker is jitter, not an improvement.
    baseline = [_csv(tmp_path, "b.csv", 370.0)]
    current = [_csv(tmp_path, "c.csv", 350.0)]

    result = compare_runs(current, baseline, threshold=0.02, min_cycles=30)

    assert not result["improvements"]


def test_min_cycles_zero_disables_the_clause(tmp_path):
    baseline = [_csv(tmp_path, "b.csv", 350.0)]
    current = [_csv(tmp_path, "c.csv", 370.0)]

    result = compare_runs(current, baseline, threshold=0.02, min_cycles=0)

    assert len(result["regressions"]) == 2


def test_defaults_are_the_measured_ones():
    # These are load-bearing: the shell wrapper documents the same numbers, and
    # the baseline that justifies them is in docs/perf_evaluation/results/.
    assert DEFAULT_THRESHOLD == 0.02
    assert DEFAULT_MIN_CYCLES == 30.0


# --- report --------------------------------------------------------------------


def test_report_names_each_side_and_its_iterations(tmp_path):
    baseline = [_csv(tmp_path, "b.csv", 1000.0)]
    current = [_csv(tmp_path, f"c{i}.csv", 1000.0) for i in range(2)]
    result = compare_runs(current, baseline, threshold=0.05)

    report = render_report(
        result,
        threshold=0.05,
        test="perf_math_matmul",
        baseline_sha="aaaa",
        current_sha="bbbb",
        baseline_iters=1,
        current_iters=2,
        baseline_label="v0.60.0",
        current_label="1a2b3c4",
    )

    assert "- baseline (v0.60.0): `aaaa` — 1 iteration(s)" in report
    assert "- current (1a2b3c4): `bbbb` — 2 iteration(s)" in report
    assert "✅ no regressions" in report


def test_report_states_both_clauses_of_the_rule(tmp_path):
    baseline = [_csv(tmp_path, "b.csv", 1000.0)]
    current = [_csv(tmp_path, "c.csv", 1000.0)]
    result = compare_runs(current, baseline)

    report = render_report(
        result,
        threshold=DEFAULT_THRESHOLD,
        min_cycles=DEFAULT_MIN_CYCLES,
        test="perf_math_matmul",
        baseline_sha="aaaa",
        current_sha="bbbb",
    )

    assert "more than 2% slower AND more than 30 cycles slower" in report


def test_report_explains_points_filtered_by_the_cycle_floor(tmp_path):
    baseline = [_csv(tmp_path, "b.csv", 350.0)]
    current = [_csv(tmp_path, "c.csv", 370.0)]
    result = compare_runs(current, baseline, threshold=0.02, min_cycles=30)

    report = render_report(
        result,
        threshold=0.02,
        min_cycles=30,
        test="perf_math_matmul",
        baseline_sha="aaaa",
        current_sha="bbbb",
    )

    assert "2 point(s) moved more than 2%" in report
    assert "30 cycles or fewer" in report


def test_new_config_reported_not_regression(tmp_path):
    baseline = [_csv(tmp_path, "b.csv", 1000.0)]
    # current has a config (tile_cnt=8) with no baseline -> a "new point"
    p = tmp_path / "c.csv"
    pd.DataFrame(
        {"marker": ["INIT"], "tile_cnt": [8], "mean(MATH_ISOLATE)": [9990.0]}
    ).to_csv(p, index=False)

    result = compare_runs([str(p)], baseline, threshold=0.05)

    assert result["new_points"]
    assert not result["regressions"]
