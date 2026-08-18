# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the perf noise-analysis script (lives in .claude/scripts).

Hardware-free: builds perf_data-shaped CSV snapshots in tmp_path.

Run: pytest test_perf_noise_analysis.py
"""

import pathlib
import sys

import pandas as pd

# The script lives in the skill's scripts dir, not on the test path; add it.
_SCRIPTS = pathlib.Path(__file__).parents[2] / ".claude" / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from perf_noise_analysis import (  # noqa: E402
    _percentile,
    _simulate_gate_deltas,
    analyze,
    load_run,
    render_report,
)


def _run_dir(tmp_path, name, values, test="perf_math_matmul"):
    """One perf_data snapshot: <run>/<test>/<test>.csv with two markers."""
    run = tmp_path / name / test
    run.mkdir(parents=True)
    pd.DataFrame(
        {
            "marker": ["INIT", "TILE_LOOP"],
            "tile_cnt": [4, 4],  # a config column -> part of the point key
            "mean(MATH_ISOLATE)": list(values),
        }
    ).to_csv(run / f"{test}.csv", index=False)
    return str(tmp_path / name)


def test_load_run_keys_points_by_test_marker_config_and_run_type(tmp_path):
    path = _run_dir(tmp_path, "run_1", [100.0, 200.0])

    points = load_run(path)

    assert set(points) == {
        ("perf_math_matmul", "INIT", (("tile_cnt", 4),), "MATH_ISOLATE"),
        ("perf_math_matmul", "TILE_LOOP", (("tile_cnt", 4),), "MATH_ISOLATE"),
    }
    assert (
        points[("perf_math_matmul", "TILE_LOOP", (("tile_cnt", 4),), "MATH_ISOLATE")]
        == 200.0
    )


def test_post_and_counters_csvs_are_ignored(tmp_path):
    path = _run_dir(tmp_path, "run_1", [100.0, 200.0])
    side = pathlib.Path(path) / "perf_math_matmul"
    pd.DataFrame({"marker": ["INIT"], "mean(MATH_ISOLATE)": [1.0]}).to_csv(
        side / "perf_math_matmul.post.csv", index=False
    )

    assert len(load_run(path)) == 2  # the .post.csv contributed nothing


def test_identical_runs_report_zero_noise(tmp_path):
    runs = [_run_dir(tmp_path, f"run_{i}", [100.0, 200.0]) for i in range(5)]

    result = analyze(runs, min_cycles=0.0)

    assert result["n_runs"] == 5
    assert all(p["cv"] == 0.0 and p["spread"] == 0.0 for p in result["points"])
    assert max(result["sims"][1]["overall"]) == 0.0


def test_noise_floor_tracks_the_observed_spread(tmp_path):
    # TILE_LOOP swings 200 -> 220 (10% of the low value); INIT is stable.
    values = [200.0, 210.0, 220.0, 205.0, 215.0]
    runs = [_run_dir(tmp_path, f"run_{i}", [100.0, v]) for i, v in enumerate(values)]

    result = analyze(runs, min_cycles=0.0)

    tile_loop = next(p for p in result["points"] if p["marker"] == "TILE_LOOP")
    assert tile_loop["min"] == 200.0 and tile_loop["max"] == 220.0
    assert abs(tile_loop["spread"] - 20.0 / 210.0) < 1e-9
    # A single-run gate could see the full 200 -> 220 swing.
    assert abs(max(result["sims"][1]["overall"]) - 0.10) < 1e-9


def test_median_of_two_is_quieter_than_single_run(tmp_path):
    values = [200.0, 210.0, 220.0, 205.0, 215.0]
    runs = [_run_dir(tmp_path, f"run_{i}", [100.0, v]) for i, v in enumerate(values)]

    result = analyze(runs, min_cycles=0.0)

    assert max(result["sims"][2]["overall"]) < max(result["sims"][1]["overall"])


def test_min_cycles_excludes_small_points_from_the_floor(tmp_path):
    # A 10-cycle point swinging by 2 cycles is 20% noise; a floor must drop it.
    runs = []
    for i, small in enumerate([10.0, 12.0, 10.0, 11.0, 12.0]):
        runs.append(_run_dir(tmp_path, f"run_{i}", [small, 1000.0]))

    with_small = analyze(runs, min_cycles=0.0)
    without_small = analyze(runs, min_cycles=100.0)

    assert max(with_small["sims"][1]["overall"]) > 0.15
    assert max(without_small["sims"][1]["overall"]) == 0.0


def test_points_missing_from_a_run_are_reported_not_compared(tmp_path):
    runs = [_run_dir(tmp_path, f"run_{i}", [100.0, 200.0]) for i in range(4)]
    partial = tmp_path / "run_4" / "perf_math_matmul"
    partial.mkdir(parents=True)
    pd.DataFrame(
        {"marker": ["INIT"], "tile_cnt": [4], "mean(MATH_ISOLATE)": [100.0]}
    ).to_csv(partial / "perf_math_matmul.csv", index=False)
    runs.append(str(tmp_path / "run_4"))

    result = analyze(runs, min_cycles=0.0)

    missing = result["incomplete"]
    assert len(missing) == 1
    assert missing[0]["marker"] == "TILE_LOOP"
    assert missing[0]["present_in"] == 4 and missing[0]["of_runs"] == 5


def test_simulated_gate_groups_are_disjoint():
    # k=2 over 5 runs: C(5,2) baselines x C(3,2) currents = 30 comparisons.
    deltas = _simulate_gate_deltas([1.0, 2.0, 3.0, 4.0, 5.0], 2)

    assert len(deltas) == 30
    # Disjoint groups mean no comparison can be a run against itself (delta 0
    # only where the medians genuinely coincide, never trivially).
    assert all(d[0] >= 0 for d in deltas)


def test_percentile_never_lands_below_an_observed_sample():
    values = sorted([0.01, 0.02, 0.03, 0.20])

    # Nearest-rank, rounding up: p99 of 4 samples is the largest one.
    assert _percentile(values, 0.99) == 0.20
    assert _percentile(values, 0.50) == 0.02


def test_report_renders_the_recommendation(tmp_path):
    values = [200.0, 210.0, 220.0, 205.0, 215.0]
    runs = [_run_dir(tmp_path, f"run_{i}", [100.0, v]) for i, v in enumerate(values)]
    result = analyze(runs, min_cycles=100.0)

    report = render_report(
        result, min_cycles=100.0, run_paths=runs, meta={"arch": "wormhole"}
    )

    assert "Recommended threshold" in report
    assert "arch: wormhole" in report
    assert "median-of-1" in report and "median-of-2" in report
