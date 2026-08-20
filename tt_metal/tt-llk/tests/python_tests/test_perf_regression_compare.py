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
from perf_regression_compare import compare_runs  # noqa: E402


def _csv(tmp_path, name, math):
    p = tmp_path / name
    pd.DataFrame(
        {
            "marker": ["INIT", "KERNEL"],
            "tile_cnt": [4, 4],  # a config column -> part of the point key
            "mean(MATH_ISOLATE)": [math, math + 10.0],
        }
    ).to_csv(p, index=False)
    return str(p)


def test_flags_regression_median_vs_median(tmp_path):
    baseline = [_csv(tmp_path, f"b{i}.csv", 100.0) for i in range(3)]
    current = [_csv(tmp_path, f"c{i}.csv", 120.0) for i in range(3)]  # +20%

    result = compare_runs(current, baseline, threshold=0.05)

    assert result["regressions"]
    assert abs(result["regressions"][0]["delta"] - 0.20) < 1e-9


def test_no_regression_within_threshold(tmp_path):
    baseline = [_csv(tmp_path, f"b{i}.csv", 100.0) for i in range(3)]
    current = [_csv(tmp_path, f"c{i}.csv", 102.0) for i in range(3)]  # +2%, under 5%

    result = compare_runs(current, baseline, threshold=0.05)

    assert not result["regressions"]
    assert result["records"]  # still compared


def test_new_config_reported_not_regression(tmp_path):
    baseline = [_csv(tmp_path, "b.csv", 100.0)]
    # current has a config (tile_cnt=8) with no baseline -> a "new point"
    p = tmp_path / "c.csv"
    pd.DataFrame(
        {"marker": ["INIT"], "tile_cnt": [8], "mean(MATH_ISOLATE)": [999.0]}
    ).to_csv(p, index=False)

    result = compare_runs([str(p)], baseline, threshold=0.05)

    assert result["new_points"]
    assert not result["regressions"]
