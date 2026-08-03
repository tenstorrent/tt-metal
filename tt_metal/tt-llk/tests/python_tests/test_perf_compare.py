# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests for compare-to-history (regression detection)."""

import pandas as pd
from helpers.perf_compare import compare_to_history
from helpers.perf_parquet import write_run_batch

_RUN_PROV = dict(
    commit_sha="abc123",
    arch="wormhole",
    run_id="run",
    timestamp="t",
    pipeline="nightly",
    pr_number=None,
)


def _batch(tmp_path, name, mean_math, tile_cnt=(4, 4)):
    df = pd.DataFrame(
        {
            "marker": ["INIT", "TILE_LOOP"],
            "tile_cnt": list(tile_cnt),
            "mean(MATH_ISOLATE)": list(mean_math),
        }
    )
    path = tmp_path / f"{name}.parquet"
    write_run_batch({"perf_a": df}, path, **dict(_RUN_PROV, run_id=name))
    return path


def test_detects_regression(tmp_path):
    history = [
        _batch(tmp_path, "h1", [100.0, 200.0]),
        _batch(tmp_path, "h2", [102.0, 198.0]),
    ]
    current = _batch(tmp_path, "cur", [130.0, 205.0])  # INIT ~+29%, TILE ~+3%

    result = compare_to_history(current, history, threshold=0.05)

    regs = {(r["marker"], r["run_type"]) for r in result["regressions"]}
    assert ("INIT", "MATH_ISOLATE") in regs
    assert ("TILE_LOOP", "MATH_ISOLATE") not in regs  # within threshold


def test_no_regression_within_threshold(tmp_path):
    history = [_batch(tmp_path, "h1", [100.0, 200.0])]
    current = _batch(tmp_path, "cur", [103.0, 202.0])  # ~3%, ~1%

    result = compare_to_history(current, history, threshold=0.05)

    assert result["regressions"] == []
    assert len(result["records"]) == 2


def test_new_config_has_no_baseline(tmp_path):
    history = [_batch(tmp_path, "h1", [100.0, 200.0], tile_cnt=(4, 4))]
    # a config not seen in history -> no baseline, reported as a new point
    current = _batch(tmp_path, "cur", [100.0, 200.0], tile_cnt=(8, 8))

    result = compare_to_history(current, history, threshold=0.05)

    assert result["regressions"] == []
    assert result["records"] == []
    assert len(result["new_points"]) == 2
