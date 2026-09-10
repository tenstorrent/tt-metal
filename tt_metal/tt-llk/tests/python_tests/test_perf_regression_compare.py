# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the canonical perf compare module (tt_metal/tt-llk/perf).

This is the module the PR gate runs, so the two-clause rule it enforces —
more than ``threshold`` slower AND more than ``min_cycles`` slower — is what
most of these tests cover. A copy of an older, percentage-only version still
lives in ``.claude/scripts/perf_regression_compare.py`` for
``perf_compare_commits.sh``; the two disagree, and the copy goes away when that
script is repointed here.

Run: pytest test_perf_regression_compare.py
"""

import pathlib
import sys

import pandas as pd

# tt-llk holds a hyphen, so `perf` is not an importable package path; add its
# directory the way both real callers reach it — by filesystem location.
_PERF = pathlib.Path(__file__).parents[2] / "perf"
sys.path.insert(0, str(_PERF))
from regression_compare import (  # noqa: E402
    DEFAULT_MIN_CYCLES,
    DEFAULT_THRESHOLD,
    _medians,
    compare_runs,
)

# Magnitudes are realistic cycle counts, not toy numbers: the verdict depends on
# an absolute cycle floor, so a test written at 100 cycles would prove nothing
# about a TILE_LOOP measured in thousands.
_INIT = 900.0
_TILE_LOOP = 2000.0


def _csv(tmp_path, name, init, tile_loop, run_type="MATH_ISOLATE", tile_cnt=4):
    path = tmp_path / name
    pd.DataFrame(
        {
            "marker": ["INIT", "TILE_LOOP"],
            "tile_cnt": [tile_cnt, tile_cnt],  # a config column -> part of the key
            f"mean({run_type})": [init, tile_loop],
        }
    ).to_csv(path, index=False)
    return str(path)


def _sides(tmp_path, base, cur, **kw):
    baseline = [_csv(tmp_path, f"b{i}.csv", *base, **kw) for i in range(3)]
    current = [_csv(tmp_path, f"c{i}.csv", *cur, **kw) for i in range(3)]
    return current, baseline


def _markers(records):
    return {r["marker"] for r in records}


# --- the two-clause rule ----------------------------------------------------


def test_both_clauses_met_is_a_regression(tmp_path):
    """+10% and +200 cycles on TILE_LOOP. Both clauses hold."""
    current, baseline = _sides(tmp_path, (_INIT, _TILE_LOOP), (_INIT, _TILE_LOOP + 200))

    result = compare_runs(current, baseline)

    assert _markers(result["regressions"]) == {"TILE_LOOP"}
    assert abs(result["regressions"][0]["delta"] - 0.10) < 1e-9


def test_percentage_alone_is_not_a_regression(tmp_path):
    """INIT moves +2.2%, which clears the threshold, but only 20 cycles.

    This is the case the cycle floor exists for: a small marker with jitter.
    """
    current, baseline = _sides(tmp_path, (_INIT, _TILE_LOOP), (_INIT + 20, _TILE_LOOP))

    result = compare_runs(current, baseline)

    assert not result["regressions"]
    assert result["noise_filtered"] == 1


def test_cycles_alone_is_not_a_regression(tmp_path):
    """TILE_LOOP moves +40 cycles, over the floor, but only +2% of a big number."""
    current, baseline = _sides(
        tmp_path, (_INIT, _TILE_LOOP), (_INIT, _TILE_LOOP * 1.019)
    )

    assert (_TILE_LOOP * 0.019) > DEFAULT_MIN_CYCLES  # the cycle clause holds
    assert 0.019 < DEFAULT_THRESHOLD  # the percentage clause does not

    result = compare_runs(current, baseline)

    assert not result["regressions"]


def test_improvement_is_symmetric(tmp_path):
    current, baseline = _sides(tmp_path, (_INIT, _TILE_LOOP), (_INIT, _TILE_LOOP - 200))

    result = compare_runs(current, baseline)

    assert _markers(result["improvements"]) == {"TILE_LOOP"}
    assert not result["regressions"]


def test_min_cycles_zero_disables_the_absolute_clause(tmp_path):
    current, baseline = _sides(tmp_path, (_INIT, _TILE_LOOP), (_INIT + 20, _TILE_LOOP))

    result = compare_runs(current, baseline, min_cycles=0)

    assert _markers(result["regressions"]) == {"INIT"}


# --- points and coverage ----------------------------------------------------


def test_unchanged_run_still_compares_every_point(tmp_path):
    current, baseline = _sides(tmp_path, (_INIT, _TILE_LOOP), (_INIT, _TILE_LOOP))

    result = compare_runs(current, baseline)

    assert not result["regressions"]
    assert len(result["records"]) == 2


def test_new_config_is_reported_not_flagged(tmp_path):
    baseline = [_csv(tmp_path, "b.csv", _INIT, _TILE_LOOP, tile_cnt=4)]
    # tile_cnt=8 exists on the current side only, so it has nothing to compare to.
    current = [_csv(tmp_path, "c.csv", _INIT, _TILE_LOOP * 4, tile_cnt=8)]

    result = compare_runs(current, baseline)

    assert len(result["new_points"]) == 2
    assert not result["regressions"]
    assert not result["records"]  # nothing was actually compared


def test_run_type_filter_keeps_only_what_was_asked_for(tmp_path):
    baseline = [_csv(tmp_path, "b.csv", _INIT, _TILE_LOOP, run_type="L1_TO_L1")]
    current = [_csv(tmp_path, "c.csv", _INIT, _TILE_LOOP + 200, run_type="L1_TO_L1")]

    assert compare_runs(current, baseline, run_types="L1_TO_L1")["regressions"]
    # A filter that matches nothing must empty the comparison, not pass it.
    assert not compare_runs(current, baseline, run_types="MATH_ISOLATE")["records"]


# --- the median ------------------------------------------------------------


def test_median_ignores_one_wild_iteration(tmp_path):
    """Three iterations, one of them absurd. The median must not move."""
    baseline = [_csv(tmp_path, f"b{i}.csv", _INIT, _TILE_LOOP) for i in range(3)]
    current = [
        _csv(tmp_path, "c0.csv", _INIT, _TILE_LOOP),
        _csv(tmp_path, "c1.csv", _INIT, _TILE_LOOP),
        _csv(tmp_path, "c2.csv", _INIT, _TILE_LOOP * 10),
    ]

    result = compare_runs(current, baseline)

    assert not result["regressions"]


def test_medians_tolerate_frames_with_different_columns(tmp_path):
    """Iterations need not agree on their sweep columns.

    A column absent from one frame must stay out of that frame's point key,
    rather than merging its rows into another point.
    """
    wide = pd.DataFrame(
        {
            "marker": ["TILE_LOOP"],
            "tile_cnt": [4],
            "math_fidelity": ["HiFi4"],
            "mean(L1_TO_L1)": [_TILE_LOOP],
        }
    )
    narrow = pd.DataFrame(
        {"marker": ["TILE_LOOP"], "tile_cnt": [4], "mean(L1_TO_L1)": [_TILE_LOOP]}
    )

    medians = _medians([wide, narrow])

    assert len(medians) == 2  # two distinct points, not one merged point


def test_medians_of_no_frames_is_empty():
    assert _medians([]) == {}
    assert _medians([pd.DataFrame()]) == {}
