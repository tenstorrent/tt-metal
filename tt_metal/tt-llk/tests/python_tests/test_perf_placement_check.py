# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for stage 2 of the perf gate (tt_metal/tt-llk/perf/placement_check.py).

Run: pytest test_perf_placement_check.py
"""

import pathlib
import sys

import pandas as pd

_PERF = pathlib.Path(__file__).parents[2] / "perf"
sys.path.insert(0, str(_PERF))
from placement_check import (  # noqa: E402
    POSITIONS,
    _value,
    decide,
    nodes,
    read_candidates,
    read_window_map,
    render,
)

_BASE = 2000.0
_NODE = "perf_x.py::test_perf_x[4]"


def _side(root, tile_loop, window="aaa", skip=()):
    """One side measured at every position: tile_loop is a value per position."""
    for pos, value in zip(POSITIONS, tile_loop):
        if pos in skip:
            continue
        path = root / pos / "perf_x" / "perf_x.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "marker": ["INIT", "TILE_LOOP"],
                "tile_cnt": [4, 4],
                "dest_acc": [False, False],
                "mean(L1_TO_L1)": [900.0, value],
            }
        ).to_csv(path, index=False)
    pd.DataFrame(
        {
            "tile_cnt": [4],
            "dest_acc": [False],
            "window_run_type": ["L1_TO_L1"],
            "window_thread": ["pack"],
            "window_hash": [window],
            "nodeid": [_NODE],
        }
    ).to_csv(root / "p0" / "perf_x.1.windowhash", index=False)
    return root


def _candidates(tmp_path, delta_pct=20.0):
    path = tmp_path / "s1.regressions.csv"
    pd.DataFrame(
        {
            "marker": ["TILE_LOOP", "INIT"],
            "run_type": ["L1_TO_L1", "L1_TO_L1"],
            "current": [_BASE * 1.2, 1000.0],
            "baseline": [_BASE, 900.0],
            "delta_pct": [delta_pct, 11.0],
            "delta_cycles": [400.0, 100.0],
            "test_module": ["perf_x", "perf_x"],
            "tile_cnt": ["4.0", "4.0"],
            "dest_acc": ["False", "False"],
        }
    ).to_csv(path, index=False)
    return read_candidates(path)


def _decide(tmp_path, base, cur):
    return decide(_candidates(tmp_path), base, cur, 0.05, 30, {})


def test_values_match_across_csv_and_typed_forms():
    assert _value(False) == _value("False") == "False"
    assert _value(4.0) == _value("4.0") == _value(4) == "4"


def test_only_tile_loop_rows_are_candidates(tmp_path):
    assert [c["run_type"] for c in _candidates(tmp_path)] == ["L1_TO_L1"]


def test_slower_everywhere_with_changed_loop_code_is_confirmed(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="aaa")
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 5, window="bbb")

    [result] = _decide(tmp_path, base, cur)

    assert result["outcome"] == "regression"
    assert result["threads"] == ["pack"]
    assert result["deltas"] == [20.0] * 5


def test_changed_loop_slower_at_some_positions_is_partial(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="aaa")
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 4 + [_BASE], window="bbb")

    [result] = _decide(tmp_path, base, cur)

    assert result["outcome"] == "partial"
    assert result["slower"] == 4


def test_same_loop_slower_at_some_positions_is_a_placement_effect(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="aaa")
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 4 + [_BASE], window="aaa")

    [result] = _decide(tmp_path, base, cur)

    assert result["outcome"] == "placement"
    assert result["slower"] == 4


def test_unchanged_loop_code_is_for_review(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="aaa")
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 5, window="aaa")

    [result] = _decide(tmp_path, base, cur)

    assert result["outcome"] == "review"


def test_missing_fingerprint_is_unmeasured(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="no-window")
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 5, window="bbb")

    [result] = _decide(tmp_path, base, cur)

    assert (result["outcome"], result["why"]) == ("unmeasured", "no window fingerprint")


def test_position_not_measured_is_unmeasured(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5)
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 5, window="bbb", skip=("p36",))

    [result] = _decide(tmp_path, base, cur)

    assert (result["outcome"], result["why"]) == (
        "unmeasured",
        "not measured at every position on both sides",
    )


def test_not_slower_anywhere_is_not_reproduced(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="aaa")
    cur = _side(tmp_path / "cur", [_BASE] * 5, window="bbb")

    [result] = _decide(tmp_path, base, cur)

    assert result["outcome"] == "not-reproduced"


def test_nodes_maps_candidates_to_their_test(tmp_path):
    _side(tmp_path / "cur", [_BASE] * 5)
    window_map = read_window_map([tmp_path / "cur" / "p0" / "perf_x.1.windowhash"])

    assert nodes(_candidates(tmp_path), window_map, cap=10) == [_NODE]
    assert nodes(_candidates(tmp_path), window_map, cap=0) == []


def test_report_shows_every_outcome(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="aaa")
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 5, window="bbb")
    [confirmed] = _decide(tmp_path, base, cur)
    others = [
        dict(confirmed, outcome=o, why="")
        for o in ("review", "partial", "placement", "not-reproduced")
    ]

    report = render([confirmed] + others, "wormhole")

    assert report.startswith("## Wormhole stage 2: ❌ 1 confirmed regression(s)")
    assert "### Confirmed regressions: all 1" in report
    assert (
        "### Slower at every position, loop code unchanged: needs a look: all 1"
        in report
    )
    assert (
        "### Loop code changed, slower at some positions only: needs a look: all 1"
        in report
    )
    assert "### Code-placement effects (hardware): all 1" in report
    assert "<summary>Not reproduced: all 1</summary>" in report
    assert "+20.0% / +20.0% / +20.0% / +20.0% / +20.0%" in report


def test_placement_only_is_not_a_failure_but_is_reported(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="aaa")
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 4 + [_BASE], window="aaa")

    report = render(_decide(tmp_path, base, cur), "wormhole")

    assert "ℹ️ no confirmed regression, code-placement effects only" in report
    assert "### Code-placement effects (hardware): all 1" in report
