# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for stage 2 of the perf gate (tt_metal/tt-llk/perf/placement_check.py).

Run: pytest tests/hardware_free/perf/test_perf_placement_check.py
"""

import json
import pathlib
import sys

import pandas as pd

_PERF = pathlib.Path(__file__).parents[3] / "perf"
sys.path.insert(0, str(_PERF))
from placement_check import (  # noqa: E402
    POSITIONS,
    _value,
    blocks,
    decide,
    gate,
    main,
    nodes,
    read_candidates,
    read_window_map,
    render,
)

_BASE = 2000.0
_NODE = "perf_x.py::test_perf_x[4]"


def _side(root, tile_loop, window="aaa", skip=(), init=900.0):
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
                "mean(L1_TO_L1)": [init, value],
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


def _candidates(tmp_path, delta_pct=20.0, init=False):
    path = tmp_path / "s1.regressions.csv"
    frame = pd.DataFrame(
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
    )
    (frame if init else frame[frame["marker"] == "TILE_LOOP"]).to_csv(path, index=False)
    return read_candidates(path)


def _decide(tmp_path, base, cur, over_cap=()):
    return decide(
        _candidates(tmp_path),
        base,
        cur,
        threshold=0.05,
        min_cycles=30,
        module_thresholds={},
        over_cap=over_cap,
    )


def test_values_match_across_csv_and_typed_forms():
    assert _value(False) == _value("False") == "False"
    assert _value(4.0) == _value("4.0") == _value(4) == "4"


def test_every_marker_is_a_candidate(tmp_path):
    assert [c["marker"] for c in _candidates(tmp_path, init=True)] == [
        "TILE_LOOP",
        "INIT",
    ]


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


def test_missing_fingerprint_is_judged_by_the_positions(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="no-window")
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 5, window="bbb")

    [result] = _decide(tmp_path, base, cur)

    assert result["outcome"] == "regression"


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

    assert nodes(_candidates(tmp_path), window_map, cap=10) == ([_NODE], [])
    assert nodes(_candidates(tmp_path), window_map, cap=0) == ([], [0])


def test_the_cap_is_shared_across_modules():
    def cand(module, delta, tile):
        return {
            "module": module,
            "run_type": "L1_TO_L1",
            "config": [("tile_cnt", str(tile))],
            "delta_pct": delta,
        }

    candidates = [cand("perf_a", 50, 1), cand("perf_a", 40, 2), cand("perf_b", 5, 1)]
    window_map = {
        ("perf_a", "L1_TO_L1", frozenset({("tile_cnt", "1")})): {"nodeid": "a1"},
        ("perf_a", "L1_TO_L1", frozenset({("tile_cnt", "2")})): {"nodeid": "a2"},
        ("perf_b", "L1_TO_L1", frozenset({("tile_cnt", "1")})): {"nodeid": "b1"},
    }

    assert nodes(candidates, window_map, cap=2) == (["a1", "b1"], [1])


def test_report_shows_every_outcome(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="aaa")
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 5, window="bbb")
    [confirmed] = _decide(tmp_path, base, cur)
    others = [
        dict(confirmed, outcome=o, why="")
        for o in ("review", "partial", "placement", "not-reproduced")
    ]

    report = render([confirmed] + others, "wormhole")

    assert report.startswith("## Wormhole stage 2: ❌ 2 slower at every code position")
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


def test_nothing_measured_is_not_called_clean(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5)
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 5, window="bbb", skip=("p36",))

    report = render(_decide(tmp_path, base, cur), "wormhole")

    assert "❌ 1 not re-measured" in report


def test_over_the_limit_makes_the_count_a_lower_bound(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5)
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 5, window="bbb")

    results = _decide(tmp_path, base, cur, over_cap=[0])
    report = render(results, "wormhole", "Run the full check.")

    assert (results[0]["outcome"], results[0]["why"]) == (
        "unmeasured",
        "over the test limit",
    )
    assert "**The confirmed count is a lower bound.** 1 stage-1 regression(s)" in report
    assert "Run the full check." in report


def test_render_joins_the_legs(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5)
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 5, window="bbb")
    for leg in ("1", "2"):
        out = tmp_path / "legs" / leg / "stage2.json"
        out.parent.mkdir(parents=True)
        out.write_text(json.dumps(_decide(tmp_path, base, cur), default=str))

    code = main(
        [
            "render",
            "--results",
            f"{tmp_path}/legs/**/stage2.json",
            "--arch",
            "blackhole",
            "--report",
            str(tmp_path / "r.md"),
        ]
    )

    assert code == 1
    assert (
        (tmp_path / "r.md")
        .read_text()
        .startswith("## Blackhole stage 2: ❌ 2 slower at every code position")
    )


def test_what_blocks(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="aaa")
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 5, window="bbb")
    [row] = _decide(tmp_path, base, cur)

    for outcome, blocking in (
        ("regression", True),
        ("review", True),
        ("unmeasured", True),
        ("partial", False),
        ("placement", False),
        ("not-reproduced", False),
    ):
        assert blocks([dict(row, outcome=outcome)]) is blocking


def test_other_markers_are_judged_by_the_positions(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="aaa")
    cur = _side(tmp_path / "cur", [_BASE] * 5, window="aaa", init=1000.0)

    results = decide(
        _candidates(tmp_path, init=True),
        base,
        cur,
        threshold=0.05,
        min_cycles=30,
        module_thresholds={},
    )

    assert [(r["marker"], r["outcome"]) for r in results] == [
        ("TILE_LOOP", "not-reproduced"),
        ("INIT", "regression"),
    ]
    assert blocks(results)


def test_positions_measured_in_parts_are_read(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="aaa")
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 5, window="bbb")
    for root in (base, cur):
        for pos in POSITIONS:
            (root / pos).rename(root / f"{pos}-1")

    [result] = _decide(tmp_path, base, cur)

    assert result["outcome"] == "regression"


def _cleared(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="aaa")
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 4 + [_BASE], window="aaa")
    return _decide(tmp_path, base, cur)


def test_gate_passes_when_every_leg_cleared_every_regression(tmp_path):
    rows = _cleared(tmp_path)

    results, notes = gate([rows, []], 2, _candidates(tmp_path), "abc", ["abc", "abc"])

    assert notes == []
    assert not blocks(results)


def test_gate_blocks_when_a_leg_left_no_result(tmp_path):
    rows = _cleared(tmp_path)

    _, notes = gate([rows], 2, _candidates(tmp_path), "abc", ["abc"])

    assert notes == ["1 of 2 leg(s) left a stage-2 result."]


def test_gate_blocks_when_a_leg_used_another_baseline(tmp_path):
    rows = _cleared(tmp_path)

    _, notes = gate([rows], 1, _candidates(tmp_path), "abc", ["def"])

    assert notes == ["A leg compared against def, not the gate's baseline abc."]


def test_gate_blocks_a_stage1_regression_no_leg_judged(tmp_path):
    results, notes = gate([[]], 1, _candidates(tmp_path), "abc", ["abc"])

    assert notes == []
    assert [(r["outcome"], r["why"]) for r in results] == [
        ("unmeasured", "no stage-2 result")
    ]
    assert blocks(results)


def test_gate_command_exit_code(tmp_path):
    rows = _cleared(tmp_path)
    leg = tmp_path / "legs" / "1"
    leg.mkdir(parents=True)
    (leg / "stage2.json").write_text(json.dumps(rows, default=str))
    (leg / "stage2.base").write_text("abc\n")
    args = [
        "gate",
        "--results",
        f"{tmp_path}/legs/**/stage2.json",
        "--bases",
        f"{tmp_path}/legs/**/stage2.base",
        "--stage1",
        str(tmp_path / "s1.regressions.csv"),
        "--gate-base",
        "abc",
        "--report",
        str(tmp_path / "r.md"),
    ]

    assert main(args + ["--legs", "1"]) == 0
    assert main(args + ["--legs", "2"]) == 1
    assert (
        "1 of 2 leg(s) left a stage-2 result. The gate keeps blocking."
        in (tmp_path / "r.md").read_text()
    )


def test_a_test_the_base_does_not_have_is_named(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="aaa")
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 5, window="bbb")
    for pos in POSITIONS:
        for f in (base / pos).rglob("*.csv"):
            f.unlink()

    [result] = _decide(tmp_path, base, cur)

    assert (result["outcome"], result["why"]) == (
        "unmeasured",
        "not measured on the base (renamed or new test?)",
    )
