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
import pytest

_PERF = pathlib.Path(__file__).parents[3] / "perf"
sys.path.insert(0, str(_PERF))
import placement_check  # noqa: E402
from placement_check import (  # noqa: E402
    POSITIONS,
    _changed,
    _tests,
    _value,
    blocks,
    decide,
    gate,
    main,
    nodes,
    read_candidates,
    read_window_map,
    render,
    write_blocking,
)

_BASE = 2000.0
_NODE = "perf_x.py::test_perf_x[4]"


def _side(root, tile_loop, window="aaa", skip=(), init=900.0):
    """One side measured at every position: tile_loop, and init if a list, give a value per position."""
    inits = init if isinstance(init, list) else [init] * len(POSITIONS)
    for pos, value, init_value in zip(POSITIONS, tile_loop, inits):
        if pos in skip:
            continue
        path = root / pos / "perf_x" / "perf_x.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "marker": ["INIT", "TILE_LOOP"],
                "tile_cnt": [4, 4],
                "dest_acc": [False, False],
                "mean(L1_TO_L1)": [init_value, value],
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


def test_other_markers_get_no_loop_label(tmp_path):
    # The TILE_LOOP fingerprint changed, but it says nothing about INIT: no threads, no "partial".
    base = _side(tmp_path / "base", [_BASE] * 5, window="aaa")
    everywhere = _side(tmp_path / "all", [_BASE] * 5, window="bbb", init=1000.0)
    some = _side(
        tmp_path / "some", [_BASE] * 5, window="bbb", init=[1000.0] * 4 + [900.0]
    )

    def init_row(cur):
        results = decide(
            _candidates(tmp_path, init=True),
            base,
            cur,
            threshold=0.05,
            min_cycles=30,
            module_thresholds={},
        )
        return next(r for r in results if r["marker"] == "INIT")

    row = init_row(everywhere)
    assert (row["outcome"], row["threads"]) == ("regression", [])
    row = init_row(some)
    assert (row["outcome"], row["threads"]) == ("placement", [])


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
    assert "> ⚠️ 1 of 2 leg(s) left a stage-2 result." in (tmp_path / "r.md").read_text()


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


def test_gate_blocks_when_a_leg_left_no_baseline_record(tmp_path):
    rows = _cleared(tmp_path)

    _, notes = gate([rows, []], 2, _candidates(tmp_path), "abc", ["abc"])

    assert notes == ["1 leg(s) left no record of their baseline."]


def test_slack_lists_only_the_rows_the_gate_blocks_on(tmp_path):
    rows = _cleared(tmp_path)
    stage1 = _candidates(tmp_path, init=True)
    out = tmp_path / "blocking.csv"

    results, notes = gate([rows], 1, stage1, "abc", ["abc"])
    write_blocking(tmp_path / "s1.regressions.csv", results, notes, out)
    assert pd.read_csv(out)["marker"].tolist() == ["INIT"]

    results, notes = gate([rows], 2, stage1, "abc", ["abc"])
    write_blocking(tmp_path / "s1.regressions.csv", results, notes, out)
    assert pd.read_csv(out)["marker"].tolist() == ["INIT", "TILE_LOOP"]


def test_slack_lists_confirmed_regressions_first(tmp_path):
    stage1 = _candidates(tmp_path, init=True)
    results = [
        dict(stage1[0], outcome="unmeasured"),
        dict(stage1[1], outcome="regression"),
    ]
    out = tmp_path / "blocking.csv"

    write_blocking(tmp_path / "s1.regressions.csv", results, [], out)

    assert pd.read_csv(out)["marker"].tolist() == ["INIT", "TILE_LOOP"]


_CODE_POSITIONS = "0 4 8 12 16 20 24 28"


def _hashes(path, pack):
    if pack is None:
        return {}
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "tile_cnt": [4],
            "dest_acc": [False],
            "window_run_type": ["L1_TO_L1"],
            "window_thread": ["pack"],
            "window_hash": [pack],
            "nodeid": [_NODE],
        }
    ).to_csv(path, index=False)
    return read_window_map([path])


def _code_decide(
    tmp_path,
    base,
    cur,
    baseline="code-a",
    head="code-b",
    candidates=(),
    stable=(),
    tests=None,
):
    return decide(
        list(candidates),
        base,
        cur,
        threshold=0.05,
        min_cycles=30,
        module_thresholds={},
        tests=_tests(_candidates(tmp_path)) if tests is None else tests,
        window_map=_hashes(tmp_path / "head.windowhash", head),
        base_map=_hashes(tmp_path / "baseline.windowhash", baseline),
        stable=stable,
    )


def _measured(tmp_path, base_values, head_values, base="code-a", head="code-b", **kw):
    return _code_decide(
        tmp_path,
        _side(tmp_path / "base", base_values, window=base),
        _side(tmp_path / "cur", head_values, window=head),
        **kw,
    )


def _outcomes(results):
    return {r["marker"]: r["outcome"] for r in results}


@pytest.mark.parametrize(
    "baseline, head, threads",
    [
        pytest.param({"pack": "code-a"}, {"pack": "code-a"}, [], id="equal"),
        pytest.param(
            {"pack": "code-a", "nodeid": "old"},
            {"pack": "code-a", "nodeid": "new"},
            [],
            id="renamed test id",
        ),
        pytest.param(
            {"math": "code-m", "pack": "code-a"},
            {"math": "code-m", "pack": "code-b"},
            ["pack"],
            id="one thread",
        ),
        pytest.param(
            {"pack": "code-a"},
            {"brisc": "code-c", "pack": "code-a"},
            ["brisc"],
            id="thread only on the head",
        ),
        pytest.param(
            {"brisc": "code-c", "pack": "code-a"},
            {"pack": "code-a"},
            ["brisc"],
            id="thread only on the baseline",
        ),
        pytest.param(
            {"pack": float("nan")}, {"pack": float("nan")}, ["pack"], id="no hash"
        ),
        pytest.param({"pack": ""}, {"pack": ""}, ["pack"], id="empty hash"),
        pytest.param({"pack": "aaa"}, {"pack": "aaa"}, ["pack"], id="loop hash"),
        pytest.param(
            {"pack": "no-objdump"}, {"pack": "no-objdump"}, ["pack"], id="no objdump"
        ),
        pytest.param({"pack": "no-code"}, {"pack": "no-code"}, ["pack"], id="no code"),
        pytest.param(None, {"pack": "code-a"}, ["all", "pack"], id="no baseline"),
        pytest.param({}, {"pack": "code-a"}, ["all", "pack"], id="empty baseline"),
        pytest.param({"pack": "code-a"}, None, ["pack"], id="no head"),
        pytest.param(None, None, ["all"], id="neither"),
    ],
)
def test_code_is_the_same_only_with_equal_code_hashes(baseline, head, threads):
    assert _changed(baseline, head) == threads


def test_code_first_selects_only_the_tests_with_changed_code(tmp_path):
    tests = _tests(_candidates(tmp_path))
    same = _hashes(tmp_path / "a.windowhash", "code-a")
    other = _hashes(tmp_path / "b.windowhash", "code-b")
    empty = _hashes(tmp_path / "c.windowhash", "")

    assert nodes(_candidates(tmp_path), same, 0, tests, same) == ([], [])
    assert nodes([], other, 0, tests, same) == ([_NODE], [])
    assert nodes([], empty, 0, tests, empty) == ([_NODE], [])
    assert nodes([], same, 0, tests, {}) == ([_NODE], [])
    assert nodes([], other, 0, set(), same) == ([], [])
    assert nodes([], {}, 0, tests, same) == ([], [])


def test_code_first_selects_a_test_once():
    keys = [("perf_a", "L1_TO_L1", frozenset({("tile_cnt", str(n))})) for n in (1, 2)]
    head = {k: {"nodeid": "a", "pack": "code-b"} for k in keys}

    assert nodes([], head, 0, set(keys), {}) == (["a"], [])


def test_same_code_passes_a_stage1_regression(tmp_path):
    results = _code_decide(
        tmp_path,
        tmp_path / "base",
        tmp_path / "cur",
        head="code-a",
        candidates=_candidates(tmp_path, init=True),
    )

    assert _outcomes(results) == {"TILE_LOOP": "same", "INIT": "same"}
    assert not blocks(results)


def test_same_code_gets_no_row_without_a_stage1_regression(tmp_path):
    assert (
        _code_decide(tmp_path, tmp_path / "base", tmp_path / "cur", head="code-a") == []
    )


@pytest.mark.parametrize(
    "base, head, outcome",
    [
        pytest.param(2000.0, [2400.0] * 5, "changed-slower", id="slower everywhere"),
        pytest.param(
            2000.0, [2600.0] + [2000.0] * 4, "changed-slower", id="mean above"
        ),
        pytest.param(2000.0, [2400.0] + [2000.0] * 4, "changed", id="mean below"),
        pytest.param(
            2000.0, [2400.0, 1600.0] + [2000.0] * 3, "changed", id="mean zero"
        ),
        pytest.param(2000.0, [2100.0] * 5, "changed", id="at the threshold"),
        pytest.param(100.0, [130.0] * 5, "changed", id="at the cycle floor"),
        pytest.param(100.0, [131.0] * 5, "changed-slower", id="above the cycle floor"),
        pytest.param(2000.0, [2000.0] * 5, "changed", id="no move"),
        pytest.param(2000.0, [1600.0] * 5, "changed", id="faster"),
    ],
)
def test_changed_code_is_slower_by_the_mean_of_the_positions(
    tmp_path, base, head, outcome
):
    results = _measured(tmp_path, [base] * 5, head)

    assert _outcomes(results) == {"INIT": "changed", "TILE_LOOP": outcome}
    assert {tuple(r["threads"]) for r in results} == {("pack",)}
    assert blocks(results)


def test_changed_code_in_a_stable_module_passes_when_not_slower(tmp_path):
    results = _measured(tmp_path, [_BASE] * 5, [_BASE] * 5, stable=["perf_x"])

    assert _outcomes(results) == {"INIT": "changed-ok", "TILE_LOOP": "changed-ok"}
    assert not blocks(results)


def test_changed_code_in_a_stable_module_blocks_when_slower(tmp_path):
    results = _measured(tmp_path, [_BASE] * 5, [_BASE * 1.2] * 5, stable=["perf_x"])

    assert _outcomes(results) == {"INIT": "changed-ok", "TILE_LOOP": "changed-slower"}
    assert blocks(results)


def test_another_stable_module_does_not_clear_the_test(tmp_path):
    results = _measured(tmp_path, [_BASE] * 5, [_BASE] * 5, stable=["perf_y"])

    assert _outcomes(results) == {"INIT": "changed", "TILE_LOOP": "changed"}


def test_module_threshold_applies_to_changed_code(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="code-a")
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 5, window="code-b")

    results = decide(
        [],
        base,
        cur,
        threshold=0.05,
        min_cycles=30,
        module_thresholds={"perf_x": 0.25},
        tests=_tests(_candidates(tmp_path)),
        window_map={},
        base_map={},
    )

    assert _outcomes(results) == {"INIT": "changed", "TILE_LOOP": "changed"}


def test_a_stage1_regression_of_changed_code_is_one_row(tmp_path):
    results = _measured(
        tmp_path, [_BASE] * 5, [_BASE * 1.2] * 5, candidates=_candidates(tmp_path)
    )

    assert sorted((r["marker"], r["delta_pct"]) for r in results) == [
        ("INIT", 0.0),
        ("TILE_LOOP", 20.0),
    ]


def test_changed_code_that_was_not_measured_blocks(tmp_path):
    [result] = _code_decide(tmp_path, tmp_path / "base", tmp_path / "cur")

    assert (result["marker"], result["outcome"], result["why"]) == (
        "",
        "unmeasured",
        "not measured at every position on both sides",
    )
    assert blocks([result])


@pytest.mark.parametrize(
    "baseline, head", [(None, "code-a"), ("code-a", None), ("", "")]
)
def test_a_test_with_no_fingerprint_reads_as_changed(tmp_path, baseline, head):
    [result] = _code_decide(
        tmp_path, tmp_path / "base", tmp_path / "cur", baseline=baseline, head=head
    )

    assert result["outcome"] == "unmeasured"


def test_changed_code_with_a_missing_position_blocks(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="code-a")
    cur = _side(tmp_path / "cur", [_BASE] * 5, window="code-b", skip=("p20",))

    results = _code_decide(tmp_path, base, cur)

    assert _outcomes(results) == {"INIT": "unmeasured", "TILE_LOOP": "unmeasured"}


def test_changed_code_that_the_base_did_not_measure_blocks(tmp_path):
    cur = _side(tmp_path / "cur", [_BASE] * 5, window="code-b")

    results = _code_decide(tmp_path, tmp_path / "base", cur)

    assert {r["why"] for r in results} == {
        "not measured on the base (renamed or new test?)"
    }
    assert _outcomes(results) == {"INIT": "unmeasured", "TILE_LOOP": "unmeasured"}


def test_the_same_code_as_the_base_clears_a_stale_baseline(tmp_path):
    results = _measured(
        tmp_path,
        [_BASE] * 5,
        [_BASE * 1.2] * 5,
        base="code-b",
        candidates=_candidates(tmp_path),
    )

    assert _outcomes(results) == {"TILE_LOOP": "same"}
    assert not blocks(results)


def test_code_changed_against_the_base_is_not_cleared_by_the_baseline(tmp_path):
    results = _measured(tmp_path, [_BASE] * 5, [_BASE * 1.2] * 5, baseline="code-b")

    assert _outcomes(results) == {"INIT": "changed", "TILE_LOOP": "changed-slower"}


@pytest.mark.parametrize("base, head", [("aaa", "aaa"), ("aaa", "code-b"), ("", "")])
def test_a_base_with_no_code_hash_does_not_clear_the_test(tmp_path, base, head):
    results = _measured(tmp_path, [_BASE] * 5, [_BASE] * 5, base=base, head=head)

    assert _outcomes(results) == {"INIT": "changed", "TILE_LOOP": "changed"}


def test_a_stage1_regression_outside_the_compared_tests_blocks(tmp_path):
    [result] = _code_decide(
        tmp_path,
        tmp_path / "base",
        tmp_path / "cur",
        head="code-a",
        candidates=_candidates(tmp_path),
        tests=set(),
    )

    assert result["outcome"] == "unmeasured"


def test_a_test_with_no_baseline_gets_no_row(tmp_path):
    assert _measured(tmp_path, [_BASE] * 5, [_BASE * 1.2] * 5, tests=set()) == []


def test_fingerprints_do_not_change_the_rule_without_the_tests(tmp_path):
    base = _side(tmp_path / "base", [_BASE] * 5, window="code-a")
    cur = _side(tmp_path / "cur", [_BASE * 1.2] * 4 + [_BASE], window="code-a")

    [result] = decide(
        _candidates(tmp_path),
        base,
        cur,
        threshold=0.05,
        min_cycles=30,
        module_thresholds={},
        window_map=_hashes(tmp_path / "head.windowhash", "code-a"),
        base_map=_hashes(tmp_path / "baseline.windowhash", "code-a"),
        stable=["perf_x"],
    )

    assert result["outcome"] == "placement"


@pytest.mark.parametrize(
    "outcome, blocking",
    [
        ("changed-slower", True),
        ("changed", True),
        ("changed-ok", False),
        ("same", False),
    ],
)
def test_which_code_first_outcomes_block(outcome, blocking):
    assert blocks([{"outcome": outcome}]) is blocking


def test_code_first_report_of_a_block(tmp_path):
    report = render(
        _measured(tmp_path, [_BASE] * 5, [_BASE * 1.2] * 5),
        "wormhole",
        code_first=True,
    )

    assert report.startswith("## Wormhole stage 2: ❌ 2 with changed code")
    assert (
        "Stage 2 measured every test with changed code with the TRISC code at 5 positions"
        in report
    )
    assert "- 2 row(s)" in report
    assert "B | code changed in |" in report
    assert "### Code changed and slower: all 1" in report
    assert (
        "### Code changed in a layout-sensitive test: needs the "
        "llk-perf-accept-regression label: all 1" in report
    )
    assert "| +20.0% / +20.0% / +20.0% / +20.0% / +20.0% | pack |" in report


def test_code_first_report_of_a_pass(tmp_path):
    cleared = _measured(tmp_path, [_BASE] * 5, [_BASE] * 5, stable=["perf_x"])
    same = _code_decide(
        tmp_path,
        tmp_path / "none",
        tmp_path / "none",
        head="code-a",
        candidates=_candidates(tmp_path),
    )

    report = render(cleared + same, "wormhole", code_first=True)

    assert report.startswith("## Wormhole stage 2: ✅ no confirmed regression")
    assert "<summary>Code changed, not slower: all 2</summary>" in report
    assert "<summary>Same code as the baseline: all 1</summary>" in report
    assert "###" not in report


def test_code_first_report_of_unmeasured_changed_code(tmp_path):
    report = render(
        _code_decide(tmp_path, tmp_path / "base", tmp_path / "cur"),
        "wormhole",
        code_first=True,
    )

    assert report.startswith("## Wormhole stage 2: ❌ 1 not re-measured")


def _gate_main(tmp_path, monkeypatch, legs, rows="[]", stage1="none.csv"):
    monkeypatch.setattr(placement_check, "POSITIONS", placement_check.POSITIONS)
    leg = tmp_path / "legs" / "1"
    leg.mkdir(parents=True, exist_ok=True)
    (leg / "stage2.json").write_text(rows)
    (leg / "stage2.base").write_text("abc\n")
    return main(
        [
            "gate",
            "--results",
            f"{tmp_path}/legs/**/stage2.json",
            "--bases",
            f"{tmp_path}/legs/**/stage2.base",
            "--stage1",
            str(tmp_path / stage1),
            "--gate-base",
            "abc",
            "--report",
            str(tmp_path / "r.md"),
            "--blocking-csv",
            str(tmp_path / "blocking.csv"),
            "--positions",
            _CODE_POSITIONS,
            "--legs",
            str(legs),
        ]
    )


def test_code_first_gate_passes_a_clean_stage1_when_every_leg_reported(
    tmp_path, monkeypatch
):
    assert _gate_main(tmp_path, monkeypatch, legs=1) == 0
    report = (tmp_path / "r.md").read_text()
    assert report.startswith("## Stage 2: ✅ no confirmed regression")
    assert "every test with changed code with the TRISC code at 8 positions" in report
    assert not (tmp_path / "blocking.csv").exists()


@pytest.mark.parametrize("legs", [0, 2])
def test_code_first_gate_blocks_a_clean_stage1_with_a_missing_leg(
    tmp_path, monkeypatch, legs
):
    assert _gate_main(tmp_path, monkeypatch, legs=legs) == 1
    assert (
        f"> ⚠️ 1 of {legs} leg(s) left a stage-2 result."
        in (tmp_path / "r.md").read_text()
    )


@pytest.mark.parametrize(
    "outcome, code",
    [("changed-slower", 1), ("changed", 1), ("changed-ok", 0), ("same", 0)],
)
def test_code_first_gate_exit_code_follows_the_rows(
    tmp_path, monkeypatch, outcome, code
):
    row = dict(
        _candidates(tmp_path)[0],
        marker="INIT",
        deltas=[0.0] * 8,
        threads=["pack"],
        slower=0,
        outcome=outcome,
        why="",
    )

    assert _gate_main(tmp_path, monkeypatch, legs=1, rows=json.dumps([row])) == code
    assert (
        "+0 / +4 / +8 / +12 / +16 / +20 / +24 / +28 B | code changed in |"
        in (tmp_path / "r.md").read_text()
    )


def test_code_first_gate_blocks_a_stage1_regression_no_leg_judged(
    tmp_path, monkeypatch
):
    _candidates(tmp_path)

    assert _gate_main(tmp_path, monkeypatch, legs=1, stage1="s1.regressions.csv") == 1
    assert pd.read_csv(tmp_path / "blocking.csv")["marker"].tolist() == ["TILE_LOOP"]


def test_code_first_gate_lists_no_stage1_row_for_a_same_code_regression(
    tmp_path, monkeypatch
):
    rows = _code_decide(
        tmp_path,
        tmp_path / "base",
        tmp_path / "cur",
        head="code-a",
        candidates=_candidates(tmp_path),
    )

    assert (
        _gate_main(
            tmp_path,
            monkeypatch,
            legs=1,
            rows=json.dumps(rows, default=str),
            stage1="s1.regressions.csv",
        )
        == 0
    )
    assert pd.read_csv(tmp_path / "blocking.csv").empty


def _leg(tmp_path, monkeypatch, baseline, head, values, points=True):
    positions = tuple(f"p{s}" for s in _CODE_POSITIONS.split())
    monkeypatch.setattr(sys.modules[__name__], "POSITIONS", positions)
    monkeypatch.setattr(placement_check, "POSITIONS", placement_check.POSITIONS)
    _candidates(tmp_path)
    _hashes(tmp_path / "baseline" / "perf-data-wormhole-1" / "x.windowhash", baseline)
    _hashes(tmp_path / "cur" / "run" / "x.windowhash", head)
    code = [
        "--points",
        str(tmp_path / ("s1.regressions.csv" if points else "none.csv")),
        "--base-map",
        f"{tmp_path}/baseline/**/*.windowhash",
        "--stable",
        "perf_y perf_z",
    ]
    main(
        [
            "nodes",
            "--candidates",
            str(tmp_path / "none.csv"),
            "--map",
            f"{tmp_path}/cur/**/*.windowhash",
            "--cap",
            "1000000",
            "--out",
            str(tmp_path / "nodes.txt"),
            *code,
        ]
    )
    selected = (tmp_path / "nodes.txt").read_text().split()
    if selected:
        _side(tmp_path / "base", [_BASE] * 8, window=baseline)
        _side(tmp_path / "cur", values, window=head)
    main(
        [
            "decide",
            "--candidates",
            str(tmp_path / "none.csv"),
            "--baseline",
            str(tmp_path / "base"),
            "--current",
            str(tmp_path / "cur"),
            "--threshold",
            "0.01",
            "--min-cycles",
            "30",
            "--positions",
            _CODE_POSITIONS,
            "--arch",
            "wormhole",
            "--report",
            str(tmp_path / "stage2.md"),
            "--map",
            f"{tmp_path}/cur/**/*.windowhash",
            *code,
        ]
    )
    return selected, json.loads((tmp_path / "stage2.json").read_text())


def test_a_leg_with_changed_code_measures_it_at_eight_positions(tmp_path, monkeypatch):
    selected, results = _leg(
        tmp_path, monkeypatch, "code-a", "code-b", [_BASE * 1.02] * 7 + [_BASE]
    )

    assert selected == [_NODE]
    assert _outcomes(results) == {"INIT": "changed", "TILE_LOOP": "changed-slower"}
    assert [len(r["deltas"]) for r in results] == [8, 8]
    assert "at 8 positions" in (tmp_path / "stage2.md").read_text()


def test_a_leg_with_the_same_code_measures_nothing(tmp_path, monkeypatch):
    assert _leg(tmp_path, monkeypatch, "code-a", "code-a", []) == ([], [])


def test_a_leg_with_no_baseline_fingerprints_measures_every_test(tmp_path, monkeypatch):
    selected, results = _leg(tmp_path, monkeypatch, None, "code-b", [_BASE] * 8)

    assert selected == [_NODE]
    assert _outcomes(results) == {"INIT": "changed", "TILE_LOOP": "changed"}


def test_a_leg_that_compared_nothing_measures_nothing(tmp_path, monkeypatch):
    assert _leg(tmp_path, monkeypatch, "code-a", "code-b", [], points=False) == ([], [])
