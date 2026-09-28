# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The derived-op test pass (task O.1): captured ttnn.bringup calls vs the forks' test cases."""

import json

import pytest

from models.demos.common.bringup.testing import fork_cases


def test_every_fork_op_maps_to_its_folder():
    m = fork_cases.op_to_fork()
    assert m["ttnn.bringup.unified_routed_expert_moe"] == "unified_routed_expert_ffn"
    assert m["ttnn.bringup.dispatch"] == "dispatch" and m["ttnn.bringup.offset_cumsum"] == "offset_cumsum"
    assert m["ttnn.bringup.rms_norm"] == "rms_norm_ttnn"  # its C++ binding (rms_norm_ttnn_nanobind.cpp)


def test_uncovered_calls_are_listed_per_fork(tmp_path, monkeypatch):
    cap = tmp_path / "fork_calls.json"
    calls = [
        {"sig": "aaa", "count": 2, "op": "ttnn.bringup.dispatch", "args": [], "kwargs": {}},
        {"sig": "bbb", "count": 1, "op": "ttnn.bringup.combine", "args": [], "kwargs": {}},
    ]
    cap.write_text(json.dumps({"calls": calls}))
    have = {"dispatch": [{"model": "m", "sig": "aaa"}, {"model": "other", "sig": "bbb"}], "combine": []}
    monkeypatch.setattr(fork_cases, "cases", lambda fork: have.get(fork, []))
    res, missing = fork_cases.check(cap, "m")
    assert res["forks"] == ["combine", "dispatch"]
    assert res["stats"] == {"forks_used": 2, "fork_calls": 2, "fork_calls_uncovered": 1}
    assert [c["sig"] for c in missing["combine"]] == ["bbb"] and "dispatch" not in missing


def test_a_call_to_an_unknown_op_stops(tmp_path):
    cap = tmp_path / "fork_calls.json"
    cap.write_text(
        json.dumps({"calls": [{"sig": "x", "count": 1, "op": "ttnn.bringup.nope", "args": [], "kwargs": {}}]})
    )
    with pytest.raises(SystemExit, match="no fork binds"):
        fork_cases.check(cap, "m")


def test_gate_commit_stages_the_task_extras_and_the_fork_calls(fx):
    from models.demos.common.bringup.core.gate import stage_paths
    from models.demos.common.bringup.core.ledger import Ledger
    from models.demos.common.bringup.core.spec import Spec

    s = Spec.load(fx())
    led = Ledger(s.bringup_dir)
    led.results_dir.mkdir(parents=True, exist_ok=True)
    for f in ("O.1.json", "O.1_extra.json", "fork_calls.json", "X.3_profile.json"):
        (led.results_dir / f).write_text("{}")
    names = lambda task: {p.rsplit("/", 1)[-1] for p in stage_paths(s, led, task)}  # noqa: E731
    got = names({"id": "O.1", "step": "optests"})
    assert {"O.1.json", "O.1_extra.json", "fork_calls.json"} <= got and "X.3_profile.json" not in got
    got = names({"id": "X.3", "step": "perf"})
    assert "X.3_profile.json" in got and "fork_calls.json" not in got


def test_source_swap_points_the_original_name_at_the_fork(monkeypatch):
    """fork_source's plugin swaps attributes for the session and puts them back."""
    import sys
    import types

    from models.demos.common.bringup.testing import fork_source as S

    orig_mod = types.ModuleType("fakepkg_orig")
    orig_mod.op = "original"
    fork_mod = types.ModuleType("fakepkg_fork")
    fork_mod.op = "fork"
    monkeypatch.setitem(sys.modules, "fakepkg_orig", orig_mod)
    monkeypatch.setitem(sys.modules, "fakepkg_fork", fork_mod)
    monkeypatch.setenv(S.ENV, json.dumps({"fakepkg_orig.op": "fakepkg_fork.op"}))
    S.pytest_configure(None)
    assert orig_mod.op == "fork"
    S.pytest_unconfigure(None)
    assert orig_mod.op == "original"


def test_source_swap_converts_enum_arguments_at_the_fork_op(monkeypatch):
    """A swap entry with convert_enums wraps the fork op: original-enum arguments become the fork enum's same-named
    members; the original enum itself stays in place for the other original ops."""
    import enum
    import sys
    import types

    from models.demos.common.bringup.testing import fork_source as S

    class Orig(enum.Enum):
        A = 0
        B = 1

    class Fork(enum.Enum):
        A = 0
        B = 1
        C = 2

    orig_mod = types.ModuleType("fakepkg_eorig")
    orig_mod.op, orig_mod.Act = (lambda *a, **k: "original"), Orig
    fork_mod = types.ModuleType("fakepkg_efork")
    fork_mod.Act = Fork

    def fork_op(x, activation=Fork.A, other=None):
        assert isinstance(activation, Fork), activation
        return (x, activation, other)

    fork_mod.op = fork_op
    monkeypatch.setitem(sys.modules, "fakepkg_eorig", orig_mod)
    monkeypatch.setitem(sys.modules, "fakepkg_efork", fork_mod)
    spec = {"op": "fakepkg_efork.op", "convert_enums": {"fakepkg_eorig.Act": "fakepkg_efork.Act"}}
    monkeypatch.setenv(S.ENV, json.dumps({"fakepkg_eorig.op": spec}))
    S.pytest_configure(None)
    try:
        assert orig_mod.Act is Orig
        assert orig_mod.op(1, activation=Orig.B, other="z") == (1, Fork.B, "z")
        assert orig_mod.op(Orig.B) == (Fork.B, Fork.A, None)  # positional too
    finally:
        S.pytest_unconfigure(None)
    assert orig_mod.op() == "original"


def test_source_outcomes_from_junit(tmp_path):
    from models.demos.common.bringup.testing import fork_source as S

    x = tmp_path / "r.xml"
    x.write_text(
        "<testsuites><testsuite>"
        '<testcase classname="m" name="a"/>'
        '<testcase classname="m" name="b"><failure message="x"/></testcase>'
        '<testcase classname="m" name="c"><skipped/></testcase>'
        "</testsuite></testsuites>"
    )
    assert S._outcomes(x) == {"m::a": "passed", "m::b": "failed", "m::c": "skipped"}
