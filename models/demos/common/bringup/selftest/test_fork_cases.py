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
