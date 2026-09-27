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
