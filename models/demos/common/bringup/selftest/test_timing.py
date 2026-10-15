# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F70: the time breakdown from agent transcripts, gate history, the run timeline and the device log."""

import json

from models.demos.common.bringup import timing as TL


def ev(t, typ, content=None, **kw):
    e = {"type": typ, "timestamp": f"2026-10-15T10:00:{t:02d}.000Z", **kw}
    if content is not None:
        e["message"] = {"content": content}
    return json.dumps(e)


def test_agent_times_split_device_tools_and_model(tmp_path):
    lines = [
        ev(0, "assistant", [{"type": "tool_use", "id": "a", "input": {"command": "scripts/run_safe_pytest.sh t.py"}}]),
        ev(20, "user", [{"type": "tool_result", "tool_use_id": "a"}]),
        ev(25, "assistant", [{"type": "tool_use", "id": "b", "input": {"command": "grep x y"}}]),
        ev(30, "user", [{"type": "tool_result", "tool_use_id": "b"}]),
        json.dumps({"type": "result", "duration_ms": 40000, "duration_api_ms": 12000}),
    ]
    p = tmp_path / "C.swa.kv.implement.2.jsonl"
    p.write_text("\n".join(lines))
    a = TL.agent_times(p)
    assert (a["task"], a["role"], a["attempt"]) == ("C.swa.kv", "implement", "2")
    assert (a["wall"], a["api"], a["device"], a["tools"]) == (40.0, 12.0, 20.0, 5.0)


def test_breakdown_gates_idle_and_device_log(tmp_path):
    bdir, rdir = tmp_path / "b", tmp_path / "r"
    (rdir / "agents").mkdir(parents=True)
    bdir.mkdir()
    state = {
        "_run": {"created": "2026-10-15T10:00:00"},
        "G.1": {
            "duration_s": 50,
            "history": [{"t": "x", "status": "PASS", "s": 30}, {"t": "y", "status": "PASS", "s": 20}],
        },
        "C.1": {"duration_s": 7, "history": [{"t": "x", "status": "PASS"}]},
    }
    (bdir / "state.json").write_text(json.dumps(state))
    for t, e, kw in (
        ("10:00:00", "start", {}),
        ("10:10:00", "exit", {"code": 3}),
        ("10:40:00", "start", {}),
        ("10:50:00", "start", {}),
        ("11:00:00", "exit", {"code": 4}),
    ):
        (rdir / TL.TIMELINE).open("a").write(json.dumps({"t": f"2026-10-15T{t}", "event": e, **kw}) + "\n")
    (rdir / TL.DEVICE_LOG).write_text(json.dumps({"wait_ms": 3000, "run_ms": 9000}) + "\n")
    tasks = {"C.1": {"device": True}, "G.1": {}}
    now = TL.local_ts("2026-10-15T11:30:00")
    res = TL.breakdown(bdir, rdir, tasks, now=now)
    t = res["totals"]
    assert t["gates on CPU"] == 50 and t["gates on device"] == 7  # history runs summed; old entries: last duration
    assert t["idle: waiting for a person"] == 1800 and t["idle: not running"] == 600  # killed at 10:40, restarted 10:50
    assert t["idle: paused (now)"] == 1800 and res["elapsed"] == 5400
    assert res["device_log"] == {"runs": 1, "lock_wait": 3.0, "run": 9.0}
    assert "waiting for a person" in TL.table(res)
