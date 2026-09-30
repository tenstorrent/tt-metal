# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F57: perf picks. The agent makes the change; the orchestrator runs the A/B report (change off and on: frozen tests,
ladder rungs, one plain profile) with a fake command runner, then waits for the owner, whose decision it applies.
And the profile guard: a perf pick whose frozen test failed in this attempt is never profiled. CPU only, no device."""

import json
import sys
from pathlib import Path

import pytest

from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.orchestrator import DONE, HUMAN, Orchestrator, decide
from models.demos.common.bringup.selftest.conftest import record_cmd
from models.demos.common.bringup.testing import accuracy_guard

MOCK = f"{sys.executable} -m models.demos.common.bringup.selftest.mock_agent"
LADDER = "models/demos/common/bringup/tests/test_ladder.py"
PROFILE = "models/demos/common/bringup/tests/test_profile.py"
GATE = (
    "scripts/run_safe_pytest.sh --run-all tests/test_c_x.py && BRINGUP_RUNG=last scripts/run_safe_pytest.sh "
    f"--no-precompile --run-all {LADDER} && TT_METAL_DEVICE_PROFILER=1 BRINGUP_PROFILE_OPS=1 "
    f"scripts/run_safe_pytest.sh --run-all --no-precompile {PROFILE}"
)
LADDER_SPEC = [
    {"name": "s64", "seq": 64, "chunk": 32},
    {"name": "last", "seq": 128, "chunk": 32, "golden": "full", "prefix_from_golden": True},
    {"name": "full", "seq": 128, "chunk": 32},
]


def pick(**kw):
    t = {
        "id": "P.1",
        "title": "experts at HiFi2",
        "step": "perf",
        "role": "perf",
        "deps": [],
        "paths": ["src"],
        "brief": {"details": "run the experts at HiFi2"},
        "ab": {"env": {"TOY_FIDELITY": "hifi4"}, "change": {"TOY_FIDELITY": "hifi2"}},
        "gate": {"cmd": GATE, "metrics": {"device_ms_experts": "< 10", "pcc_chunk_out": ">= 0.97"}},
    }
    t.update(kw)
    return t


def fake_runner(calls):
    """Stands in for the device: the change (no TOY_FIDELITY) fails the component check and runs faster."""

    def run_cmd(cmd, env, log):
        on = env.get("TOY_FIDELITY", "hifi2") == "hifi2"
        out = Path(env[M.RESULTS_ENV])
        calls.append({"cmd": cmd, "on": on, "dir": out, "rung": env.get("BRINGUP_RUNG"), "ab": env.get("BRINGUP_AB")})
        rec = {}
        text = ""
        if "test_ladder.py" in cmd:
            rec = {"pcc_layer_L00": 0.999, "pcc_layer_L01": 0.990 if on else 0.995, "pcc_final_hidden": 0.99}
            rec.update(pcc_logits_tail=0.998, top1_match=1.0, top5_overlap=0.95 if on else 1.0)
        elif "test_profile.py" in cmd:
            rec = {"device_ms_total": 90.0 if on else 100.0, "device_ms_experts": 8.0 if on else 12.0}
            rec["pcc_chunk_out"] = 0.99
        elif on:
            text = "FAIL auto_experts_L02_vs_cpu: rel=0.0168 rel_limit=0.0137 (rel 0.01677 > 0.01367)\n"
            text += "FAILED tests/test_c_x.py::test_component - AssertionError\n"
        out.mkdir(parents=True, exist_ok=True)
        (out / "P.1.json").write_text(json.dumps({"metrics": {k: {"value": v} for k, v in rec.items()}}))
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"$ {cmd}\n{text}")
        return 1 if (text and on) else 0

    return run_cmd


@pytest.fixture
def mk(sandbox, monkeypatch, tmp_path):
    script = tmp_path / "script.json"
    monkeypatch.setenv("BRINGUP_AGENT_CMD", MOCK)
    monkeypatch.setenv("MOCK_AGENT_SCRIPT", str(script))
    (sandbox.repo / "src").mkdir()
    (sandbox.repo / "src/impl.txt").write_text("hifi4\n")
    sandbox.git("add", "src")
    sandbox.git("commit", "-q", "-m", "src")
    sandbox.write_spec(ladder=LADDER_SPEC)
    calls, lines = [], []

    def make(tasks, actions):
        script.write_text(json.dumps(actions))
        sandbox.tasks(*tasks)
        o = Orchestrator(sandbox.spec, echo=lines.append)
        o.run_cmd = fake_runner(calls)
        return o

    make.calls, make.lines = calls, lines
    make.agents = (
        lambda: script.with_suffix(".calls").read_text().split("\n")[:-1]
        if script.with_suffix(".calls").exists()
        else []
    )
    return make


CHANGE = {
    "P.1.perf.1.md": {
        "write": {
            "src/impl.txt": "hifi2\n",
            "src/new_path.py": "import os\nF = os.environ.get('TOY_FIDELITY', 'hifi2')\n",
        }
    }
}


def test_the_ab_report_runs_off_and_on_then_waits_for_the_owner(mk):
    o = mk([pick()], CHANGE)
    assert o.run() == HUMAN, mk.lines
    assert mk.agents() == ["P.1.perf.1.md bringup-engineer"]
    st = o.led.state()["P.1"]
    assert "last_run" not in st  # no gate before or after the agent: the gate never ran
    d = o.run_dir / "ab" / "P.1"
    # per config: the frozen test, the ladder at the gate's rung and at the spec's last rung, one plain profile
    got = [(c["on"], c["dir"].relative_to(d).as_posix()) for c in mk.calls]
    names = ["test_c_x", "last", "full", "profile"]
    assert got == [(False, f"old/{n}") for n in names] + [(True, f"new/{n}") for n in names]
    assert all(c["ab"] == "1" for c in mk.calls)  # the profile guard lets the orchestrator's own profile through
    assert [c["rung"] for c in mk.calls if "test_ladder.py" in c["cmd"]] == ["last", "full", "last", "full"]
    prof = [c["cmd"] for c in mk.calls if "test_profile.py" in c["cmd"]]
    assert len(prof) == 2 and all("BRINGUP_PROFILE_OPS" not in c and "TT_METAL_DEVICE_PROFILER=1" in c for c in prof)
    rec = st["ab"]
    assert rec["table"] == str(d / "table.md") and rec["env"] == {"TOY_FIDELITY": "hifi4"}
    assert rec["ladder"]["last"]["new"]["min layer PCC"] == 0.990 and rec["ladder"]["last"]["new"]["min layer"] == "L01"
    assert rec["profile"]["new"]["device_ms_experts"] == 8.0 and rec["profile"]["old"]["device_ms_total"] == 100.0
    assert rec["tests"]["test_c_x"]["new"]["rc"] == 1 and rec["tests"]["test_c_x"]["old"]["rc"] == 0
    assert rec["gate_on"] == {"ok": False, "failing": ["test_c_x: exit code 1"]}  # thresholds hold, the test fails
    table = (d / "table.md").read_text()
    assert "apply or not" in table and "| test_c_x | FAIL (rc 1) | pass |" in table
    assert "test_c_x on: FAIL auto_experts_L02_vs_cpu: rel=0.0168 rel_limit=0.0137" in table
    assert "| min layer PCC | 0.9900 (L01) | 0.9950 (L01) | 0.9900 (L01) | 0.9950 (L01) |" in table
    assert "| device_ms_experts (gate < 10) | 8.0 | 12.0 |" in table and "- test_c_x: exit code 1" in table
    assert "apply P.1 or not" in st["waiting"] and str(d / "table.md") in st["waiting"]
    # nothing reruns while the owner has not decided: no agent, no measurement
    n = len(mk.calls)
    assert o.run() == HUMAN and len(mk.calls) == n and mk.agents() == ["P.1.perf.1.md bringup-engineer"]


def test_without_an_ab_switch_the_change_is_measured_on_only(mk):
    o = mk([pick(ab=None)], CHANGE)
    assert o.run() == HUMAN
    assert [c["on"] for c in mk.calls] == [True] * 4
    rec = o.led.state()["P.1"]["ab"]
    assert "no `ab` switch" in rec["note"] and "no `ab` switch" in (Path(rec["table"])).read_text()


def test_an_agent_that_changed_nothing_measures_nothing(mk):
    o = mk([pick()], {})
    assert o.run() == HUMAN and mk.calls == []
    assert "changed nothing" in o.led.state()["P.1"]["ab"]["note"]


def test_reject_reverts_the_change_and_lets_the_dependents_run(mk, sandbox):
    after = {"id": "X.3", "title": "final", "step": "perf", "deps": ["P.1"], "gate": {"cmd": record_cmd(n=1)}}
    o = mk([pick(), after], CHANGE)
    assert o.run() == HUMAN
    assert decide(o, "P.1", "reject") == 0
    assert o.run() == DONE, mk.lines
    assert (sandbox.repo / "src/impl.txt").read_text() == "hifi4\n" and not (sandbox.repo / "src/new_path.py").exists()
    assert o.led.status("P.1") == "REJECTED" and o.led.status("X.3") == "PASS"
    assert "[toy][P.1] experts at HiFi2 (rejected by the owner)" in sandbox.git("log", "--format=%s")


def test_accept_runs_the_gate_and_commits_the_change(mk, sandbox):
    o = mk([pick(gate={"cmd": record_cmd(device_ms_experts=8.0), "metrics": {"device_ms_experts": "< 10"}})], CHANGE)
    assert o.run() == HUMAN
    assert decide(o, "P.1", "accept", note="HiFi2 is fine end to end") == 0
    assert o.run() == DONE, mk.lines
    assert o.led.status("P.1") == "PASS" and sandbox.git("log", "-1", "--format=%s") == "[toy][P.1] experts at HiFi2"
    assert sandbox.git("show", "HEAD:src/impl.txt") == "hifi2"
    assert o.led.state()["P.1"]["ab"]["note"] == "HiFi2 is fine end to end"


def test_an_accepted_change_that_fails_the_gate_waits_again(mk):
    o = mk([pick(gate={"cmd": record_cmd(device_ms_experts=12.0), "metrics": {"device_ms_experts": "< 10"}})], CHANGE)
    assert o.run() == HUMAN
    decide(o, "P.1", "accept")
    assert o.run() == HUMAN and o.led.status("P.1") == "FAIL"
    assert "accepted, but the gate fails" in o.led.state()["P.1"]["waiting"]


def test_a_new_brief_voids_the_report_and_decide_needs_one(mk, sandbox):
    o = mk([pick()], {**CHANGE, "P.1.perf.2.md": {}})
    assert o.run() == HUMAN
    sandbox.tasks(pick(brief={"details": "another idea"}))
    assert decide(o, "P.1", "accept") == 1  # the report was about the old brief
    assert o.run() == HUMAN and mk.agents()[-1] == "P.1.perf.1.md bringup-engineer" and len(mk.agents()) == 2


# ---------------------------------------------------------------- the profile guard
@pytest.fixture
def guard(sandbox, monkeypatch):
    sandbox.tasks(pick(), {"id": "C.1", "title": "component", "step": "implement", "gate": {"cmd": GATE}})
    monkeypatch.delenv("BRINGUP_AB", raising=False)
    monkeypatch.setenv(M.TASK_ENV, "P.1")
    return sandbox.spec


def as_test(monkeypatch, node):
    """What pytest sets while the frozen test runs (set inside the test body: pytest rewrites it per phase)."""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", f"{node} (call)")


def refused(spec) -> str:
    try:
        accuracy_guard.check(spec)
    except accuracy_guard.AccuracyFailed as e:
        return str(e)
    return ""


def test_a_failed_frozen_test_blocks_the_profile_until_it_passes(guard, monkeypatch):
    as_test(monkeypatch, "tests/test_c_x.py::test_component[box]")
    assert refused(guard) == ""
    accuracy_guard.note(guard, False)
    assert accuracy_guard.failing(guard, "P.1") == ["tests/test_c_x.py::test_component[box]"]
    assert "refusing to profile P.1" in refused(guard)
    monkeypatch.setenv("BRINGUP_AB", "1")
    assert refused(guard) == ""  # the orchestrator's A/B profile
    monkeypatch.delenv("BRINGUP_AB")
    accuracy_guard.note(guard, True)
    assert refused(guard) == "" and not accuracy_guard.marker(guard, "P.1").exists()


def test_the_guard_ignores_other_tasks_and_tests(guard, monkeypatch):
    as_test(monkeypatch, "tests/test_other.py::test_component")
    accuracy_guard.note(guard, False)  # not one of the pick's frozen tests
    assert accuracy_guard.failing(guard, "P.1") == []
    monkeypatch.setenv(M.TASK_ENV, "C.1")
    as_test(monkeypatch, "tests/test_c_x.py::test_component")
    accuracy_guard.note(guard, False)  # a component task, not a perf pick
    accuracy_guard.write(guard, "C.1", ["x"])
    assert refused(guard) == ""


def test_run_profile_checks_the_guard_first(guard):
    from models.demos.common.bringup.testing.profile import run_profile

    accuracy_guard.write(guard, "P.1", ["tests/test_c_x.py::test_component"])
    try:
        run_profile(guard, None)
    except accuracy_guard.AccuracyFailed as e:
        assert "refusing to profile P.1" in str(e)
    else:
        raise AssertionError("run_profile profiled a pick whose frozen test failed")


def test_each_agent_attempt_starts_with_a_clean_marker(mk, sandbox):
    o = mk([pick()], {})
    accuracy_guard.write(o.spec, "P.1", ["tests/test_c_x.py::test_component"])
    o.run()
    assert not accuracy_guard.marker(o.spec, "P.1").exists()


def test_the_switch_is_set_explicitly_on_both_sides(mk):
    """The agent may leave either default in the tree: 'on' runs with ab.change, 'off' with ab.env."""
    o = mk([pick()], CHANGE)
    assert o.run() == HUMAN
    envs = {c["dir"].parent.name for c in mk.calls if c["on"]}
    assert envs == {"new"} and not any(c["on"] for c in mk.calls if c["dir"].parent.name == "old")


def test_a_switch_the_code_never_reads_is_invalid(mk):
    o = mk([pick(ab={"env": {"NOPE_SWITCH": "a"}, "change": {"NOPE_SWITCH": "b"}})], CHANGE)
    assert o.run() == HUMAN
    rec = o.led.state()["P.1"]["ab"]
    assert rec["note"].startswith("INVALID") and "NOPE_SWITCH" in rec["note"] and not mk.calls


def test_bit_identical_sides_are_flagged(mk):
    o = mk([pick()], CHANGE)
    run = fake_runner(mk.calls)
    o.run_cmd = lambda cmd, env, log: run(cmd, dict(env, TOY_FIDELITY="hifi4"), log)  # the switch does nothing
    assert o.run() == HUMAN
    assert o.led.state()["P.1"]["ab"]["note"].startswith("INVALID: off and on gave bit-identical")


def test_a_measured_side_is_reused(mk):
    o = mk([pick()], CHANGE)
    assert o.run() == HUMAN
    n = len(mk.calls)
    assert decide(o, "P.1", "rerun") == 0
    assert o.run() == HUMAN
    assert len(mk.calls) == n  # same code and switch: both sides reused, nothing re-run
    assert sum(": reused (" in x for x in mk.lines) == 2
