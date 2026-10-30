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
        "gate": {"cmd": GATE, "metrics": {"device_ms_experts": "< 10"}},  # F138: accuracy only `limit` (tests below)
        "ab_mode": "full",  # F122: these tests cover the full A/B; the quick one has its own
    }
    t.update(kw)
    return t


QUICK_LIMITS = {}


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
        elif "BRINGUP_QUICK=1" in cmd:  # F122: one process a side
            rec = {"pcc_layer_L00": 0.999, "pcc_layer_L01": 0.990 if on else 0.995, "pcc_chunk_out": 0.99}
            rec.update(chunk_wall_ms=250.0 if on else 840.0, chunk_wall_ms_spread=4.0, host_transfers_per_layer=0)
            rec.update(QUICK_LIMITS)  # F131: the quick profile records the ladder's limits
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
    # F155: the profile measures with the real-time profiler (a clean wall run first, no device profiler build)
    assert len(prof) == 2 and all("BRINGUP_PROFILE_OPS" not in c and "TT_METAL_DEVICE_PROFILER" not in c for c in prof)
    assert all("BRINGUP_PROFILE_WALL=1" in c and "BRINGUP_WALL_CLEAN=1" in c for c in prof)
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


def test_a_side_whose_process_crashed_runs_once_more(mk):
    """F207: the old side's profile died of a bus error while loading weights (DeepSeek-V4.1 P.2); the report ran it
    once more instead of recording the off side as failed."""
    o = mk([pick()], CHANGE)
    real, crashed = o.run_cmd, []

    def run_cmd(cmd, env, log):
        if "test_profile.py" in cmd and env.get("TOY_FIDELITY") == "hifi4" and not crashed:
            crashed.append(cmd)
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("Fatal Python error: Bus error\n")
            return -7
        return real(cmd, env, log)

    o.run_cmd = run_cmd
    assert o.run() == HUMAN, mk.lines
    rec = o.led.state()["P.1"]["ab"]
    assert crashed and rec["profile"]["old"]["device_ms_total"] == 100.0, rec["profile"]
    assert any("F207" in line for line in mk.lines)
    assert list((o.run_dir / "ab" / "P.1" / "old" / "profile").glob("profile.crashed.log"))


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
    assert o.run() == HUMAN and mk.agents()[-1] == "P.1.perf.2.md bringup-engineer" and len(mk.agents()) == 2  # F107


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
    assert sum(" reused (same code and switch)" in x for x in mk.lines) == 8  # F116: per run, 4 a side


def test_a_side_killed_before_any_test_is_measured_again_on_rerun(mk):
    """F228: DeepSeek-V4.1 P.16's 'on' side was SIGKILLed at model load (rc 1, no metric); `decide --rerun` then
    reused it as 'same code and switch' and the report showed nothing for the change."""
    o = mk([pick(ab_mode="quick")], CHANGE)
    real, killed = o.run_cmd, []

    def run_cmd(cmd, env, log):
        if env.get("TOY_FIDELITY", "hifi2") == "hifi2" and not killed:
            killed.append(cmd)
            Path(env[M.RESULTS_ENV]).mkdir(parents=True, exist_ok=True)
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("Killed\n")
            return 1
        return real(cmd, env, log)

    o.run_cmd = run_cmd
    assert o.run() == HUMAN, mk.lines
    assert killed and o.led.state()["P.1"]["ab"]["quick"]["new"]["chunk_wall_ms"] is None
    n = len(mk.calls)
    assert decide(o, "P.1", "rerun") == 0
    assert o.run() == HUMAN, mk.lines
    assert [c["on"] for c in mk.calls[n:]] == [True]  # the killed side ran again, the finished one was reused
    assert o.led.state()["P.1"]["ab"]["quick"]["new"]["chunk_wall_ms"] == 250.0


def test_a_flip_pick_measures_existing_switches_with_no_code_change(mk, sandbox):
    """F233: ab.flip: the switch is already in the code and the agent changes nothing; both sides are still measured."""
    (sandbox.repo / "src/sw.py").write_text("import os\nF = os.environ.get('TOY_FIDELITY', 'hifi2')\n")
    sandbox.git("add", "src")
    sandbox.git("commit", "-q", "-m", "switch")
    ab = {"env": {"TOY_FIDELITY": "hifi4"}, "change": {"TOY_FIDELITY": "hifi2"}, "flip": True}
    o = mk([pick(ab=ab, ab_mode="quick")], {})
    assert o.run() == HUMAN, mk.lines
    rec = o.led.state()["P.1"]["ab"]
    assert sorted(c["on"] for c in mk.calls) == [False, True], mk.calls
    assert rec["note"].startswith("no code change") and rec["quick"]["new"]["chunk_wall_ms"] == 250.0


def test_without_flip_an_unchanged_tree_is_not_measured(mk):
    o = mk([pick(ab_mode="quick")], {})
    assert o.run() == HUMAN and mk.calls == []


def before_mode(mk, sandbox, **kw):
    """F230: a pick without a switch (a branch merge): the tree itself is the switch. The fake runner reads the tree
    (src/impl.txt) instead of a variable; positions record their metrics with _p<pos>, as testing/profile.py does."""
    o = mk(
        [pick(ab={"before": True, "env": {"TOY_TABLE": "resident"}}, ab_mode="quick", ab_positions=[51200], **kw)],
        CHANGE,
    )
    real = o.run_cmd

    def run_cmd(cmd, env, log):
        tree = (sandbox.repo / "src/impl.txt").read_text().strip()
        if "BRINGUP_POS=" in cmd:
            on = tree == "hifi2"
            mk.calls.append(
                {"cmd": cmd, "on": on, "dir": Path(env[M.RESULTS_ENV]), "rung": None, "ab": env.get("TOY_TABLE")}
            )
            pos = cmd.split("BRINGUP_POS=")[1].split()[0]
            rec = {"device_ms_total": 90.0 if on else 100.0, "device_ms_experts": 8.0 if on else 12.0}
            rec.update(chunk_wall_ms=250.0 if on else 300.0, host_launch_ms=150.0 if on else 180.0)
            out = Path(env[M.RESULTS_ENV])
            out.mkdir(parents=True, exist_ok=True)
            (out / "P.1.json").write_text(
                json.dumps({"metrics": {f"{k}_p{pos}": {"value": v} for k, v in rec.items()}})
            )
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(f"$ {cmd}\n")
            return 0
        return real(cmd, dict(env, TOY_FIDELITY=tree), log)

    o.run_cmd = run_cmd
    return o


def test_a_merge_pick_measures_before_then_lets_the_agent_change_the_tree(mk, sandbox):
    """F230 (owner 2026-10-30): 'before' runs on the tree before the agent's merge, 'after' on the merged tree; one
    table with the position rows (device and host) on and off."""
    o = before_mode(mk, sandbox)
    assert o.run() == HUMAN, mk.lines
    assert [c["on"] for c in mk.calls] == [False, False, True, True], mk.calls  # quick + p51200 before, then after
    assert any("A/B before: measuring the tree before the change" in x for x in mk.lines)
    rec = o.led.state()["P.1"]["ab"]
    assert rec["positions"]["p51200"]["old"]["device_ms_total"] == 100.0
    assert rec["positions"]["p51200"]["new"]["host_launch_ms"] == 150.0
    table = Path(rec["table"]).read_text()
    assert "measured before the agent started" in table and "position 51,200" in table and "TOY_TABLE=resident" in table
    assert [c["ab"] for c in mk.calls if "BRINGUP_POS=" in c["cmd"]] == ["resident", "resident"]  # ab.env, both sides
    assert "| device_ms_total | 90.0000 | 100.0000 | -10.0 |" in table, table
    assert not rec["note"], rec["note"]


def test_a_merge_picks_before_side_is_never_measured_on_the_changed_tree(mk, sandbox):
    """F230: decide --rerun measures 'after' again and keeps 'before' (the tree no longer holds it)."""
    o = before_mode(mk, sandbox)
    assert o.run() == HUMAN
    n = len(mk.calls)
    assert decide(o, "P.1", "rerun") == 0
    assert o.run() == HUMAN, mk.lines
    assert mk.calls[n:] == [] or all(c["on"] for c in mk.calls[n:])  # nothing off on the merged tree


def test_a_merge_pick_without_its_before_side_on_a_changed_tree_waits(mk, sandbox):
    """F230: no 'before' measured and the tree already changed under the task's paths: a person decides."""
    o = before_mode(mk, sandbox)
    (sandbox.repo / "src/impl.txt").write_text("hifi2\n")
    assert o.run() == HUMAN, mk.lines
    assert mk.calls == [] and mk.agents() == []
    assert "is not measured and the tree already holds changes" in o.led.state()["P.1"]["waiting"]


def test_a_setting_name_switches_its_variable(mk, monkeypatch):
    """F103: ab names a model setting (EXPERTS_IMPL), the code reads its variable (DS41_EXPERTS_IMPL): the report sets
    the variable on both sides (DeepSeek-V4.1 P.1 measured bfp8 against bfp8: an identical table); a value outside the
    setting's choices is refused."""
    from models.demos.common.bringup.core.model_settings import Setting, Settings
    from models.demos.common.bringup.testing import model_precision

    toy = Settings("TOY_", {"FIDELITY": Setting("hifi4", ("hifi4", "hifi2"))})
    monkeypatch.setattr(model_precision, "model_settings", lambda spec, fresh=False: [toy])
    o = mk([pick(ab={"env": {"FIDELITY": "hifi4"}, "change": {"FIDELITY": "hifi2"}})], CHANGE)
    assert o.run() == HUMAN, mk.lines
    assert [c["on"] for c in mk.calls] == [False] * 4 + [True] * 4
    rec = o.led.state()["P.1"]["ab"]
    assert rec["env"] == {"TOY_FIDELITY": "hifi4"} and rec["on"] == {"TOY_FIDELITY": "hifi2"} and not rec["note"]
    assert model_precision.switch_env(None, {"TOY_FIDELITY": "hifi2", "OTHER": "1"}) == (
        {"TOY_FIDELITY": "hifi2", "OTHER": "1"},
        ["OTHER"],
    )
    with pytest.raises(ValueError, match="takes one of"):
        model_precision.switch_env(None, {"FIDELITY": "lofi"})


def test_accepting_a_setting_switch_ships_it_as_the_default(tmp_path, monkeypatch):
    """F104: an accepted pick behind a model setting becomes its default (tt/settings.py rewritten in place), so the
    accept gate and every later task run the change (DeepSeek-V4.1 P.2: the opt-in EXPERTS_KERNEL stayed `loop`, the
    accept gate measured the old path and failed device_ms_experts < 2522 at 2522)."""
    from types import SimpleNamespace

    from models.demos.common.bringup.core.model_settings import Setting, Settings
    from models.demos.common.bringup.testing import model_precision

    (tmp_path / "tt").mkdir()
    src = (
        "TABLE = {\n"
        '    "KERNEL": Setting(\n        "loop",\n        ("loop", "unified"),\n        "the experts",\n    ),\n'
        '    "FIDELITY": Setting("HiFi4", ("HiFi4", "HiFi2")),\n'
        '    "FP32": Setting(True),\n'
        "}\n"
    )
    (tmp_path / "tt" / "settings.py").write_text(src)
    table = {"KERNEL": Setting("loop", ("loop", "unified")), "FIDELITY": Setting("HiFi4"), "FP32": Setting(True)}
    monkeypatch.setattr(model_precision, "model_settings", lambda spec, fresh=False: [Settings("TOY_", table)])
    spec = SimpleNamespace(model_dir=str(tmp_path))
    done = model_precision.set_defaults(spec, {"KERNEL": "unified", "TOY_FP32": "0", "OTHER_VAR": "1"})
    assert done == ["KERNEL: 'loop' -> 'unified'", "FP32: True -> False"]
    out = (tmp_path / "tt" / "settings.py").read_text()
    assert '"KERNEL": Setting(\n        "unified",\n        ("loop", "unified")' in out
    assert '"FP32": Setting(False)' in out and '"FIDELITY": Setting("HiFi4", ("HiFi4", "HiFi2"))' in out
    assert model_precision.set_defaults(spec, {"FIDELITY": "HiFi4"}) == []  # already the default: no edit


def test_the_accept_reuses_the_on_side_when_the_code_is_unchanged(mk, sandbox):
    """F105: the accept gate takes the A/B 'on' side's measurement (same commands, same code, every command rc 0)
    instead of running the gate again (DeepSeek-V4.1 P.2: the accept repeated ~50 min of tests the A/B had just run);
    a ledger commit in between (state, results, tasks.yaml) is no code change. Changed code: the gate runs."""
    gate = (
        f"BRINGUP_RUNG=last scripts/run_safe_pytest.sh --no-precompile --run-all {LADDER} && TT_METAL_DEVICE_PROFILER=1 "
        f"scripts/run_safe_pytest.sh --run-all --no-precompile {PROFILE}"
    )
    o = mk([pick(gate={"cmd": gate, "metrics": {"device_ms_experts": "< 10"}})], CHANGE)
    assert o.run() == HUMAN, mk.lines
    n = len(mk.calls)
    sandbox.git("commit", "-q", "--allow-empty", "-m", "a ledger commit in between")
    assert decide(o, "P.1", "accept") == 0
    assert o.run() == DONE, mk.lines
    assert len(mk.calls) == n  # nothing measured again
    assert o.led.status("P.1") == "PASS" and any("not run again" in ln for ln in mk.lines)
    got = M.load("P.1", o.led.results_dir)
    assert got["device_ms_experts"]["value"] == 8.0 and got["pcc_layer_L01"]["value"] == 0.990  # on side, rung last
    assert sandbox.git("show", "HEAD:src/impl.txt") == "hifi2"


def test_the_accept_runs_the_gate_when_the_code_changed_since_the_ab(mk, sandbox):
    gate = f"BRINGUP_RUNG=last scripts/run_safe_pytest.sh --no-precompile --run-all {LADDER}"
    o = mk([pick(gate={"cmd": gate, "metrics": {"device_ms_experts": "< 10"}})], CHANGE)
    assert o.run() == HUMAN, mk.lines
    (sandbox.repo / "src/impl.txt").write_text("hifi2 tuned\n")  # the code moved after the report
    assert o.ab_on_reuse(o.led.task("P.1"), o.led.state()["P.1"]["ab"]) is None


def test_a_resumed_pick_continues_the_attempt_numbering(mk):
    """F107 (F95 for perf picks): after earlier perf attempts (here two, stopped by the overseer), the next agent run
    is attempt 3, not 1 again (DeepSeek-V4.1 P.3 restarted at attempt 1 after a resume, reusing attempt 1's brief)."""
    o = mk([pick()], {"P.1.perf.3.md": CHANGE["P.1.perf.1.md"]})
    runs = [{"role": "perf", "attempt": a, "t": f"2026-10-18T0{a}:00:00"} for a in (1, 2)]
    o.led.update("P.1", agent_runs=runs)
    assert o.run() == HUMAN, mk.lines
    assert mk.agents() == ["P.1.perf.3.md bringup-engineer"]


def test_the_report_measures_an_earlier_attempts_change(mk, sandbox):
    """F108: a later attempt that changes nothing (the change is already in the tree from an earlier attempt) still
    gets a report of that change (DeepSeek-V4.1 P.4 attempt 2: 'the agent changed nothing', nothing measured)."""
    (sandbox.repo / "src/impl.txt").write_text("hifi2\n")  # an earlier attempt's uncommitted change
    (sandbox.repo / "src/new_path.py").write_text("import os\nF = os.environ.get('TOY_FIDELITY', 'hifi2')\n")
    o = mk([pick()], {})
    assert o.run() == HUMAN, mk.lines
    rec = o.led.state()["P.1"]["ab"]
    assert "changed nothing" not in rec["note"] and len(mk.calls) == 8
    assert "src/impl.txt" in rec["changed"]


def test_a_pick_waiting_for_the_owner_does_not_hold_up_independent_tasks(mk, sandbox):
    """F109: while a pick waits for a person, the run goes on with tasks that do not depend on it, but only when the
    pick has nothing uncommitted in the tree (another task's commit would take it along); its dependents wait.
    (DeepSeek-V4.1: the owner asked to run P.5 regardless of the P.4 decision.)"""
    other = {"id": "X.9", "title": "independent", "step": "perf", "deps": [], "gate": {"cmd": record_cmd(n=1)}}
    after = {"id": "X.10", "title": "after the pick", "step": "perf", "deps": ["P.1"], "gate": {"cmd": record_cmd(n=1)}}
    o = mk([pick(), other, after], CHANGE)
    assert o.run() == HUMAN, mk.lines
    assert o.led.status("X.9") != "PASS"  # the pick's change is uncommitted: the run stops at the pick
    sandbox.git("add", "src")
    sandbox.git("commit", "-q", "-m", "the pick's switch, opt-in")
    assert o.run() == HUMAN, mk.lines
    assert o.led.status("X.9") == "PASS" and o.led.status("X.10") != "PASS"
    assert any("waits for a person; the run goes on" in ln for ln in mk.lines) or o.led.state()["P.1"].get("waiting")


def test_a_waiting_pick_is_judged_by_its_own_change_not_its_neighbours(mk, sandbox):
    """F109: a waiting pick whose own change is committed may be passed even when another task's uncommitted change
    sits under the same paths (DeepSeek-V4.1: P.5's uncommitted tt/ files made P.4 look unsafe, the run stopped)."""
    other = {"id": "X.9", "title": "independent", "step": "perf", "deps": [], "gate": {"cmd": record_cmd(n=1)}}
    o = mk([pick(), other], CHANGE)
    assert o.run() == HUMAN, mk.lines
    sandbox.git("add", "src")
    sandbox.git("commit", "-q", "-m", "the pick's switch, opt-in")
    (sandbox.repo / "src/neighbour.py").write_text("x = 1\n")  # someone else's uncommitted work under src/
    assert o.run() == HUMAN, mk.lines
    assert o.led.status("X.9") == "PASS"


def _apply_change(sandbox):
    for p, text in CHANGE["P.1.perf.1.md"]["write"].items():
        (sandbox.repo / p).write_text(text)


def test_an_agents_measure_run_is_the_reports_on_side(mk, sandbox):
    """F116: `orchestrator measure` runs the change-on side's frozen test as the report would; the report reuses it."""
    from models.demos.common.bringup.orchestrator import measure

    o = mk([pick()], CHANGE)
    _apply_change(sandbox)  # the agent's change, as its last edit left it
    assert measure(o, "P.1", ["test_c_x"]) == 1  # the toy change fails its component check (rc 1)
    d = o.run_dir / "ab" / "P.1"
    assert [(c["on"], c["dir"].relative_to(d).as_posix()) for c in mk.calls] == [(True, "new/test_c_x")]
    assert o.run() == HUMAN, mk.lines  # the mock agent writes the same change: the code key is unchanged
    runs = [(c["on"], c["dir"].relative_to(d).as_posix()) for c in mk.calls[1:]]
    assert (True, "new/test_c_x") not in runs and (True, "new/profile") in runs and (False, "old/test_c_x") in runs
    assert any("test_c_x reused" in ln for ln in mk.lines)
    rec = o.led.state()["P.1"]["ab"]
    assert rec["tests"]["test_c_x"]["new"]["rc"] == 1  # the reused result, in the table


def test_a_code_edit_after_measure_measures_again(mk, sandbox):
    from models.demos.common.bringup.orchestrator import measure

    o = mk(
        [pick()], {"P.1.perf.1.md": {"write": dict(CHANGE["P.1.perf.1.md"]["write"], **{"src/impl.txt": "hifi2b\n"})}}
    )
    _apply_change(sandbox)
    measure(o, "P.1", ["test"])
    assert o.run() == HUMAN, mk.lines  # the agent's last edit differs from what was measured
    d = o.run_dir / "ab" / "P.1"
    assert [c["dir"].relative_to(d).as_posix() for c in mk.calls].count("new/test_c_x") == 2


def test_every_frozen_test_runs_on_both_sides_at_the_shipped_settings(mk):
    """F137 (owner): no max-precision device run: the gate's frozen tests run on each side with that side's switch
    (the F120 / F121 split, which took them from the side max precision pinned, is gone)."""
    o = mk([pick()], CHANGE)
    assert o.run() == HUMAN, mk.lines
    d = o.run_dir / "ab" / "P.1"
    runs = [(c["on"], c["dir"].relative_to(d).as_posix()) for c in mk.calls]
    assert (False, "old/test_c_x") in runs and (True, "new/test_c_x") in runs
    rec = o.led.state()["P.1"]["ab"]
    assert rec["tests"]["test_c_x"]["new"]["rc"] == 1 and "shared" not in rec["tests"]["test_c_x"]["new"]
    assert not hasattr(o, "gate_tests_side") and not hasattr(o, "switches_pinned")


def test_the_quick_ab_runs_one_process_a_side(mk):
    """F122: the default A/B: one run a side (the last chunk after the golden prefix, every layer, clean wall)."""
    gate = {
        "cmd": GATE,
        "metrics": {"device_ms_experts": "< 10"},
    }
    o = mk([pick(ab_mode="quick", gate=gate)], CHANGE)
    assert o.run() == HUMAN, mk.lines
    d = o.run_dir / "ab" / "P.1"
    assert [(c["on"], c["dir"].relative_to(d).as_posix()) for c in mk.calls] == [
        (False, "old/quick"),
        (True, "new/quick"),
    ]
    rec = o.led.state()["P.1"]["ab"]
    assert rec["quick"]["new"]["min layer PCC"] == 0.990 and rec["quick"]["old"]["chunk_wall_ms"] == 840.0
    assert rec["gate_on"]["ok"] and rec["gate_on"]["partial"]  # device_ms_experts waits for the full gate
    table = (d / "table.md").read_text()
    assert "Quick A/B" in table and "Frozen accuracy tests" not in table


def test_a_quick_pick_is_accepted_now_and_its_full_gate_runs_once_before_the_final_task(mk, sandbox):
    """F122: accept on the quick A/B (no run), the pick's full gate pending; before the final task the pending gates
    run once, picks with the same gate command sharing one run."""
    full = record_cmd(device_ms_experts=8.0, err_layer_L00=0.01, err_layer_L00_limit=0.02, pcc_chunk_out=0.99)
    gate = {"cmd": full, "metrics": {"device_ms_experts": "< 10", "err_layer_L*": "<= limit"}}
    final = {"id": "X.3", "title": "final", "step": "perf", "deps": ["P.1", "P.2"], "gate": {"cmd": record_cmd(n=1)}}
    p2 = pick(id="P.2", ab_mode="quick", gate=gate, deps=["P.1"])
    o = mk(
        [pick(ab_mode="quick", gate=gate), p2, final], dict(CHANGE, **{"P.2.perf.1.md": {"write": {"src/b.txt": "x"}}})
    )
    assert o.run() == HUMAN
    assert decide(o, "P.1", "accept") == 0
    assert o.run() == HUMAN  # P.1 accepted without a run; P.2's quick A/B waits
    st = o.led.state()["P.1"]
    assert st["status"] == "PASS" and st.get("full_pending")
    assert decide(o, "P.2", "accept") == 0
    assert o.run() == DONE, mk.lines[-4:]
    assert not o.led.state()["P.1"].get("full_pending") and not o.led.state()["P.2"].get("full_pending")
    assert sum("full gate (quick-accepted pick)" in ln for ln in mk.lines) == 1  # P.1 shares P.2's run
    # F175: the newest pick runs first (its own units already passed on this code: the delta); the older one shares it
    assert any("[P.2] full gate (quick-accepted pick)" in ln for ln in mk.lines)
    assert "the same gate as P.2" in Path(o.led.state()["P.1"]["log"]).read_text()


@pytest.mark.parametrize("limits", [True, False])
def test_a_quick_ab_judges_limit_gates_on_the_limits_it_recorded(mk, monkeypatch, limits):
    """F131: a `>= limit` gate passes on a quick A/B that recorded its limits and fails without them; the table shows
    value / limit / share of the margin used."""
    lim = {"pcc_layer_L00_limit": 0.995, "pcc_layer_L01_limit": 0.985, "pcc_state_kv_L01": 0.999}
    monkeypatch.setattr(sys.modules[__name__], "QUICK_LIMITS", dict(lim, pcc_state_kv_L01_limit=0.99) if limits else {})
    gate = {"cmd": GATE, "metrics": {"pcc_layer_L*": ">= limit", "device_ms_experts": "< 10"}}
    o = mk([pick(ab_mode="quick", gate=gate)], CHANGE)
    assert o.run() == HUMAN, mk.lines
    g = o.led.state()["P.1"]["ab"]["gate_on"]
    assert g["ok"] is limits, g
    table = (o.run_dir / "ab" / "P.1" / "table.md").read_text()
    if limits:
        assert "| pcc_layer_L01 | 0.99000 / 0.98500 (0.67) | 0.99500 / 0.98500 (0.33) |" in table, table
        assert "pcc_state_kv_L01" in table
    else:
        assert "no pcc_layer_L00_limit recorded" in " ".join(g["failing"]) and "Accuracy vs" not in table


def test_a_pick_accepted_after_the_final_task_runs_its_full_gate_at_the_end(mk, sandbox):
    """F188: the final task already passed, so it cannot run a later quick pick's full gate; the end of the run does."""
    full = record_cmd(device_ms_experts=8.0, err_layer_L00=0.01, err_layer_L00_limit=0.02, pcc_chunk_out=0.99)
    gate = {"cmd": full, "metrics": {"device_ms_experts": "< 10", "err_layer_L*": "<= limit"}}
    final = {"id": "X.3", "title": "final", "step": "perf", "deps": [], "gate": {"cmd": record_cmd(n=1)}}
    o = mk([final, pick(ab_mode="quick", gate=gate, deps=["X.3"])], CHANGE)
    assert o.run() == HUMAN and o.led.status("X.3") == "PASS"
    assert decide(o, "P.1", "accept") == 0
    assert o.run() == DONE, mk.lines[-4:]
    assert not o.led.state()["P.1"].get("full_pending")
    assert sum("[P.1] full gate (quick-accepted pick)" in ln for ln in mk.lines) == 1


def test_every_real_time_profile_records_op_rows(fx, monkeypatch):
    """F187 (owner): the per-op rows come with every real-time profile, so the dashboard never shows an older code's
    op list; the device profiler's slow op mode only when asked."""
    from models.demos.common.bringup.core.spec import Spec
    from models.demos.common.bringup.testing.profile import want_ops

    monkeypatch.delenv("BRINGUP_PROFILE_OPS", raising=False)
    s = Spec.load(fx())
    assert want_ops(s, rt=True) and not want_ops(s)
    assert want_ops(Spec.load(fx(perf={"op_profile": True})))
    assert not want_ops(Spec.load(fx(perf={"op_profile": False})), rt=True)
    monkeypatch.setenv("BRINGUP_PROFILE_OPS", "1")
    assert want_ops(s)
