# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.31 (OWNER 2026-11-13), item 5: gates that run back to back are ONE run (core/multigate.py). The quick picks'
full gates and the milestone gate after them: every pending unit of every task in one process on one loaded model,
each unit once however many tasks name it, each task its own verdict and metrics; a failing unit fails only the tasks
that name it. Also: a unit asking for the precompile pass still batches (and the log says so), a unit that cannot
share says why, and a switch at the value a test reads unset is the same model. CPU only: a stand-in batch process."""

import stat

from models.demos.common.bringup import orchestrator as O
from models.demos.common.bringup.core import batch as B
from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.core import multigate
from models.demos.common.bringup.core.gate import run_gate
from models.demos.common.bringup.selftest.conftest import PY
from models.demos.common.bringup.selftest.test_core import task
from models.demos.common.bringup.selftest.test_gate_one_process import CONTRACT, FAKE, LADDER, NOT_SHARED, OWN

# the stand-in runs past a failure when the plan says so (a multi-task plan: stop_on_fail False)
FAKE_GOING = FAKE.replace(
    "        if rc:\n            rc_all = 1\n            break",
    "        if rc:\n            rc_all = 1\n            if plan.get('stop_on_fail', True):\n                break",
)
assert FAKE_GOING != FAKE


def _setup(sandbox, monkeypatch, tmp_path):
    root = sandbox.repo.parent
    (root / "fake_runner.py").write_text(FAKE_GOING)
    runner = sandbox.repo / "scripts/run_safe_pytest.sh"
    runner.parent.mkdir(exist_ok=True)
    runner.write_text(f'#!/bin/bash\nexec {PY} {root / "fake_runner.py"} "$@"\n')
    runner.chmod(runner.stat().st_mode | stat.S_IEXEC)
    calls = tmp_path / "calls.txt"
    monkeypatch.setenv("FAKE_CALLS", str(calls))
    monkeypatch.setenv("PYTHONPATH", str(M._REPO))
    sandbox.write_spec(tests={"one_process": True})
    return calls


LIMITS = {
    "ladder_failed": "== 0",
    "contract_checks_failed": "== 0",
    "served_bytes_differ": "== 0",
    "gate_model_loads": "== 1",
}


def _task(tid, cmd, step="perf"):
    metrics = {k: v for k, v in LIMITS.items() if k == "gate_model_loads" or k.split("_")[0] in cmd}
    return task(tid, cmd, step=step, device=True, paths=["src"], metrics=metrics)


def _three(sandbox):
    """P.1 / P.2: quick picks (perf); X.1: the milestone. The ladder in all three, the contract in two."""
    return sandbox.tasks(
        _task("P.1", f"{LADDER} && {CONTRACT}"),
        _task("P.2", f"{LADDER} && {OWN}"),
        _task("X.1", f"{LADDER} && {CONTRACT}"),
    )


def _batches(calls):
    return [ln for ln in calls.read_text().splitlines() if ln.startswith("batch ")] if calls.exists() else []


def test_three_tasks_overlapping_one_process_one_load_each_unit_once(sandbox, monkeypatch, tmp_path):
    calls = _setup(sandbox, monkeypatch, tmp_path)
    led = _three(sandbox)
    pre = multigate.prerun(sandbox.spec, led, ["P.1", "P.2", "X.1"], echo=lambda s: None)
    lines = calls.read_text().splitlines()
    assert len(lines) == 1 and lines[0].startswith("batch "), lines  # ONE process
    names = lines[0].split()[1:]
    # determinism (FIX.25: one per task, the same runs) once, the ladder once, the contract once, the own test once
    assert len(names) == 4, names
    assert sum("test_determinism" in n for n in names) == 1 and sum("test_ladder" in n for n in names) == 1
    for tid in ("P.1", "P.2", "X.1"):
        res = run_gate(sandbox.spec, led, tid, pre=pre.get(tid))
        assert res.verdict == "PASS", res.summary()
        m = M.load(tid, led.results_dir)
        assert m["gate_model_loads"]["value"] == 1  # each task: the one load of the shared process
        assert m["ladder_failed"]["value"] == 0 and m["determinism_max_abs_diff"]["value"] == 0
        assert "multi-task shared process (FIX.31)" in res.log.read_text()
    assert len(calls.read_text().splitlines()) == 1  # the gates ran nothing more
    assert "contract_checks_failed" in M.load("X.1", led.results_dir)  # (the run P.1's process made, X.1's too)
    assert "served_bytes_differ" not in M.load("X.1", led.results_dir)


def test_a_failing_unit_fails_only_the_tasks_that_name_it(sandbox, monkeypatch, tmp_path):
    calls = _setup(sandbox, monkeypatch, tmp_path)
    monkeypatch.setenv("FAKE_FAIL", "served_bytes_differ")  # P.2's own test
    led = _three(sandbox)
    pre = multigate.prerun(sandbox.spec, led, ["P.1", "P.2", "X.1"], echo=lambda s: None)
    verdicts = {tid: run_gate(sandbox.spec, led, tid, pre=pre.get(tid)).verdict for tid in ("P.1", "P.2", "X.1")}
    assert verdicts == {"P.1": "PASS", "P.2": "FAIL", "X.1": "PASS"}
    assert len(calls.read_text().splitlines()) == 1  # past the failure, in the same process


def test_units_the_gates_would_reuse_are_not_run_again(sandbox, monkeypatch, tmp_path):
    calls = _setup(sandbox, monkeypatch, tmp_path)
    led = _three(sandbox)
    assert run_gate(sandbox.spec, led, "P.1").verdict == "PASS"  # determinism, ladder, contract: passed
    n = len(calls.read_text().splitlines())
    plan = multigate.plan(sandbox.spec, led, ["P.1", "P.2", "X.1"])
    assert [u for _, u, *_ in plan["todo"]] == ["P.2#test_decoder_skip_decode"]  # only what never ran
    multigate.prerun(sandbox.spec, led, ["P.1", "P.2", "X.1"], echo=lambda s: None)
    assert len(calls.read_text().splitlines()) == n + 1


def test_the_orchestrator_runs_the_pending_full_gates_and_the_milestone_together(monkeypatch, tmp_path):
    seen = {}

    class Fake:
        spec = type("S", (), {"get": staticmethod(lambda k, d=None: True if k == "tests.one_process" else d)})()
        led = type(
            "L", (), {"state": lambda self: {"P.1": {"full_pending": True}}, "tasks": lambda self: {"P.1": {}}}
        )()
        _pre = {}
        echo = print

        def gate(self, tid, reuse=None):
            seen.setdefault("gated", []).append(tid)
            return type("R", (), {"verdict": "PASS"})()

    def fake_prerun(spec, led, tids, echo=None):
        seen["prerun"] = list(tids)
        return {"P.1": {"u": {}}}

    monkeypatch.setattr(multigate, "prerun", fake_prerun)
    f = Fake()
    f.led.update = lambda *a, **k: None
    f.led.task = lambda tid: {"gate": {"cmd": "x"}}
    f.led.results_dir = tmp_path
    assert O.Orchestrator.pending_full_gates(f, include=["X.1"]) is None
    assert seen["prerun"] == ["P.1", "X.1"] and seen["gated"] == ["P.1"] and f._pre == {"P.1": {"u": {}}}


def test_a_precompile_unit_batches_and_the_log_says_why_one_does_not(sandbox, monkeypatch, tmp_path):
    """X.3's units had no --no-precompile and core/batch.parse refused them: ~4 model loads."""
    calls = _setup(sandbox, monkeypatch, tmp_path)
    pre_cmd = LADDER.replace(" --no-precompile", "")
    contract = CONTRACT.replace(" --no-precompile", "")
    assert B.parse_all(pre_cmd) is None and B.why_not(pre_cmd) is None  # the launch form shares
    led = sandbox.tasks(_task("X.1", f"{pre_cmd} && {contract} && {NOT_SHARED}", step="overseer_fix"))
    res = run_gate(sandbox.spec, led, "X.1")
    log = res.log.read_text()
    batches = _batches(calls)
    assert len(batches) == 1 and "test_ladder" in batches[0] and "test_contract" in batches[0]
    assert "cannot share the model" in log  # NOT_SHARED: a process of its own, and why
    form, notes = B.launch_form(pre_cmd)
    assert "--no-precompile" in form and "compile on demand" in notes[0]


def test_a_switch_at_its_test_value_is_the_same_model_but_resident_is_not():
    assert B.test_value("DS41_ENGRAM_TABLE") == (True, "mmap")  # F222: the test value, not the default (resident)
    names = {"DS41_ENGRAM_TABLE"}
    assert B.model_key({"DS41_ENGRAM_TABLE": "mmap"}, names) == B.model_key({}, names)
    assert B.model_key({"DS41_ENGRAM_TABLE": "resident"}, names) != B.model_key({}, names)
    unknown = {"DS41_NO_SUCH_SWITCH"}
    assert B.model_key({"DS41_NO_SUCH_SWITCH": "x"}, unknown) != B.model_key({}, unknown)
