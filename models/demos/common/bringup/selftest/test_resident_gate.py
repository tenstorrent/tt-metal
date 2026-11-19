# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F281 / F283: gate units on the resident runner (core/batch.drive) and the owner's one-load rule (core/batch
one_load_errors, core/gate pre-check and validate, testing/batch's runtime refusal). CPU only."""

import json
from pathlib import Path

import pytest

from models.demos.common.bringup import resident
from models.demos.common.bringup.core import batch as B
from models.demos.common.bringup.testing import batch as TB

T = "models/demos/common/bringup/tests/"
R = "scripts/run_safe_pytest.sh --no-precompile --run-all"
DET = f"{R} {T}test_determinism.py"
LADDER = f"{R} {T}test_ladder.py"
PROFILE = f"BRINGUP_PROFILE_WALL=1 {R} {T}test_profile.py"


@pytest.fixture
def call_hooks(monkeypatch):
    """A model whose X_SKIP switch is per run (but at value 'tail'), as DeepSeek-V4.1's DECODER_SKIP (but replay128)."""
    monkeypatch.setattr(B, "_CALL_HOOKS", [[("X_", lambda env: () if env.get("X_SKIP") == "tail" else ("SKIP",))]])


def test_drive_submits_to_a_serving_resident_runner_and_marks_determinism_fresh(tmp_path, monkeypatch):
    plan = B.write_plan(
        tmp_path / "plan.json",
        [
            {"name": "u0.0", "group": "u0", "env": {}, "args": [f"{T}test_determinism.py"], "log": None},
            {"name": "u1.0", "group": "u1", "env": {"A": "1"}, "args": [f"{T}test_ladder.py"], "log": str(tmp_path / "u1.log")},
        ],  # fmt: skip
    )
    sent = {}

    def submit(d, p, env, fresh=False):
        sent.update(d=d, plan=json.loads(Path(p).read_text()), env=env, fresh=fresh)
        Path(f"{p}.u0.0.log").write_text("determinism output\n")
        B.results_path(Path(p)).write_text(
            json.dumps({"running": None, "runs": {"u0.0": {"rc": 0}, "u1.0": {"rc": 0}}})
        )
        return {"id": "r", "rc": 0, "runs": {}, "timing": {"first_chunk_s": 12.0, "model_ready_s": 3.0}}

    monkeypatch.setattr(resident, "for_env", lambda env, echo=None: tmp_path / "res")
    monkeypatch.setattr(resident, "submit_plan", submit)
    spawned, out = [], []
    rc = B.drive(plan, lambda c, k: spawned.append(c) or 0, env={"BRINGUP_TASK": "T"}, echo=out.append)
    assert rc == 0 and not spawned and sent["fresh"] is True and sent["env"]["BRINGUP_TASK"] == "T"
    assert sent["plan"]["entries"][0]["log"] == f"{plan}.u0.0.log"  # a run without a log gets one, echoed after
    text = "".join(out)
    assert "determinism output" in text and "first chunk 12.0 s" in text
    assert set(B.load_results(plan)["runs"]) == {"u0.0", "u1.0"}


def test_drive_takes_the_old_path_without_a_runner_and_a_dead_runner_fails_the_plan(tmp_path, monkeypatch):
    plan = B.write_plan(
        tmp_path / "plan.json", [{"name": "a", "group": "a", "env": {}, "args": [f"{T}test_ladder.py"]}]
    )
    monkeypatch.setattr(resident, "for_env", lambda env, echo=None: None)
    spawned = []
    B.drive(plan, lambda c, k: spawned.append(c) or 0)
    assert len(spawned) == 1 and B.command(plan) in spawned[0]
    monkeypatch.setattr(resident, "for_env", lambda env, echo=None: tmp_path)
    monkeypatch.setattr(
        resident, "submit_plan", lambda *a, **k: {"rc": 2, "error": "the device process HANG", "runs": {}}
    )
    B.results_path(plan).unlink(missing_ok=True)
    assert B.drive(plan, lambda c, k: 0, echo=lambda t: None) == 2
    assert B.load_results(plan)["error"] == "the device process HANG"


def test_for_env_serves_a_healthy_runner_restarts_a_yielded_one_and_never_a_stopped_one(tmp_path, monkeypatch):
    d = tmp_path / "res"
    monkeypatch.setattr(resident, "state_dir", lambda spec: d)
    env = {"BRINGUP_SPEC": "s.yaml"}
    assert resident.for_env(env) is None  # nothing there
    resident._write(d / "status.json", {"phase": "yielded"})
    started = []
    monkeypatch.setattr(resident, "start", lambda dd, spec, echo=None: started.append(spec))
    resident.for_env(env, echo=lambda t: None)
    assert started == ["s.yaml"]
    resident._write(d / "status.json", {"phase": "stopped"})
    assert resident.for_env(env) is None and started == ["s.yaml"]
    resident._write(d / "supervisor.json", {"pid": __import__("os").getpid()})
    resident._write(d / "status.json", {"phase": "ready"})
    assert resident.for_env(env) == d
    assert resident.for_env(dict(env, BRINGUP_RESIDENT="off")) is None
    assert resident.for_env(dict(env, BRINGUP_RESIDENT_DIR=str(d))) is None  # (inside the runner itself)


# ---------------------------------------------------------------- F283: one model load a gate
def test_one_load_a_gate_counts_process_groups_of_shared_model_runs(call_hooks):
    """F283 / F285: one full-model load (the accuracy units with the determinism check) and one executed-layers load
    (the performance units) a gate."""
    ok = [("t#test_determinism", DET), ("t#a", LADDER), ("t#b", f"X_SKIP=decode {PROFILE}")]
    assert B.one_load_errors(ok, {}) == []
    assert [[u for u, _ in g] for g in B.gate_groups(ok, {})] == [["t#test_determinism", "t#a"], ["t#b"]]
    assert [B.group_class(g) for g in B.gate_groups(ok, {})] == ["full", "executed"]
    # the determinism check runs at the configuration of the gate's accuracy units: one process with them
    det = [("t#test_determinism", DET), ("t#a", f"X_LOAD=1 {LADDER}")]
    assert [[u for u, _ in g] for g in B.gate_groups(det, {})] == [["t#test_determinism", "t#a"]]
    entries = B.gate_entries(det, {}, {}, {"t#test_determinism": "d.log", "t#a": "a.log"})
    assert entries[0]["env"] == {"X_LOAD": "1"}
    two = [("t#a", LADDER), ("t#b", f"X_LOAD=1 {LADDER}")]  # two accuracy configurations
    err = B.one_load_errors(two, {})
    assert len(err) == 1 and "2 full-layer model loads" in err[0] and "[t#a] [t#b]" in err[0] and "X_LOAD" in err[0]
    perf = [("t#a", PROFILE), ("t#b", f"X_SKIP=tail {PROFILE}")]  # a load-time value of the per-run switch
    assert "2 executed-layer model loads" in B.one_load_errors(perf, {})[0]
    other = [("t#a", LADDER), ("t#c", f"{R} models/x/tests/test_own_process.py"), ("t#b", LADDER)]
    assert B.one_load_errors(other, {}) == []  # a unit of its own between them: the group still loads once
    assert [[u for u, _ in g] for g in B.gate_groups(other, {})] == [["t#a", "t#b"], ["t#c"]]


def test_layer_classes(tmp_path):
    assert B.layer_class(f"{T}test_profile.py") == "executed" and B.layer_class(f"{T}test_ladder.py") == "full"
    t = tmp_path / "models/x/tests/test_p.py"
    t.parent.mkdir(parents=True)
    t.write_text('BATCH_SHARED = True\nBATCH_LAYERS = "executed"\n')
    assert B.layer_class("models/x/tests/test_p.py", tmp_path) == "executed"
    t.write_text("BATCH_SHARED = True\n")
    assert B.layer_class("models/x/tests/test_p.py", tmp_path) == "full"


def test_in_the_runner_a_performance_run_never_takes_another_plans_full_model(monkeypatch):
    monkeypatch.setenv(TB.RESIDENT_ENV, "/tmp/r")
    TB._SUBSETS[(0, 1)] = (0, 1, 2)
    TB._MODELS["plan"] = object()
    TB._RUN["plan"] = object()
    try:
        assert not TB._serves_layers([0, 1, 2], [0, 1])
        TB._RUN["plan"] = TB._MODELS["plan"]  # the same plan (the accept gate's fresh process)
        assert TB._serves_layers([0, 1, 2], [0, 1])
    finally:
        TB._MODELS.clear()
        TB._SUBSETS.clear()


def test_per_run_switches_never_split_a_process(call_hooks):
    names = {"X_SKIP"}
    assert B.model_key({"X_SKIP": "decode"}, names) == B.model_key({}, names)
    assert B.model_key({"X_SKIP": "tail"}, names) != B.model_key({}, names)


def test_a_run_needing_a_second_model_fails_when_the_rule_is_on(tmp_path, monkeypatch):
    calls = []

    def main(args, plugins=None):
        name = TB._RUN["name"]
        calls.append(name)
        if name == "b":  # the second run needs another model than the process holds
            TB._FRESH.update(why="b needs another model", by="a", same_plan=same["plan"])
            return 1
        return 0

    same = {"plan": True}
    monkeypatch.setenv(B.ONE_LOAD_ENV, "1")
    plan = {"entries": [{"name": n, "env": {}, "args": ["t.py"], "log": str(tmp_path / f"{n}.log")} for n in "abc"]}
    assert TB.run_plan("MESH", (1, 1), dict(plan, one_load=True), None, main=main) == ["b"]  # a gate's plan says it
    calls.clear()
    failed = TB.run_plan("MESH", (1, 1), plan, None, main=main)
    assert failed == ["b"] and not TB.DEFERRED and calls == ["a", "b", "c"]
    assert "F283" in (tmp_path / "b.log").read_text()
    same["plan"] = False  # the model an earlier plan left (the resident runner's last request): a new process
    assert TB.run_plan("MESH", (1, 1), plan, None, main=main) == [] and TB.DEFERRED == ["b", "c"]
    same["plan"] = True
    monkeypatch.delenv(B.ONE_LOAD_ENV)
    calls.clear()
    assert TB.run_plan("MESH", (1, 1), plan, None, main=main) == [] and TB.DEFERRED == ["b", "c"]


def test_the_gate_refuses_and_validate_fails_a_task_needing_two_loads(sandbox, call_hooks):
    from models.demos.common.bringup.core import gate as G

    spec = sandbox.spec
    spec.data.setdefault("tests", {})["one_process"] = True
    task = {"id": "P.9", "gate": {"cmd": f"{LADDER} && X_LOAD=1 {LADDER}"}}
    assert G.one_load_errors(spec, task, {"P.9": task})
    assert G.one_load(spec) and B.ONE_LOAD_ENV not in G.gate_env(spec, sandbox.ledger, "P.9")  # (the plan says it)
    spec.data["tests"]["one_load"] = False
    assert G.one_load_errors(spec, task, {"P.9": task}) == []


def test_an_accept_gate_runs_its_performance_units_first_then_one_full_process(sandbox, monkeypatch, tmp_path):
    """F285: the performance units' process (the runner's executed-layers model) runs before the accuracy units'
    fresh full-model process with the determinism check, so the runner's model is not evicted for a second load."""
    from models.demos.common.bringup.core.gate import run_gate
    from models.demos.common.bringup.selftest import test_gate_one_process as TG

    calls = TG._setup(sandbox, monkeypatch, tmp_path)
    sandbox.write_spec(tests={"one_process": True})
    task = TG._gate(f"{TG.LADDER} && {TG.CONTRACT} && BRINGUP_PROFILE_WALL=1 {R} {T}test_profile.py")
    task["gate"]["metrics"].pop("gate_model_loads")
    led = sandbox.tasks(task)
    res = run_gate(sandbox.spec, led, "A.1")
    lines = calls.read_text().splitlines()
    assert len(lines) == 2 and "test_profile" in lines[0] and "test_determinism" in lines[1], (lines, res.summary())
