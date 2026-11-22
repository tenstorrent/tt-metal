# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F281 / F283: gate units on the resident runner (core/batch.drive) and the owner's one-load rule (core/batch
one_load_errors, core/gate pre-check and validate, testing/batch's runtime refusal). CPU only."""

import json
import os
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
    with pytest.raises(resident.Refused, match="meant to serve this run but is yielded"):  # F286: never a fresh one
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


# ---------------------------------------------------------------- F286: P.84 (2026-11-19): ~10 device opens in one task
def test_a_single_run_unit_goes_to_a_yielded_runner_never_a_process_of_its_own(sandbox, monkeypatch, tmp_path):
    """P.84: after the runner stepped aside, each test_skip_positions unit (one run: no batch plan) started a safe-
    runner process of its own. Now it goes to the runner as a plan; a runner that cannot come back refuses it."""
    from models.demos.common.bringup.core.gate import run_gate
    from models.demos.common.bringup.selftest import test_gate_one_process as TG

    calls = TG._setup(sandbox, monkeypatch, tmp_path)
    sandbox.write_spec(tests={"one_process": True})
    task = TG._gate(f"BRINGUP_PROFILE_WALL=1 {R} {T}test_profile.py")
    task["gate"]["metrics"] = {}
    led = sandbox.tasks(task)
    d = tmp_path / "res"
    monkeypatch.setattr(resident, "wanted_dir", lambda env=None: d)
    sent = []

    def submit(dd, plan, env, fresh=False):
        data = json.loads(Path(plan).read_text())
        sent.append((fresh, [e["args"][0].rsplit("/", 1)[-1] for e in data["entries"]]))
        runs = {e["name"]: {"rc": 0, "metrics": {}, "files": [], "s": 1.0} for e in data["entries"]}
        B.results_path(Path(plan)).write_text(json.dumps({"running": None, "runs": runs}))
        return {"rc": 0, "runs": runs, "timing": {}}

    monkeypatch.setattr(resident, "for_env", lambda env, echo=None: d)
    monkeypatch.setattr(resident, "submit_plan", submit)
    run_gate(sandbox.spec, led, "A.1")
    spawned = [c for c in calls.read_text().splitlines()] if calls.exists() else []
    assert spawned == [] and any("test_profile.py" in names for _, names in sent), (spawned, sent)

    def refuse(env, echo=None):
        raise resident.Refused("the resident runner is dead")

    monkeypatch.setattr(resident, "for_env", refuse)
    res = run_gate(sandbox.spec, led, "A.1", fresh=True)
    assert res.verdict == "FAIL" and not calls.exists() and "the resident runner is dead" in res.log.read_text()


def test_the_ab_says_fail_when_a_side_of_its_test_failed(monkeypatch):
    """P.84: the table said the gate with the change on would PASS while both sides of its only test exited 1."""
    from types import SimpleNamespace

    from models.demos.common.bringup.orchestrator import Orchestrator

    items = [("test_skip_positions", "test", "cmd", {}), ("p145408", "position", "cmd", {})]
    o = SimpleNamespace(
        ab_items=lambda task: items,
        resident_serves=lambda: True,
        led=SimpleNamespace(tasks=lambda: {}),
        ab_determinism_failing=lambda *a: [],
    )
    o.ab_failed_runs = lambda task, rec: Orchestrator.ab_failed_runs(o, task, rec)
    o._gate_on = lambda *a: Orchestrator._gate_on(o, *a)
    o.spec = None
    monkeypatch.setattr("models.demos.common.bringup.core.gate.unit_states", lambda spec, led, tid: {})
    rec = {"tests": {"test_skip_positions": {"old": {"rc": 1}, "new": {"rc": 1}}}, "positions": {}}
    monkeypatch.setattr("models.demos.common.bringup.orchestrator.gate_limits", lambda task, tasks: {})
    got = Orchestrator.gate_on(o, {"id": "P.84"}, rec, Path("/nonexistent"))
    assert (
        not got["ok"] and "test_skip_positions: exit code 1" in got["failing"] and "p145408: not run" in got["failing"]
    )
    rec["tests"]["test_skip_positions"]["new"]["rc"] = 0
    rec["positions"]["p145408"] = {"new": {"rc": 0}}
    assert Orchestrator.gate_on(o, {"id": "P.84"}, rec, Path("/nonexistent"))["ok"]


def test_the_ab_says_fail_when_a_gate_unit_of_the_task_failed(monkeypatch):
    """F293 (PB.7): the quick A/B side passed, but the agent's own gate run had test_glued_oracle FAIL; the answer
    'would the gate pass' must say no and name the unit."""
    from types import SimpleNamespace

    from models.demos.common.bringup.orchestrator import Orchestrator

    o = SimpleNamespace(spec=None, led=None, _gate_on=lambda *a: {"ok": True, "failing": [], "partial": True})
    monkeypatch.setattr(
        "models.demos.common.bringup.core.gate.unit_states",
        lambda spec, led, tid: {"PB.7#test_glued_oracle": "FAIL", "PB.7#test_packed_oracle": "PASS"},
    )
    got = Orchestrator.gate_on(o, {"id": "PB.7"}, {}, Path("/nonexistent"))
    assert not got["ok"] and got["failing"] == ["PB.7#test_glued_oracle: FAIL in the task's last gate run"]
    monkeypatch.setattr("models.demos.common.bringup.core.gate.unit_states", lambda spec, led, tid: {"u": "PASS"})
    assert Orchestrator.gate_on(o, {"id": "PB.7"}, {}, Path("/nonexistent"))["ok"]


def test_task_accounting_counts_safe_runner_processes_and_runner_restarts(tmp_path):
    hm = tmp_path / "hostmem.jsonl"
    hm.write_text('{"task": "P.84"}\n{"task": "P.84"}\n{"task": "X.9"}\n')
    (tmp_path / "res").mkdir()
    (tmp_path / "res" / "requests.jsonl").write_text(
        json.dumps({"task": "P.84", "accounting": {"device_opens": 1, "model_loads": 1}})
        + "\n"
        + json.dumps({"task": "P.84", "accounting": {"device_opens": 0, "model_loads": 0}})
        + "\n"
    )
    a = resident.task_accounting(tmp_path / "res", hm, "P.84")
    assert a == {"opens": 3, "process_opens": 2, "runner_opens": 1, "runner_loads": 1}


def test_ab_sides_with_the_same_run_name_and_another_load_switch_go_to_a_new_process(tmp_path, monkeypatch):
    """F287 (P.84 A/B rerun): the off side's request loaded the model with MOE_REPLICAS off; the on side's run, also
    named "test_skip_positions.0", needs it on. It failed as if it had asked for two models itself; it must go to a new
    process (the runner restarts with that configuration: one open, one load, reported)."""
    from types import SimpleNamespace

    loads = []

    class Model:
        def __init__(self, layers):
            loads.append(os.environ.get("X_REPLICAS"))
            self.layers, self.state = list(layers), None

        def sync(self):
            pass

    hooks = SimpleNamespace(
        device_model=lambda mesh, s, layers, lm_head: Model(layers),
        batch_settings=lambda: {"REPLICAS": os.environ.get("X_REPLICAS")},
    )
    spec = SimpleNamespace(data={"target": {"seq": 128, "chunk": 32}}, hooks=lambda: hooks)
    monkeypatch.setenv("BRINGUP_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("BRINGUP_TASK", "P.84")

    def main(args, plugins=None):
        try:
            TB.device_model(spec, "MESH", [0, 1], lm_head=False)
        except TB.NeedsFreshProcess:
            return 1
        return 0

    def side(v):
        e = {"name": "test_skip_positions.0", "env": {"X_REPLICAS": v}, "args": ["t.py"], "key": "k"}
        return {"entries": [dict(e, log=str(tmp_path / f"{v}.log"))]}

    TB.release()
    try:
        assert TB.run_plan("MESH", (1, 1), side("off"), None, main=main, keep=True) == []  # the off side, held
        failed = TB.run_plan("MESH", (1, 1), side("on"), None, main=main, keep=True)
        assert failed == [] and TB.DEFERRED == ["test_skip_positions.0"] and loads == ["off"]
        assert "fresh process" in (tmp_path / "on.log").read_text()
    finally:
        TB.release()


def test_ab_items_keep_every_unit_of_the_same_test_file(monkeypatch):
    """F288 (P.84 attempt 2): two test_skip_positions units differing only in BRINGUP_POS_TOKENS were one A/B item
    (keyed by the file's stem): each side measured a different text. Now one item a unit, the same names on both
    sides, and both units of a side in one process (one model configuration)."""
    from types import SimpleNamespace

    from models.demos.common.bringup.orchestrator import Orchestrator

    skip = "models/demos/deepseek_v41_flash_d_p/tests/test_skip_positions.py"
    a = f"BRINGUP_POS_TOKENS=agentic DS41_DECODER_SKIP=decode SKIP_POS=145408 {R} {skip}"
    p = f"BRINGUP_POS_TOKENS=prose DS41_DECODER_SKIP=decode SKIP_POS=145408 {R} {skip}"
    o = SimpleNamespace(
        ab_quick=lambda t: False,
        ab_rungs=lambda t: [],
        ab_position_items=lambda t: [],
        ab_timing_only=lambda items: items,
        resident_serves=lambda: False,
        ab_test_names=Orchestrator.ab_test_names,
    )
    task = {"id": "P.84", "gate": {"cmd": f"{a} && {p}"}}
    items = [x for x in Orchestrator.ab_items(o, task) if x[1] == "test"]
    names = [n for n, _, _, _ in items]
    assert len(items) == 2 and len(set(names)) == 2, names
    assert any("agentic" in n for n in names) and any("prose" in n for n in names)
    assert names == ["test_skip_positions__BRINGUP_POS_TOKENS-agentic", "test_skip_positions__BRINGUP_POS_TOKENS-prose"]
    assert names == [x[0] for x in Orchestrator.ab_items(o, task) if x[1] == "test"]  # (the same on both sides)
    runs = [B.parse_all(c) for _, _, c, _ in items]
    keys = {B.model_key(r[0]["env"], {k for rs in runs for k in rs[0]["env"]}) for r in runs}
    assert len(keys) == 1  # one model configuration: both units in one runner request a side


def test_a_plans_model_is_loaded_for_its_furthest_declared_position(tmp_path, monkeypatch):
    """F288 (P.84: run0 at 770048 loaded the model, run1 at 1024000 then needed a new process)."""
    from types import SimpleNamespace

    t = tmp_path / "models/x/tests/test_pts.py"
    t.parent.mkdir(parents=True)
    t.write_text('BATCH_SHARED = True\nBATCH_POS_VARS = ("SKIP_POS",)\n')
    monkeypatch.setattr(B, "_REPO", tmp_path)
    spec = SimpleNamespace(get=lambda k, d=None: {"target.chunk": 5120, "deploy.seq": 1044480}.get(k, d))
    runs = [
        {"env": {"SKIP_POS": "145408,770048"}, "args": ["models/x/tests/test_pts.py"]},
        {"env": {"SKIP_POS": "145408,1024000,2000000"}, "args": ["models/x/tests/test_pts.py"]},  # (past deploy)
    ]
    assert B.hints(runs, spec)[B.MAX_SEQ_ENV] == str(1024000 + 5120)


def test_a_plans_model_is_loaded_for_a_declared_max_seq(tmp_path, monkeypatch):
    """PB.T: a test declaring BATCH_MAX_SEQ gets a load covering it (a test without position variables too), never past
    deploy.seq; a plan without such a test gets no hint from it."""
    from types import SimpleNamespace

    for name, body in (
        ("test_iso.py", "BATCH_SHARED = True\nBATCH_MAX_SEQ = 114688\n"),
        ("test_plain.py", "BATCH_SHARED = True\n"),
    ):
        t = tmp_path / "models/x/tests" / name
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(body)
    monkeypatch.setattr(B, "_REPO", tmp_path)
    spec = SimpleNamespace(
        get=lambda k, d=None: {"target.chunk": 8192, "target.seq": 56320, "deploy.seq": 1044480}.get(k, d)
    )
    iso = {"env": {}, "args": ["models/x/tests/test_iso.py"]}
    plain = {"env": {}, "args": ["models/x/tests/test_plain.py"]}
    assert B.declared_max_seq("models/x/tests/test_iso.py") == 114688
    assert B.hints([plain, iso], spec)[B.MAX_SEQ_ENV] == "114688"
    assert B.hints([iso], spec)[B.MAX_SEQ_ENV] == "114688"
    assert B.hints([plain, plain], spec)[B.MAX_SEQ_ENV] == "56320"
    small = SimpleNamespace(get=lambda k, d=None: {"target.chunk": 8192, "deploy.seq": 65536}.get(k, d))
    assert B.MAX_SEQ_ENV not in B.hints([iso], small)
