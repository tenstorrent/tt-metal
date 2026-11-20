# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F269 (owner 2026-11-18): a batch process holds ONE model. A run that needs another model than the one loaded
(2026-11-17: ab/P.72/old/test_full_prefill, its runtime sized for the deployment context) is never loaded again in
place (the released model's device DRAM did not come back: Out of Memory at the reload); its process ends at that
run and core/batch.drive runs it and the rest in a fresh process. CPU only: fake runs, a fake model, a fake driver
that plays each process (testing/batch.run_plan with the module state of a new process)."""

import json
import shlex
from pathlib import Path

from models.demos.common.bringup.core import batch as B
from models.demos.common.bringup.selftest.test_one_process import Model, fake_spec, shared  # noqa: F401
from models.demos.common.bringup.testing import batch as TB

LAYERS = {
    "t_a": [0, 1],
    "t_b": [0, 1],
    "t_fail": [0, 1],
    "t_full": [0],
    "t_after": [0],
}  # t_full: another model (other layers)


def _entries(tmp_path, tests):
    return [
        {"name": f"r{i}", "group": f"r{i}", "env": {}, "args": [t], "run_all": True,
         "log": str(tmp_path / f"r{i}.log"), "key": "k"}
        for i, t in enumerate(tests, 1)
    ]  # fmt: skip


def _processes(mesh, log):
    """A fake driver's process: a fresh module state (no model, nothing deferred), then run_plan over its plan.
    ``log``: per process [(run, outcome)] and its model loads."""

    def main(args, plugins):
        name = TB._RUN["name"]
        try:
            m = TB.device_model(fake_spec(), mesh, LAYERS[args[0]], lm_head=False)
        except TB.NeedsFreshProcess:
            log[-1]["runs"].append((name, "needs another model"))
            return 1  # (as pytest reports a test whose fixture raised)
        log[-1]["runs"].append((name, "ran"))
        log[-1]["models"].append(id(m))
        return 1 if args[0] == "t_fail" else 0

    def run(cmd, k):
        assert f"{B.DRIVEN_ENV}=1 " in cmd
        plan = Path(shlex.split(cmd.split(f"{B.PLAN_ENV}=")[1])[0])
        TB._MODELS.clear()  # a new process: no model of the last one (never released into: it ended)
        loads0 = Model.loads
        log.append({"k": k, "runs": [], "models": []})
        failed = TB.run_plan(mesh, (1, 1), json.loads(plan.read_text()), plan, main=main)
        log[-1]["loads"] = Model.loads - loads0
        return 1 if failed else 0

    return run


def test_a_run_needing_another_model_runs_in_a_fresh_process(shared, tmp_path, capsys):
    """Runs 1-2 share process A's model; run 3 needs another: process A ends there (no reload in place) and process
    B runs 3 and 4 on its own load. Each run has one result, in the plan's results file, as from one process."""
    mesh, log = object(), []
    plan = B.write_plan(tmp_path / "plan.json", _entries(tmp_path, ["t_a", "t_b", "t_full", "t_after"]))
    rc = B.drive(plan, _processes(mesh, log))
    assert rc == 0
    assert [p["k"] for p in log] == [0, 1]  # two processes
    a, b = log
    assert a["runs"] == [("r1", "ran"), ("r2", "ran"), ("r3", "needs another model")]
    assert a["loads"] == 1 and len(set(a["models"])) == 1  # process A: one load, never a second one in place
    assert b["runs"] == [("r3", "ran"), ("r4", "ran")] and b["loads"] == 1  # process B: a fresh load for run 3
    assert "WARNING" not in capsys.readouterr().out  # (the reload-in-place warning is gone)
    got = B.load_results(plan)
    assert sorted(got["runs"]) == ["r1", "r2", "r3", "r4"] and all(r["rc"] == 0 for r in got["runs"].values())
    assert got["running"] is None and got["deferred"] == [] and got["processes"] == 2
    assert got["runs"]["r3"]["loads"] == 1 and got["runs"]["r1"]["loads"] == 1 and got["runs"]["r2"]["loads"] == 0
    fresh = B.load_results(Path(f"{plan}.fresh1.json"))  # the fresh process's own results: runs 3 and 4 only
    assert sorted(fresh["runs"]) == ["r3", "r4"]
    assert "F269" in (tmp_path / "r3.log").read_text()  # the run's own log says why it moved


def test_the_process_that_ends_early_leaves_the_rest_unrecorded(shared, tmp_path):
    """The first process alone: the run that needs another model and the runs after it have no result (no rc: not
    failed, not passed, never filed), only ``deferred``; a failed group's later run is skipped, not deferred."""
    mesh, log = object(), []
    entries = _entries(tmp_path, ["t_a", "t_fail", "t_full", "t_after"])
    entries[1]["group"] = entries[3]["group"] = "g"  # r2 fails: its group's r4 is skipped, not sent on
    plan = B.write_plan(tmp_path / "plan.json", entries)
    assert _processes(mesh, log)(f"{B.DRIVEN_ENV}=1 {B.command(plan)}", 0) == 1
    got = B.load_results(plan)
    assert got["runs"]["r1"]["rc"] == 0 and got["runs"]["r2"]["rc"] == 1 and "r3" not in got["runs"]
    assert got["runs"]["r4"] == {"rc": None, "skipped": True}
    assert got["deferred"] == ["r3"] and TB.DEFERRED == ["r3"] and got["running"] is None


def test_a_run_needing_two_models_itself_fails_and_is_never_reloaded_in_place(shared, tmp_path):
    mesh = object()
    calls = []

    def main(args, plugins):
        TB.device_model(fake_spec(), mesh, [0, 1], lm_head=False)
        try:
            TB.device_model(fake_spec(), mesh, [0], lm_head=False)
        except TB.NeedsFreshProcess:
            calls.append("refused")
            return 0  # (even a test that swallows it)
        return 0

    failed = TB.run_plan(mesh, (1, 1), {"entries": _entries(tmp_path, ["t_a"])}, None, main=main)
    assert failed == ["r1"] and TB.DEFERRED == [] and calls == ["refused"] and Model.loads == 1


def test_the_driver_stops_when_a_fresh_process_runs_none_of_its_runs(tmp_path):
    plan = B.write_plan(tmp_path / "plan.json", _entries(tmp_path, ["t_a", "t_b"]))
    n = []

    def run(cmd, k):
        p = Path(shlex.split(cmd.split(f"{B.PLAN_ENV}=")[1])[0])
        names = [e["name"] for e in json.loads(p.read_text())["entries"]]
        done = {"r1": {"rc": 0}} if k == 0 else {}
        B.results_path(p).write_text(json.dumps({"running": None, "runs": done, "deferred": names[k == 0 :]}))
        n.append(k)
        return 0

    B.drive(plan, run)
    assert n == [0, 1]  # the fresh process deferred all it got: no third process
    got = B.load_results(plan)
    assert list(got["runs"]) == ["r1"] and got["deferred"] == ["r2"]  # r2 never ran: callers count it as not run
