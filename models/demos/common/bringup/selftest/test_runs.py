# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F2: freeze with the zero-stub check, resume, rerun, fork, compare, agent-definition hashes. CPU only."""

import pytest

from models.demos.common.bringup.core.gate import run_gate
from models.demos.common.bringup.core.ledger import Ledger
from models.demos.common.bringup.core.runs import (
    FreezeError,
    agent_hashes,
    compare,
    fork,
    freeze_task,
    init_run,
    rerun_from,
    resume_point,
)
from models.demos.common.bringup.selftest.conftest import PY, record_cmd

# A component test in miniature: the module under test is chosen by BRINGUP_IMPL, like the real templates.
CHECK = """
import os
from models.demos.common.bringup.core import metrics as M
impl = os.environ.get("BRINGUP_IMPL", "device")
M.record("pcc_out", {values}[impl])
"""


def write_check(sandbox, values, name="tests/check.py"):
    f = sandbox.repo / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(CHECK.format(values=values))
    return name


def comp_task(tid, test, deps=()):
    return {
        "id": tid,
        "title": f"component {tid}",
        "deps": list(deps),
        "tests": [test],
        "gate": {"cmd": f"{PY} {test}", "metrics": {"pcc_out": ">= 0.99"}},
    }


def test_freeze_validates_with_reference_and_stub_then_commits(sandbox):
    t = write_check(sandbox, {"reference": 1.0, "stub": 0.0, "device": 0.995})
    led = sandbox.tasks(comp_task("C.1", t))
    rec = freeze_task(sandbox.spec, led, "C.1")
    assert rec["reference"] == "PASS" and rec["stub"] == "FAIL" and list(rec["files"]) == [t]
    assert led.task("C.1")["frozen"]["files"][t] == rec["files"][t]
    assert sandbox.git("log", "-1", "--format=%s") == "[toy][C.1][freeze] component C.1"
    assert set(sandbox.git("show", "--name-only", "--format=", "HEAD").split()) == {"bringup/tasks.yaml", t}
    assert led.status("C.1") == "TODO"  # the stub and reference runs never touch state
    assert run_gate(sandbox.spec, led, "C.1").verdict == "PASS"
    (sandbox.repo / t).write_text((sandbox.repo / t).read_text().replace("0.995", "0.5").replace("0.0", "0.999"))
    res = run_gate(sandbox.spec, led, "C.1")
    assert res.verdict == "FAIL" and "frozen file changed" in res.summary()


def test_freeze_rejects_a_test_that_passes_a_zero_stub(sandbox):
    t = write_check(sandbox, {"reference": 1.0, "stub": 1.0, "device": 1.0})
    led = sandbox.tasks(comp_task("C.1", t))
    with pytest.raises(FreezeError, match="zero stub"):
        freeze_task(sandbox.spec, led, "C.1")
    assert "frozen" not in led.task("C.1")


def test_freeze_rejects_a_test_that_fails_the_reference(sandbox):
    t = write_check(sandbox, {"reference": 0.9, "stub": 0.0, "device": 1.0})
    led = sandbox.tasks(comp_task("C.1", t))
    with pytest.raises(FreezeError, match="CPU reference"):
        freeze_task(sandbox.spec, led, "C.1")


def test_freeze_without_stub_check(sandbox):
    t = write_check(sandbox, {"reference": 1.0, "stub": 1.0, "device": 1.0})
    task = comp_task("C.1", t)
    task["stub_check"] = False
    led = sandbox.tasks(task)
    rec = freeze_task(sandbox.spec, led, "C.1", commit=False)
    assert "stub" not in rec and t in rec["files"]


def chain(sandbox):
    return sandbox.tasks(
        {"id": "A", "title": "a", "gate": {"cmd": record_cmd(v=1), "metrics": {"v": "== 1"}}},
        {"id": "B", "title": "b", "deps": ["A"], "gate": {"cmd": record_cmd(v=2), "metrics": {"v": "== 2"}}},
        {"id": "C", "title": "c", "deps": ["B"], "gate": {"cmd": "exit 1"}},
        {"id": "D", "title": "d", "deps": ["A"], "gate": {"cmd": "true"}},
    )


def test_resume_and_rerun(sandbox):
    led = chain(sandbox)
    for t in "ABC":
        run_gate(sandbox.spec, led, t)
    assert resume_point(led) == "C"  # the failed step
    assert rerun_from(led, "B") == ["B", "C"]
    assert led.status("A") == "PASS" and led.status("B") == "TODO" and resume_point(led) == "B"


def test_fork_from_a_passed_commit_shares_artifacts(sandbox):
    led = chain(sandbox)
    init_run(sandbox.spec, led, "runA")
    for t in "AB":
        assert run_gate(sandbox.spec, led, t, commit=True).commit
    fspec = fork(sandbox.spec, led, "A", "runB")
    fled = Ledger(fspec.bringup_dir)
    assert fspec.repo == sandbox.spec.run_dir("runB") / "worktree"
    assert (fspec.repo / ".git").exists()
    assert fspec.golden_root == sandbox.spec.golden_root and fspec.tt_cache() == sandbox.spec.tt_cache()
    assert fled.status("A") == "PASS" and fled.status("B") == "TODO"
    run = fled.state()["_run"]
    assert run["name"] == "runB" and run["branch"] == "bringup/toyspec/runB"
    assert run["forked_from"] == {
        "run": "runA",
        "task": "A",
        "commit": sandbox.git("log", "-1", "--format=%h", "HEAD~1"),
    }
    assert led.status("B") == "PASS"  # the parent run is untouched
    with pytest.raises(RuntimeError, match="passed task"):
        fork(sandbox.spec, led, "C", "runC")


def test_compare_reports_attempts_defs_and_metric_deltas(tmp_path):
    a, b = Ledger(tmp_path / "a"), Ledger(tmp_path / "b")
    tasks = {"tasks": [{"id": "X", "title": "x", "gate": {"cmd": "true"}}]}
    a.write_tasks(tasks)
    b.write_tasks(tasks)
    a.update("X", status="PASS", attempts=0, metrics={"pcc": 0.99}, agent={"defs": {"impl.md": {"blob": "1"}}})
    b.update("X", status="PASS", attempts=2, metrics={"pcc": 0.995}, agent={"defs": {"impl.md": {"blob": "2"}}})
    (row,) = compare(a, b)
    assert row["attempts"] == (0, 2) and row["agent_defs_changed"] == ["impl.md"]
    assert row["metric_deltas"] == {"pcc": pytest.approx(0.005)}


def test_agent_hashes_detect_local_edits(sandbox):
    f = sandbox.repo / "agents/impl.md"
    f.parent.mkdir()
    f.write_text("v1\n")
    sandbox.git("add", "agents/impl.md")
    sandbox.git("commit", "-q", "-m", "agent")
    clean = agent_hashes(sandbox.repo, ["agents/impl.md", "agents/missing.md"])
    assert clean["agents/impl.md"]["dirty"] is False and clean["agents/missing.md"]["blob"] is None
    f.write_text("v2\n")
    dirty = agent_hashes(sandbox.repo, ["agents/impl.md"])
    assert (
        dirty["agents/impl.md"]["dirty"] is True and dirty["agents/impl.md"]["blob"] != clean["agents/impl.md"]["blob"]
    )
