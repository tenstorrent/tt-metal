# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F150: a gate result is stamped with what it measured; a gate on unchanged work is never run twice."""

import json

from models.demos.common.bringup import overseer as OV
from models.demos.common.bringup.core import stamps
from models.demos.common.bringup.core import verify as V
from models.demos.common.bringup.core.gate import gate_units, run_gate
from models.demos.common.bringup.orchestrator import DONE, now
from models.demos.common.bringup.selftest.conftest import PY, record_cmd
from models.demos.common.bringup.selftest.test_core import task
from models.demos.common.bringup.selftest.test_orchestrator import impl_task, orch  # noqa: F401 (fixture)


def counted(sandbox, cmd: str) -> str:
    """The command, counting its runs outside the repo (a file it changes in the tree would change the code key)."""
    return f"echo run >> {sandbox.repo.parent / 'runs.txt'} && {cmd}"


def runs(sandbox) -> int:
    p = sandbox.repo.parent / "runs.txt"
    return p.read_text().count("run") if p.exists() else 0


PROFILE = f" && {PY} -c \"open('bringup/results/A.1_profile.json', 'w').write('{{}}')\""


def test_an_unchanged_gate_is_reused_and_a_one_byte_change_runs_it(sandbox):
    (sandbox.repo / "src").mkdir()
    (sandbox.repo / "src/impl.py").write_text("x = 1\n")
    cmd = counted(sandbox, record_cmd(pcc_out=0.995)) + PROFILE
    led = sandbox.tasks(task("A.1", cmd, metrics={"pcc_*": ">= 0.99"}, paths=["src"]))
    assert run_gate(sandbox.spec, led, "A.1", commit=True).verdict == "PASS" and runs(sandbox) == 1
    led.reset(["A.1"])  # a rerun of the same code (e.g. after a resume)
    res = run_gate(sandbox.spec, led, "A.1", commit=True)
    assert res.verdict == "PASS" and res.reused == ["A.1"] and runs(sandbox) == 1
    assert "reused: same code" in res.log.read_text()
    assert (sandbox.repo / "bringup/results/A.1_profile.json").exists()  # its outputs come back with it
    st = led.state()["A.1"]
    assert st["metrics"] == {"pcc_out": 0.995} and st["stamp"]["reused"] == ["A.1"]
    assert run_gate(sandbox.spec, led, "A.1", fresh=True).reused == [] and runs(sandbox) == 2  # --fresh
    (sandbox.repo / "src/impl.py").write_text("x = 2\n")  # one byte
    res = run_gate(sandbox.spec, led, "A.1", commit=True)
    assert res.verdict == "PASS" and res.reused == [] and runs(sandbox) == 3


def test_a_fail_is_never_reused(sandbox):
    led = sandbox.tasks(task("A.1", counted(sandbox, record_cmd(pcc_out=0.5)), metrics={"pcc_*": ">= 0.99"}))
    assert run_gate(sandbox.spec, led, "A.1").verdict == "FAIL"
    assert run_gate(sandbox.spec, led, "A.1").verdict == "FAIL" and runs(sandbox) == 2


def test_a_gate_with_a_missing_metric_is_not_kept_as_a_pass(sandbox):
    led = sandbox.tasks(task("A.1", counted(sandbox, record_cmd(other=1)), metrics={"pcc_*": ">= 0.99"}))
    assert run_gate(sandbox.spec, led, "A.1").verdict == "FAIL"
    assert run_gate(sandbox.spec, led, "A.1").verdict == "FAIL" and runs(sandbox) == 2


def test_the_code_key_follows_content_not_commits(sandbox):
    (sandbox.repo / "src").mkdir()
    (sandbox.repo / "src/a.py").write_text("a = 1\n")
    k0 = stamps.code_key(sandbox.repo)
    sandbox.git("add", "src/a.py")
    sandbox.git("commit", "-q", "-m", "a")
    assert stamps.code_key(sandbox.repo) == k0  # committed, same content
    (sandbox.repo / "bringup").mkdir(exist_ok=True)
    (sandbox.repo / "bringup/state.json").write_text("{}")
    (sandbox.repo / "NOTES.md").write_text("x")
    assert stamps.code_key(sandbox.repo) == k0  # the run's records and markdown measure nothing
    (sandbox.repo / "src/a.py").write_text("a = 2\n")
    assert stamps.code_key(sandbox.repo) != k0


def test_a_fix_jobs_unit_of_an_unchanged_passed_task_is_reused(sandbox):
    """A fix job's gate reruns its gate tasks' commands as units (F79); a unit whose task passed on the same code is
    restored, not run. The code key is the whole tree: a change anywhere it covers runs every unit."""
    (sandbox.repo / "src").mkdir()
    (sandbox.repo / "src/a.py").write_text("a = 1\n")
    (sandbox.repo / "src/b.py").write_text("b = 1\n")
    a = task("C.1", counted(sandbox, record_cmd(pcc_a=0.999)), metrics={"pcc_a": ">= 0.99"}, paths=["src/a.py"])
    b = task(
        "C.2", f"cat src/b.py >/dev/null && {counted(sandbox, record_cmd(pcc_b=0.999))}", metrics={"pcc_b": ">= 0.99"}
    )
    led = sandbox.tasks(a, b)
    for t in ("C.1", "C.2"):
        assert run_gate(sandbox.spec, led, t, commit=True).verdict == "PASS"
    job = OV.queue_fix(sandbox.spec.bringup_dir, "Fix b.", ["C.1", "C.2"], tasks=led.tasks())
    led = sandbox.tasks(a, b, OV.fix_task(job, led.tasks()))
    assert [u for u, _ in gate_units(led.task("FIX.1"), led.tasks())] == ["C.1", "C.2"]
    res = run_gate(sandbox.spec, led, "FIX.1", force=True)
    assert res.verdict == "PASS" and res.reused == ["C.1", "C.2"] and runs(sandbox) == 2
    (sandbox.repo / "src/b.py").write_text("b = 2\n")  # the fix's change: every unit's code key moves
    res = run_gate(sandbox.spec, led, "FIX.1", force=True)
    assert res.verdict == "PASS" and res.reused == [] and runs(sandbox) == 4
    got = json.loads((sandbox.repo / "bringup/results/FIX.1.json").read_text())["metrics"]
    assert set(got) == {"pcc_a", "pcc_b"}
    (sandbox.repo / "src/b.py").write_text("b = 1\n")  # reverted: the kept runs of that code count again
    assert run_gate(sandbox.spec, led, "FIX.1", force=True).reused == ["C.1", "C.2"] and runs(sandbox) == 4


def test_verify_reuses_a_kept_pass_unless_forced(sandbox, tmp_path):
    led = sandbox.tasks(task("C.1", counted(sandbox, record_cmd(pcc_out=0.999)), metrics={"pcc_out": ">= 0.99"}))
    assert run_gate(sandbox.spec, led, "C.1").verdict == "PASS" and runs(sandbox) == 1
    res = V.verify(sandbox.spec, ["C.1"], out=tmp_path / "v")
    assert not res["failed"] and res["tasks"][0].get("reused") and runs(sandbox) == 1
    assert json.loads((tmp_path / "v/results/C.1.json").read_text())["metrics"]["pcc_out"]["value"] == 0.999
    res = V.verify(sandbox.spec, ["C.1"], out=tmp_path / "v2", force=True)
    assert not res["failed"] and not res["tasks"][0].get("reused") and runs(sandbox) == 2


def test_a_resumed_task_reuses_its_unchanged_pass_without_an_agent(orch, sandbox):  # noqa: F811
    """F87 gates a resumed task first; on unchanged code that gate is the kept PASS, not a run (DeepSeek-V4.1: resumed
    fix jobs re-ran 45-70 min gates on code unchanged since their last run)."""
    gate = {"cmd": counted(sandbox, f"{PY} tests/check.py"), "metrics": {"pcc_out": ">= 0.99"}}
    o = orch([impl_task(tests=None, gate=gate)], {"C.1.implement.1.md": {"write": {"src/impl.txt": "0.999"}}})
    assert o.run() == DONE and runs(sandbox) == 2  # the pre-agent check and the gate after the agent
    o.led.update("C.1", status="TODO", history_add={"t": now(), "status": "RESUMED"})  # e.g. a crash before the commit
    calls = len(orch.calls())
    assert o.run() == DONE
    assert runs(sandbox) == 2 and len(orch.calls()) == calls and o.led.status("C.1") == "PASS"
    assert any("reused: same code" in ln for ln in orch.lines)
