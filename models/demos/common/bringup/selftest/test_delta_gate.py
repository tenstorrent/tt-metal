# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F169: a gate never restarts (owner, 2026-10-22). Every test is its own unit; a unit that passed in its task's gate
stays passed until the code under test changes; after a failure only the delta runs (DeepSeek-V4.1 FIX.10 re-ran a
~40 min gate four times after waiver / framework fixes and a timeout)."""

from models.demos.common.bringup import overseer as OV
from models.demos.common.bringup import watch as W
from models.demos.common.bringup.core import stamps
from models.demos.common.bringup.core.gate import delta_status, run_gate
from models.demos.common.bringup.orchestrator import DONE, STOPPED, now
from models.demos.common.bringup.selftest.conftest import PY
from models.demos.common.bringup.selftest.test_core import task
from models.demos.common.bringup.selftest.test_orchestrator import impl_task, orch  # noqa: F401 (fixture)

UNIT = """import sys, pathlib
from models.demos.common.bringup.core import metrics as M
root = pathlib.Path(sys.argv[1]); name = pathlib.Path(sys.argv[2]).stem.replace("test_c_", "")
with open(root / "runs.txt", "a") as f:
    f.write(f"run {name}\\n")
M.record(f"pcc_{name}", 0.995)
if (root / f"infra_{name}").exists():
    print("E   Failed: Timeout (>300.0s) from pytest-timeout.")
    sys.exit(1)
sys.exit(1 if (root / f"fail_{name}").exists() else 0)
"""
NAMES = ("a", "b", "c", "d", "e")


def _loop(sandbox, tid="A.1", names=NAMES):
    root = sandbox.repo.parent
    (root / "unit.py").write_text(UNIT)
    return f"for t in {' '.join(names)}; do {PY} {root / 'unit.py'} {root} tests/test_c_${{t}}.py || exit 1; done", root


def _ran(root) -> list[str]:
    p = root / "runs.txt"
    return [ln.split()[1] for ln in p.read_text().splitlines()] if p.exists() else []


def _five(sandbox):
    cmd, root = _loop(sandbox)
    led = sandbox.tasks(task("A.1", cmd, metrics={"pcc_*": ">= 0.99"}, paths=["src"]))
    (sandbox.repo / "src").mkdir(exist_ok=True)
    (sandbox.repo / "src/impl.py").write_text("x = 1\n")
    return led, root


def test_a_failed_gate_runs_only_the_delta_after_a_waiver_and_a_framework_fix(sandbox):
    led, root = _five(sandbox)
    (root / "fail_c").write_text("")
    res = run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "FAIL" and _ran(root) == ["a", "b", "c"]
    assert "  units passed: A.1#a, A.1#b" in res.lines
    assert any(ln.startswith("  units remaining: A.1#c, A.1#d, A.1#e") for ln in res.lines)
    assert led.state()["A.1"]["units"] == {"A.1#a": "PASS", "A.1#b": "PASS", "A.1#c": "FAIL"} | {
        "A.1#d": "not run",
        "A.1#e": "not run",
    }
    # the overseer's fixes: a waiver in the spec, a framework file; neither is code under test
    sandbox.write_spec(tests={"waivers": {"c": "a format limit (owner)"}})
    fw = sandbox.repo / "models/demos/common/bringup/testing"
    fw.mkdir(parents=True, exist_ok=True)
    (fw / "component.py").write_text("# fixed\n")
    (root / "fail_c").unlink()
    res = run_gate(sandbox.spec, led, "A.1", commit=True)  # gate --continue
    assert res.verdict == "PASS" and res.reused == ["A.1#a", "A.1#b"] and res.commit
    assert _ran(root) == ["a", "b", "c", "c", "d", "e"]  # 1 and 2 never ran again
    assert set(led.state()["A.1"]["metrics"]) == {f"pcc_{n}" for n in NAMES}


def test_a_change_to_the_code_under_test_runs_every_unit(sandbox):
    led, root = _five(sandbox)
    assert run_gate(sandbox.spec, led, "A.1").verdict == "PASS" and len(_ran(root)) == 5
    mdir = sandbox.spec.model_dir / "tt"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "model.py").write_text("y = 1\n")  # the model's code: an agent's change
    led.reset(["A.1"])
    assert run_gate(sandbox.spec, led, "A.1").verdict == "PASS" and _ran(root)[5:] == list(NAMES)
    (sandbox.repo / "src/impl.py").write_text("x = 2\n")  # the task's own paths count too
    assert run_gate(sandbox.spec, led, "A.1").verdict == "PASS" and _ran(root)[10:] == list(NAMES)


def test_an_infrastructure_stop_is_retried_alone(sandbox):
    led, root = _five(sandbox)
    (root / "infra_d").write_text("")
    assert run_gate(sandbox.spec, led, "A.1").verdict == "FAIL" and _ran(root) == ["a", "b", "c", "d"]
    assert led.state()["A.1"]["units"]["A.1#d"] == "STOPPED"
    (root / "infra_d").unlink()
    assert run_gate(sandbox.spec, led, "A.1").verdict == "PASS" and _ran(root)[4:] == ["d", "e"]


def test_unit_runs_one_unit_and_rerun_names_passed_ones(sandbox):
    led, root = _five(sandbox)
    (root / "fail_c").write_text("")
    assert run_gate(sandbox.spec, led, "A.1").verdict == "FAIL"
    (root / "fail_c").unlink()
    one = run_gate(sandbox.spec, led, "A.1", units=["A.1#c"])  # gate --unit: see the fix work, in seconds
    assert one.verdict == "PASS" and _ran(root)[3:] == ["c"] and led.status("A.1") == "FAIL"
    assert dict(delta_status(sandbox.spec, led, "A.1"))["A.1#c"].startswith("kept")
    assert run_gate(sandbox.spec, led, "A.1").verdict == "PASS" and _ran(root)[4:] == ["d", "e"]
    led.reset(["A.1"])
    assert run_gate(sandbox.spec, led, "A.1", rerun=["A.1#a"]).verdict == "PASS" and _ran(root)[6:] == ["a"]
    assert run_gate(sandbox.spec, led, "A.1", units=["A.1#zz"]).verdict == "FAIL"  # unknown unit: refused


def test_a_resume_after_an_overseer_fix_runs_the_delta_and_starts_no_agent(orch, sandbox):  # noqa: F811
    cmd, root = _loop(sandbox, "C.1", ("a", "b", "c"))
    gate = {"cmd": cmd, "metrics": {"pcc_*": ">= 0.99"}}
    o = orch([impl_task(tests=None, gate=gate)], {})
    (root / "fail_b").write_text("")
    assert run_gate(sandbox.spec, o.led, "C.1").verdict == "FAIL" and _ran(root) == ["a", "b"]
    OV.classify(sandbox.spec.bringup_dir, "C.1", "waiver", "the owner's waiver covers b")
    o.led.update("C.1", status="TODO", history_add={"t": now(), "status": "RESUMED"})
    assert o.run() == STOPPED and orch.calls() == []  # still failing: stopped for the overseer, no agent
    assert any("STOPPED (overseer-classified waiver)" in ln for ln in orch.lines)
    assert _ran(root) == ["a", "b", "b"]  # only the failed unit ran
    (root / "fail_b").unlink()
    o.led.update("C.1", status="TODO", history_add={"t": now(), "status": "RESUMED"})
    assert o.run() == DONE and orch.calls() == [] and o.led.status("C.1") == "PASS"
    assert _ran(root) == ["a", "b", "b", "b", "c"]
    assert any("gate unit 2/3 C.1#b PASS" in ln for ln in orch.lines)
    assert OV.classification(sandbox.spec.bringup_dir, "C.1") is None  # cleared once the gate passed


def test_the_watch_wakes_on_a_failed_unit():
    assert W.PATTERN.search("  [C.1] gate unit 3/5 C.1#c FAIL (12 s)")
    assert W.PATTERN.search("  [C.1] gate unit 4/5 C.1#d STOPPED (300 s)")
    assert not W.PATTERN.search("  [C.1] gate unit 1/5 C.1#a PASS (40 s)")


def test_the_delta_record_survives_in_the_run_dir(sandbox):
    led, root = _five(sandbox)
    run_gate(sandbox.spec, led, "A.1")
    d = stamps.Delta(sandbox.spec.run_dir("default"), "A.1").all()
    assert set(d) == {f"A.1#{n}" for n in NAMES} and all(r["verdict"] == "PASS" for r in d.values())
