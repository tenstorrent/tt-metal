# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F170: the framework-waste audit (2026-10-22, reports/framework_waste_audit.md), group A: each test reproduces a
way the orchestrator spent the owner's time on DeepSeek-V4.1 and checks it no longer does."""

import pytest

from models.demos.common.bringup import orchestrator as O
from models.demos.common.bringup.core import stamps
from models.demos.common.bringup.core.gate import gate_units, run_gate
from models.demos.common.bringup.orchestrator import DONE, PAUSED, STOPPED, now
from models.demos.common.bringup.selftest.conftest import PY
from models.demos.common.bringup.selftest.test_core import task
from models.demos.common.bringup.selftest.test_orchestrator import impl_task, orch  # noqa: F401 (fixture)

# a stand-in for one test file of a gate: under a pytest/ directory, so the gate sees a test run (TEST_RUN)
UNIT = """import sys, pathlib
from models.demos.common.bringup.core import metrics as M
root = pathlib.Path(sys.argv[1]); name = sys.argv[2]
with open(root / "runs.txt", "a") as f:
    f.write(f"run {name}\\n")
M.record(f"pcc_{name}", 0.995)
if (root / f"timeout_{name}").exists():
    (root / f"timeout_{name}").unlink()  # times out once
    print("E   Failed: Timeout (>300.0s) from pytest-timeout.")
    sys.exit(1)
if (root / f"box_{name}").exists():
    (root / f"box_{name}").unlink()
    print("TT_THROW: Device 1: Timed out while waiting for active ethernet core 31-25")
    sys.exit(1)
sys.exit(1 if (root / f"fail_{name}").exists() else 0)
"""


def _unit(sandbox):
    root = sandbox.repo.parent
    (root / "pytest").mkdir(exist_ok=True)
    (root / "pytest" / "unit.py").write_text(UNIT)
    return root, lambda name: f"{PY} {root / 'pytest' / 'unit.py'} {root} {name}"


def _ran(root) -> list[str]:
    p = root / "runs.txt"
    return [ln.split()[1] for ln in p.read_text().splitlines()] if p.exists() else []


def test_a_git_error_never_gives_a_constant_code_key(tmp_path):
    """Audit item 16: a failed git call returned an empty file map, so every tree hashed alike and a PASS that never ran
    on this code could be reused."""
    with pytest.raises(stamps.GitError):
        stamps.code_key(tmp_path)  # not a repository


def test_every_test_segment_of_a_gate_is_its_own_unit(sandbox):
    """Audit item 5: X.3 (ladder && profile && positions) was one unit: a stop in its last test re-ran the full-target
    ladder and both profile runs. Each test segment is a unit; a setup step travels with its test."""
    root, run = _unit(sandbox)
    cmd = f"{run('ladder')} && echo setup > /dev/null && {run('profile')} && {run('positions')}"
    led = sandbox.tasks(task("X.3", cmd, metrics={"pcc_*": ">= 0.99"}, paths=["src"]))
    assert len(gate_units(led.task("X.3"))) == 3
    assert "echo setup" in gate_units(led.task("X.3"))[1][1]  # the setup step runs with the test after it
    (root / "fail_positions").write_text("")
    assert run_gate(sandbox.spec, led, "X.3").verdict == "FAIL" and _ran(root) == ["ladder", "profile", "positions"]
    (root / "fail_positions").unlink()
    assert run_gate(sandbox.spec, led, "X.3", commit=True).verdict == "PASS"
    assert _ran(root) == ["ladder", "profile", "positions", "positions"]  # the ladder and the profile were kept


def test_the_agents_gate_check_is_kept_and_the_orchestrators_gate_runs_only_the_rest(sandbox):
    """Audit item 2: the agent ran the gate by hand, then the orchestrator ran the same gate again (5-70 min per
    attempt). `gate --check` records each unit; the orchestrator's gate afterwards runs nothing the agent passed."""
    root, run = _unit(sandbox)
    cmd = f"for t in a b c; do {PY} {root / 'pytest' / 'unit.py'} {root} $t || exit 1; done"
    led = sandbox.tasks(task("C.1", cmd, metrics={"pcc_*": ">= 0.99"}, paths=["src"]))
    (root / "fail_c").write_text("")
    res = run_gate(sandbox.spec, led, "C.1", check=True)  # the agent's run
    assert res.verdict == "FAIL" and led.status("C.1") == "TODO" and not res.commit  # no state, no commit
    (root / "fail_c").unlink()
    assert run_gate(sandbox.spec, led, "C.1", check=True).verdict == "PASS" and _ran(root) == ["a", "b", "c", "c"]
    res = run_gate(sandbox.spec, led, "C.1", commit=True)  # the orchestrator's gate after the agent
    assert res.verdict == "PASS" and res.commit and _ran(root) == ["a", "b", "c", "c"]  # nothing ran again


def test_the_brief_names_the_gate_check_and_the_units(orch, sandbox):  # noqa: F811
    o = orch([impl_task()], {})
    text = o.brief(o.led.task("C.1"), "implement", 1, "").read_text()
    assert "gate C.1 --spec" in text and "--check" in text and "`C.1`:" in text


def test_a_pause_stops_before_the_next_agent_attempt(orch):  # noqa: F811
    """Audit item 6: the pause was honoured only after an abort; a failed gate started the next agent anyway."""
    o = orch([impl_task(tests=[])], {})
    (o.spec.repo / "src/impl.txt").write_text("0.1")  # existing work: the gate runs first and fails
    o.pause_file.write_text("")
    o.led.update("C.1", status="TODO", history_add={"t": now(), "status": "RESUMED"})
    o.pause_file.unlink()
    gate = o.gate

    def gate_then_pause(tid, reuse=None):  # the overseer pauses while the gate runs
        res = gate(tid, reuse)
        o.pause_file.write_text("")
        return res

    o.gate = gate_then_pause
    with pytest.raises(SystemExit) as e:
        o.run()
    assert e.value.code == PAUSED and orch.calls() == []


def test_an_agent_that_changed_nothing_is_not_measured_again(orch, sandbox):  # noqa: F811
    """Audit item 4: after every attempt the gate ran again on an unchanged tree (DS41: six attempts aborted after
    25-38 s, each after a 5-48 min gate). The failure of the last gate stands."""
    sandbox.write_spec(agents={"policy": {"implement": {"attempts": 2, "escalate": "stop"}}})
    counted = f"echo run >> {sandbox.repo.parent / 'gates.txt'} && {PY} tests/check.py"
    o = orch([impl_task(tests=[], gate={"cmd": counted, "metrics": {"pcc_out": ">= 0.99"}})], {})
    (o.spec.repo / "src/impl.txt").write_text("0.1")
    assert o.run() == STOPPED and len(orch.calls()) == 2
    assert (sandbox.repo.parent / "gates.txt").read_text().count("run") == 1  # the check before the agents only


def test_an_agent_that_crashed_without_changes_is_not_charged(orch, sandbox, monkeypatch):  # noqa: F811
    """Audit item 4: an API outage burned the attempts (and a gate each) in minutes."""
    monkeypatch.setattr(O, "CRASH_WAIT_S", 0)
    sandbox.write_spec(agents={"policy": {"implement": {"attempts": 1, "escalate": "stop"}}})
    o = orch(
        [impl_task(tests=[])],
        {"C.1.implement.1.md": {"exit": 1}, "C.1.implement.2.md": {"write": {"src/impl.txt": "1.0"}}},
    )
    assert o.run() == DONE and orch.calls() == [
        "C.1.implement.1.md bringup-engineer",
        "C.1.implement.2.md bringup-engineer",
    ]


FW_FAIL = (
    f"{PY} -c \"print('models/demos/common/bringup/testing/component_checks.py:174: KeyError'); "
    'print(\\"E   KeyError: \'component_col_factor\'\\"); raise SystemExit(1)"'
)


def test_a_framework_error_on_a_task_that_passed_before_stops_for_the_overseer(orch, sandbox):  # noqa: F811
    """Audit item 4: a framework bug (DS41 FIX.9: a renamed defaults key) started agents that cannot edit framework
    files. A programming error in the framework's code on a task that passed before stops for the overseer."""
    o = orch([impl_task(tests=[], gate={"cmd": FW_FAIL, "metrics": {}})], {})
    (o.spec.repo / "src/impl.txt").write_text("1.0")
    o.led.update("C.1", status="TODO", history_add={"t": now(), "status": "PASS"})
    o.led.update("C.1", history_add={"t": now(), "status": "RESUMED"})
    assert o.run() == STOPPED and orch.calls() == []
    assert "framework's own code" in o.led.state()["C.1"]["reason"][0]
    assert O.framework_error("tests/test_x.py:3: AssertionError") is None  # a failed check is the model's


def test_a_first_check_failing_in_the_framework_still_starts_the_agent(orch):  # noqa: F811
    """No implementation yet: a missing hook fails inside the framework too; that is the agent's work, not a stop."""
    o = orch([impl_task(tests=[], gate={"cmd": FW_FAIL, "metrics": {}})], {})
    (o.spec.repo / "src/impl.txt").write_text("1.0")
    o.run()
    assert orch.calls()[:1] == ["C.1.implement.1.md bringup-engineer"]


def test_a_timeout_or_a_box_error_is_retried_once_not_a_run_stop(orch, sandbox):  # noqa: F811
    """Audit items 7 and 17: a timeout under CPU load stopped the whole run for a person (DS41 FIX.10); a box error
    likewise. One automatic retry of the stopped unit (after a quick box check for a box error)."""
    root, run = _unit(sandbox)
    o = orch([impl_task(tests=[], gate={"cmd": f"{run('a')} && {run('b')}", "metrics": {"pcc_*": ">= 0.99"}})], {})
    (o.spec.repo / "src/impl.txt").write_text("1.0")
    (root / "timeout_b").write_text("")
    assert o.run() == DONE and orch.calls() == [] and _ran(root) == ["a", "b", "b"]
    o.led.reset(["C.1"])
    (root / "box_a").write_text("")
    o.BOX_CHECK = "true"
    assert o.run() == DONE and orch.calls() == []
    o.led.reset(["C.1"])
    (root / "box_a").write_text("")
    o.BOX_CHECK = "false"  # the box check fails: a person resets the board
    o.led.update("C.1", history_add={"t": now(), "status": "RESUMED"})
    (o.spec.repo / "src/impl.txt").write_text("1.0 ")  # new code under test: every unit runs
    assert o.run() == STOPPED and "reset the board" in o.led.state()["C.1"]["reason"][0]


def test_every_device_test_takes_the_box_timeout(fx, monkeypatch):
    """Audit item 7: frozen component and swap tests had no timeout mark, so pytest.ini's 300 s applied."""
    from models.demos.common.bringup.testing import harness

    monkeypatch.setenv("BRINGUP_SPEC", fx(box={"mesh": [1, 1], "test_timeout_s": 1234}))
    monkeypatch.setenv("BRINGUP_IMPL", "reference")

    def test_x(mesh_device):
        pass

    fn = harness.mesh_parametrize(test_x)
    assert [m.args for m in fn.pytestmark if m.name == "timeout"] == [(1234,)]


def test_the_debugger_keeps_a_pass_with_problems_and_honours_an_abort(orch, sandbox):  # noqa: F811
    """Audit items 9 and 6: the debugger threw away a passing gate for a lint-level problem (more attempts, a deferral
    attempt, then STOPPED over a committed PASS)."""
    sandbox.write_spec(
        agents={"policy": {"implement": {"attempts": 1, "escalate": "debugger", "debugger_attempts": 2}}}
    )
    o = orch(
        [impl_task(tests=[])],
        {
            "C.1.implement.1.md": {"write": {"src/impl.txt": "0.2"}},
            "C.1.implement.101.md": {"write": {"src/impl.txt": "1.0"}, "bash": ["pytest tests/check.py"]},
        },
    )
    assert o.run() == DONE
    assert [c for c in orch.calls() if ".101." in c or ".102." in c] == ["C.1.implement.101.md ttnn-expert-debugger"]
    assert o.led.state()["C.1"].get("review")  # the problem is filed for review, the pass stands


# ---- group B


def test_the_env_key_ignores_where_kernels_compile_and_how_many_cpu_threads():
    """Audit item 12: a shell that differed only in these re-ran every unit."""
    base = {"BRINGUP_IMPL": "device", "TT_METAL_HOME": "/r"}
    k = stamps.env_key(base)
    assert k == stamps.env_key(dict(base, BRINGUP_CPU_THREADS="8", TT_METAL_JIT_SERVER_ENDPOINT="h:1"))
    assert k != stamps.env_key(dict(base, BRINGUP_IMPL="reference"))


def test_a_fully_reused_gate_runs_no_pre_commit(sandbox, monkeypatch):
    """Audit item 22: pre-commit ran over every task path before every committing gate, reused ones included."""
    from models.demos.common.bringup.core import gate as G

    calls = []
    monkeypatch.setattr(G, "format_paths", lambda repo, paths: calls.append(list(paths)))
    (sandbox.repo / "src").mkdir()
    (sandbox.repo / "src/impl.py").write_text("x = 1\n")
    (sandbox.repo / "src/other.py").write_text("y = 1\n")
    sandbox.git("add", "src/other.py")
    sandbox.git("commit", "-q", "-m", "other")
    led = sandbox.tasks(task("A.1", "true", paths=["src"]))
    assert run_gate(sandbox.spec, led, "A.1", commit=True).verdict == "PASS"
    assert calls == [["src/impl.py"]]  # only the changed file
    led.reset(["A.1"])
    assert run_gate(sandbox.spec, led, "A.1", commit=True).verdict == "PASS" and calls == [["src/impl.py"]]


def test_a_unit_is_stamped_on_the_tree_as_it_starts(sandbox):
    """Audit item 15: an edit during a gate filed the later units under the gate's starting key."""
    from models.demos.common.bringup.core.gate import unit_stamps

    root, run = _unit(sandbox)
    (sandbox.repo / "src").mkdir()
    (root / "pytest" / "edit.py").write_text("open('src/impl.py', 'w').write('z = 2')\n")  # a unit that edits code
    cmd = f"{run('a')} && {PY} {root / 'pytest' / 'edit.py'} && {run('b')}"
    led = sandbox.tasks(task("A.1", cmd, metrics={"pcc_*": ">= 0.99"}, paths=["src"]))
    assert run_gate(sandbox.spec, led, "A.1").verdict == "PASS"
    store = stamps.Store(sandbox.spec.run_dir("default"))
    env = O.gate_env(sandbox.spec, led, "A.1")
    last, _, _, sid = unit_stamps(sandbox.spec, led.task("A.1"), env)[-1]
    assert store.get(last, sid) is not None  # the last unit is filed under the key of the tree it ran on


def test_a_dashboard_line_wakes_the_watch_once(tmp_path):
    """Audit item 20: DASHBOARD STALE two minutes after every gate, and a red dashboard made every watch return at once."""
    import os

    from models.demos.common.bringup import watch as W

    (tmp_path / W.PID_FILE).write_text(f"{os.getpid()} now\n")
    (tmp_path / "orchestrator.log").write_text("")
    red = lambda: "DASHBOARD RED: 1 passed tasks show red"  # noqa: E731
    assert W.watch(tmp_path, 0.05, poll=0.01, out=lambda *_: None, stale=red) == W.EVENT
    assert W.watch(tmp_path, 0.05, poll=0.01, out=lambda *_: None, stale=red) == W.TIMEOUT


def test_a_cpu_job_next_to_device_work_takes_a_quarter_of_the_cores(monkeypatch):
    """Audit item 19: the full-model goldens took every core at normal priority and starved a live gate (FIX.10)."""
    import os

    from models.demos.common.bringup.core import metrics as Mx

    monkeypatch.setattr(Mx, "device_work_running", lambda: True)
    monkeypatch.setattr(Mx, "_physical_cores", lambda: 16)
    monkeypatch.setattr(os, "nice", lambda n: 0)
    monkeypatch.delenv("BRINGUP_CPU_THREADS", raising=False)
    monkeypatch.delenv(Mx.TASK_ENV, raising=False)
    assert Mx.cpu_threads() == 4
    monkeypatch.setenv(Mx.TASK_ENV, "G.1")  # the gate's own CPU job: the box is its
    assert Mx.cpu_threads() == 16


def test_an_agent_is_charged_only_for_knowledge_errors_it_introduced(orch):  # noqa: F811
    """Audit item 10: a pre-existing format error in a shared knowledge file failed every attempt."""
    o = orch([impl_task(tests=[])], {"C.1.implement.1.md": {"write": {"src/impl.txt": "1.0"}}})
    seen = iter([["old error"], ["old error"]])
    o.knowledge_errors = lambda: next(seen)
    assert o.run() == DONE
    assert not any("old error" in p for r in o.led.state()["C.1"]["agent_runs"] for p in r["problems"])


def test_the_ab_key_sees_an_untracked_module_and_ignores_a_commit_of_the_same_content(orch, sandbox):  # noqa: F811
    """Audit item 14: an agent's new, untracked module could change after its A/B side was measured without changing
    the key (an accept could commit unmeasured code); a commit of the same content changed it (a needless re-measure).
    """
    o = orch([impl_task()], {})
    t = o.led.task("C.1")
    k0 = o.ab_code_key(t)
    (sandbox.repo / "src/new.py").write_text("a = 1\n")
    k1 = o.ab_code_key(t)
    assert k1 != k0
    sandbox.git("add", "src/new.py")
    sandbox.git("commit", "-q", "-m", "same content")
    assert o.ab_code_key(t) == k1
