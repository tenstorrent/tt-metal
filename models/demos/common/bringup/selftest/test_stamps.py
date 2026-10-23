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
        "C.2",
        f"cat src/b.py >/dev/null && {counted(sandbox, record_cmd(pcc_b=0.999))}",
        metrics={"pcc_b": ">= 0.99"},
        paths=["src/b.py"],
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


def test_a_unit_is_rerun_only_when_what_it_reads_changed(sandbox):
    """F162: DeepSeek-V4.1 FIX.10 re-ran 25-45 min of passed tests three times for edits they could not see (a spec
    waiver for another step, an intake re-approval, records). The code key now covers what a unit can depend on."""
    (sandbox.repo / "src").mkdir()
    (sandbox.repo / "src/impl.py").write_text("x = 1\n")
    s = sandbox.spec
    k0 = stamps.code_key(sandbox.repo, s)
    same = {  # none of these can change a measurement
        "bringup/approvals.yaml": "intake: me\n",
        "bringup/tasks.yaml": "tasks: []\n",
        "models/demos/common/bringup/dev/BREADCRUMBS.md": "x\n",
        "models/demos/common/bringup/knowledge/k.md": "x\n",
        "models/demos/common/bringup/skill/bringup/SKILL.md": "x\n",
        "models/demos/common/bringup/selftest/test_x.py": "x = 1\n",
        "models/demos/othermodel/bringup/spec.yaml": "model: other\n",  # another model's bring-up records
        "models/demos/othermodel/tests/bringup/test_c_blk_a.py": "x = 1\n",  # and its frozen tests
    }
    for p, text in same.items():
        (sandbox.repo / p).parent.mkdir(parents=True, exist_ok=True)
        (sandbox.repo / p).write_text(text)
        assert stamps.code_key(sandbox.repo, sandbox.spec) == k0, p
    s = sandbox.write_spec(tests={"waivers": {"experts": "bfp4 (owner)"}})
    assert stamps.code_key(sandbox.repo, s) == k0  # a waiver is stamped per unit, not in the code key
    changes = {
        "src/impl.py": "x = 2\n",  # the model's code
        "ttnn/ttnn/bringup/fork/op.py": "y = 1\n",  # a fork
        "models/demos/common/bringup/testing/component.py": "z = 1\n",  # the test logic
        "models/demos/othermodel/bringup/hooks.py": "h = 1\n",  # Python another model's bring-up may share
    }
    for p, text in changes.items():
        k = stamps.code_key(sandbox.repo, s)
        (sandbox.repo / p).parent.mkdir(parents=True, exist_ok=True)
        (sandbox.repo / p).write_text(text)
        assert stamps.code_key(sandbox.repo, s) != k, p
    k = stamps.code_key(sandbox.repo, s)
    s = sandbox.write_spec(target={"seq": 512, "chunk": 64})  # any other spec field still counts
    assert stamps.code_key(sandbox.repo, s) != k


def test_a_waiver_reruns_only_the_units_that_read_it():
    """F162: a waiver changes the component test of its step and every swap test (F158: it reaches the block), never
    the ladder, the contract or the profile, and not another step's component test."""

    class S:
        def __init__(self, w):
            self.w = w

        def get(self, key, default=None):
            return self.w if key == "tests.waivers" else default

    a, b = S({"experts": "bfp4"}), S({"experts": "bfp4", "router": "TF32"})
    comp = "scripts/run_safe_pytest.sh --run-all models/demos/m/tests/bringup/test_c_swa_experts.py"
    swap = "scripts/run_safe_pytest.sh --run-all models/demos/m/tests/bringup/test_swap_swa_12_experts.py"
    router = "scripts/run_safe_pytest.sh --run-all models/demos/m/tests/bringup/test_c_swa_router.py"
    ladder = "BRINGUP_RUNG=last scripts/run_safe_pytest.sh --run-all models/demos/common/bringup/tests/test_ladder.py"
    folder = "scripts/run_safe_pytest.sh --run-all models/demos/m/tests/bringup"
    assert stamps.waiver_key(a, comp) == stamps.waiver_key(b, comp) != ""  # the router waiver: not its step
    assert stamps.waiver_key(a, swap) != stamps.waiver_key(b, swap)
    assert stamps.waiver_key(a, router) == "" != stamps.waiver_key(b, router)
    assert stamps.waiver_key(a, ladder) == stamps.waiver_key(b, ladder) == ""
    assert stamps.waiver_key(a, folder) != stamps.waiver_key(b, folder)


UNIT_SCRIPT = """import sys, pathlib
from models.demos.common.bringup.core import metrics as M
root = pathlib.Path(sys.argv[1]); name = pathlib.Path(sys.argv[2]).stem.replace("test_c_", "")
with open(root / "runs.txt", "a") as f:
    f.write(f"run {name}\\n")
M.record(f"pcc_{name}", 0.995)
sys.exit(1 if (root / f"fail_{name}").exists() else 0)
"""


def _looped(sandbox, names=("alpha", "beta", "gamma")):
    """A gate segment `for t in ...; do <one test> || exit 1; done` over three tests (DS41 FIX.10's expert loop)."""
    root = sandbox.repo.parent
    (root / "unit.py").write_text(UNIT_SCRIPT)
    loop = f"for t in {' '.join(names)}; do {PY} {root / 'unit.py'} {root} tests/test_c_${{t}}.py || exit 1; done"
    return sandbox.tasks(task("A.1", loop, metrics={"pcc_*": ">= 0.99"})), root


def _ran(root) -> list[str]:
    p = root / "runs.txt"
    return [ln.split()[1] for ln in p.read_text().splitlines()] if p.exists() else []


def test_a_looped_gate_reuses_the_tests_that_passed(sandbox):
    """F166: DeepSeek-V4.1 FIX.10: a loop of 8 tests was one gate unit, so a stop at the 7th re-ran all 8 (~25 min).
    Each test of a literal loop is its own unit: after a failure of the 3rd, the re-run runs only the 3rd; the metrics
    of all three are merged as before."""
    led, root = _looped(sandbox)
    assert [u for u, _ in gate_units(led.task("A.1"))] == ["A.1#alpha", "A.1#beta", "A.1#gamma"]
    (root / "fail_gamma").write_text("")
    assert run_gate(sandbox.spec, led, "A.1").verdict == "FAIL" and _ran(root) == ["alpha", "beta", "gamma"]
    (root / "fail_gamma").unlink()
    res = run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "PASS" and res.reused == ["A.1#alpha", "A.1#beta"]
    assert _ran(root) == ["alpha", "beta", "gamma", "gamma"]
    assert led.state()["A.1"]["metrics"] == {"pcc_alpha": 0.995, "pcc_beta": 0.995, "pcc_gamma": 0.995}
    log = res.log.read_text()
    assert "A.1#alpha: reused" in log and "tests/test_c_gamma.py" in log


def test_a_waiver_reruns_only_the_looped_test_of_its_step(sandbox):
    """F166 with F162: across gates (another task's gate reusing the stamps) a waiver for one step changes only that
    step's looped component test; F169: within the task's own gate a waiver re-runs nothing that passed."""
    led, root = _looped(sandbox)
    assert run_gate(sandbox.spec, led, "A.1").verdict == "PASS" and len(_ran(root)) == 3
    sandbox.write_spec(tests={"waivers": {"beta": "a format limit (owner)"}})
    led.reset(["A.1"])
    res = run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "PASS" and len(res.reused) == 3 and len(_ran(root)) == 3  # F169: kept in its own gate
    stamps.Delta(sandbox.spec.run_dir("default"), "A.1").drop(["A.1#alpha", "A.1#beta", "A.1#gamma"])
    led.reset(["A.1"])
    res = run_gate(sandbox.spec, led, "A.1")  # no own record: F162's stamps decide
    assert res.verdict == "PASS" and res.reused == ["A.1#alpha", "A.1#gamma"] and _ran(root)[3:] == ["beta"]


def test_only_a_literal_loop_or_a_plain_file_list_is_split():
    """F166: the split keeps what it cannot read exactly as one unit; a task without such segments is unchanged."""
    from models.demos.common.bringup.core.gate import gate_command, split_segment

    r = "scripts/run_safe_pytest.sh"
    assert split_segment(f"{r} --run-all a/test_x.py b/test_y.py") == [
        ("test_x", f"{r} --run-all a/test_x.py"),
        ("test_y", f"{r} --run-all b/test_y.py"),
    ]
    assert len(split_segment(f"{r} --run-all -k fast a/test_x.py b/test_y.py")) == 1  # a flag with a value
    assert len(split_segment(f"{r} -p plug a/test_x.py b/test_y.py")) == 1
    assert len(split_segment("for t in a b; do echo $t && false; done")) == 1  # not one command
    assert len(split_segment("for t in a b; do echo fixed; done")) == 1  # no loop variable in the body
    assert split_segment(f"X=1 {r} --run-all t.py") == [("", f"X=1 {r} --run-all t.py")]
    t = task("A.1", f"{r} --run-all t.py && {r} --run-all u.py")
    # F170: two test segments are two units (their command, unrolled, is unchanged)
    assert [u for u, _ in gate_units(t)] == ["A.1#t", "A.1#u"] and gate_command(t) == t["gate"]["cmd"]
    one = task("A.1", f"{r} --run-all t.py")
    assert gate_units(one) == [("A.1", one["gate"]["cmd"])]  # one segment: one unit named as its task
    loop = task("A.1", f"for t in p q; do {r} --run-all test_c_${{t}}.py || exit 1; done && {r} --run-all l.py")
    assert [u for u, _ in gate_units(loop)] == ["A.1#p", "A.1#q", "A.1#l"]
    assert gate_command(loop) == f"{r} --run-all test_c_p.py && {r} --run-all test_c_q.py && {r} --run-all l.py"


def test_a_docstring_or_setting_text_edit_keeps_the_code_under_test():
    """F173: DeepSeek-V4.1 P.22 re-ran 18 accuracy units because a Setting's explanation text changed in tt/settings.py.
    Comments, formatting, docstrings and a Setting's `why` do not change the semantic hash; code and defaults do."""
    from models.demos.common.bringup.core.stamps import semantic_py_hash as h

    base = 'def f(x):\n    """doc"""\n    return x + 1  # c\nS = Setting("fused", ("fused", "ops"), "old text")\n'
    assert h(base) == h(base.replace('"""doc"""', '"""other doc"""').replace("# c", "# d"))
    assert h(base) == h(base.replace('"old text"', '"new, much longer text"'))
    assert h(base) == h(base.replace("return x + 1", "return (x\n            + 1)"))
    assert h(base) != h(base.replace("x + 1", "x + 2"))
    assert h(base) != h(base.replace('Setting("fused"', 'Setting("ops"'))
    assert h("def (") != h("def ( ")  # unparsable: hashed as text
