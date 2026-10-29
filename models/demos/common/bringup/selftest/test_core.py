# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F1: spec, ledger, gate runner, metrics. CPU only."""

import json
import os
import subprocess
import sys

import pytest

from models.demos.common.bringup.core import freeze
from models.demos.common.bringup.core.gate import check_metrics, device_policy_errors, run_gate
from models.demos.common.bringup.core.ledger import Ledger, LedgerError
from models.demos.common.bringup.core.spec import Spec, parse_layers
from models.demos.common.bringup.selftest.conftest import PY, record_cmd


def task(tid, cmd, deps=(), **kw):
    t = {"id": tid, "title": f"task {tid}", "deps": list(deps), "gate": {"cmd": cmd, "metrics": kw.pop("metrics", {})}}
    t.update(kw)
    return t


# ---------------------------------------------------------------- gate verdicts
def test_pass_commits_only_declared_paths(sandbox):
    (sandbox.repo / "src").mkdir()
    (sandbox.repo / "src/impl.py").write_text("x = 1\n")
    (sandbox.repo / "unrelated.py").write_text("other agent's work\n")
    led = sandbox.tasks(task("A.1", record_cmd(pcc_out=0.995), metrics={"pcc_*": ">= 0.99"}, paths=["src"]))
    res = run_gate(sandbox.spec, led, "A.1", commit=True)
    assert res.verdict == "PASS", res.summary()
    assert res.commit
    assert sandbox.git("log", "-1", "--format=%s") == "[toy][A.1] task A.1"
    committed = set(sandbox.git("show", "--name-only", "--format=", "HEAD").split())
    assert committed == {
        "bringup/state.json",
        "bringup/results/A.1.json",
        "bringup/tasks.yaml",
        "bringup/.gitignore",
        "src/impl.py",
    }
    assert "unrelated.py" in sandbox.git("status", "--porcelain")
    st = led.state()["A.1"]
    assert st["status"] == "PASS" and st["metrics"] == {"pcc_out": 0.995} and st["attempts"] == 0
    assert sandbox.git("status", "--porcelain", "bringup") == ""  # nothing written after the commit


def test_threshold_miss_fails_and_counts_attempts(sandbox):
    led = sandbox.tasks(task("A.1", record_cmd(pcc_out=0.95), metrics={"pcc_*": ">= 0.99"}))
    for n in (1, 2):
        res = run_gate(sandbox.spec, led, "A.1", commit=True)
        assert res.verdict == "FAIL" and res.commit is None
        assert led.state()["A.1"]["attempts"] == n
    assert any("FAIL     pcc_out = 0.95" in line for line in res.lines)
    assert sandbox.git("rev-list", "--count", "HEAD") == "1"


def test_missing_metric_and_missing_artifact_fail(sandbox):
    led = sandbox.tasks(
        task("A.1", record_cmd(other=1), metrics={"pcc_*": ">= 0.99"}),
        task("A.2", record_cmd(pcc=1.0), metrics={"pcc": ">= 0.99"}, artifacts=["out/model.bin"]),
    )
    r1 = run_gate(sandbox.spec, led, "A.1")
    r2 = run_gate(sandbox.spec, led, "A.2")
    assert r1.verdict == "FAIL" and any("MISSING  pcc_*" in x for x in r1.lines)
    assert r2.verdict == "FAIL" and any("MISSING artifact out/model.bin" in x for x in r2.lines)


def test_nonzero_exit_fails_even_with_good_metrics(sandbox):
    led = sandbox.tasks(task("A.1", record_cmd(pcc=1.0) + " && exit 3", metrics={"pcc": ">= 0.99"}))
    assert run_gate(sandbox.spec, led, "A.1").verdict == "FAIL"


def test_blocked_until_deps_pass(sandbox):
    led = sandbox.tasks(task("A.1", "exit 1"), task("A.2", "true", deps=["A.1"]))
    assert run_gate(sandbox.spec, led, "A.2").verdict == "BLOCKED"
    assert led.status("A.2") == "TODO"
    assert run_gate(sandbox.spec, led, "A.2", force=True).verdict == "PASS"


def test_frozen_file_change_fails_before_running(sandbox):
    t = sandbox.repo / "tests/test_x.py"
    t.parent.mkdir()
    t.write_text("assert True\n")
    marker = sandbox.repo / "ran"
    led = sandbox.tasks(task("A.1", f"touch {marker}", frozen={"files": freeze.hash_paths(sandbox.repo, ["tests"])}))
    assert run_gate(sandbox.spec, led, "A.1").verdict == "PASS"
    marker.unlink()
    t.write_text("assert True  # relaxed\n")
    res = run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "FAIL" and "frozen file changed: tests/test_x.py" in res.summary()
    assert not marker.exists()


def test_metrics_are_pinned_to_the_ledger_results_dir(sandbox):
    led = sandbox.tasks(task("A.1", record_cmd(v=2), metrics={"v": "== 2"}))
    run_gate(sandbox.spec, led, "A.1")
    data = json.loads((sandbox.repo / "bringup/results/A.1.json").read_text())
    assert data["task"] == "A.1" and data["metrics"]["v"]["value"] == 2


def test_gate_pins_pythonpath(sandbox, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/nonexistent/other-checkout")
    cmd = f"{PY} -c \"import os, models.demos.common.bringup.core.spec as s; assert os.environ['PYTHONPATH'] == str(s.CODE_ROOT)\""
    led = sandbox.tasks(task("A.1", cmd))
    assert run_gate(sandbox.spec, led, "A.1").verdict == "PASS"


# ---------------------------------------------------------------- device policy and hangs
@pytest.mark.parametrize(
    "cmd,device,bad",
    [
        ("scripts/run_safe_pytest.sh --run-all tests/test_x.py", True, False),
        ("scripts/tt-probe.sh probe < p.py", True, False),
        ("python tests/test_x.py", True, True),
        ("pytest tests/test_x.py", False, True),
        ("python -m pytest tests/test_x.py", False, True),
        ("TT_METAL_X=1 scripts/run_safe_pytest.sh t.py", True, False),
        ("scripts/run_safe_pytest.sh t.py && python post.py", True, True),
        ("python check_plan.py", False, False),
        ("scripts/run_safe_pytest.sh t.py && python -m models.demos.common.bringup.knowledge.check", True, False),
        ("scripts/run_safe_pytest.sh t.py && python -m models.demos.common.bringup.testing.profiler", True, True),
    ],
)
def test_device_policy(cmd, device, bad):
    errs = device_policy_errors({"gate": {"cmd": cmd}, "device": device})
    assert bool(errs) == bad, errs


def test_hang_is_its_own_verdict_and_keeps_the_triage(sandbox):
    runner = sandbox.repo / "scripts/run_safe_pytest.sh"
    runner.parent.mkdir()
    runner.write_text(
        "#!/bin/bash\nmkdir -p generated/tt-triage\necho 'cb_wait_front' > generated/tt-triage/triage.txt\nexit 2\n"
    )
    runner.chmod(0o755)
    led = sandbox.tasks(task("A.1", "scripts/run_safe_pytest.sh tests/test_x.py", device=True))
    res = run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "HANG"
    assert res.log.with_suffix(".triage.txt").read_text().strip() == "cb_wait_front"
    assert led.status("A.1") == "HANG"


# ---------------------------------------------------------------- ledger
def test_topo_order_downstream_and_runnable(sandbox):
    led = sandbox.tasks(
        task("B", "true", deps=["A"]), task("A", "true"), task("C", "true", deps=["B"]), task("D", "true", deps=["A"])
    )
    assert led.topo_order() == ["A", "B", "C", "D"]
    assert led.downstream("B") == ["B", "C"]
    assert led.runnable() == ["A"]
    led.update("A", status="PASS")
    assert led.runnable() == ["B", "D"]
    led.reset(["A"])
    assert led.status("A") == "TODO" and led.state()["A"]["history"][-1]["status"] == "RESET"


def test_ledger_validation(sandbox):
    led = sandbox.tasks(task("A", "true", deps=["B"]), task("B", "true", deps=["A"], metrics={"x": "~ 1"}))
    errs = led.validate()
    assert any("cycle" in e for e in errs) and any("bad threshold" in e for e in errs)
    led = sandbox.tasks(task("A", "true"), task("A", "true"))
    with pytest.raises(LedgerError):
        led.tasks()


def test_concurrent_state_updates_do_not_lose_verdicts(sandbox):
    led = sandbox.tasks(*[task(f"T{i}", "true") for i in range(16)])
    code = (
        "import sys; from models.demos.common.bringup.core.ledger import Ledger; "
        f"L = Ledger({str(led.dir)!r}); [L.update(sys.argv[1], status='PASS', n=k) for k in range(20)]"
    )
    procs = [subprocess.Popen([sys.executable, "-c", code, f"T{i}"]) for i in range(16)]
    assert all(p.wait() == 0 for p in procs)
    st = Ledger(led.dir).state()
    assert all(st[f"T{i}"]["status"] == "PASS" for i in range(16))


def test_check_metrics_non_numeric_is_a_failure():
    ok, lines = check_metrics({"x": ">= 1"}, {"x": {"value": "nan-ish"}})
    assert not ok


# ---------------------------------------------------------------- spec
def test_parse_layers():
    assert parse_layers("0,2-4") == [0, 2, 3, 4]
    assert parse_layers([5, "1-2"]) == [1, 2, 5]
    assert parse_layers("all", 3) == [0, 1, 2]


def model_spec(**over):
    d = {
        "model": "m",
        "hf_id": "org/m",
        "model_dir": "models/demos/m",
        "hooks": "models.demos.m.bringup.hooks",
        "num_layers": 4,
        "box": {"mesh": [1, 4]},
        "target": {"seq": 8192, "chunk": 4096},
        "ladder": [
            {"name": "s8192", "seq": 8192, "chunk": 4096},
            {"name": "last", "seq": 8192, "chunk": 4096, "golden": "s8192"},
        ],
        "block_types": {"dense": {"layers": [0]}, "moe": {"layers": "1-3"}},
        "state": {"kind": "kv", "tensors": ["key", "value"]},
        "paths": {"art": "/tmp/art"},
    }
    d.update(over)
    return Spec(d)


def test_spec_valid_and_paths():
    s = model_spec()
    assert s.validate() == []
    assert s.block_type_of(2) == "moe" and s.representative_layer("moe") == 1
    assert str(s.tt_cache()) == "/tmp/art/m/tt_cache/1x4/v1"
    assert str(s.golden_root) == "/tmp/art/m/golden"


def test_spec_subset_moves_the_representative_layer():
    s = model_spec(layers=[0, 3])
    assert s.representative_layer("moe") == 3


@pytest.mark.parametrize(
    "over,needle",
    [
        ({"block_types": {"dense": {"layers": [0, 1]}}}, "cover every layer"),
        ({"ladder": [{"name": "a", "seq": 100, "chunk": 30}]}, "not a multiple"),
        ({"ladder": [{"name": "a", "seq": 64, "chunk": 32, "golden": "b"}]}, "does not exist"),
        (
            {"ladder": [{"name": "a", "seq": 64, "chunk": 32, "golden": "b"}, {"name": "b", "seq": 128, "chunk": 32}]},
            "differ",
        ),
        ({"layers": [7]}, "out of range"),
        ({"state": {"kind": "kv"}}, "state.tensors"),
        ({"box": {"mesh": [4]}}, "box.mesh"),
        ({"checkpoint": {"config": {"sliding_window": 1024}}}, "must exceed the sliding window"),
    ],
)
def test_spec_validation_errors(over, needle):
    errs = model_spec(**over).validate()
    assert any(needle in e for e in errs), errs


def test_cli_status_and_validate(sandbox):
    sandbox.tasks(task("A.1", "true"))
    out = subprocess.run(
        [sys.executable, "-m", "models.demos.common.bringup", "status", "--spec", str(sandbox.spec_path)],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0 and "A.1" in out.stdout and "TODO" in out.stdout
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.common.bringup",
            "validate",
            "--ledger-only",
            "--spec",
            str(sandbox.spec_path),
        ],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0 and "valid" in out.stdout


def test_cpu_threads_are_physical_cores(monkeypatch):
    """F40: CPU gates use one torch thread per physical core (F171: when no device work shares the box)."""
    import psutil

    from models.demos.common.bringup.core import metrics as Mx

    monkeypatch.setattr(Mx, "device_work_running", lambda: False)
    monkeypatch.delenv("BRINGUP_CPU_THREADS", raising=False)
    assert Mx.cpu_threads() == (psutil.cpu_count(logical=False) or os.cpu_count())


def test_state_keeps_only_gated_metrics_when_a_test_records_many(sandbox):
    """F52: a swap test records hundreds of informational metrics; state.json keeps the gated ones, results all."""
    many = {f"info_{i:03d}": 1.0 for i in range(70)}
    led = sandbox.tasks(task("A.1", record_cmd(pcc_out=0.995, **many), metrics={"pcc_*": ">= 0.99"}))
    res = run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "PASS", res.summary()
    assert led.state()["A.1"]["metrics"] == {"pcc_out": 0.995}
    full = json.loads((led.results_dir / "A.1.json").read_text())["metrics"]
    assert len(full) == 71 and full["info_069"]["value"] == 1.0


def test_state_keeps_the_gated_metrics_and_results_keep_all(sandbox):
    """F129: state.json carries the gated metrics only (the 500 KB hook); results/<task>.json has every one."""
    led = sandbox.tasks(task("A.1", record_cmd(pcc_out=0.995, info=2.0), metrics={"pcc_*": ">= 0.99"}))
    run_gate(sandbox.spec, led, "A.1")
    assert led.state()["A.1"]["metrics"] == {"pcc_out": 0.995}


def refusing_hook(sandbox, text: str):
    """A pre-commit hook that refuses while bringup/REFUSE exists (the overseer's framework fix removes it)."""
    hook = sandbox.repo / ".git" / "hooks" / "pre-commit"
    hook.write_text(f"#!/bin/sh\nif [ -e bringup/REFUSE ]; then echo '{text}' >&2; exit 1; fi\nexit 0\n")
    hook.chmod(0o755)
    (sandbox.repo / "bringup").mkdir(exist_ok=True)
    (sandbox.repo / "bringup/REFUSE").write_text("")


def counted(cmd: str, sandbox) -> str:
    """The gate command, appending a line to runs.txt every time it runs."""
    return f"echo run >> {sandbox.repo / 'runs.txt'} && {cmd}"


def test_a_refused_commit_is_pending_and_commits_without_a_rerun(sandbox):
    """F149 (was F110): a gate whose checks pass but whose commit git refuses is COMMIT_PENDING with the hook's output,
    never PASS with the change uncommitted (DeepSeek-V4.1 P.5) and never a FAIL that re-runs the gate (FIX.6 / FIX.5:
    35 and 70 min of passed checks re-run for a commit). Its outputs stay; commit-only commits them, running nothing."""
    from models.demos.common.bringup.core.gate import commit_only

    refusing_hook(sandbox, "prefer-expect-error: use the expect_error fixture")
    (sandbox.repo / "src").mkdir()
    (sandbox.repo / "src/impl.py").write_text("x = 1\n")
    cmd = (
        counted(record_cmd(pcc_out=0.995), sandbox)
        + f" && {PY} -c \"open('bringup/results/A.1_profile.json', 'w').write('{{}}')\""
    )
    led = sandbox.tasks(task("A.1", cmd, metrics={"pcc_*": ">= 0.99"}, paths=["src"]), task("A.2", "true", ["A.1"]))
    res = run_gate(sandbox.spec, led, "A.1", commit=True)
    assert res.verdict == "COMMIT_PENDING" and not res.commit and res.exit_code == 3
    assert any("prefer-expect-error" in ln for ln in res.lines)
    st = led.state()["A.1"]
    assert st["status"] == "COMMIT_PENDING" and st["metrics"] == {"pcc_out": 0.995}
    assert any("prefer-expect-error" in ln for ln in st["commit_pending"]["hook"])
    assert led.runnable() == ["A.1"]  # not done: its dependent waits for the commit

    # still refused: a gate run on the pending task retries the commit only; nothing runs, the outputs stay
    res = run_gate(sandbox.spec, led, "A.1", commit=True)
    assert res.verdict == "COMMIT_PENDING" and (sandbox.repo / "runs.txt").read_text() == "run\n"
    assert (sandbox.repo / "bringup/results/A.1_profile.json").exists()
    assert (sandbox.repo / "bringup/results/A.1.json").exists()

    (sandbox.repo / "bringup/REFUSE").unlink()  # the overseer's fix
    res = commit_only(sandbox.spec, led, "A.1")
    assert res.verdict == "PASS" and res.commit, res.summary()
    assert (sandbox.repo / "runs.txt").read_text() == "run\n"
    committed = set(sandbox.git("show", "--name-only", "--format=", "HEAD").split())
    assert {"src/impl.py", "bringup/results/A.1.json", "bringup/results/A.1_profile.json"} <= committed
    st = led.state()["A.1"]
    assert st["status"] == "PASS" and not st.get("commit_pending")
    assert sandbox.git("status", "--porcelain", "bringup", "src") == ""


def test_a_gate_commit_leaves_the_generated_dashboard_pages_out(sandbox):
    """F210: the dashboard pages are re-exported after every gate and published as artifacts; a gate commit never
    takes them (DeepSeek-V4.1 FIX.8: the repo's 500 KB hook refused a passed gate over two 0.9 MB pages)."""
    (sandbox.repo / "src").mkdir()
    (sandbox.repo / "src/impl.py").write_text("x = 1\n")
    dash = sandbox.repo / "bringup/dashboard"
    dash.mkdir(parents=True, exist_ok=True)
    (dash / "index.html").write_text("<p>dashboard</p>\n")
    led = sandbox.tasks(task("A.1", record_cmd(pcc_out=0.995), metrics={"pcc_*": ">= 0.99"}, paths=["src", "bringup"]))
    res = run_gate(sandbox.spec, led, "A.1", commit=True)
    assert res.verdict == "PASS", res.summary()
    committed = set(sandbox.git("show", "--name-only", "--format=", "HEAD").split())
    assert "src/impl.py" in committed and "bringup/dashboard/index.html" not in committed
    assert "bringup/dashboard/" in sandbox.git("status", "--porcelain", "--untracked-files=all")


def test_commit_only_names_a_file_too_large_and_stays_pending(sandbox):
    """F149: a refusal by the repo's large-file hook names the file; the task stays pending for the overseer's fix."""
    from models.demos.common.bringup.core.gate import commit_only

    refusing_hook(sandbox, "bringup/results/A.1_profile.json (613 KB) exceeds the 500 KB limit.")
    led = sandbox.tasks(task("A.1", record_cmd(pcc_out=0.995), metrics={"pcc_*": ">= 0.99"}))
    res = run_gate(sandbox.spec, led, "A.1", commit=True)
    assert res.verdict == "COMMIT_PENDING"
    assert led.state()["A.1"]["commit_pending"]["too_large"] == ["bringup/results/A.1_profile.json"]
    res = commit_only(sandbox.spec, led, "A.1")
    assert res.verdict == "COMMIT_PENDING" and any("too large for the repo" in ln for ln in res.lines)


def test_commit_only_ignores_the_records_the_framework_writes_under_the_tasks_paths(sandbox):
    """F211: a task whose paths hold its ledger (DeepSeek-V4.1 FIX.8: the whole model dir) is still committed by
    commit-only after the refusal wrote state.json: the framework's own records are not the task's files."""
    from models.demos.common.bringup.core.gate import commit_only

    refusing_hook(sandbox, "some hook refuses")
    (sandbox.repo / "src").mkdir()
    (sandbox.repo / "src/impl.py").write_text("x = 1\n")
    led = sandbox.tasks(task("A.1", record_cmd(pcc_out=0.995), metrics={"pcc_*": ">= 0.99"}, paths=["src", "bringup"]))
    res = run_gate(sandbox.spec, led, "A.1", commit=True)
    assert res.verdict == "COMMIT_PENDING"
    (sandbox.repo / ".git/hooks/pre-commit").write_text("#!/bin/sh\nexit 0\n")  # the overseer's fix (outside the paths)
    led.update("A.1", reason=["a later state.json write"])  # the framework writes its records after the refusal
    res = commit_only(sandbox.spec, led, "A.1")
    assert res.verdict == "PASS" and res.commit, res.lines


def test_commit_only_refuses_when_the_tasks_files_changed(sandbox):
    """F149: commit-only commits what the gate tested; a task file changed since then needs the gate."""
    from models.demos.common.bringup.core.gate import commit_only

    refusing_hook(sandbox, "refused")
    (sandbox.repo / "src").mkdir()
    (sandbox.repo / "src/impl.py").write_text("x = 1\n")
    led = sandbox.tasks(task("A.1", record_cmd(pcc_out=0.995), metrics={"pcc_*": ">= 0.99"}, paths=["src"]))
    assert run_gate(sandbox.spec, led, "A.1", commit=True).verdict == "COMMIT_PENDING"
    (sandbox.repo / "bringup/REFUSE").unlink()
    (sandbox.repo / "src/impl.py").write_text("x = 2\n")
    res = commit_only(sandbox.spec, led, "A.1")
    assert res.verdict == "COMMIT_PENDING" and any("changed since its gate passed" in ln for ln in res.lines)
    led.reset(["A.1"])
    assert led.status("A.1") == "TODO" and "commit_pending" not in led.state()["A.1"]


def test_commit_only_takes_a_fix_to_the_files_the_hook_named(sandbox):
    """F154: the hook names a file (DeepSeek-V4.1 P.18: prefer-expect-error on a fork unit test); fixing it is the
    change it asks for and commits without a re-run. A change to any other task file still needs the gate."""
    from models.demos.common.bringup.core.gate import commit_only

    refusing_hook(sandbox, "src/test_x.py:3: use the expect_error fixture")
    (sandbox.repo / "src").mkdir()
    (sandbox.repo / "src/impl.py").write_text("x = 1\n")
    (sandbox.repo / "src/test_x.py").write_text("raises\n")
    led = sandbox.tasks(
        task("A.1", counted(record_cmd(pcc_out=0.995), sandbox), metrics={"pcc_*": ">= 0.99"}, paths=["src"])
    )
    assert run_gate(sandbox.spec, led, "A.1", commit=True).verdict == "COMMIT_PENDING"
    (sandbox.repo / "bringup/REFUSE").unlink()
    (sandbox.repo / "src/test_x.py").write_text("expect_error\n")
    (sandbox.repo / "src/impl.py").write_text("x = 2\n")
    res = commit_only(sandbox.spec, led, "A.1")
    assert res.verdict == "COMMIT_PENDING" and any("src/impl.py (not named by the hook)" in ln for ln in res.lines)
    (sandbox.repo / "src/impl.py").write_text("x = 1\n")
    res = commit_only(sandbox.spec, led, "A.1")
    assert res.verdict == "PASS" and res.commit, res.summary()
    assert any("as the hook asked (F154): src/test_x.py" in ln for ln in res.lines)
    assert (sandbox.repo / "runs.txt").read_text() == "run\n"
    assert "src/test_x.py" in sandbox.git("show", "--name-only", "--format=", "HEAD").split()


def test_cpu_threads_env_override(monkeypatch):
    """F165: BRINGUP_CPU_THREADS caps the torch threads of a CPU job that shares the box with a live bring-up
    (DeepSeek-V4.1 full-model goldens on 8 of 16 cores while the 2x2 run's swap tests ran their CPU references)."""
    from models.demos.common.bringup.core.metrics import cpu_threads

    monkeypatch.setenv("BRINGUP_CPU_THREADS", "8")
    assert cpu_threads() == 8
    monkeypatch.delenv("BRINGUP_CPU_THREADS")
    assert cpu_threads() >= 1


def test_attempt_and_ab_histories_live_beside_state_json(sandbox):
    """F180: DeepSeek-V4.1 Z.1's commit was refused: state.json reached 504 KB (agent_runs 137 KB, ab 85 KB). Those
    histories now live in history_state.json (uncommitted); state() merges them back for every reader."""
    led = sandbox.tasks(task("A.1", "true"))
    led.update("A.1", agent_runs=[{"attempt": 1, "role": "fix"}], ab={"decision": "accept"}, status="PASS")
    raw = json.loads(led.state_path.read_text())["A.1"]
    assert "agent_runs" not in raw and "ab" not in raw and raw["status"] == "PASS"
    st = led.state()["A.1"]
    assert st["agent_runs"] == [{"attempt": 1, "role": "fix"}] and st["ab"] == {"decision": "accept"}
    led.update("A.1", agent_runs=st["agent_runs"] + [{"attempt": 2, "role": "fix"}])
    assert len(led.state()["A.1"]["agent_runs"]) == 2


def test_a_spec_rule_that_contradicts_realistic_precision_is_refused():
    """F208: DeepSeek-V4.1's 2026-10-14 line 'HiFi4, fp32 accumulation, bf16 KV cache' survived F123 in agents.rules and
    the serving contract followed it (bf16 KV caches where the reference stores fp8 / fp4)."""
    from models.demos.common.bringup.core.spec import stale_precision_rules

    stale = {
        "agents": {"rules": ["Use 2D fabric.", "Precision during bring-up: HiFi4, fp32 accumulation, bf16 KV cache."]}
    }
    errs = stale_precision_rules(stale)
    assert len(errs) == 1 and errs[0].startswith("agents.rules[1]")
    ok = {
        "agents": {
            "rules": [
                "SDPA at HiFi4 for the streaming path (owner exception, measured).",
                "fp32 accumulation only where measured to matter.",
            ]
        }
    }
    assert stale_precision_rules(ok) == []
    assert stale_precision_rules({}) == []


def test_a_gate_that_needs_device_time_but_only_profiles_the_wall_is_refused_up_front(sandbox):
    """F212: device_ms_* gated while every test_profile.py segment is wall-only (BRINGUP_PROFILE_WALL=1): the gate adds
    no profiled run, the metric is MISSING and no agent attempt can pass it (DeepSeek-V4.1 P.6). The ledger and the
    gate's pre-check name it; without the flag the gate adds the wall run and the profiled run itself (F114)."""
    from models.demos.common.bringup.core.gate import with_clean_wall
    from models.demos.common.bringup.core.ledger import wall_only_device_metrics

    prof = "scripts/run_safe_pytest.sh models/demos/common/bringup/tests/test_profile.py"
    bad = {
        "step": "perf",
        "gate": {"cmd": f"true && BRINGUP_PROFILE_WALL=1 {prof}", "metrics": {"device_ms_total": "<= 230"}},
    }
    good = {"step": "perf", "gate": {"cmd": f"true && {prof}", "metrics": {"device_ms_total": "<= 230"}}}
    wall = {"step": "perf", "gate": {"cmd": f"BRINGUP_PROFILE_WALL=1 {prof}", "metrics": {"chunk_wall_ms": "< 500"}}}
    assert wall_only_device_metrics(bad) and "device_ms_total" in wall_only_device_metrics(bad)[0]
    assert wall_only_device_metrics(good) == [] and wall_only_device_metrics(wall) == []
    segs = with_clean_wall(good["gate"]["cmd"]).split("&&")
    assert any("BRINGUP_PROFILE_WALL=1" in s for s in segs) and any("BRINGUP_WALL_CLEAN=1" in s for s in segs)
    led = sandbox.tasks(task("A.1", "true", metrics={"device_ms_total": "<= 230"}))
    t = led.tasks()["A.1"]
    t["gate"]["cmd"] = f"BRINGUP_PROFILE_WALL=1 {prof}"
    led.tasks_path.write_text(__import__("yaml").safe_dump({"tasks": list(led.tasks().values())[:0] + [t]}))
    assert any("wall-only" in e for e in led.validate())
