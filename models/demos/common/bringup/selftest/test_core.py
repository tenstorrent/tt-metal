# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F1: spec, ledger, gate runner, metrics. CPU only."""

import json
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
