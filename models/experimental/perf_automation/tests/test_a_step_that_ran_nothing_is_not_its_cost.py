"""A step that reused a record ran nothing on the device, so its wall time is not a duration to budget by.

An image-edit port on a 32-chip mesh, 2026-10-07: a --persist relaunch at the same HEAD reused the correctness
gate's baseline record and returned in 0.618 s, and that was logged as the gate's observed cost
("pcc": [0.618]). Every "pcc"-derived budget -- the full-pipeline run's hard backstop among them -- then
sat on its 30 s floor, and every full-pipeline check of the run (10+ minutes on that model) was
SIGKILLed at 30 s. The kill was itself hidden for the first attempts: _run_full_pipeline_ms bound a local
`_sp`, the module's own `subprocess` alias, so its timeout handler raised UnboundLocalError instead.

Device steps are stubbed; the real run.py / perf_mcp.py functions run. Nothing here touches a device.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parents[1]


@pytest.fixture
def run(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("cc_run_reused_cost_ut", str(_PA / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    recorded = []
    monkeypatch.setattr(m, "record_observed", lambda root, op, s: recorded.append((op, s)))
    monkeypatch.setattr(m, "cc_env", lambda repo_root, devices: {})
    monkeypatch.setattr(m, "adaptive_timer", lambda *a, **k: 60)
    import agent.probes as probes

    monkeypatch.setattr(probes, "adaptive_backstop", lambda *a, **k: 60)
    m._recorded = recorded
    return m


def _gate_answer(run, monkeypatch, doc, rc=0):
    line = "PCC_BASELINE=" + json.dumps(doc) if doc is not None else "no record printed"
    monkeypatch.setattr(run, "_run_device_step", lambda *a, **k: (rc, line + "\n"))


def test_a_reused_gate_record_logs_no_cost(run, monkeypatch, tmp_path):
    _gate_answer(run, monkeypatch, {"status": "ok", "recorded": True, "reused": True, "failed_tests": []})
    run._pcc_gate_baseline(tmp_path, {}, "all")
    assert run._recorded == [], "a record reused at this HEAD ran nothing; its wall time is not the gate's cost"


def test_a_gate_that_ran_is_observed_as_before(run, monkeypatch, tmp_path):
    _gate_answer(run, monkeypatch, {"status": "ok", "recorded": True, "reused": False, "failed_tests": []})
    run._pcc_gate_baseline(tmp_path, {}, "all")
    assert [op for op, _ in run._recorded] == ["pcc"]


def test_a_killed_gate_is_still_observed(run, monkeypatch, tmp_path):
    """A kill is the strongest evidence the budget was too small; it must still be logged."""
    _gate_answer(run, monkeypatch, None, rc=None)
    run._pcc_gate_baseline(tmp_path, {}, "all")
    assert [op for op, _ in run._recorded] == ["pcc"]


def test_the_device_step_no_longer_observes_the_gate_itself(run, monkeypatch, tmp_path):
    """One observer, not two: the step must not log the reused record behind the caller's back."""
    seen = {}

    def _step(*a, **k):
        seen.update(k)
        return 0, 'PCC_BASELINE={"recorded": true, "reused": true, "failed_tests": []}\n'

    monkeypatch.setattr(run, "_run_device_step", _step)
    run._pcc_gate_baseline(tmp_path, {}, "all")
    assert "observe_op" not in seen


def test_a_disabled_bookend_logs_no_cost_and_keeps_its_shape(run, monkeypatch, tmp_path):
    monkeypatch.setenv("PERF_MCP_FULLPIPE_E2E", "0")
    assert run._fullpipe_e2e(tmp_path, {}, "all", "BEFORE") == (None, ""), "callers unpack two values"
    assert run._recorded == []


def test_a_bookend_that_ran_is_observed_as_pcc(run, monkeypatch, tmp_path):
    monkeypatch.setenv("PERF_MCP_FULLPIPE_E2E", "1")
    monkeypatch.setattr(run, "_run_device_step", lambda *a, **k: (None, ""))  # killed: still a real run
    assert run._fullpipe_e2e(tmp_path, {}, "all", "MEASURE") == (None, "")
    assert [op for op, _ in run._recorded] == ["pcc"]


def test_the_measured_case_one_sub_second_observation_floors_the_backstop(tmp_path, monkeypatch):
    """The arithmetic the fix keeps out of the ledger: [0.618] alone makes the full-pipeline backstop 30 s."""
    import agent.probes as probes

    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"config": {"timeout": 10800}}))
    (run_dir / "observed_durations.json").write_text(json.dumps({"pcc": [0.618]}))
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(run_dir / "manifest.json"))
    monkeypatch.delenv("PERF_MCP_MEASURE_BACKSTOP", raising=False)
    assert probes.adaptive_backstop(3600) == 30


# --------------------------------------------------------------------------------------------------
# no function rebinds a module it also uses: the `_sp` UnboundLocalError, as a class
# --------------------------------------------------------------------------------------------------


def _module_aliases(tree) -> set:
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return names


def _shadowing(path: Path) -> list:
    """Functions that bind (by assignment, loop or `as`) a name the module imports AND use that name as a
    module (`name.attr`). Python makes the name local to the whole function, so the module use raises
    UnboundLocalError. A local re-import of the same module is not flagged: it binds the same module."""
    tree = ast.parse(path.read_text())
    mods = _module_aliases(tree)
    hits = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bound, used, declared = set(), set(), set()
        for n in ast.walk(fn):
            if isinstance(n, (ast.Global, ast.Nonlocal)):
                declared |= set(n.names)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                bound.add(n.id)
            elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and isinstance(n.value.ctx, ast.Load):
                used.add(n.value.id)
        hits += [
            "%s:%d %s() rebinds %r" % (path.name, fn.lineno, fn.name, x)
            for x in sorted((bound & used & mods) - declared)
        ]
    return hits


def test_no_tool_function_rebinds_a_module_it_imports():
    repo = _PA.parents[2]
    files = subprocess.run(
        ["git", "ls-files", "models/experimental/perf_automation/*.py", "scripts/tt_hw_planner/*.py"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    ).stdout.split()
    files = [repo / f for f in files if "/tests/" not in f]
    assert files, "the tool's sources were not found"
    assert [h for f in files for h in _shadowing(f)] == []


def test_the_scan_catches_the_original_defect(tmp_path):
    src = tmp_path / "m.py"
    src.write_text(
        "import subprocess as _sp\n"
        "def f(xs):\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as exc:\n"
        "        isinstance(exc, _sp.TimeoutExpired)\n"
        "    for x in xs:\n"
        "        _sp = x\n"
    )
    assert _shadowing(src) == ["m.py:2 f() rebinds '_sp'"]
