# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HARD STRESS: the engine must run the matmul sweep OUT OF PROCESS, never in-process.

Root cause (reproduced on device): the sweep opens the 1x8 fabric mesh to benchmark matmuls.
close_mesh_device does NOT release the UMD device cluster -- it is held until the owning PROCESS
exits. So if the engine opens the mesh in-process for the sweep, the op-sig probe CHILD the engine
spawns next deadlocks at open_mesh_device on chips the parent still holds (isolation: parent-opens-
in-process then child-opens hangs 150s; two separate subprocess opens are clean). Running the sweep
as a subprocess makes it release the mesh on exit, so the probe + every round open cleanly.

  s1  OUT OF PROCESS: _invoke_matmul_sweep spawns matmul_sweep.py's CLI, not an in-process call
  s2  CLI SHAPE: node + --out/--pcc/--iters/--max-shapes are all forwarded; --case only when given
  s3  ENV: the child gets TT_METAL_HOME=repo and PYTHONPATH with perf_automation + repo
  s4  SUMMARY: the caller reads the summary back from --out (what the subprocess writes)
  s5  ISOLATION: a subprocess that errors/leaves no file does not crash the engine path
  s6  WIRING: _matmul_sweep_after_discovery (opt-in) routes through the subprocess and prints counts
  s7  NO REGRESSION: _invoke_matmul_sweep calls subprocess.run and no longer calls run_prepass in-proc
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"
sys.path.insert(0, str(_PA))


def _run():
    spec = importlib.util.spec_from_file_location("cc_run_subproc_stress", str(_CC / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _wire(monkeypatch, write_summary=None, boom=None):
    m = _run()
    calls = {}

    def fake_run(cmd, cwd=None, env=None, timeout=None, **kw):
        calls["cmd"] = list(cmd)
        calls["cwd"] = cwd
        calls["env"] = dict(env or {})
        calls["timeout"] = timeout
        if boom is not None:
            raise boom
        if write_summary is not None:
            oi = cmd.index("--out")
            Path(cmd[oi + 1]).write_text(json.dumps(write_summary))

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    return m, calls


def _invoke(m, tmp, case=None, pcc=0.99, iters=5, max_shapes=0):
    return m._invoke_matmul_sweep(
        node="models/demos/x/tests/e2e/test_main_perf.py::test_main_perf",
        case=case,
        out_path=str(tmp / "matmul_sweep.json"),
        pcc_threshold=pcc,
        iters=iters,
        max_shapes=max_shapes,
        repo_root=tmp,
    )


# --------------------------------------------------------------------------- s1
def test_s1_runs_out_of_process(monkeypatch, tmp_path):
    m, calls = _wire(monkeypatch, write_summary={"shapes": 1, "seeded": 1})
    _invoke(m, tmp_path)
    assert calls, "no subprocess spawned -- the sweep is still running in-process (will wedge the probe)"
    assert calls["cmd"][0] == sys.executable, calls["cmd"]
    assert calls["cmd"][1].endswith("matmul_sweep.py"), calls["cmd"]
    assert "test_main_perf" in calls["cmd"][2], calls["cmd"]


# --------------------------------------------------------------------------- s2
def test_s2_cli_forwards_all_knobs(monkeypatch, tmp_path):
    m, calls = _wire(monkeypatch, write_summary={"shapes": 1})
    _invoke(m, tmp_path, pcc=0.97, iters=9, max_shapes=12)
    cmd = calls["cmd"]
    for flag, val in (("--out", None), ("--pcc", "0.97"), ("--iters", "9"), ("--max-shapes", "12")):
        assert flag in cmd, f"{flag} not forwarded: {cmd}"
        if val is not None:
            assert cmd[cmd.index(flag) + 1] == val, f"{flag} value wrong: {cmd}"
    assert "--case" not in cmd, "no case was given, --case must be absent"


@pytest.mark.parametrize("case", ["test_main_perf", "device_params0"])
def test_s2_case_forwarded_when_given(monkeypatch, tmp_path, case):
    m, calls = _wire(monkeypatch, write_summary={"shapes": 1})
    _invoke(m, tmp_path, case=case)
    cmd = calls["cmd"]
    assert "--case" in cmd and cmd[cmd.index("--case") + 1] == case, cmd


def test_s2_repo_root_forwarded(monkeypatch, tmp_path):
    """The CLI must receive --repo-root: matmul_sweep.py's main() would otherwise fall back to a
    derived root, and the perf-test node is a RELATIVE path -- a wrong root means the op-sig probe
    runs from the wrong dir, collects no test, and enumerates ZERO matmuls."""
    m, calls = _wire(monkeypatch, write_summary={"shapes": 1})
    _invoke(m, tmp_path)
    cmd = calls["cmd"]
    assert "--repo-root" in cmd and cmd[cmd.index("--repo-root") + 1] == str(tmp_path), cmd


# --------------------------------------------------------------------------- s3
def test_s3_child_env(monkeypatch, tmp_path):
    m, calls = _wire(monkeypatch, write_summary={"shapes": 1})
    _invoke(m, tmp_path)
    env = calls["env"]
    assert env.get("TT_METAL_HOME") == str(tmp_path), env.get("TT_METAL_HOME")
    pp = env.get("PYTHONPATH", "")
    assert "perf_automation" in pp, f"perf_automation not on PYTHONPATH: {pp}"
    assert str(tmp_path) in pp, f"repo root not on PYTHONPATH: {pp}"
    assert calls["cwd"] == str(tmp_path), calls["cwd"]


# --------------------------------------------------------------------------- s4
def test_s4_summary_read_back_from_out(monkeypatch, tmp_path):
    summary = {"ok": True, "shapes": 76, "seeded": 76, "improved": 54}
    m, calls = _wire(monkeypatch, write_summary=summary)
    got = _invoke(m, tmp_path)
    assert got == summary, got


# --------------------------------------------------------------------------- s5
def test_s5_no_file_returns_none_not_crash(monkeypatch, tmp_path):
    m, calls = _wire(monkeypatch, write_summary=None)  # subprocess writes nothing
    got = _invoke(m, tmp_path)
    assert got is None, got


def test_s5_subprocess_raise_propagates_to_caller_guard(monkeypatch, tmp_path):
    m, calls = _wire(monkeypatch, boom=RuntimeError("spawn failed"))
    with pytest.raises(RuntimeError):  # allow-pytest.raises: no expect_error fixture
        _invoke(m, tmp_path)


# --------------------------------------------------------------------------- s6
def test_s6_after_discovery_routes_through_subprocess(monkeypatch, tmp_path):
    m, calls = _wire(monkeypatch, write_summary={"shapes": 76, "seeded": 76})
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    pipe = {"task": "main", "perf_test": "models/demos/x/tests/e2e/test_main_perf.py", "case": "test_main_perf"}
    m._matmul_sweep_after_discovery(tmp_path, tmp_path, [pipe], "all")
    assert calls, "after_discovery did not spawn the sweep subprocess"
    assert calls["cmd"][1].endswith("matmul_sweep.py") and "--case" in calls["cmd"], calls["cmd"]


def test_s6_after_discovery_optout_spawns_nothing(monkeypatch, tmp_path):
    m, calls = _wire(monkeypatch, write_summary={"shapes": 1})
    monkeypatch.delenv("PERF_MCP_MATMUL_SWEEP", raising=False)
    pipe = {"task": "main", "perf_test": "p.py", "case": None}
    m._matmul_sweep_after_discovery(tmp_path, tmp_path, [pipe], "all")
    assert not calls, "sweep spawned despite opt-out"


# --------------------------------------------------------------------------- s7
def test_s7_source_spawns_and_no_inproc_run_prepass():
    src = (_CC / "run.py").read_text()
    i = src.index("def _invoke_matmul_sweep(")
    j = src.index("\ndef ", i + 1)
    body = src[i:j]
    assert "subprocess.run(" in body, "the sweep invocation must spawn a subprocess"
    assert (
        "run_prepass(" not in body
    ), "run_prepass must NOT be called in-process -- that holds the mesh and wedges the probe"
    assert "matmul_sweep.py" in body, "must invoke the matmul_sweep CLI"
