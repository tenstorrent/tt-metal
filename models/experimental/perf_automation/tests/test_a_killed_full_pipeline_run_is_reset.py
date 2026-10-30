"""A device run perf_mcp kills for making no progress is reset on the way out, for every caller.

Run 1791980272 (2026-10-14, minicpm5_2b, --devices 0): check_full_pipeline_latency stalled in
fetch_queue_reserve_back, _adaptive_run SIGKILLed it at 662 s, and nothing reset the board. The only
reset on that path sat inside the trace-region grow, which stops once the region is at its ceiling
(it was), and the gate never told note_crash it had crashed. Four back-to-back 662 s attempts followed
on the same board, with the device-recovery state still reading reset_fails=0.

These pin: the kill resets through the one shared rule (probes.reset_after_ended_child) with the run's
own env and output; a run that ends by itself is not reset; the grow loop no longer resets a second
time; and the gate's crashes and readings reach the crash streak like measure_candidate's and
check_pcc's. Nothing here opens or resets a device.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent import probes as PR
from cc_optimize import perf_mcp as pm


@pytest.fixture
def resets(monkeypatch):
    seen = []
    for name in ("reset_after_ended_child", "reset_after_killed_child"):
        monkeypatch.setattr(
            PR, name, lambda error_text="", env=None, _n=name: seen.append((error_text, env, _n)) or True
        )
    # Not the thing under test, and both read the real host.
    monkeypatch.setattr(PR, "wait_for_memory_headroom_before_device_work", lambda label: None)
    monkeypatch.setattr(PR, "memory_cap_preexec_fn", lambda: None)
    return seen


def _child(code: str):
    return [sys.executable, "-c", code]


# ---- the kill ----------------------------------------------------------------------------------------


def test_a_run_killed_for_no_progress_is_reset_with_its_own_env_and_output(resets, tmp_path):
    env = dict(os.environ, PERF_TEST_MARK="child-env")
    cmd = _child("import time; print('loading weights', flush=True); time.sleep(120)")
    with pytest.raises(subprocess.TimeoutExpired):  # allow-pytest.raises: repo-root conftest bypassed
        pm._adaptive_run(cmd, str(tmp_path), env, "full-pipeline", stall_s=1, backstop=600)
    assert len(resets) == 1, "the kill must be followed by exactly one reset"
    text, child_env, how = resets[0]
    assert how == "reset_after_killed_child", "a stall kill is counted toward the repeat-hang reset"
    assert child_env is env, "the rule decides from the killed child's own env (its chip count)"
    assert "loading weights" in text, "the killed run's output is the evidence the reset is given"


def test_a_run_that_ends_by_itself_is_not_reset(resets, tmp_path):
    r = pm._adaptive_run(_child("print('done')"), str(tmp_path), dict(os.environ), "smoke", stall_s=30)
    assert r.returncode == 0 and "done" in r.stdout
    assert resets == [], "only a kill is evidence; a finished run leaves the board as it was"


def test_a_failing_reset_still_raises_the_timeout(monkeypatch, tmp_path):
    def boom(**_k):
        raise RuntimeError("tt-smi missing")

    monkeypatch.setattr(PR, "reset_after_ended_child", boom)
    monkeypatch.setattr(PR, "reset_after_killed_child", boom)
    monkeypatch.setattr(PR, "wait_for_memory_headroom_before_device_work", lambda label: None)
    monkeypatch.setattr(PR, "memory_cap_preexec_fn", lambda: None)
    with pytest.raises(subprocess.TimeoutExpired):  # allow-pytest.raises: repo-root conftest bypassed
        pm._adaptive_run(_child("import time; time.sleep(120)"), str(tmp_path), dict(os.environ), "x", 1, 2)


def test_the_reset_is_the_shared_rule_for_a_killed_child():
    """No parallel policy: a single chip still meets the veto, a fabric is reset whatever it says."""
    import inspect

    src = inspect.getsource(PR.reset_after_ended_child)
    assert "reset_is_mandatory_after_kill" in src


# ---- the grow loop does not reset a second time ------------------------------------------------------


@pytest.fixture
def stalled_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(tmp_path / "m"))
    monkeypatch.setattr(pm, "_FULLPIPE_SAMPLES", 1, raising=False)
    monkeypatch.setattr(pm, "_MODEL_ROOT", tmp_path / "m", raising=False)
    monkeypatch.setattr(pm, "_MANIFEST", {"perf_test_resolved": {"path": "t.py", "case": None}}, raising=False)
    monkeypatch.setattr(pm, "_recover_if_board_is_dead", lambda *a, **k: False)
    calls = SimpleNamespace(regions=[], direct_resets=0)

    def stalled(cmd, repo, env, label):
        calls.regions.append(int(env.get("TT_PERF_TRACE_REGION") or 0))
        raise subprocess.TimeoutExpired(cmd, 662)

    def direct_reset(**_k):
        calls.direct_resets += 1
        return True

    monkeypatch.setattr(pm, "_adaptive_run", stalled)
    monkeypatch.setattr(PR, "_device_reset", direct_reset)
    return calls


@pytest.fixture
def _keep_environ():
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)


def test_at_the_ceiling_the_stall_is_reported_and_not_reset_again(stalled_gate, monkeypatch, _keep_environ):
    monkeypatch.setenv("TT_PERF_TRACE_REGION", str(pm._TRACE_REGION_MAX))
    ms, _m, err, _p = pm._run_full_pipeline_ms()
    assert ms is None and "timed out" in err
    assert stalled_gate.regions == [pm._TRACE_REGION_MAX], "nothing left to grow: one attempt"
    assert stalled_gate.direct_resets == 0, "_adaptive_run owns the reset after its own kill"


def test_below_the_ceiling_the_region_still_grows(stalled_gate, monkeypatch, _keep_environ):
    start = max(1, pm._TRACE_REGION_MAX // 4)
    monkeypatch.setenv("TT_PERF_TRACE_REGION", str(start))
    pm._run_full_pipeline_ms()
    assert stalled_gate.regions[:3] == [start, start * 2, pm._TRACE_REGION_MAX], "the grow is unchanged"
    assert stalled_gate.direct_resets == 0, "the board was reset once per kill, inside _adaptive_run"


# ---- the gate reaches the crash streak ---------------------------------------------------------------


@pytest.fixture
def gate(monkeypatch):
    seen = SimpleNamespace(crashes=[], oks=0)
    monkeypatch.setattr(pm, "_note_device_crash", lambda where, text="": seen.crashes.append((where, text)))
    monkeypatch.setattr(pm, "_note_device_ok", lambda: setattr(seen, "oks", seen.oks + 1))
    monkeypatch.setattr(pm, "_emit_fullpipe", lambda result: result)
    return seen


_DEVICE_FAULT = "run failed: TT_FATAL: Read unexpected run_mailbox value from core 25-16 (assert.hpp:104)"


def test_a_device_fault_is_counted_like_the_other_device_tools(gate, monkeypatch):
    assert not pm._is_measurement_failure(_DEVICE_FAULT), "the fixture must carry a device-fault signature"
    monkeypatch.setattr(pm, "_measure_full_pipeline_guarded", lambda: (None, None, _DEVICE_FAULT, None))
    v = pm._check_full_pipeline_latency_body()
    assert v["status"] == "crash"
    assert gate.crashes == [("check_full_pipeline_latency", _DEVICE_FAULT)]
    assert gate.oks == 0


def test_a_failure_with_no_device_signature_is_not_counted(gate, monkeypatch):
    """The exclusion check_pcc and measure_candidate apply: an UNKNOWN failure is not a wedge. A bare
    timeout is one of those -- the kill that produced it was already reset inside _adaptive_run."""
    msg = "run failed: Command '['pytest']' timed out after 662 seconds"
    assert pm._is_measurement_failure(msg)
    monkeypatch.setattr(pm, "_measure_full_pipeline_guarded", lambda: (None, None, msg, None))
    assert pm._check_full_pipeline_latency_body()["status"] == "crash"
    assert gate.crashes == []


def test_a_zero_headline_does_not_clear_the_streak(gate, monkeypatch):
    monkeypatch.setattr(pm, "_measure_full_pipeline_guarded", lambda: (0.0, "trace", None, None))
    v = pm._check_full_pipeline_latency_body()
    assert v["status"] == "crash" and gate.oks == 0


def test_a_positive_reading_clears_the_streak(gate, monkeypatch):
    class _Stop(Exception):
        pass

    monkeypatch.setattr(pm, "_measure_full_pipeline_guarded", lambda: (12.5, "trace", None, None))

    def stop(*_a, **_k):
        raise _Stop()

    # Everything after the streak update is the existing verdict logic; stop right after it.
    monkeypatch.setattr(pm, "_fullpipe_mode", stop)
    with pytest.raises(_Stop):  # allow-pytest.raises: repo-root conftest bypassed
        pm._check_full_pipeline_latency_body()
    assert gate.oks == 1 and gate.crashes == []
