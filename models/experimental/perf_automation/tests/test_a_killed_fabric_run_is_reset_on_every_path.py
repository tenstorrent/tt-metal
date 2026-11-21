"""A SIGKILLed multi-chip run is reset on every path that kills one, and a single-chip run still is not.

The temperature veto in device_recovery.recover() cancels a reset when every ARC answers and the error
text names no dead board. After a multi-chip fabric run is killed that is exactly the wrong answer: the
fabric stays wedged while every chip keeps publishing its temperature. Measured 2026-09-27 on a WH
Galaxy: the profiler's hang retry, the perf-test builder's hang handler and the round watchdog all
reset with no evidence, every reset was vetoed ("no reset issued"), and every later profile opened the
mesh with "Read unexpected run_mailbox value" 40 times and then hung or returned no device data.

These pin: the kill rule's answers (unchanged for the caller that already had it), each killing path
passing it, the failed run's text reaching the reset, the new signature, and the hung attempt's log
being kept instead of overwritten.
"""

import inspect
import subprocess
from pathlib import Path

import pytest

from agent import device_recovery as dr
from agent import probes
from cc_optimize import run as run_mod

_DIRTY_START = (
    "Metal | While initializing device 0, active ethernet dispatch core 25-17 detected as still running, "
    "issuing exit signal. (risc_firmware_initializer.cpp:4)\n"
    "Read unexpected run_mailbox value: 0x40 (expected 0x80 or 0x0)\n"
    "critical | Always | TT_FATAL: Read unexpected run_mailbox value from core 25-16 (assert.hpp:104)\n"
)


@pytest.fixture
def no_devices_env(monkeypatch):
    monkeypatch.delenv(dr.DEVICES_ENV, raising=False)


# ---- the rule -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec,expected",
    [("all", True), ("", True), ("0,1", True), ("0,1,2,3", True), ("0", False), ("3", False), ("single", False)],
)
def test_the_rule_resets_a_fabric_and_spares_one_chip(spec, expected, no_devices_env):
    assert dr.reset_is_mandatory_after_kill(spec) is expected


def test_an_unreadable_spec_widens_to_a_fabric(no_devices_env):
    assert dr.reset_is_mandatory_after_kill("chip-a") is True, "an uncountable spec is UNKNOWN, not one"


def test_the_childs_own_chip_count_wins(no_devices_env):
    assert dr.reset_is_mandatory_after_kill("0", {"device_count": "8"}) is True
    assert dr.reset_is_mandatory_after_kill("all", {"mesh_chips": "1"}) is False


def test_the_orchestrators_spec_is_read_when_the_caller_has_none(monkeypatch):
    monkeypatch.setenv(dr.DEVICES_ENV, "all")
    assert dr.reset_is_mandatory_after_kill(env={}) is True
    monkeypatch.setenv(dr.DEVICES_ENV, "single")
    assert dr.reset_is_mandatory_after_kill(env={}) is False


def test_no_spec_anywhere_keeps_the_veto(no_devices_env):
    assert dr.reset_is_mandatory_after_kill(env={}) is False, "outside the orchestrator nothing says fabric"


def test_the_callers_that_already_had_the_rule_answer_as_before(monkeypatch):
    # run.py's name, signature and its own _chip_count are unchanged
    monkeypatch.setattr(run_mod, "_chip_count", lambda d: 8)
    assert run_mod._reset_is_mandatory_after_kill("0,1,2,3") is True
    monkeypatch.setattr(run_mod, "_chip_count", lambda d: 1)
    assert run_mod._reset_is_mandatory_after_kill("0") is False
    assert run_mod._reset_is_mandatory_after_kill("all") is True
    assert run_mod._reset_is_mandatory_after_kill(None) is True


def test_there_is_one_chip_count_parser():
    from agent import before_loop

    assert before_loop._requested_chip_count is dr.requested_chip_count


# ---- the signature --------------------------------------------------------------------------------


def test_a_mesh_opened_on_a_still_running_fabric_is_a_dead_board():
    assert dr.is_dead_board(_DIRTY_START) is True


def test_a_clean_open_is_not():
    clean = "Metal | Initializing Fabric\nMetal | Fabric initialized on 32 devices\n[perf] batch=32 steps=2\n"
    assert dr.is_dead_board(clean) is False


# ---- the veto now gets its evidence ----------------------------------------------------------------


@pytest.fixture
def warm_board(monkeypatch):
    """Every ARC publishing a temperature, nothing to reap: the state the veto reads as healthy."""
    monkeypatch.setattr(dr, "reap_device_holders", lambda: [])
    monkeypatch.setattr(dr, "_board_needs_reset", lambda: False)
    monkeypatch.setattr(dr, "_live_temps", lambda: [30.5] * 32)
    monkeypatch.setattr(dr, "device_is_healthy", lambda: True)
    monkeypatch.setattr(dr, "recovery_exhausted", lambda: False)
    issued = []
    return issued, (lambda target: issued.append(target) or True)


def test_the_veto_is_unchanged_without_evidence(warm_board):
    issued, reset = warm_board
    dr.recover("t", reset, error_text="")
    assert issued == [], "no evidence, warm chips: the veto still has the last word"


def test_a_fabric_kill_is_reset_through_the_same_veto(warm_board, no_devices_env):
    issued, reset = warm_board
    dr.recover("t", reset, error_text="", fault_is_certain=dr.reset_is_mandatory_after_kill("all"))
    assert issued, "the kill of a multi-chip run is the evidence"


def test_a_single_chip_kill_still_meets_the_veto(warm_board, no_devices_env):
    issued, reset = warm_board
    dr.recover("t", reset, error_text="", fault_is_certain=dr.reset_is_mandatory_after_kill("0"))
    assert issued == [], "the 2026-08-17 single-chip case keeps its protection"


def test_a_dirty_start_log_is_reset_on_its_own_evidence(warm_board):
    issued, reset = warm_board
    dr.recover("t", reset, error_text=_DIRTY_START)
    assert issued


# ---- each killing path hands it over ---------------------------------------------------------------


def _collect_one(cmd, cwd, env=None, capture_output=None, text=None, timeout=None):
    return subprocess.CompletedProcess(cmd, 0, "t.py::test_full[S128]\n1 test collected in 0.1s\n", "")


def _hang_then_pass(first_log):
    calls = {"n": 0}

    def execute(cmd, cwd, env, timeout_s, log_path):
        calls["n"] += 1
        if calls["n"] == 1:
            Path(log_path).write_text(first_log)
            raise probes.TracyHangError("tracy run made no forward progress for 600s")
        Path(log_path).write_text("second attempt\n")
        d = Path(cmd[cmd.index("-o") + 1]) / "reports" / "ts1"
        d.mkdir(parents=True, exist_ok=True)
        (d / "ops_perf_results_ts1.csv").write_text("OP CODE,X\nMatmul,1\n")
        return 0

    return execute


@pytest.fixture(autouse=True)
def _clear_node_cache():
    probes._NODE_ID_CACHE.clear()
    yield
    probes._NODE_ID_CACHE.clear()


@pytest.mark.parametrize("spec,certain", [("all", True), ("single", False)])
def test_the_profilers_hang_retry_passes_the_kill_and_the_log(tmp_path, monkeypatch, spec, certain):
    monkeypatch.setenv(dr.DEVICES_ENV, spec)
    seen = []
    rp = probes.make_run_profiled(
        tmp_path,
        "t.py",
        "S128",
        execute=_hang_then_pass(_DIRTY_START),
        collect_runner=_collect_one,
        device_reset=lambda **kw: seen.append(kw) or True,
    )
    rp("e2e", 1, 128, tmp_path / "profiles", 0)
    assert len(seen) == 1
    assert seen[0].get("fault_is_certain", False) is certain, "certainty is passed only for a fabric"
    assert "unexpected run_mailbox value" in seen[0]["error_text"], "the hung run's own output reaches the reset"
    kept = tmp_path / "profiles" / "run0_tracy.log.attempt1"
    assert kept.read_text() == _DIRTY_START, "the hung attempt's log is kept, not overwritten by the retry"
    assert (tmp_path / "profiles" / "run0_tracy.log").read_text() == "second attempt\n"


def test_a_kept_attempt_log_is_invisible_to_the_log_readers(tmp_path):
    d = tmp_path / "profiles"
    d.mkdir()
    (d / "run0_tracy.log.attempt1").write_text("TRACE_PER_TOKEN_MS=1.0\n")
    assert sorted(d.glob("*_tracy.log")) == []


def test_the_perf_test_builders_hang_handler_passes_the_kill(tmp_path, monkeypatch):
    from agent import perf_test_gen as ptg

    monkeypatch.setenv(dr.DEVICES_ENV, "all")
    test_file = tmp_path / "test_x.py"
    test_file.write_text("def test_x():\n    pass\n")
    seen = []

    def execute(cmd, cwd, env, timeout_s, log_path, stall_timeout_s=0):
        Path(log_path).write_text(_DIRTY_START)
        raise probes.TracyHangError("no forward progress for 300s")

    monkeypatch.setattr(probes, "_execute", execute)
    monkeypatch.setattr(probes, "_device_reset", lambda **kw: seen.append(kw) or True)
    monkeypatch.setattr(probes, "wait_for_memory_headroom_before_device_work", lambda *a, **k: None)
    monkeypatch.setattr(probes, "_await_cool", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(probes, "_cool_before_remeasure", lambda *a, **k: (True, 0.0), raising=False)
    monkeypatch.setenv("PERF_MCP_DEVICE_DISRUPT_RETRIES", "0")
    ptg._run_perf_node(str(test_file), {}, timeout_s=5)
    assert seen, "the handler reset"
    assert seen[0].get("fault_is_certain") is True
    assert "unexpected run_mailbox value" in seen[0]["error_text"]


def test_both_watchdog_reclaims_declare_the_kill():
    src = inspect.getsource(run_mod)
    for anchor in (
        "error_text=_tail_lines(agent_log, 40)",
        'error_text=_tail_lines(str(kernel_log) + ".agent.log", 40)',
    ):
        i = src.index(anchor)
        window = src[i : i + 200]
        assert "after_kill=_reset_is_mandatory_after_kill(devices)" in window, anchor
