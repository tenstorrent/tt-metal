"""A device run killed for no progress TWICE in a row, with nothing working between, is reset even on a
single chip.

Run 1791998839 (2026-10-14, minicpm5_2b, --devices 0): board 0 hung in fetch_queue_reserve_back on every
large upload while every ARC answered and the one-op work probe passed. So recover()'s single-chip veto
skipped each kill's reset, and five stalled runs in a row -- the PCC gate and the full-pipeline gate, on
the edit and on the clean HEAD -- went to the same hung chip for ~80 minutes.

These pin: one kill keeps the veto (the 2026-08-17 protection); a second with no working run between
resets whatever the veto says, on both routes a kill reaches recovery (a killing caller through
probes.reset_after_killed_child, a gate's stalled report through note_crash); a working run or a
verified reset clears the streak; a ceiling kill, a heat exit and an ordinary crash are not counted.
Nothing here opens or resets a device.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from agent import device_recovery as dr
from agent import probes as PR

_STALL = "check_pcc %s for 600s -- no log growth; Process group killed; log: /tmp/x/run.log" % PR.KILL_STALL
_CEILING = "profile %s: 4000s of a 3600s ceiling; it was still moving" % PR.KILL_CEILING
_PLAIN = "run failed: AssertionError: shapes differ"


@pytest.fixture
def warm_single_chip(monkeypatch, tmp_path):
    """One chip, every ARC warm, the one-op probe passing: everything the veto reads as healthy."""
    monkeypatch.setenv("TT_RECOVERY_STATE", str(tmp_path / "recovery.json"))
    monkeypatch.setenv(dr.DEVICES_ENV, "0")
    monkeypatch.setattr(dr, "reap_device_holders", lambda: [])
    monkeypatch.setattr(dr, "_board_needs_reset", lambda: False)
    monkeypatch.setattr(dr, "_live_temps", lambda: [55.0])
    monkeypatch.setattr(dr, "_device_runs_work", lambda: True)
    monkeypatch.setattr(dr, "device_is_healthy", lambda: True)
    issued = []
    return issued, (lambda target: issued.append(target) or True)


@pytest.fixture
def through_probes(warm_single_chip, monkeypatch):
    """probes._device_reset routed into recover() with the fixture's reset, as the real one is."""
    issued, reset = warm_single_chip

    def _device_reset(error_text="", config_target="", fault_is_certain=False):
        return dr.recover("t", reset, error_text, config_target, fault_is_certain=fault_is_certain)

    monkeypatch.setattr(PR, "_device_reset", _device_reset)
    return issued


# ---- a killing caller (probes.reset_after_killed_child) ----------------------------------------------


def test_one_kill_keeps_the_veto(through_probes):
    PR.reset_after_killed_child(error_text="")
    assert through_probes == [], "one stall is ambiguous: a slow op or a contended device"


def test_a_second_kill_with_nothing_working_between_resets(through_probes):
    PR.reset_after_killed_child(error_text="")
    PR.reset_after_killed_child(error_text="")
    assert len(through_probes) == 1, "the hang a warm ARC and a passing probe both miss"


def test_a_working_run_between_clears_the_streak(through_probes):
    PR.reset_after_killed_child(error_text="")
    dr.note_ok()
    PR.reset_after_killed_child(error_text="")
    assert through_probes == []


def test_a_verified_reset_clears_the_streak(through_probes):
    PR.reset_after_killed_child(error_text="")
    PR.reset_after_killed_child(error_text="")
    PR.reset_after_killed_child(error_text="")
    assert len(through_probes) == 1, "the board came back; the next kill starts a new streak"


def test_a_heat_exit_is_not_counted(through_probes):
    PR.reset_after_ended_child(error_text="")
    PR.reset_after_ended_child(error_text="")
    assert through_probes == [], "a board that ended itself to cool is not hung"
    assert dr.CONSEC_KILLS["n"] == 0


# ---- a gate's stalled report (note_crash) ------------------------------------------------------------


def test_two_stalled_gate_runs_reset_through_note_crash(warm_single_chip):
    issued, reset = warm_single_chip
    assert dr.note_crash("check_pcc", reset, _STALL) is False
    assert issued == []
    dr.note_crash("check_pcc", reset, _STALL)
    assert len(issued) == 1


def test_a_kill_and_a_stalled_gate_share_one_streak(warm_single_chip, through_probes):
    issued, reset = warm_single_chip
    PR.reset_after_killed_child(error_text="")
    dr.CONSEC_CRASH["n"] = 0
    dr.note_crash("check_pcc", reset, _STALL)
    assert len(issued) == 1


@pytest.mark.parametrize("text", [_CEILING, _PLAIN])
def test_other_crashes_keep_the_veto(warm_single_chip, text):
    issued, reset = warm_single_chip
    dr.note_crash("check_pcc", reset, text)
    dr.note_crash("check_pcc", reset, text)
    assert issued == [], "unchanged: two ordinary crashes on a warm single chip are still vetoed"
    assert dr.CONSEC_KILLS["n"] == 0


def test_the_stall_wording_has_one_owner():
    assert dr.killed_for_no_progress(_STALL)
    assert not dr.killed_for_no_progress(_CEILING)
    assert not dr.killed_for_no_progress("")


# ---- perf_mcp's own kill picks the counted reset only for a stall ------------------------------------


@pytest.fixture
def mcp_resets(monkeypatch):
    seen = []
    for name in ("reset_after_ended_child", "reset_after_killed_child"):
        monkeypatch.setattr(PR, name, lambda error_text="", env=None, _n=name: seen.append(_n) or True)
    monkeypatch.setattr(PR, "wait_for_memory_headroom_before_device_work", lambda label: None)
    monkeypatch.setattr(PR, "memory_cap_preexec_fn", lambda: None)
    from cc_optimize import perf_mcp

    return perf_mcp, seen


def test_a_backstop_kill_of_a_moving_run_is_not_counted(mcp_resets, tmp_path):
    pm, seen = mcp_resets
    cmd = [sys.executable, "-c", "import time\nwhile True: print('step', flush=True); time.sleep(1)"]
    with pytest.raises(subprocess.TimeoutExpired):  # allow-pytest.raises: repo-root conftest bypassed
        pm._adaptive_run(cmd, str(tmp_path), dict(os.environ), "moving", stall_s=60, backstop=6)
    assert seen == ["reset_after_ended_child"], "still moving at the backstop is not a hang"
