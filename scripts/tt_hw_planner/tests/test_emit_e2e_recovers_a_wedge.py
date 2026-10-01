# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""emit-e2e recovers a wedged board through the shared recovery, and notices a wedge that is not a hang.

2026-09-25, WH Galaxy: the e2e gate hung mid-collective, and from then on every round failed at
device-open in seconds with "NOC0 is hung on PCIe device ID 9". emit-e2e reset only on a timeout,
and then with its own `tt-smi -r 0,1,2,3` -- a plain `-r` that does not reset a Galaxy, aimed at
chips that were not the hung one. Each round failed identically until the run was stopped by hand.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.tt_hw_planner.commands import emit_e2e as E
from models.experimental.perf_automation.agent import probes as _PR

NOC_HANG = "E   RuntimeError: NOC0 is hung on PCIe device ID 9.\nLocation: umd/device/tt_device/tt_device.cpp:90"


def _shared(monkeypatch, ok=True, exhausted=False):
    """Stub the shared recovery at its entry point and record what emit-e2e hands it."""
    from models.experimental.perf_automation.agent import device_recovery as _dr
    from models.experimental.perf_automation.agent import probes as _pr

    seen = []
    monkeypatch.setattr(_pr, "tt_smi_bin", lambda: __file__)  # any existing path
    monkeypatch.setattr(_pr, "_device_reset", lambda **k: seen.append(k) or ok)
    monkeypatch.setattr(_dr, "recovery_exhausted", lambda: exhausted)
    monkeypatch.delenv("TT_HW_PLANNER_RESET_CHIPS", raising=False)
    return seen


def test_the_reset_goes_through_the_shared_recovery_with_the_evidence(monkeypatch):
    seen = _shared(monkeypatch)
    assert "verified" in E._reset_device(error_text=NOC_HANG)
    assert seen == [{"error_text": NOC_HANG, "config_target": ""}]


def test_an_operator_target_is_still_honoured(monkeypatch):
    seen = _shared(monkeypatch)
    monkeypatch.setenv("TT_HW_PLANNER_RESET_CHIPS", "8,9")
    E._reset_device()
    assert seen[0]["config_target"] == "8,9"


def test_a_reset_that_did_not_bring_the_board_back_says_so(monkeypatch):
    _shared(monkeypatch, ok=False)
    assert "did NOT bring the board back" in E._reset_device(error_text=NOC_HANG)
    _shared(monkeypatch, ok=False, exhausted=True)
    assert "EXHAUSTED" in E._reset_device(error_text=NOC_HANG)


def test_a_wedge_in_a_failure_triggers_a_reset(monkeypatch):
    seen = _shared(monkeypatch)
    assert E._recover_if_wedged(NOC_HANG)
    assert seen and seen[0]["error_text"] == NOC_HANG


def test_an_ordinary_failure_does_not_reset_the_board(monkeypatch):
    seen = _shared(monkeypatch)
    assert E._recover_if_wedged("AssertionError: Gate 3: min image PCC 0.72 < 0.99") is None
    assert E._recover_if_wedged("") is None
    assert seen == []


def test_the_e2e_gate_resets_a_board_that_wedges_without_hanging(monkeypatch, tmp_path):
    """End to end through _run_deterministic_gates: rc=1 in seconds, the wedge only in the output."""
    seen = _shared(monkeypatch)
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tests" / "e2e").mkdir(parents=True)
    (demo / "tests" / "e2e" / "test_e2e_m.py").write_text("def test_e2e():\n    pass\n")
    monkeypatch.setenv("E2E_REQUIRE_ON_DEVICE", "0")

    # The gate runs pytest through probes._execute now (progress watchdog, not a stopwatch): it
    # streams to a log and returns rc, so the double writes what the gate will read back.
    runs = []

    def _exec(cmd, cwd, env, timeout_s, log_path, **k):
        runs.append(1)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(NOC_HANG + "\n1 error in 7.4s")
        return 1

    monkeypatch.setattr(_PR, "_execute", _exec)
    ok, reasons = E._run_deterministic_gates(demo, 0.99, 60)
    assert ok is False
    assert any("the device was wedged" in r for r in reasons), reasons
    assert seen and "NOC0 is hung" in seen[0]["error_text"]
    assert len(runs) == 2, "a wedge-caused failure is re-run once on the recovered board"


def test_a_hang_hands_its_partial_output_to_the_recovery(monkeypatch, tmp_path):
    seen = _shared(monkeypatch)
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tests" / "e2e").mkdir(parents=True)
    (demo / "tests" / "e2e" / "test_e2e_m.py").write_text("def test_e2e():\n    pass\n")
    monkeypatch.setenv("E2E_REQUIRE_ON_DEVICE", "0")

    # A STALL, NOT A CLOCK. The watchdog raises only when the log stopped growing AND no CPU was
    # burned, so what the gate reports is "made no forward progress" -- it no longer accuses the
    # fabric merely because a slow run outlived a typed number.
    def _hang(cmd, cwd, env, timeout_s, log_path, **k):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("[e2e] stage dp_gather\n")
        raise _PR.TracyHangError("tracy run made no forward progress for 600s; log: %s" % log_path)

    monkeypatch.setattr(_PR, "_execute", _hang)
    ok, reasons = E._run_deterministic_gates(demo, 0.99, 60)
    assert ok is False
    assert any("made no forward progress" in r for r in reasons), reasons
    assert seen and "dp_gather" in seen[0]["error_text"], "the partial output must still reach recovery"


def test_emit_e2e_stamps_its_run_before_any_device_work(monkeypatch):
    """Unstamped, emit-e2e counted reset failures under the empty stamp, which never expires."""
    from models.experimental.perf_automation.agent import device_recovery as _dr

    monkeypatch.delenv("PERF_MCP_RUN_ID", raising=False)
    monkeypatch.setattr("scripts.tt_hw_planner.commands.optimize.invalid_trace_flag_error", lambda: None)
    from models.experimental.perf_automation.agent import probes as _pr

    prepared = []
    monkeypatch.setattr(_pr, "prepare_device_reset", lambda box="": prepared.append(box))
    stamped = []
    monkeypatch.setattr(E, "_emit_e2e_phase_a", lambda args: stamped.append(_dr._run_stamp()) or 0)
    assert E.cmd_emit_e2e(object()) == 0
    assert stamped and stamped[0], "phase A ran before the run was stamped"
    assert prepared, "the device reset was not prepared at startup"


def test_a_previous_runs_failed_resets_do_not_block_this_run(monkeypatch, tmp_path):
    """2026-09-25: reset_fails=3 under the empty stamp made every later emit-e2e refuse to reset."""
    from models.experimental.perf_automation.agent import device_recovery as _dr

    state = tmp_path / "recovery.json"
    state.write_text('{"run": "", "reset_fails": %d}' % _dr.RESET_FAIL_LIMIT)
    monkeypatch.setattr(_dr, "state_path", lambda: state)
    monkeypatch.delenv("PERF_MCP_RUN_ID", raising=False)
    assert _dr.recovery_exhausted() is True  # the latch, as it was
    _dr.stamp_run()
    assert _dr.recovery_exhausted() is False
