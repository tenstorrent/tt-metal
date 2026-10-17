# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A gate step that failed because the BOARD was wedged is re-run once on the recovered board.

Measured on a WH Galaxy, 2026-09-25/26: after a fabric run exits, the next device open can fail with
"Timed out waiting for ETH heartbeat ... Stuck at 0xabcd...", or hang. The gate recovered the board
and still reported the rescued step as failed, so each fix-loop round (1-2 h at B=32) was lost to a
board the previous process had left wedged. G6's block-stack probe had it worst: no recovery at all,
a message whose "tail" was start-up log lines instead of the error, and a 4 h timeout that ended by
letting G6 pass unchecked.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.tt_hw_planner.commands import emit_e2e as E
from models.experimental.perf_automation.agent import perf_adapter as PA
from models.experimental.perf_automation.agent import probes as _PR

ETH_WEDGE = (
    "2026-09-26 05:17:16.725 | info | UMD | Established firmware bundle version: 19.10.0\n"
    "RuntimeError: Timed out waiting for ETH heartbeat on device ASIC ID: 591434138682143042, "
    "ETH core e2-0 (NOC0) to advance. Stuck at 0xabcde601\n"
)


@pytest.fixture
def board(monkeypatch):
    """Record every recovery the gate asks for; the board comes back unless told otherwise."""
    calls = {"recoveries": [], "comes_back": True}

    def _reset(error_text="", config_target="", fault_is_certain=False):
        calls["recoveries"].append((error_text, fault_is_certain))
        return calls["comes_back"]

    # Patched at the SHARED reset (probes._device_reset), which every version of the gate reaches,
    # so these tests run -- and fail -- against the gate as it was before this change too.
    monkeypatch.setattr(_PR, "_device_reset", _reset)
    monkeypatch.setattr(_PR, "tt_smi_bin", lambda: __file__)
    monkeypatch.delenv("TT_HW_PLANNER_RESET_CHIPS", raising=False)
    return calls


def _attempts(*results):
    seq = list(results)
    ran = []

    def run_once():
        ran.append(1)
        return seq.pop(0)

    return run_once, ran


# --- the rule: retry only a failure that is the board's, at most once, only on a recovered board ----


def test_a_passing_step_runs_once(board):
    run, ran = _attempts(E._StepResult(True, "ok"))
    res, notes = E._retry_after_wedge("step", run)
    assert res.ok and len(ran) == 1 and notes == [] and board["recoveries"] == []


def test_a_model_failure_is_not_retried_and_the_board_is_left_alone(board):
    run, ran = _attempts(E._StepResult(False, "AssertionError: Gate 3: min image PCC 0.72 < 0.95"))
    res, notes = E._retry_after_wedge("step", run)
    assert not res.ok and len(ran) == 1 and notes == [] and board["recoveries"] == []


def test_a_wedge_is_recovered_and_the_step_run_once_more(board):
    run, ran = _attempts(E._StepResult(False, ETH_WEDGE), E._StepResult(True, "1 passed"))
    res, notes = E._retry_after_wedge("step", run)
    assert res.ok and len(ran) == 2
    assert board["recoveries"] == [(ETH_WEDGE, False)], "evidence goes to the recovery"


def test_the_retry_happens_at_most_once(board):
    run, ran = _attempts(E._StepResult(False, ETH_WEDGE), E._StepResult(False, ETH_WEDGE))
    res, notes = E._retry_after_wedge("step", run)
    assert not res.ok and len(ran) == 2 and len(board["recoveries"]) == 1
    assert any("failed again" in n for n in notes)


def test_no_retry_on_a_board_that_did_not_come_back(board):
    board["comes_back"] = False
    run, ran = _attempts(E._StepResult(False, ETH_WEDGE))
    res, notes = E._retry_after_wedge("step", run)
    assert not res.ok and len(ran) == 1 and notes and "wedged" in notes[0]


def test_a_stall_is_a_certain_fault(board):
    """A stalled step was killed mid-run: the telemetry veto must not cancel the reset."""
    run, ran = _attempts(E._StepResult(False, "partial", stalled=True), E._StepResult(True, "ok"))
    res, _ = E._retry_after_wedge("step", run)
    assert res.ok and board["recoveries"] == [("partial", True)]


# --- the e2e step ---------------------------------------------------------------------------------


def _demo(tmp_path):
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tests" / "e2e").mkdir(parents=True)
    (demo / "tests" / "e2e" / "test_e2e_m.py").write_text("def test_e2e():\n    pass\n")
    return demo


def test_the_e2e_gate_passes_when_the_rerun_on_the_recovered_board_passes(board, monkeypatch, tmp_path):
    outputs = [ETH_WEDGE + "1 error in 7.4s", PA.batch_report_line(32) + "\n1 passed in 1624s"]
    rcs = [1, 0]

    def _exec(cmd, cwd, env, timeout_s, log_path, **k):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(outputs.pop(0))
        return rcs.pop(0)

    monkeypatch.setenv("E2E_REQUIRE_ON_DEVICE", "0")
    monkeypatch.setattr(_PR, "_execute", _exec)
    ok, reasons = E._run_deterministic_gates(_demo(tmp_path), 0.95, 60, batch=32)
    assert not [r for r in reasons if r.startswith(("G2/G3", "G3 batch"))], reasons
    assert len(board["recoveries"]) == 1


# --- G6 block stacks ------------------------------------------------------------------------------


@pytest.fixture
def two_sections(monkeypatch):
    from models.experimental.perf_automation.agent import layer_depth

    monkeypatch.setattr(layer_depth, "declared_section_depths", lambda **k: [32, 28])
    monkeypatch.setattr(E, "_missing_stack_knobs", lambda demo_dir, n: [])


def _probe_runs(monkeypatch, *outcomes):
    seq = list(outcomes)

    def _run(cmd, **k):
        o = seq.pop(0)
        return subprocess.CompletedProcess(cmd, 0 if "STACKS=" in o else 1, o, "" if "STACKS=" in o else ETH_WEDGE)

    monkeypatch.setattr(E.subprocess, "run", _run)
    return seq


def test_g6_retries_a_wedged_board_and_then_checks_the_model(board, two_sections, monkeypatch, tmp_path):
    left = _probe_runs(monkeypatch, "boom", "STACKS=7\n")
    assert E._block_stack_gate(tmp_path, "m", 60) is None
    assert left == [] and len(board["recoveries"]) == 1


def test_g6_shows_the_real_error_not_start_up_logs(board, two_sections, monkeypatch, tmp_path):
    board["comes_back"] = False
    _probe_runs(monkeypatch, "boom")
    reason = E._block_stack_gate(tmp_path, "m", 60)
    assert reason and "Timed out waiting for ETH heartbeat" in reason
    assert "Established firmware bundle" not in reason


def test_a_g6_hang_is_bounded_and_is_not_a_pass(board, two_sections, monkeypatch, tmp_path):
    """It used to return None ('could not run') after the whole caller budget, letting G6 through."""
    seen = []

    def _run(cmd, **k):
        seen.append(k.get("timeout"))
        raise subprocess.TimeoutExpired(cmd, k.get("timeout"), output=b"", stderr=b"")

    monkeypatch.setattr(E.subprocess, "run", _run)
    reason = E._block_stack_gate(tmp_path, "m", 14400)
    assert reason and "made no progress" in reason
    assert seen and max(seen) == E._g6_probe_timeout_s(14400) < 14400
    assert all(c[1] is True for c in board["recoveries"]), "a killed probe resets for certain"


def test_an_unrunnable_g6_probe_is_still_not_a_model_defect(board, two_sections, monkeypatch, tmp_path):
    def _cannot(cmd, **k):
        raise OSError("no interpreter")

    monkeypatch.setattr(E.subprocess, "run", _cannot)
    assert E._block_stack_gate(tmp_path, "m", 60) is None
