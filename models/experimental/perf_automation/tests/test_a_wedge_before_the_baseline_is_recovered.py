"""Pin: a board that dies BEFORE the baseline must still reach device recovery.

optimize's per-op loop is wedge-aware -- a wedge is recorded and the board reset between attempts,
which is why a table can show `·wedge` on one knob and a win on the next. The baseline measurement
that runs BEFORE that loop was not: every failure exit returned the error upward without asking
whether the board had died. A wedge therefore surfaced as "the workload printed no timing marker",
is_dead_board was never called, consec_crash stayed at 0, and the reset that already existed was
never issued -- so a dead board failed every retry identically for as long as the run lasted
(observed 2026-09-24: 65 minutes of "nothing recorded yet").

Two holes, both fixed:
  * the evidence never reached the check (all three baseline exits);
  * the check would not have matched it anyway -- UMD's "Firmware startup error" was absent from
    DEAD_BOARD_SIGS, and that is what the wedged board actually printed.
"""

from __future__ import annotations

import inspect

from models.experimental.perf_automation.agent import device_recovery


def test_the_firmware_startup_failure_counts_as_a_dead_board():
    """UMD's generic device-init failure means the card never came up."""
    text = (
        "RuntimeError: Firmware startup error on device 1 at core 0-10 over NOC0: "
        "scratch_status=0x0, postcode=0x57, message_id=0x0 (Timed out after 300000 ms)"
    )
    assert device_recovery.is_dead_board(text)


def test_the_signatures_that_already_worked_still_do():
    for text in ("Read 0xffffffff over PCIe ID 3", "board should be reset", "pcie link down", "device hang"):
        assert device_recovery.is_dead_board(text), text


def test_a_healthy_failure_is_not_a_dead_board():
    """Widening the list must not make ordinary failures trigger a reset."""
    for text in (
        "no TRACE_PER_TOKEN_MS or FORWARD_WALL_MS in output (workload did not run full-pipeline)",
        "Gate 3: min image PCC 0.720134 < 0.99",
        "decode did not advance between iterations: step 3",
        "",
        None,
    ):
        assert not device_recovery.is_dead_board(text), repr(text)


def _baseline_source() -> str:
    from models.experimental.perf_automation.cc_optimize import perf_mcp

    return inspect.getsource(perf_mcp._run_full_pipeline_ms)


def test_every_failure_exit_of_the_baseline_consults_the_recovery():
    """Not one exit: all of them. A wedge can surface as a stalled step, as a captured
    exception, or as missing markers, and each one used to return blind."""
    src = _baseline_source()
    assert src.count("_recover_if_board_is_dead(") >= 3, (
        "each baseline failure exit must hand its own text to the recovery; "
        f"found {src.count('_recover_if_board_is_dead(')}"
    )
    for evidence in ("decode_stuck", "last_err"):
        assert f"_recover_if_board_is_dead({evidence}" in src, f"{evidence} exit is still blind"
    assert 'locals().get("out")' in src, "the marker-missing exit must pass the captured output"


def test_the_helper_reuses_the_shared_primitives():
    """No parallel reset logic: one predicate, one recover, both already in the file."""
    from models.experimental.perf_automation.cc_optimize import perf_mcp

    src = inspect.getsource(perf_mcp._recover_if_board_is_dead)
    assert "_is_dead_board(" in src
    assert "_recover_device(" in src
    assert "tt-smi" not in src, "the helper must not issue resets itself"


def test_the_helper_is_silent_on_a_non_device_failure():
    """It must return False without touching the hardware when the text is not a wedge."""
    from models.experimental.perf_automation.cc_optimize import perf_mcp

    assert perf_mcp._recover_if_board_is_dead("", "test") is False
    assert perf_mcp._recover_if_board_is_dead(None, "test") is False
    assert perf_mcp._recover_if_board_is_dead("PCC 0.98 below threshold", "test") is False
