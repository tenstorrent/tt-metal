"""Recovery stops when it said it would, and says when a reset cannot possibly help.

THE DEFECT, from run 39 on 2026-08-05/06. The board wedged and the tool reset it 34 times against a
RESET_FAIL_LIMIT of 3, spending ~100 minutes, then sat dead until morning -- about ten hours lost.

The limit was not missing. It was COUNTED in one place and ENFORCED in none:

    recover()                 increments RESET_FAILS, returns False, never refuses
    recovery_exhausted()      reads it -- and had exactly ONE consumer in the whole tool,
                              perf_mcp.termination_check, on the path where profiling RAISES
    the four sites that reset  probes:693, run.py:1911, perf_mcp._recover_device:351, and
                              note_crash's dead-board branch -- none consult it

recover()'s own docstring promised the opposite: "which board, how many tries, whether it worked and
when to give up are decided here so no caller can decide it differently". Counting was; stopping was
not. And `Read 0xffffffff` matches is_dead_board, whose branch resets IMMEDIATELY on every crash,
bypassing even the two-strike counter -- so the one signature guaranteed to appear on a dead board was
also the one that retried hardest.

THE SECOND HALF. Three failures tell you resets are not working; they do not tell you why, or what to
do instead. The kernel does:

    tenstorrent tenstorrent!2: Failed to set initial power state: -22

`tt-smi -r` reaches the card OVER PCIe and asks its board-management firmware to cycle power. When
that firmware is the thing refusing, the request has nowhere to land -- no reset can clear it, only a
host reboot. On this box that line appeared 714 times across a reboot-less day while 34 resets
achieved nothing, and NOTHING else in the kernel log marked the fault: no thermal trip, no PCIe AER,
no OOM, no hung task. The tool read no kernel log at all, so it could only infer "unrecoverable" from
repeated failure, and reported it as such -- which is not something an operator can act on.

Reading it turns a ten-hour silent stall into a halt, in minutes, that names the fix.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DR_PATH = Path(__file__).resolve().parent.parent / "agent" / "device_recovery.py"


@pytest.fixture()
def dr(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_RECOVERY_STATE", str(tmp_path / "rec.json"))
    monkeypatch.delenv("TT_RECOVERY_FAIL_LIMIT", raising=False)
    monkeypatch.delenv("PERF_MCP_RESET_FAIL_LIMIT", raising=False)
    import agent.device_recovery as m

    importlib.reload(m)
    # never touch a real board, and never read a real kernel log, from a test
    monkeypatch.setattr(m, "device_is_healthy", lambda *a, **k: False)
    monkeypatch.setattr(m, "_kernel_tail", lambda: "")
    monkeypatch.setattr(m, "targets_for", lambda *a, **k: ["0,1"])
    return m


# ---------------------------------------------------------------- the limit is enforced where it is counted


def test_recover_refuses_once_the_limit_is_reached(dr):
    """THE FIX. Before this, recover() counted to 34 and never declined."""
    tries = []
    for _ in range(dr.RESET_FAIL_LIMIT):
        dr.recover("t", lambda tgt: tries.append(tgt))
    n_before = len(tries)
    assert dr.recover("t", lambda tgt: tries.append(tgt)) is False
    assert len(tries) == n_before, "it reset again after the limit"


def test_the_count_matches_the_configured_limit(dr):
    """Called far more times than the limit -- as run 39's callers did -- only `limit` resets issue.
    (recover() returns False on ANY failed reset, not only on exhaustion, so the caller keeps asking;
    that is exactly the loop that reached 34.)"""
    tries = []
    for _ in range(dr.RESET_FAIL_LIMIT + 10):
        dr.recover("t", lambda tgt: tries.append(tgt))
    assert len(tries) == dr.RESET_FAIL_LIMIT, tries


@pytest.mark.parametrize("limit", [1, 2, 5])
def test_the_limit_is_configurable(tmp_path, monkeypatch, limit):
    monkeypatch.setenv("TT_RECOVERY_STATE", str(tmp_path / ("rec-%d.json" % limit)))
    monkeypatch.setenv("TT_RECOVERY_FAIL_LIMIT", str(limit))
    import agent.device_recovery as m

    importlib.reload(m)
    monkeypatch.setattr(m, "device_is_healthy", lambda *a, **k: False)
    monkeypatch.setattr(m, "_kernel_tail", lambda: "")
    monkeypatch.setattr(m, "targets_for", lambda *a, **k: ["0,1"])
    tries = []
    for _ in range(limit + 6):
        m.recover("t", lambda tgt: tries.append(tgt))
    assert len(tries) == limit


def test_every_entry_point_is_closed_by_the_one_guard(dr):
    """note_crash's DEAD-BOARD branch resets immediately on every crash, bypassing the two-strike
    counter -- so the signature guaranteed to appear on a dead board retried hardest. It funnels
    through recover(), so the guard closes it too."""
    tries = []
    reset = lambda tgt: tries.append(tgt)  # noqa: E731
    for _ in range(dr.RESET_FAIL_LIMIT + 4):
        dr.note_crash("t", reset, error_text="Read 0xffffffff over PCIe ID 2")
    assert len(tries) == dr.RESET_FAIL_LIMIT, tries


def test_a_success_clears_the_count(dr, monkeypatch):
    """A limit that never resets would retire a board that recovers normally."""
    dr.recover("t", lambda tgt: None)
    monkeypatch.setattr(dr, "device_is_healthy", lambda *a, **k: True)
    assert dr.recover("t", lambda tgt: None) is True
    assert dr.recovery_exhausted() is False


# ---------------------------------------------------------------- the kernel's verdict short-circuits


def test_a_board_management_fault_is_recognised():
    import agent.device_recovery as m

    line = "tenstorrent tenstorrent!2: Failed to set initial power state: -22"
    assert m.board_needs_host_reboot(line) is True
    assert m.board_needs_host_reboot("tenstorrent: pin_user_pages_longterm failed: -14") is False
    assert m.board_needs_host_reboot("") is False


def test_an_unreadable_kernel_log_falls_back_to_counting(dr, monkeypatch):
    """Recovery must never DEPEND on dmesg: a host with dmesg_restrict=1 and no journal still gets the
    old behaviour, bounded by the limit."""
    monkeypatch.setattr(dr, "_kernel_tail", lambda: "")
    tries = []
    for _ in range(dr.RESET_FAIL_LIMIT + 6):
        dr.recover("t", lambda tgt: tries.append(tgt))
    assert len(tries) == dr.RESET_FAIL_LIMIT


def test_the_kernel_probe_is_bounded_and_best_effort():
    """It runs inside recovery, on a host that is already misbehaving. It may not hang, and it may not
    raise."""
    src = DR_PATH.read_text()
    i = src.index("def _kernel_tail")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "timeout=" in body, "an unbounded dmesg would hang recovery"
    assert "except Exception" in body and 'return ""' in body


# ---------------------------------------------------------------- the halt names the action


def test_the_halt_tells_the_operator_to_reboot():
    """ "unrecoverable after N attempts" is an inference from repeated failure and not actionable. When
    the kernel has diagnosed it, the halt says so."""
    src = (Path(__file__).resolve().parent.parent / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index('"halt": "needs_host_reboot"')
    win = src[max(0, i - 600) : i + 600]
    assert "board_needs_host_reboot" in win
    assert "REBOOT THE HOST" in win
    assert "device_unrecoverable" in win, "the non-diagnosed case must keep its own halt reason"


# ---------------------------------------------------------------- the count belongs to ONE run


def test_a_count_from_another_run_reads_as_zero(dr, monkeypatch):
    """THE REGRESSION THE GUARD CREATED, fixed at the lifetime rather than patched.

    state_path() keys the file by (model, task), which OUTLIVES the run -- so "resets have stopped
    working" was inherited by every later run on that model. Harmless while nothing read the count;
    once recover() began REFUSING at the limit it was a latch. Run 39's dead board left
    reset_fails=34 in a file that survived the board being fixed, a host reboot and a fresh run on
    healthy hardware, which then halted before its first round with all four chips at 45C."""
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-A")
    for _ in range(dr.RESET_FAIL_LIMIT + 4):
        dr.recover("t", lambda tgt: None)
    assert dr.recovery_exhausted() is True
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-B")
    assert dr.recovery_exhausted() is False, "a previous run's failures still gate this one"


def test_the_new_run_actually_gets_its_resets_back(dr, monkeypatch):
    """Not just the flag -- the behaviour."""
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-A")
    for _ in range(dr.RESET_FAIL_LIMIT + 2):
        dr.recover("t", lambda tgt: None)
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-B")
    tries = []
    for _ in range(dr.RESET_FAIL_LIMIT + 3):
        dr.recover("t", lambda tgt: tries.append(tgt))
    assert len(tries) == dr.RESET_FAIL_LIMIT, tries


def test_within_one_run_the_count_still_accumulates(dr, monkeypatch):
    """It must survive the PROCESS -- the process holding it is the one a device fault kills -- just
    not the run."""
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-A")
    for _ in range(dr.RESET_FAIL_LIMIT):
        dr.recover("t", lambda tgt: None)
    assert dr.recovery_exhausted() is True
    assert dr.recovery_exhausted() is True, "re-reading the file lost the count"


def test_a_successful_measurement_does_not_reset_the_backstop(dr, monkeypatch):
    """note_ok clears the CRASH STREAK, not the reset count. A board alternating working and wedging
    would otherwise clear it on every good measurement -- restoring the unbounded retrying the limit
    exists to stop."""
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-A")
    for _ in range(dr.RESET_FAIL_LIMIT + 1):
        dr.recover("t", lambda tgt: None)
    dr.note_ok()
    assert dr.recovery_exhausted() is True


def test_the_run_id_is_forwarded_to_the_mcp_server():
    """perf_mcp counts in its OWN process. An unforwarded stamp puts the two sides in different runs,
    each reading the other's failures as zero, and the backstop never triggers."""
    src = (Path(__file__).resolve().parent.parent / "cc_optimize" / "run.py").read_text()
    i = src.index('for _k in ("PERF_MCP_STATE_DIR"')
    assert "PERF_MCP_RUN_ID" in src[i : i + 200], src[i : i + 200]


def test_the_stamp_is_not_overwritten_on_a_restart():
    """The supervisor restarts the child. A fresh stamp there would hand every restart a new budget,
    which is the latch's opposite failure: never stopping."""
    src = (Path(__file__).resolve().parent.parent / "cc_optimize" / "run.py").read_text()
    i = src.index("def _stamp_run_id")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "if not cur:" in body, body[-400:]


# ---------------------------------------------------------------- the kernel verdict is the REAL stop


def test_note_ok_still_clears_the_crash_streak(dr):
    """Unchanged behaviour: it cleared CONSEC_CRASH before and must keep doing so."""
    dr.note_crash("t", lambda tgt: None, error_text="something ambiguous")
    dr.note_ok()
    tries = []
    dr.note_crash("t", lambda tgt: tries.append(tgt), error_text="something ambiguous")
    assert tries == [], "one ambiguous crash after an OK must not reset -- the two-strike rule"


def test_the_kernel_line_does_not_gate_recovery(dr, monkeypatch):
    """SUPERSEDED DESIGN. An earlier revision refused in ZERO attempts when the driver had logged
    "Failed to set initial power state", on the reading that no reset could then work.

    It is transient. The message fires whenever a device is OPENED while its ARC is not ready: a
    wedged board produced 714 across a day on this box, and a HEALTHY run produced 4 in an hour while
    continuing to optimize. Gating on it declares a working board dead at the first fault -- worse
    than the unbounded retrying the check was added to stop."""
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-A")
    monkeypatch.setattr(dr, "_kernel_tail", lambda: "tenstorrent!0: Failed to set initial power state: -22")
    tries = []
    dr.recover("t", lambda tgt: tries.append(tgt))
    assert tries, "a transient kernel message refused a reset outright"


def test_the_kernel_line_still_explains_a_failure(dr, monkeypatch):
    """Its actual job: after the resets have failed, say WHY -- "reboot the host" is actionable,
    "unrecoverable after N attempts" is not. Run 39 sat dead until morning for want of that."""
    assert dr.board_needs_host_reboot("tenstorrent!2: Failed to set initial power state: -22") is True
    assert dr.board_needs_host_reboot("tenstorrent: pin_user_pages_longterm failed: -14") is False


def test_the_kernel_read_is_scoped_to_this_boot(dr):
    """`journalctl -k` without -b returns the last N kernel lines across EVERY boot, so a fault from a
    previous boot reads as live. On this box: chips died, the host rebooted, all four came back
    healthy at 1e52 -- and the check still answered True from yesterday's log."""
    src = DR_PATH.read_text()
    i = src.index('"journalctl"')
    assert '"-b"' in src[i : i + 120], src[i : i + 120]
