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


def test_it_does_not_even_try_once_when_the_kernel_has_ruled_it_out(dr, monkeypatch):
    """Three failures tell you resets are not working; the kernel tells you they CANNOT work. There is
    no value in spending the three."""
    monkeypatch.setattr(dr, "_kernel_tail", lambda: "tenstorrent!2: Failed to set initial power state: -22")
    tries = []
    assert dr.recover("t", lambda tgt: tries.append(tgt)) is False
    assert tries == [], "it reset a board the driver had already refused"
    assert dr.recovery_exhausted() is True, "the run must halt, not keep polling"


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


# ---------------------------------------------------------------- the limit must not be a one-way door


def test_a_working_device_clears_the_reset_count(dr):
    """THE REGRESSION THE GUARD CREATED. RESET_FAILS cleared in exactly one place -- inside recover(),
    after a reset came back healthy. Harmless while nothing read it; once recover() REFUSED at the
    limit, clearing required a successful reset and resetting was refused. A one-way door.

    Run 39's dead board left reset_fails=34 in the DURABLE state file. It survived the board being
    fixed, a host reboot and a fresh run on healthy hardware, then halted that run before its first
    round with the board idling at 45C. A profile that completes is proof the device is fine, and
    proof outranks a stale count."""
    for _ in range(dr.RESET_FAIL_LIMIT + 5):
        dr.recover("t", lambda tgt: None)
    assert dr.recovery_exhausted() is True
    dr.note_ok()
    assert dr.recovery_exhausted() is False, "a healthy device did not clear the count"


def test_after_clearing_it_will_reset_again(dr):
    """Not just the flag -- the behaviour. The next crash must actually get its resets back."""
    for _ in range(dr.RESET_FAIL_LIMIT + 2):
        dr.recover("t", lambda tgt: None)
    dr.note_ok()
    tries = []
    for _ in range(dr.RESET_FAIL_LIMIT + 3):
        dr.recover("t", lambda tgt: tries.append(tgt))
    assert len(tries) == dr.RESET_FAIL_LIMIT, tries


def test_note_ok_still_clears_the_crash_streak(dr):
    """Unchanged behaviour: it cleared CONSEC_CRASH before and must keep doing so."""
    dr.note_crash("t", lambda tgt: None, error_text="something ambiguous")
    dr.note_ok()
    tries = []
    dr.note_crash("t", lambda tgt: tries.append(tgt), error_text="something ambiguous")
    assert tries == [], "one ambiguous crash after an OK must not reset -- the two-strike rule"


def test_the_kernel_verdict_is_not_cleared_by_a_stale_count(dr, monkeypatch):
    """note_ok clears a COUNT, not a diagnosis. If the driver still reports the board-management
    fault, recovery must stay refused however healthy the counter looks."""
    dr.note_ok()
    monkeypatch.setattr(dr, "_kernel_tail", lambda: "tenstorrent!2: Failed to set initial power state: -22")
    tries = []
    assert dr.recover("t", lambda tgt: tries.append(tgt)) is False
    assert tries == []
