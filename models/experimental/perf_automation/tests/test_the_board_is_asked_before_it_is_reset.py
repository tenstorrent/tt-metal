# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A timeout means SLOW. Resetting four healthy chips over it is what broke this board three times.

WHAT HAPPENED, runs 13, 17 and 20. An op ran longer than its budget, and the watchdog did this:

    termination_check KILLED after 1806s (hard limit) (likely a device wedge / leaked mesh)
      -- killed the whole process group + reclaimed device + tt-smi -r 0,1,2,3 rc=0

Nothing looked at the device. `reset_on_timeout: bool = True` is a default with no check behind it,
and "likely a device wedge" is a fixed string in the message, not a conclusion. The board was healthy;
`error_text` is consulted only to choose WHICH chips to reset, never WHETHER to.

The reset then produced `Failed to set initial power state: -22` -- a board-management fault no PCIe
reset clears. Measured 2026-08-15: once in that state a further `tt-smi -r` hung for 300 s, was
killed, and the fault count rose 30 -> 34. Only a host reboot recovered it. So the reset is not a
neutral act, and firing one on no evidence costs a machine.

THE EVIDENCE IS CHEAP. On this host a live board answers in 0.24 s and a wedged one does not answer
at all -- both measured, before and after a reboot. That is a clean discriminator and the run already
asks the same question at startup.

WHY NOT THE STARTUP PROBE ITSELF. tt_smi_probe raises on an unrecognised board_type, and it rejects
this host's own `p300c`. A board that answers with a name the table lacks is ALIVE; treating that as
death would reset healthy hardware, which is the failure being fixed. Liveness asks only whether
tt-smi came back naming any device.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

_RUN = (_PA / "cc_optimize" / "run.py").read_text()
_PROBES = (_PA / "agent" / "probes.py").read_text()


def _timeout_handler():
    i = _RUN.index("except subprocess.TimeoutExpired")
    return _RUN[i : i + 2500]


def test_a_timeout_no_longer_resets_unconditionally():
    """THE BUG, in one line: the reset had no condition on the device at all."""
    body = _timeout_handler()
    assert "_device_answers()" in body, "the timeout path still resets without asking the board"


def test_a_responsive_board_is_not_reset():
    body = _timeout_handler()
    i = body.index("_device_answers()")
    assert "NOT reset" in body[i : i + 300], "a board that answered is still reset"


def test_a_silent_board_is_still_reset():
    """The recovery must survive: a genuinely wedged device still gets reclaimed."""
    body = _timeout_handler()
    assert "_reclaim_device(" in body, "a wedged board is no longer recovered"


def test_liveness_does_not_reuse_the_probe_that_rejects_this_board():
    """tt_smi_probe raises on board_type 'p300c' -- the name these very chips report."""
    i = _RUN.index("def _device_answers(")
    body = _RUN[i : _RUN.index("\ndef ", i + 1)]
    assert "device_is_responsive" in body
    assert "tt_smi_probe" not in body, "liveness would call an unrecognised board dead"


def test_the_liveness_bar_is_only_that_a_device_was_named():
    """No arch mapping, no schema: answering at all is the whole question."""
    i = _PROBES.index("def device_is_responsive(")
    body = _PROBES[i : _PROBES.index("\ndef ", i + 1)]
    assert "device_info" in body
    assert "board_to_arch" not in body, "liveness still depends on recognising the board"


def test_liveness_answers_false_rather_than_raising(monkeypatch):
    """Any failure must mean 'reset as before', never an exception inside the timeout handler."""
    from agent import probes

    monkeypatch.setattr(probes.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no tt-smi")))
    assert probes.device_is_responsive(1) is False


def test_liveness_is_bounded_well_under_tt_smis_own_timeout():
    """120 s to learn the board is dead turns diagnosis into another stall."""
    import re

    m = re.search(r'_LIVENESS_PROBE_S = float\(os\.environ\.get\("PERF_MCP_LIVENESS_PROBE_S", "(\d+)"\)\)', _RUN)
    assert m and float(m.group(1)) <= 30, "the liveness probe is not promptly bounded"


def test_an_empty_device_list_is_not_alive(monkeypatch):
    """tt-smi returning valid JSON with no devices is a machine with no board, not a live one."""
    from agent import probes

    class _P:
        stdout = '{"device_info": []}'

    monkeypatch.setattr(probes.subprocess, "run", lambda *a, **k: _P())
    assert probes.device_is_responsive(1) is False
