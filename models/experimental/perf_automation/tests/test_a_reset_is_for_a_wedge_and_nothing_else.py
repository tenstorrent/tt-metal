# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The recovery broke the board twice in one day. Both times the board was fine before it.

2026-08-17, run 7. A full-depth measurement outran its three-hour backstop and was killed -- correctly:
the watchdog watches tree CPU and the process was genuinely computing, so the stall clock kept
resetting and only the absolute bound fired. Then the recovery ran:

    tt-smi -r 0,1,2,3  rc=1   [DEVICE STILL UNHEALTHY]

and two chips stopped reporting. They were healthy seconds earlier -- all four had been publishing
97-102C throughout the measurement. The reset is what killed them.

TWO SEPARATE MISTAKES, both in one line.

FIRST, IT RESET A LIVE BOARD. The liveness question went to tt-smi, which must OPEN the device to
answer and therefore hangs exactly when the board is busy or still held by a process:

    tt-smi -s          opens the device; ~0.27 s idle, HANGS under load
    sysfs temp1_input  a file read; 0.0003 s, answered the whole time

The probe timed out, the tool concluded "dead", and reset a board whose every chip was reporting.
The cheap signal was available and unused -- and it is the one that still works precisely when the
expensive one cannot.

SECOND, IT RESET FOUR CHIPS TO RECOVER ONE. The run was MESH_DEVICE=P150, mesh 1x1: one chip. A
multi-chip reset halts each chip and brings it back in turn, so a sequence that errors partway leaves
the ones already halted DOWN -- no firmware, no telemetry. rc=1 says it errored, and the two dead
afterwards were devices 2 and 3: the tail of the list, and chips the run had never touched.

THE RULE. A reset is for a wedge. Skipping is decided from telemetry that does not touch the device,
and it can only ever CANCEL a reset, never cause one. Three states, because the middle one is real:

    every chip reports    nothing wedged   -> skip
    some chips report     partly down      -> reset (the silent ones do not recover alone)
    no chip reports       fully wedged     -> reset

Holders are reaped either way: a leaked process is worth clearing whether or not the board is sick.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _recover(monkeypatch, temps, dead=()):
    """Run the recovery with a mocked thermometer; returns the reset targets it actually attempted."""
    from agent import device_recovery as dr
    from agent import probes

    monkeypatch.setattr(probes, "board_telemetry", lambda: (list(temps), list(dead)))
    monkeypatch.setattr(dr, "reap_device_holders", lambda: [])
    monkeypatch.setattr(dr, "recovery_exhausted", lambda: False)
    monkeypatch.setattr(dr, "device_is_healthy", lambda: True)
    monkeypatch.setattr(dr, "targets_for", lambda *a, **k: ["0,1,2,3"])
    tried = []
    dr.recover("test", lambda tgt: tried.append(tgt))
    return tried


# ------------------------------------------------------------------ the three states


def test_a_board_where_every_chip_reports_is_not_reset(monkeypatch):
    """THE 2026-08-17 CASE. All four were publishing 97-102C and the tool reset them anyway."""
    assert _recover(monkeypatch, [97.0, 101.0, 97.0, 102.0]) == []


def test_a_cool_healthy_board_is_not_reset_either(monkeypatch):
    """Nothing about the temperature matters -- only whether the chip can answer at all."""
    assert _recover(monkeypatch, [55.5, 57.1, 56.0, 58.2]) == []


def test_a_board_with_a_silent_chip_IS_reset(monkeypatch):
    """The middle state, and it must not be skipped: a chip publishing all-ones does not come back
    on its own, so a majority answering is not a reason to leave half a board dead."""
    assert _recover(monkeypatch, [55.5, 57.1], dead=["hwmon5", "hwmon6"]) == ["0,1,2,3"]


def test_a_board_where_nothing_answers_IS_reset(monkeypatch):
    """The wedge a reset actually exists for."""
    assert _recover(monkeypatch, []) == ["0,1,2,3"]


# ------------------------------------------------------------------ how the decision is made


def test_the_decision_never_opens_the_device():
    """tt-smi answers by OPENING the device, so it hangs exactly when a reset is being considered --
    a busy or held board. It timed out on 2026-08-17 and a healthy board was reset on that timeout."""
    src = (_PA / "agent" / "device_recovery.py").read_text()
    i = src.index("def _board_needs_reset")
    body = src[i : src.index("\ndef ", i + 1)]
    # CODE ONLY: the docstring names tt-smi to explain why it is NOT consulted, and asserting over
    # prose would forbid recording the reason.
    body = body.split('"""', 2)[-1]
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    assert "tt_smi" not in code and "tt-smi" not in code, "the skip decision opens the device again"
    assert "board_telemetry" in code


def test_the_gate_can_only_cancel_a_reset_never_cause_one(monkeypatch):
    """It is a veto, not a trigger. If it cannot tell, the old unconditional behaviour stands --
    a recovery that refuses to run because its own check broke is worse than one reset too many."""
    from agent import device_recovery as dr

    monkeypatch.setattr(dr, "_live_temps", lambda: (_ for _ in ()).throw(OSError("no sysfs")))
    assert dr._board_needs_reset() is True


def test_holders_are_reaped_even_when_the_reset_is_skipped(monkeypatch):
    """A leaked process is worth clearing whether or not the board is sick -- and on this box a
    lingering holder is what made the previous reset fail in the first place."""
    from agent import device_recovery as dr
    from agent import probes

    reaped = {"n": 0}
    monkeypatch.setattr(probes, "board_telemetry", lambda: ([60.0, 61.0], []))
    monkeypatch.setattr(dr, "reap_device_holders", lambda: reaped.__setitem__("n", reaped["n"] + 1) or [])
    monkeypatch.setattr(dr, "recovery_exhausted", lambda: False)
    monkeypatch.setattr(dr, "targets_for", lambda *a, **k: ["0"])
    dr.recover("test", lambda tgt: None)
    assert reaped["n"] == 1, "the reap was skipped along with the reset"


def test_every_reset_path_goes_through_this_gate():
    """Five call sites reset -- three _reclaim_device in run.py, two _device_reset in perf_mcp -- and
    a rule enforced at four of them is not a rule. They all route through recover(), so the gate is
    placed there rather than at each caller."""
    src = (_PA / "agent" / "device_recovery.py").read_text()
    i = src.index("def recover(")
    body = src[i : src.index("\ndef ", i + 1) if "\ndef " in src[i + 1 :] else len(src)]
    assert "_board_needs_reset()" in body, "recover() no longer consults the gate"
    assert body.index("reap_device_holders()") < body.index("_board_needs_reset()"), "reap must precede the veto"
