# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A halt must say what happened and what to do about IT, not about the other halt.

termination_check has two halt paths and they disagreed about where the reason lives:

    tt-lang rung   {"halt": True,                 "halt_reason": "..."}
    dead device    {"halt": "needs_host_reboot",  "error": "..."}     <- no halt_reason

The supervisor reads `halt_reason`, so a dead board reported an EMPTY reason -- printed under a
message hardcoded to "install tt-lang first, then re-run". The operator was told to install a
toolchain for a card that needed a host reboot, and the run sat dead until morning. The diagnosis
existed the whole time; it was written to a key nobody read.

So: one key for the reason, and the REMEDY is looked up from the gate's own name for the condition
rather than assumed. A halt kind with no entry falls back to a stated default instead of borrowing
whichever remedy happens to be first.
"""

import importlib.util
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))
sys.path.insert(0, str(_PA.parent.parent.parent))


def _run():
    spec = importlib.util.spec_from_file_location("cc_run_halt", str(_PA / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["cc_run_halt"] = m
    spec.loader.exec_module(m)
    return m


def test_a_reboot_halt_does_not_tell_the_operator_to_install_tt_lang():
    m = _run()
    assert "reboot the host" in m._HALT_REMEDY["needs_host_reboot"]
    assert "tt-lang" not in m._HALT_REMEDY["needs_host_reboot"]
    assert "tt-lang" not in m._HALT_REMEDY["device_unrecoverable"]


def test_an_unnamed_halt_falls_back_to_a_stated_default():
    """Not to whichever remedy sorts first: a wrong instruction reads as authoritative."""
    m = _run()
    assert m._HALT_REMEDY.get("some_future_halt") is None
    assert m._HALT_REMEDY[""]  # the fallback exists and is explicit


def test_the_gate_parses_the_kind_alongside_the_reason(monkeypatch):
    m = _run()
    out = "CANSTOP=False\nHALT=True\nHALTKIND=needs_host_reboot\nHALTREASON=board-management fault\n"
    monkeypatch.setattr(m, "_run_device_proc", lambda *a, **k: (0, out))
    monkeypatch.setattr(m, "cc_env", lambda *a, **k: {})
    monkeypatch.setattr(m, "_python_bin", lambda r: "python")
    monkeypatch.setattr(m, "_measure_backstop", lambda r: 1)
    monkeypatch.setattr(m, "adaptive_timer", lambda *a, **k: 1)
    st = m._gate_status(Path("/nonexistent"), {}, "0")
    assert st["halt"] is True and st["kind"] == "needs_host_reboot"
    assert st["reason"] == "board-management fault"
    assert "reboot the host" in m._HALT_REMEDY[st["kind"]]


def test_the_reason_is_read_from_either_key():
    """`error` is where the device halt has always written it, and dropping that key would trade one
    silent halt for another."""
    m = _run()
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _gate_status")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "halt_reason" in body and "error" in body


def test_the_device_halt_now_carries_halt_reason():
    """Fixed at the SOURCE: both halts write the reason to the same key."""
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index('"halt": "needs_host_reboot"')
    assert '"halt_reason"' in src[i : i + 1200], "the device halt still has no halt_reason"
