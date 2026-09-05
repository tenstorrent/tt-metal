"""Above the abort line the STEP is ended and re-run; the optimize run itself survives.

The safety ceiling holds work at a boundary, which is enough whenever boundaries come round. A
baseline measurement is one continuous job -- you cannot pause halfway through timing something --
and with a cold kernel cache it runs twelve minutes instead of two. Measured 2026-08-29: the last
boundary passed at 74C, heavy work began, and the board reached 90C in four and a half minutes and
95C over the next ten with no boundary in between. Every ceiling check was correctly silent; there
was simply nothing left to check.

Killing the whole run would be a heavy answer: the round's ledger, baseline and best-so-far are all
still good. Ending the CHILD releases the device, the launch gate cools before the relaunch, and the
caller sees one slow step instead of a failure.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]


def _load_run():
    spec = importlib.util.spec_from_file_location("cc_optimize_run", _PA / "cc_optimize" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stub(mod, hot_for):
    """Replace the inner runner with one that reports `hot_for` consecutive thermal aborts."""
    calls = {"n": 0}

    def once(*_a, **_k):
        calls["n"] += 1
        mod._THERMAL_ABORTED[0] = calls["n"] <= hot_for
        return ("rc%d" % calls["n"], "out")

    mod._run_device_proc = once
    return calls


def test_a_cool_step_is_run_exactly_once():
    mod = _load_run()
    calls = _stub(mod, hot_for=0)
    assert mod._run_device_step() == ("rc1", "out")
    assert calls["n"] == 1, "a cool step must not be retried"


def test_a_step_aborted_on_heat_is_rerun():
    mod = _load_run()
    calls = _stub(mod, hot_for=1)
    result = mod._run_device_step()
    assert calls["n"] == 2, "the step was not re-run after a thermal abort"
    assert result == ("rc2", "out"), "the caller got the aborted attempt's result, not the retry's"


def test_retries_are_bounded():
    """A board at the abort line on every attempt is not fixed by trying forever."""
    mod = _load_run()
    calls = _stub(mod, hot_for=99)
    mod._run_device_step()
    assert calls["n"] == 1 + mod._THERMAL_ABORT_RETRIES, "retries are not bounded"


def test_the_abort_flag_does_not_leak_between_calls():
    """A stale flag would make the NEXT step retry for a temperature that has already passed."""
    mod = _load_run()
    _stub(mod, hot_for=1)
    mod._run_device_step()
    assert mod._THERMAL_ABORTED[0] is False, "the abort flag was left set after the retry succeeded"


def test_the_abort_line_sits_above_the_hold_and_below_where_chips_died():
    """95C: above the 90C hold so it only fires where holding cannot, below the 98.7C failures."""
    mod = _load_run()
    mcp = mod._perf_mcp()
    assert mcp._ABORT_CEILING_C > mcp._SAFETY_CEILING_C, "the abort line must sit above the hold"
    assert mcp._ABORT_CEILING_C < 98.0, "the abort line must leave room before the board stops answering"
