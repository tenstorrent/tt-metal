"""A round ends when the gate allows it, not when the agent feels finished.

The band veto made can_stop false while any stack sits above its own achievable band -- but the
agent was never told. Its prompt closed with "LEAVE CLEAN ... Report start->final", and that is what
ended 8 of the 10 rounds of voxtral 2026-09-03, each with a tidy summary while can_stop was still
false and ~150 ms sat reachable in one stack. A gate the worker cannot see is not a gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
if str(PERF) not in sys.path:
    sys.path.insert(0, str(PERF))

_SRC = (PERF / "cc_optimize" / "run.py").read_text(encoding="utf-8")


def _prompt() -> str:
    i = _SRC.index('_PROMPT = """')
    return _SRC[i + 13 : _SRC.index('"""', i + 13)]


def test_the_agent_is_told_which_stacks_still_owe_the_band():
    p = _prompt()
    assert "stages_short_of_achievable" in p, "the gate's own field must reach the worker"


def test_the_prompt_still_formats():
    _prompt().format(model="m", task="t", metric="device_ms")


def test_wrapping_up_is_conditional_on_the_gate():
    """The closing instruction was unconditional, so it read as a cue to stop. It now sits behind
    finish_round(), which is a gate the tool can refuse rather than a sentence it cannot check."""
    p = _prompt()
    i = p.index("LEAVE CLEAN")
    before = p[max(0, i - 400) : i]
    assert "finished=true" in before, "LEAVE CLEAN must sit behind the verdict, not stand alone"
    assert "finish_round()" in before


def test_the_usual_reasons_to_stop_early_are_refused():
    p = _prompt()
    for reason in ("last lever failed", "progress feels slow", "exhausted"):
        assert reason in p, reason


def test_beating_the_band_is_not_a_fault():
    """Only stacks ABOVE the band are listed; going past it is the goal."""
    p = _prompt()
    assert "beating it is fine" in p.lower()


def test_a_genuine_dead_end_can_still_finish_cleanly():
    """Not an infinite loop: everything reachable measured and recorded is a real ending, and
    finish_round is told to answer finished=true for it."""
    p = _prompt()
    assert "no material op has an attempt left to try" in p
    assert "say WHICH and by how many ms" in p, "the next round has to know where to start"


def test_the_agent_may_not_end_a_round_the_gate_refused():
    p = _prompt()
    assert "Do not end a round without calling it, and do not end one it refused." in p
