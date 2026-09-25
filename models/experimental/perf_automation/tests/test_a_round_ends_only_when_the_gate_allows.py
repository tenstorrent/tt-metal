"""Ending a round goes through the gate, the same way ending a run does.

can_stop is enforced in CODE: the loop reads it and will not exit while it is false. The round-level
condition was only a sentence in the prompt, and nothing checked it -- the loop calls the agent, the
agent's process exits, and `rounds += 1` accepts that unconditionally.

voxtral 2026-09-04: thirteen consecutive rounds ended with the agent writing a progress summary while
the prompt stack sat 125 ms above its band. Every one was accepted, because no code ever asked.
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
for _p in (PERF, PERF / "cc_optimize"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_spec = _ilu.spec_from_file_location("_pm_finish", PERF / "cc_optimize" / "perf_mcp.py")
_pm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pm)


def _finish():
    f = _pm.finish_round
    for a in ("fn", "func", "_fn", "__wrapped__"):
        if hasattr(f, a):
            return getattr(f, a)
    return f


SHORT = [{"stage": "prompt", "measured_ms": 159.5, "achievable_ms": 34.8, "over_by_ms": 124.6, "binds": "memory"}]


def _arrange(monkeypatch, short, left):
    monkeypatch.setattr(_pm, "_stages_short_of_achievable", lambda: list(short))
    monkeypatch.setattr(_pm, "_untried_material_ops", lambda b, a: list(left))
    monkeypatch.setattr(_pm, "_load_attempts_all", lambda: [])
    monkeypatch.setattr(_pm, "_record_round_finish", lambda v: None)


def test_a_round_that_ends_early_is_refused(monkeypatch):
    """THE DEFECT. Stacks owe their band and ops remain untried -- that is not an ending."""
    _arrange(monkeypatch, SHORT, [{"op": "A"}, {"op": "B"}])
    r = _finish()()
    assert r["finished"] is False
    assert "NOT FINISHED" in r["why"]
    assert "prompt" in r["why"], "it must name the stack that still owes"
    assert "124.6ms over" in r["why"], "and by how much"


def test_the_refusal_hands_back_the_work(monkeypatch):
    """A refusal that does not say what to do next is just an obstruction."""
    _arrange(monkeypatch, SHORT, [{"op": "A"}, {"op": "B"}])
    r = _finish()()
    assert r["untried_material_ops"], "the ops with no attempt must come back with the refusal"
    assert r["stages_short_of_achievable"] == SHORT
    assert "top of the LOOP" in r["why"]


def test_stacks_short_with_nothing_left_is_still_refused(monkeypatch):
    """BEHAVIOUR CHANGE (2026-09-06), deliberate. This used to be the "genuine ending" exit and it
    was the one that fired: five voxtral rounds in a row ended on it with prefill 101.8 ms above its
    band, each reported as a clean finish. A band that cannot be reached is not a band that is met.
    The round still ends -- the agent exits regardless -- but it ends recorded as refused, and the
    run stops on its budget saying the work was unfinished."""
    _arrange(monkeypatch, SHORT, [])
    r = _finish()()
    assert r["finished"] is False
    assert "above its own achievable band" in r["why"]


def test_every_stack_in_band_with_rungs_left_is_refused(monkeypatch):
    """The other half. In band is not finished while ops still owe a rung they have never tried."""
    _arrange(monkeypatch, [], [{"op": "A"}])
    r = _finish()()
    assert r["finished"] is False
    assert "rung they are next owed" in r["why"]


def test_both_halves_met_ends_cleanly(monkeypatch):
    _arrange(monkeypatch, [], [])
    r = _finish()()
    assert r["finished"] is True
    assert "inside its achievable band" in r["why"] and "nothing reachable is left" in r["why"]


def test_a_check_that_cannot_run_does_not_itself_refuse(monkeypatch):
    """A broken reader must not ADD a reason to refuse -- it reports nothing outstanding, so the
    verdict rests on the band alone, which is measured separately and still works."""

    def _boom(*a, **k):
        raise RuntimeError("gate unavailable")

    monkeypatch.setattr(_pm, "_stages_short_of_achievable", list)
    monkeypatch.setattr(_pm, "_untried_material_ops", _boom)
    monkeypatch.setattr(_pm, "_record_round_finish", lambda v: None)
    r = _finish()()
    assert r["finished"] is True, "unable to tell must not mean unable to finish"


def test_the_verdict_is_recorded_for_the_loop(monkeypatch, tmp_path):
    """The loop must be able to tell a round that ASKED from one that just exited."""
    monkeypatch.setattr(_pm, "_round_finish_path", lambda: tmp_path / "finish.json")
    monkeypatch.setattr(_pm, "_stages_short_of_achievable", lambda: [])
    _finish()()
    got = _pm.read_round_finish()
    assert got.get("finished") is True and got.get("at")


def test_no_verdict_reads_as_never_asked(monkeypatch, tmp_path):
    monkeypatch.setattr(_pm, "_round_finish_path", lambda: tmp_path / "absent.json")
    assert _pm.read_round_finish() == {}


def test_the_prompt_routes_the_ending_through_it():
    src = (PERF / "cc_optimize" / "run.py").read_text(encoding="utf-8")
    i = src.index('_PROMPT = """')
    prompt = src[i + 13 : src.index('"""', i + 13)]
    assert "finish_round()" in prompt
    j = prompt.index("LEAVE CLEAN")
    assert "finished=true" in prompt[max(0, j - 200) : j], "LEAVE CLEAN must sit behind the verdict"


def test_the_loop_notices_a_round_that_never_asked():
    src = (PERF / "cc_optimize" / "run.py").read_text(encoding="utf-8")
    assert "_last_round_finish(" in src
    assert "WITHOUT calling finish_round" in src
    assert "finish_round REFUSED it" in src
