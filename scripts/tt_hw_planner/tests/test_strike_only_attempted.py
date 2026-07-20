"""Pin the 2026-06-04 design fix: the consec-same-class strike counter
must only increment for modules the LLM actually attempted this iter.

Background: seamless-m4t 2026-06-03 run retired 3 of 5 cap-out
components after only 2 LLM attempts each because the strike counter
incremented every iter the module sat in the failure list — including
the many iters where the LLM was working on other modules. With the
new gate, untouched modules are frozen at their existing strike count
and pytest still runs (regression detection intact).
"""

from __future__ import annotations

import re
from pathlib import Path

_AUTO_ITER_PATH = Path(__file__).resolve().parent.parent / "_cli_helpers" / "auto_iterate.py"


def _source() -> str:
    return _AUTO_ITER_PATH.read_text(encoding="utf-8")


def test_strike_loop_gates_on_attempted_this_iter() -> None:
    """The post-iter pytest analysis loop that increments
    `consecutive_same_class_attempts` must check whether the failed
    component was actually attempted this iter. Untouched modules must
    `continue` without incrementing."""
    src = _source()
    # The strike loop is the one at the post-iter pytest analysis site
    # that calls `_record_failure_for_component(failed_comp, ...)`.
    loop_idx = src.find("for failed_comp in set(last_failed_components or []):")
    assert loop_idx != -1, "post-iter strike loop not found"

    # Look at the ~1500 chars BEFORE the loop — the gate setup must be
    # right above it (attempted_this_iter set construction).
    pre_loop = src[max(0, loop_idx - 2000) : loop_idx]
    assert "attempted_this_iter" in pre_loop, (
        "strike loop must be preceded by an `attempted_this_iter` set "
        "that aggregates iter_target_component + _parallel_extra_jobs"
    )
    assert "iter_target_component" in pre_loop, "`attempted_this_iter` must include iter_target_component"
    assert "_parallel_extra_jobs" in pre_loop, "`attempted_this_iter` must include parallel-extra components"

    # And the body of the loop must have the gate that skips non-attempted
    # components.
    body = src[loop_idx : loop_idx + 2000]
    assert "attempted_this_iter" in body, "gate must reference attempted_this_iter"
    # Specifically: the `continue` must fire when failed_comp not in attempted_this_iter
    gate_pat = re.compile(
        r"if\s+attempted_this_iter\s+and\s+failed_comp\s+not\s+in\s+attempted_this_iter\s*:",
    )
    gate_match = gate_pat.search(body)
    assert gate_match, (
        "gate must start with `if attempted_this_iter and failed_comp "
        "not in attempted_this_iter:` (so untouched modules skip the "
        "strike-counter increment)"
    )
    # And there must be a `continue` somewhere in the next ~800 chars
    # (after the gate's `if`, any number of comment lines, then
    # `continue`). Don't pin the exact spacing.
    body_after_gate = body[gate_match.end() : gate_match.end() + 800]
    assert "continue" in body_after_gate.split("\n_record_failure_for_component")[0], (
        "gate body must `continue` (skip the strike increment) for "
        "untouched components before falling through to "
        "_record_failure_for_component"
    )


def test_parallel_extras_inherit_iter_model() -> None:
    """Reminder pin: parallel-extras inherit the primary target's model
    via `model=_iter_model`. This is what makes the tier ladder apply
    to them. If this changes, the new strike-gate test must be revised
    (the attended set composition depends on parallel_extra_jobs)."""
    src = _source()
    body_start = src.find("_parallel_extra_jobs.append(")
    assert body_start != -1
    body = src[body_start : body_start + 1500]
    assert "model=_iter_model" in body, (
        "parallel-extras must inherit `_iter_model`; if changed, the "
        "strike-gate fix needs review because parallel-extras may now "
        "have their own tier picks"
    )


def test_strike_loop_falls_back_when_parallel_extras_not_in_scope() -> None:
    """Defensive: on early-exit / recovery code paths, `_parallel_extra_jobs`
    may not be defined. The gate must NOT crash with NameError and must
    NOT freeze every counter (which would stall the loop). Behavior on
    NameError: fall back to legacy `increment all` so we don't lose
    forward progress on edge cases."""
    src = _source()
    loop_idx = src.find("for failed_comp in set(last_failed_components or []):")
    pre_loop = src[max(0, loop_idx - 2000) : loop_idx]
    assert "except NameError" in pre_loop, (
        "attempted_this_iter construction must guard against NameError "
        "(parallel_extra_jobs undefined on some code paths)"
    )
    # And on NameError it must reset to empty set so the `if attempted_this_iter`
    # check short-circuits to legacy behavior
    nameerror_idx = pre_loop.find("except NameError")
    nameerror_body = pre_loop[nameerror_idx : nameerror_idx + 400]
    assert "attempted_this_iter = set()" in nameerror_body, (
        "on NameError, attempted_this_iter must reset to empty set so "
        "the `if attempted_this_iter and ...` gate falls through to "
        "legacy increment-all behavior — not freeze every counter"
    )
