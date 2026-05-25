"""Map observed evidence to suspect-confidence updates.

This module bridges :mod:`.evidence` (what we observed) and
:mod:`.hypothesis` (what we believe). It has no state of its own;
its job is to package up the pattern rules so :class:`text.TextComparator`
can drive a :class:`HypothesisState` without duplicating
domain-knowledge across modules.

The architecture is:

::

    TextEvidence (this iter)              HypothesisState (running)
            │                                       │
            ▼                                       ▼
    ┌──────────────────────────────────────────────────────┐
    │  diagnose.apply_priors(state, evidence)              │
    │      adjusts confidences once at loop start          │
    └──────────────────────────────────────────────────────┘
            │
            ▼                                       │
    diagnose.score_iteration_delta(state,           │
            evidence_before, evidence_after,        │
            edited_files) ────────────────────►  state.update_from_iteration(...)

The split between the two functions matters: at loop start we know
nothing has been tried, so the only signal is "what failure mode
SHAPE is the evidence telling us about?" (priors). After each
iteration we have a delta — same prompt, same model, but the
agent edited some files — so we can ask "did those edits actually
help?" (deltas).

The "verdict improved / worsened" decision
-----------------------------------------
We define progress as ANY of:

  * prefix_match_count grew by >= 4 tokens (the medgemma "first
    36 tokens fine" pattern would be improved to "first 50
    tokens fine").
  * collapse_position moved later by >= 8 tokens (the collapse
    is happening further in; useful even if the model still
    eventually collapses).
  * mismatch_ratio dropped by >= 0.1.

We define regression as ALL of:

  * prefix_match_count dropped by >= 4 tokens.
  * collapse_position is either unchanged or moved earlier.
  * mismatch_ratio rose by >= 0.05.

"Unchanged" (between the two) is the most common outcome.
"""

from __future__ import annotations

from typing import Any, Sequence

from .hypothesis import HypothesisState


PREFIX_GROWTH_IS_PROGRESS = 4
COLLAPSE_DELAY_IS_PROGRESS = 8
MISMATCH_DROP_IS_PROGRESS = 0.10

PREFIX_DROP_IS_REGRESSION = 4
MISMATCH_RISE_IS_REGRESSION = 0.05


def apply_priors(state: HypothesisState, evidence: Any) -> None:
    """At loop start, narrow ``state`` to suspects that match the
    observed evidence shape. Side-effect on ``state``.

    Idempotent: calling it twice with the same evidence has the
    same effect as calling it once (no compounding demotion)."""
    state.update_from_evidence_shape(evidence)


def score_iteration_delta(
    state: HypothesisState,
    *,
    evidence_before: Any,
    evidence_after: Any,
    edited_files: Sequence[str],
) -> None:
    """Apply per-iteration updates to ``state``. Side-effect.

    Improvements PROMOTE suspects whose files were edited;
    regressions DEMOTE them; null deltas demote softly (the test
    was performed and the answer was "no").
    """
    improved = _verdict_improved(evidence_before, evidence_after)
    worsened = _verdict_worsened(evidence_before, evidence_after)
    state.update_from_iteration(
        edited_files=edited_files,
        evidence_before=evidence_before,
        evidence_after=evidence_after,
        verdict_improved=improved,
        verdict_worsened=worsened,
    )


def _getattr_or(obj: Any, name: str, default: Any) -> Any:
    """Like :func:`getattr` with a default, but also substitutes the
    default when the attribute exists but is ``None``. Renamed from
    ``_safe_attr`` to avoid a naming clash with
    :func:`op_emitter._safe_attr` (which sanitizes dotted names into
    python identifiers — a completely different operation).
    """
    try:
        v = getattr(obj, name, default)
        return v if v is not None else default
    except Exception:
        return default


_safe_attr = _getattr_or


def _verdict_improved(before: Any, after: Any) -> bool:
    """ANY of the three progress signals fires."""
    p_before = _getattr_or(before, "prefix_match_count", 0)
    p_after = _getattr_or(after, "prefix_match_count", 0)
    if p_after - p_before >= PREFIX_GROWTH_IS_PROGRESS:
        return True

    cb = _getattr_or(before, "collapse_position", None)
    ca = _getattr_or(after, "collapse_position", None)

    if cb is not None and ca is None:
        return True

    if cb is not None and ca is not None and ca - cb >= COLLAPSE_DELAY_IS_PROGRESS:
        return True

    mb = _getattr_or(before, "mismatch_ratio", 0.0)
    ma = _getattr_or(after, "mismatch_ratio", 0.0)
    if mb - ma >= MISMATCH_DROP_IS_PROGRESS:
        return True

    return False


def _verdict_worsened(before: Any, after: Any) -> bool:
    """ALL of the regression signals fire."""
    p_before = _getattr_or(before, "prefix_match_count", 0)
    p_after = _getattr_or(after, "prefix_match_count", 0)
    if p_before - p_after < PREFIX_DROP_IS_REGRESSION:
        return False

    cb = _getattr_or(before, "collapse_position", None)
    ca = _getattr_or(after, "collapse_position", None)
    if cb is not None and ca is not None and ca > cb:
        return False

    mb = _getattr_or(before, "mismatch_ratio", 0.0)
    ma = _getattr_or(after, "mismatch_ratio", 0.0)
    if ma - mb < MISMATCH_RISE_IS_REGRESSION:
        return False

    return True


__all__ = [
    "COLLAPSE_DELAY_IS_PROGRESS",
    "MISMATCH_DROP_IS_PROGRESS",
    "MISMATCH_RISE_IS_REGRESSION",
    "PREFIX_DROP_IS_REGRESSION",
    "PREFIX_GROWTH_IS_PROGRESS",
    "apply_priors",
    "score_iteration_delta",
]
