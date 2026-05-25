"""G8: convergence + budget bail.

A tiny linear fit on the verdict-mismatch history. The fit decides:

  * Is the verdict moving toward zero?
  * If so, at the observed slope, will we hit zero before the budget
    runs out?
  * If no movement at all over the last K iterations, declare
    "stagnant" and bail (or escalate).

This is generic across categories because the only input is the
``mismatch_ratio`` scalar -- every comparator produces one (it's the
fraction of compared elements that don't match).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


STAGNANT_DELTA = 0.02
STAGNANT_WINDOW = 2


@dataclass
class ConvergenceVerdict:
    """Output of :func:`predict_convergence`."""

    progress_score: float
    stagnant: bool
    predicted_iters_to_zero: Optional[int]
    will_hit_zero_by: Optional[int]
    note: str


def progress_score(history: Sequence[float]) -> float:
    """Scalar in [-1, 1]: how strongly mismatch is trending to zero.

    > 0  -> trending down (good)
    ~ 0  -> stagnant
    < 0  -> getting worse"""
    if len(history) < 2:
        return 0.0

    pts = list(history[-6:])
    n = len(pts)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(pts) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, pts))
    den = sum((x - mean_x) ** 2 for x in xs) or 1e-9
    slope = num / den

    return max(-1.0, min(1.0, -slope * 10.0))


def is_stagnant(history: Sequence[float]) -> bool:
    if len(history) < STAGNANT_WINDOW + 1:
        return False
    window = history[-STAGNANT_WINDOW:]
    deltas = [abs(window[i] - window[i - 1]) for i in range(1, len(window))]
    return all(d < STAGNANT_DELTA for d in deltas)


def predict_convergence(
    history: Sequence[float],
    *,
    iters_remaining: int,
) -> ConvergenceVerdict:
    """Predict whether the current trajectory will hit zero within the
    remaining iters."""
    score = progress_score(history)
    stagnant = is_stagnant(history)
    if not history:
        return ConvergenceVerdict(
            progress_score=0.0,
            stagnant=False,
            predicted_iters_to_zero=None,
            will_hit_zero_by=None,
            note="no-history",
        )
    last = history[-1]

    pts = list(history[-6:])
    n = len(pts)
    if n < 2 or score <= 0:
        return ConvergenceVerdict(
            progress_score=score,
            stagnant=stagnant,
            predicted_iters_to_zero=None,
            will_hit_zero_by=None,
            note="no-progress" if score <= 0 else "insufficient-history",
        )
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(pts) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, pts))
    den = sum((x - mean_x) ** 2 for x in xs) or 1e-9
    slope = num / den
    intercept = mean_y - slope * mean_x
    if slope >= 0:
        return ConvergenceVerdict(
            progress_score=score,
            stagnant=stagnant,
            predicted_iters_to_zero=None,
            will_hit_zero_by=None,
            note="non-negative-slope",
        )

    iters_to_zero = -intercept / slope - (n - 1)
    iters_to_zero_int = int(max(1, round(iters_to_zero)))
    will_hit = iters_to_zero_int if iters_to_zero_int <= iters_remaining else None
    return ConvergenceVerdict(
        progress_score=score,
        stagnant=stagnant,
        predicted_iters_to_zero=iters_to_zero_int,
        will_hit_zero_by=will_hit,
        note="ok",
    )


__all__ = [
    "ConvergenceVerdict",
    "STAGNANT_DELTA",
    "STAGNANT_WINDOW",
    "is_stagnant",
    "predict_convergence",
    "progress_score",
]
