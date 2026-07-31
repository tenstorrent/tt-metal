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
from typing import Dict, List, Optional, Sequence, Tuple


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


CAP_EXTEND_BUMP = 2
CAP_EXTEND_MAX_PER_COMPONENT = 1
CAP_EXTEND_MIN_PCC_FOR_TRAJECTORY = 0.50


@dataclass(frozen=True)
class CapVerdict:
    """Output of :func:`should_extend_component_cap`.

    ``extend`` -> grant ``bump`` more attempts before falling back.
    ``reason`` -> human-readable trace for the iter banner / log."""

    extend: bool
    bump: int
    reason: str


def should_extend_component_cap(
    *,
    component: str,
    consecutive_same_class: int,
    effective_cap: int,
    pcc_history: Sequence[float],
    last_pcc: Optional[float],
    last_failure_class: str,
    graduated_this_run: Sequence[str],
    extensions_used_for_this_component: int,
) -> CapVerdict:
    """Brain (G8) decision: extend a single component's per-component
    attempt cap, or route it to CPU fallback.

    A favorable verdict requires *all* of:
      * not already extended ``CAP_EXTEND_MAX_PER_COMPONENT`` times,
      * last failure is PCC-only (crashes need a different fix),
      * current PCC clears ``CAP_EXTEND_MIN_PCC_FOR_TRAJECTORY``,
      * AND either a favorable mismatch trajectory (progress_score > 0)
        OR run-wide momentum (at least one graduation this run).
    """
    if extensions_used_for_this_component >= CAP_EXTEND_MAX_PER_COMPONENT:
        return CapVerdict(
            extend=False,
            bump=0,
            reason=(
                f"`{component}` already extended {extensions_used_for_this_component} "
                f"time(s); at per-component max ({CAP_EXTEND_MAX_PER_COMPONENT})"
            ),
        )

    if last_failure_class != "PCC_ONLY":
        return CapVerdict(
            extend=False,
            bump=0,
            reason=(
                f"`{component}` last failure class is `{last_failure_class}` "
                f"(not PCC-only); cap extension is futile for crashes/shape errors"
            ),
        )

    if last_pcc is None or last_pcc < CAP_EXTEND_MIN_PCC_FOR_TRAJECTORY:
        return CapVerdict(
            extend=False,
            bump=0,
            reason=(f"`{component}` PCC too low ({last_pcc}); needs a structural fix, " f"not more iterations"),
        )

    favorable_trajectory = progress_score(pcc_history) > 0
    if favorable_trajectory:
        return CapVerdict(
            extend=True,
            bump=CAP_EXTEND_BUMP,
            reason=(
                f"`{component}` favorable trajectory + high PCC ({last_pcc:.3f}); "
                f"granting +{CAP_EXTEND_BUMP} attempts"
            ),
        )

    if graduated_this_run:
        return CapVerdict(
            extend=True,
            bump=CAP_EXTEND_BUMP,
            reason=(
                f"`{component}` high PCC ({last_pcc:.3f}) + run momentum "
                f"({len(graduated_this_run)} graduated this run); "
                f"granting +{CAP_EXTEND_BUMP} attempts"
            ),
        )

    return CapVerdict(
        extend=False,
        bump=0,
        reason=(
            f"`{component}` high PCC ({last_pcc:.3f}) but no progress "
            f"(stagnant trajectory, no graduations this run); "
            f"extension unlikely to help"
        ),
    )


BUDGET_EXTEND_BUMP_FLOOR = 2
BUDGET_EXTEND_MAX_PENDING = 2
BUDGET_EXTEND_MAX_PER_RUN = 1


@dataclass(frozen=True)
class BudgetVerdict:
    """Output of :func:`should_extend_budget`.

    ``extend`` -> grant ``bump`` more whole-run iterations before giving up.
    ``reason`` -> human-readable trace for the AUTO-EXTEND banner / log."""

    extend: bool
    bump: int
    reason: str


def should_extend_budget(
    *,
    pending_components: Sequence[str],
    pcc_history_per_component: Dict[str, Sequence[float]],
    graduated_this_run: Sequence[str],
    max_iters: int,
    extensions_used: int,
) -> BudgetVerdict:
    """Brain (G8) run-level decision: extend the whole-run iter budget once
    more, or stop. The run-level analog of :func:`should_extend_component_cap`.

    Extends only when the residue is small enough that one more block of iters
    is plausibly decisive AND there is momentum — either a favorable mismatch
    trajectory on a pending component (``progress_score > 0``) or at least one
    graduation this run. All policy lives here so it tunes in ONE place.
    """
    pending = list(pending_components)
    bump = max(BUDGET_EXTEND_BUMP_FLOOR, max_iters // 2)

    if not pending:
        return BudgetVerdict(extend=False, bump=0, reason="no-pending: nothing left to extend the budget for")

    if extensions_used >= BUDGET_EXTEND_MAX_PER_RUN:
        return BudgetVerdict(
            extend=False,
            bump=0,
            reason=(f"already extended {extensions_used} time(s); at per-run " f"cap ({BUDGET_EXTEND_MAX_PER_RUN})"),
        )

    if len(pending) > BUDGET_EXTEND_MAX_PENDING:
        return BudgetVerdict(
            extend=False,
            bump=0,
            reason=(
                f"residue too large ({len(pending)} pending > "
                f"{BUDGET_EXTEND_MAX_PENDING}); a broadly-failing run needs a "
                f"structural fix, not one more iter block"
            ),
        )

    fav_comps = [c for c in pending if progress_score(pcc_history_per_component.get(c, []) or []) > 0]
    favorable = bool(fav_comps)
    has_momentum = bool(graduated_this_run)
    fav_str = ", ".join(fav_comps)

    if favorable and has_momentum:
        return BudgetVerdict(
            extend=True,
            bump=bump,
            reason=(
                f"favorable trajectory on {fav_str} + run momentum "
                f"({len(list(graduated_this_run))} graduations this run); +{bump} iters"
            ),
        )
    if favorable:
        return BudgetVerdict(
            extend=True,
            bump=bump,
            reason=f"favorable trajectory on {fav_str}; +{bump} iters",
        )
    if has_momentum:
        return BudgetVerdict(
            extend=True,
            bump=bump,
            reason=(
                f"run momentum ({len(list(graduated_this_run))} graduations this "
                f"run) is the proxy for progress; +{bump} iters"
            ),
        )
    return BudgetVerdict(
        extend=False,
        bump=0,
        reason="no momentum: stagnant trajectory and no graduations this run",
    )


__all__ = [
    "BUDGET_EXTEND_BUMP_FLOOR",
    "BUDGET_EXTEND_MAX_PENDING",
    "BUDGET_EXTEND_MAX_PER_RUN",
    "BudgetVerdict",
    "CAP_EXTEND_BUMP",
    "CAP_EXTEND_MAX_PER_COMPONENT",
    "CAP_EXTEND_MIN_PCC_FOR_TRAJECTORY",
    "CapVerdict",
    "ConvergenceVerdict",
    "STAGNANT_DELTA",
    "STAGNANT_WINDOW",
    "is_stagnant",
    "predict_convergence",
    "progress_score",
    "should_extend_budget",
    "should_extend_component_cap",
]
