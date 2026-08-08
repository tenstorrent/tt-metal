"""I-5 · Counters & exit policy (PLAN section 5, CHECK_EXIT).

Pure function over the state dict. The goal metric is named and DIRECTIONAL
(PLAN section 5): wall_ms (min), fps (max), throughput_tok_s (max), ... The
direction lives in the state's `metric` block and is never inferred from the
metric name.

  metric = {name, unit, direction, baseline, current, target}
  "target met":  direction == "min" -> current <= target
                 direction == "max" -> current >= target

Precedence:
  1. target met            -> DONE
  2. budget exceeded       -> STOPPED
  3. max iterations hit    -> STOPPED
  4. no untried levers      -> STOPPED (floor)
  5. otherwise             -> continue

Counters (iteration, cost_usd) live in state.json and survive resume.
"""

from __future__ import annotations

from typing import Any, Literal

ExitDecision = Literal["continue", "DONE", "STOPPED"]

_VALID_DIRECTIONS = ("min", "max")


def _target_met(metric: dict[str, Any]) -> bool:
    """True if the directional metric has reached its target."""
    current = metric.get("current")
    target = metric.get("target")
    if current is None or target is None:
        return False
    direction = metric.get("direction")
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"metric.direction must be one of {_VALID_DIRECTIONS}, got {direction!r}")
    if direction == "min":
        return current <= target
    return current >= target


def check_exit(state: dict[str, Any]) -> ExitDecision:
    """Decide continue / DONE / STOPPED from checkpoint counters + metric block."""
    # 1. Goal metric reached (per direction).
    if _target_met(state.get("metric") or {}):
        return "DONE"

    # 2. Cost budget exhausted.
    budget = state.get("budget_usd")
    if budget is not None and state.get("cost_usd", 0.0) >= budget:
        return "STOPPED"

    # 3. Iteration cap hit.
    max_iter = state.get("max_iter")
    if max_iter is not None and state.get("iteration", 0) >= max_iter:
        return "STOPPED"

    # 4. No untried levers left for the current bucket (floor).
    candidates = state.get("candidates") or []
    tried = set(state.get("tried") or [])
    if candidates and all(c in tried for c in candidates):
        return "STOPPED"

    return "continue"
