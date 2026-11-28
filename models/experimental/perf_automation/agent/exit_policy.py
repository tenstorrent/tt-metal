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
  2. max iterations hit    -> STOPPED
  3. no untried levers      -> STOPPED (floor)
  4. otherwise             -> continue

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


def _band_decision(state: dict[str, Any]) -> ExitDecision | None:
    """Band-driven stop (plan §3.3), opt-in via ``state['target_band']``. Consumes the
    perf_target status: IN_BAND -> DONE (reached the achievable ceiling); BELOW_BAND /
    ABOVE_BAND -> continue (ladder keeps escalating; ABOVE_BAND additionally flags active_bytes/
    floor as suspect, asserted upstream — never a stop). UNKNOWN falls through to the existing
    policy. Returns None when the band is not enabled or not decisive, so default behaviour is
    unchanged."""
    if not state.get("target_band"):
        return None
    status = str(state.get("perf_status") or "").upper()
    if status == "IN_BAND":
        return "DONE"
    if status in ("BELOW_BAND", "ABOVE_BAND"):
        return "continue"
    return None


def check_exit(state: dict[str, Any]) -> ExitDecision:
    """Decide continue / DONE / STOPPED from checkpoint counters + metric block."""
    band = _band_decision(state)
    if band is not None:
        return band

    # 1. Goal metric reached (per direction).
    if _target_met(state.get("metric") or {}):
        return "DONE"

    # 2. Iteration cap hit.
    max_iter = state.get("max_iter")
    if max_iter is not None and state.get("iteration", 0) >= max_iter:
        return "STOPPED"

    # 3. Per-bucket lever exhaustion is handled by the CHECK_EXIT handler, which
    # advances to the next-slowest bucket (via exhausted_buckets) and only STOPs
    # once ALL buckets are exhausted. So it is intentionally NOT a hard stop here.

    return "continue"
