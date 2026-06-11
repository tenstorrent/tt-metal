"""DECIDE handler (PLAN 8.8) — REAL, pure. keep vs discard. No agent, no device.

By the time DECIDE runs, GATE_PCC passed and REMEASURE produced before/after (and
the run spread). Keep only a real improvement per metric.direction beyond the
noise floor; otherwise discard (no_gain). Edit/run crashes never reach here —
they were absorbed by REPAIR (8.5.2); an unmeasurable lever is discarded by
REMEASURE (8.7).

NOTE: _NOISE_FLOOR is a PLACEHOLDER (see progress.txt — noise-floor deferred). The
agreed fix is to use the MEASURED run spread (last_decision["spread"]) instead of
a constant; the spread is already recorded, so swapping this is a one-line change.
"""

from __future__ import annotations

from .. import states

_NOISE_FLOOR = 0.05  # PLACEHOLDER — not a measurement; see progress.txt


def decide(ctx) -> str:
    d = ctx.state.get("last_decision") or {}
    before, after = d.get("before"), d.get("after")
    direction = ctx.state["metric"].get("direction", "min")

    improved = (
        before is not None
        and after is not None
        and (
            (direction == "min" and after <= before - _NOISE_FLOOR)
            or (direction == "max" and after >= before + _NOISE_FLOOR)
        )
    )
    d["result"] = "keep" if improved else "discard"
    if not improved:
        d["reason"] = "no_gain"
    ctx.state["last_decision"] = d
    ctx.log_event(states.DECIDE, "info", f"{d['result']} ({before} -> {after}, dir={direction})")
    return states.COMMIT if improved else states.REVERT
