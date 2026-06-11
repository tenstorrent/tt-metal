"""REMEASURE handler (PLAN 8.7) — REAL median + variance + iter profile.

Re-profiles the edited model (measurement injectable via ctx.deps["measure_runner"]),
takes the MEDIAN device_ms over the runs, records the run-to-run spread (for the
deferred noise-floor decision), and writes the re-bucketed iter profile that COMMIT
promotes to current_profile. A measurement crash is infra (post-PCC) -> discard
reason measure_failed (the one crash path that does NOT go to REPAIR).
"""

from __future__ import annotations

import json
import statistics

from .. import states


def remeasure(ctx) -> str:
    before = ctx.state["metric"]["current"]
    runner = ctx.deps.get("measure_runner") or _default_runner()

    try:
        profiles = runner(ctx)
    except Exception as exc:  # infra flake, not an edit bug
        ctx.state["last_decision"] = {
            "result": "discard",
            "reason": "measure_failed",
            "before": before,
            "error": str(exc),
        }
        ctx.log_event(states.REMEASURE, "warn", f"measure failed: {exc}")
        return states.REVERT
    if not profiles:
        ctx.state["last_decision"] = {"result": "discard", "reason": "measure_failed", "before": before}
        return states.REVERT

    devs = [p["device_ms"] for p in profiles]
    median_dev = statistics.median(devs)
    after = round(median_dev, 4)
    spread = round(max(devs) - min(devs), 4) if len(devs) > 1 else 0.0
    rep = min(profiles, key=lambda p: abs(p["device_ms"] - median_dev))  # representative profile

    rel = f"profiles/iter_{ctx.state.get('iteration', 0):02d}_profile.json"
    (ctx.run.dir / rel).write_text(json.dumps(rep, indent=2, sort_keys=True))

    ctx.state["last_decision"] = {
        "before": before,
        "after": after,
        "spread": spread,
        "runs": len(devs),
        "pcc": (ctx.state.get("last_verdict") or {}).get("pcc"),
        "profile": rel,
    }
    ctx.log_event(states.REMEASURE, "info", f"after={after} spread={spread} runs={len(devs)}")
    return states.DECIDE


def _default_runner():
    from ..measure import measure_runs

    return measure_runs
