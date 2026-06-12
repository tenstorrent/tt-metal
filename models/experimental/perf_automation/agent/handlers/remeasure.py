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


def _op_count(profile):
    return sum(int(b.get("count", 0)) for b in (profile.get("buckets") or []))


def _comparable(baseline, iter_profile, tol=0.25):
    """Is the iter profile structurally comparable to the baseline? Guards
    against trusting a partial/garbage capture (e.g. tracy logging 27 ops
    instead of 308 -> a false 22x 'win'). Returns (ok, reason)."""
    b_ops = _op_count(baseline)
    if b_ops == 0:
        return True, None  # no baseline op count -> nothing to compare against
    i_ops = _op_count(iter_profile)
    ratio = i_ops / b_ops
    if not (1 - tol) <= ratio <= (1 + tol):
        return False, f"op_count_mismatch: iter {i_ops} vs baseline {b_ops} ops ({ratio:.2f}x)"
    bbuckets = baseline.get("buckets") or []
    if bbuckets:
        dom = max(bbuckets, key=lambda b: b.get("device_ms", 0)).get("id")
        iter_ids = {b.get("id") for b in (iter_profile.get("buckets") or [])}
        if dom and dom not in iter_ids:
            return False, f"dominant_bucket_missing: baseline '{dom}' absent in iter profile"
    return True, None


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

    # comparability guard: a profile structurally unlike the baseline (op count
    # collapsed, dominant bucket vanished) is an untrustworthy capture, not a win.
    measurement_ok, measurement_reason = True, None
    try:
        measurement_ok, measurement_reason = _comparable(ctx.baseline_profile(), rep)
    except Exception:  # baseline unreadable -> skip the guard, don't block
        pass

    ctx.state["last_decision"] = {
        "before": before,
        "after": after,
        "spread": spread,
        "runs": len(devs),
        "pcc": (ctx.state.get("last_verdict") or {}).get("pcc"),
        "profile": rel,
        "measurement_ok": measurement_ok,
        "measurement_reason": measurement_reason,
    }
    if not measurement_ok:
        ctx.log_event(states.REMEASURE, "warn", f"profile not comparable to baseline: {measurement_reason}")
    ctx.log_event(states.REMEASURE, "info", f"after={after} spread={spread} runs={len(devs)}")
    return states.DECIDE


def _default_runner():
    from ..measure import measure_runs

    return measure_runs
