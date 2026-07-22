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
from ..opclass import STRUCTURAL_OP_CLASSES
from ..probes import PerfRunFailed


def _op_count(profile):
    return sum(int(b.get("count", 0)) for b in (profile.get("buckets") or []))


def _bucket_count(profile, bucket_id):
    return sum(int(b.get("count", 0)) for b in (profile.get("buckets") or []) if b.get("id") == bucket_id)


def _comparable(baseline, iter_profile, tol=0.25, floor_ms=None, floor_margin=0.5, tp_regime=False):
    """Is the iter profile a TRUSTWORTHY measurement (a complete capture of the same model), not a
    crashed/partial capture? Returns (ok, reason).

    GENERALIZED, model-agnostic discriminator (`floor_ms` given): the crash-vs-fusion question is
    answered by PHYSICS, not op-count. A device_ms BELOW the model's own roofline floor is
    physically impossible -> it's a crashed/partial capture. Above the floor, a *changed* op-count
    (even a big DROP) is a LEGITIMATE fusion/batching (e.g. MoE expert batching, FFN L1-fusion) and
    must be accepted -- on ANY model, with no per-model tuning (the floor is each model's own
    physics from roofline.py). This replaces the old op-count-DROP heuristic, which false-positived
    on legitimate op-reducing optimizations. `floor_margin` (one generic constant) absorbs
    roofline-ESTIMATE imprecision.

    Op-count INFLATION stays a hard reject (it's a measurement double-count glitch, never an
    optimization). When `floor_ms` is None (caller has no roofline), fall back to the old op-count-
    drop heuristic for backward compatibility (conservative)."""
    if iter_profile.get("capture_partial"):
        return False, f"partial_capture: profiler dropped markers ({iter_profile.get('capture_partial')})"
    b_ops = _op_count(baseline)
    if b_ops == 0:
        return True, None  # no baseline op count -> nothing to compare against
    i_ops = _op_count(iter_profile)
    if tp_regime:
        i_ops -= _bucket_count(iter_profile, "ccl")
    ratio = i_ops / b_ops
    if ratio > (1 + tol):
        return False, f"op_count_inflated: iter {i_ops} vs baseline {b_ops} ops ({ratio:.2f}x)"

    b_pt = float(baseline.get("per_token_ms") or 0.0)
    i_pt = float(iter_profile.get("per_token_ms") or 0.0)
    bfw = float(baseline.get("forward_wall_ms") or 0.0)
    ifw = float(iter_profile.get("forward_wall_ms") or 0.0)
    b_dev = float(baseline.get("device_ms") or 0.0)
    i_dev = float(iter_profile.get("device_ms") or 0.0)
    if b_pt > 0 and i_pt > 0:
        ref_base, ref_iter, ref_kind = b_pt, i_pt, "trace per-token"
    elif bfw > 0 and ifw > 0:
        ref_base, ref_iter, ref_kind = bfw, ifw, "end-to-end forward wall"
    else:
        ref_base = ref_iter = 0.0
        ref_kind = ""
    if ref_base > 0 and b_dev > 0:
        device_drop = (b_dev - i_dev) / b_dev
        ref_drop = (ref_base - ref_iter) / ref_base
        if device_drop > 0.25 and ref_drop < 0.05:
            return False, (
                f"capture_incomplete: device_ms dropped {device_drop:.0%} ({b_dev:.2f}->{i_dev:.2f}) but "
                f"{ref_kind} held flat ({ref_base:.2f}->{ref_iter:.2f}ms, {ref_drop:.0%}) -- "
                f"partial profiler capture, not a real speedup"
            )

    if floor_ms:
        # PHYSICS guard: below the model's roofline floor == impossible == crashed/partial capture.
        i_ms = float(iter_profile.get("device_ms") or 0.0)
        if i_ms > 0 and i_ms < floor_ms * floor_margin:
            return False, (
                f"below_roofline_floor: {i_ms:.4f}ms < {floor_margin:.0%} of modeled floor "
                f"{floor_ms:.4f}ms -- physically impossible, crashed/partial capture (not a fusion)"
            )
        # complete capture above the floor: an op-count change here is a LEGITIMATE fusion. Accept.
        return True, None

    # BACKWARD-COMPAT (no roofline floor available): the original op-count-drop heuristic.
    icounts = {b.get("id"): int(b.get("count", 0)) for b in (iter_profile.get("buckets") or [])}
    for b in baseline.get("buckets") or []:
        bid = b.get("id")
        bc = int(b.get("count", 0))
        if bid in STRUCTURAL_OP_CLASSES and bc > 0:
            ic = icounts.get(bid, 0)
            if ic < (1 - tol) * bc:
                return False, (
                    f"structural_op_dropped: '{bid}' {bc}->{ic} ops (<{1 - tol:.0%} of baseline) -- "
                    f"partial/crashed capture (lost the forward tail), not a fusion"
                )
    bbuckets = baseline.get("buckets") or []
    if bbuckets:
        dom = max(bbuckets, key=lambda b: b.get("device_ms", 0)).get("id")
        if dom and dom not in icounts:
            return False, f"dominant_bucket_missing: baseline '{dom}' absent in iter profile"
    return True, None


def _same_op_graph(before: dict, after: dict) -> bool:
    """True iff two profiles have a byte-identical op-class signature (an inert edit never exercised by the workload)."""

    def sig(p):
        return sorted(
            (str(b.get("id")), int(b.get("count", 0)), round(float(b.get("device_ms", 0.0)), 4))
            for b in (p.get("buckets") or [])
            if b.get("id") != "host_overhead"
        )

    bsig, asig = sig(before), sig(after)
    return bool(bsig) and bsig == asig


def _op_delta_evidence(before: dict, after: dict) -> str:
    """Per-bucket count/time before->after, as measured ground truth for the inert-repair agent."""

    def by_id(p):
        return {b.get("id"): b for b in (p.get("buckets") or []) if b.get("id") != "host_overhead"}

    bb, ab = by_id(before), by_id(after)
    rows = []
    for bid in sorted(set(bb) | set(ab)):
        bc = bb.get(bid, {}).get("count", 0)
        ac = ab.get(bid, {}).get("count", 0)
        bm = round(float(bb.get(bid, {}).get("device_ms", 0.0)), 3)
        am = round(float(ab.get(bid, {}).get("device_ms", 0.0)), 3)
        tag = "" if (bc == ac and bm == am) else "  <-- CHANGED"
        rows.append(f"  {bid:12s} count {bc}->{ac}   device_ms {bm}->{am}{tag}")
    return "\n".join(rows)


def remeasure(ctx) -> str:
    before = ctx.state["metric"]["current"]
    runner = ctx.deps.get("measure_runner") or _default_runner()

    try:
        profiles = runner(ctx)
    except PerfRunFailed as exc:
        ctx.state["last_verdict"] = {"status": "crash", "error": exc.error}
        if ctx.state.get("code_fix_attempts", 0) < states.code_fix_budget(ctx.state.get("selected_lever")):
            ctx.log_event(states.REMEASURE, "warn", f"perf run crashed (repairable): {exc.error}")
            return states.REPAIR_CODE
        ctx.state["last_decision"] = {
            "result": "discard",
            "reason": "edit_failed",
            "before": before,
            "error": exc.error,
        }
        ctx.log_event(states.REMEASURE, "warn", f"perf run crashed, repair budget exhausted: {exc.error}")
        return states.REVERT
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

    metric_name = (ctx.state.get("metric") or {}).get("name", "device_ms")
    vals = [p.get(metric_name, p["device_ms"]) for p in profiles]
    median_val = statistics.median(vals)
    after = round(median_val, 4)
    spread = round(max(vals) - min(vals), 4) if len(vals) > 1 else 0.0
    rep = min(profiles, key=lambda p: abs(p.get(metric_name, p["device_ms"]) - median_val))  # representative profile

    rel = f"profiles/iter_{ctx.state.get('iteration', 0):02d}_profile.json"
    (ctx.run.dir / rel).write_text(json.dumps(rep, indent=2, sort_keys=True))

    # comparability guard: a profile structurally unlike the baseline (op count
    # collapsed, dominant bucket vanished) is an untrustworthy capture, not a win.
    measurement_ok, measurement_reason = True, None
    try:
        base = ctx.baseline_profile()
        # GENERALIZED guard: compute this model's roofline floor (physics) so the crash-vs-fusion
        # call is physics-based, not op-count-based (which false-positives on legitimate fusions).
        floor_ms = None
        try:
            from .. import roofline

            floor_ms = roofline.residual_report(base, (ctx.manifest or {}).get("env", {})).get("modeled_floor_ms")
        except Exception:
            pass
        measurement_ok, measurement_reason = _comparable(base, rep, floor_ms=floor_ms)
    except Exception:  # baseline unreadable -> skip the guard, don't block
        pass

    op_graph_identical = False
    op_delta = None
    if metric_name == "device_ms":
        try:
            op_graph_identical = _same_op_graph(ctx.current_profile(), rep)
            op_delta = _op_delta_evidence(ctx.current_profile(), rep)
        except Exception:
            pass

    ctx.state["last_decision"] = {
        "before": before,
        "after": after,
        "spread": spread,
        "runs": len(vals),
        "pcc": (ctx.state.get("last_verdict") or {}).get("pcc"),
        "profile": rel,
        "measurement_ok": measurement_ok,
        "measurement_reason": measurement_reason,
        "op_graph_identical": op_graph_identical,
        "op_delta": op_delta,
    }
    if not measurement_ok:
        ctx.log_event(states.REMEASURE, "warn", f"profile not comparable to baseline: {measurement_reason}")
    _counts = {b.get("id"): b.get("count") for b in (rep.get("buckets") or [])}
    ctx.log_event(
        states.REMEASURE,
        "info",
        f"after={after} spread={spread} runs={len(vals)} "
        f"counts(matmul={_counts.get('matmul')},datamove={_counts.get('datamove')},eltwise={_counts.get('eltwise')})",
    )
    return states.DECIDE


def _default_runner():
    from ..measure import measure_runs

    return measure_runs
