"""Mock handlers — the walking skeleton (PLAN 8.11).

These let the engine walk SELECT -> ... -> DONE TODAY with no API key and no
hardware, so integration is "swap one mock for a real module and stay green."

IMPORTANT for the team: the CONTROL FLOW here (the `return states.X` routing,
the repair-counter checks) is the real design — keep it. Replace only the leaf
`_work` functions marked `# MOCK leaf` / `# TODO`. That is what keeps the state
graph correct-by-construction while you fill in the actual work.
"""

from __future__ import annotations

import json

from .. import states

_MOCK_USAGE = {"tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0, "latency_s": 0.0}


# ============================ Member 1 — decide & act ========================
def select(ctx) -> str:
    # TODO(member1): query() over ctx.state["candidates"], enum-constrained to
    # the closed list, fallback to untried[0] on invalid/limit. Record agent call.
    cands = ctx.state.get("candidates") or []
    tried = set(ctx.state.get("tried") or [])
    untried = [c for c in cands if c not in tried]
    ctx.state["selected_lever"] = untried[0] if untried else (cands[0] if cands else None)
    ctx.state["code_fix_attempts"] = 0  # counters reset per NEW lever
    ctx.state["pcc_fix_attempts"] = 0
    ctx.record_agent_call(states.SELECT, "select", "mock", _MOCK_USAGE)
    return states.APPLY


def apply(ctx) -> str:
    # TODO(member1): record git_sha_clean (clean HEAD), then edit sub-agent
    # applies ctx.state["selected_lever"] to manifest.pathmap.model_files.
    ctx.state["git_sha_clean"] = ctx.state.get("git_sha_clean") or "MOCKSHA"
    return states.VERIFY


def verify(ctx) -> str:
    verdict = _verify_edit(ctx)  # leaf
    ctx.state["last_verdict"] = verdict
    if verdict["status"] == "ok":
        return states.GATE_PCC
    if ctx.state.get("code_fix_attempts", 0) < states.MAX_CODE_FIX:
        return states.REPAIR_CODE
    ctx.state["last_decision"] = {"result": "discard", "reason": "edit_failed"}
    return states.REVERT


def _verify_edit(ctx) -> dict:
    # TODO(member1): ast.parse each edited file, then import in a subprocess.
    return {"status": "ok"}  # MOCK leaf: "ok" | "parse_error" | "import_error"


def repair_code(ctx) -> str:
    # TODO(member1): edit sub-agent fixes from ctx.state["last_verdict"]["error"];
    # prompt MUST say "keep the optimization, do not delete it" (lazy-fix guard).
    ctx.state["code_fix_attempts"] = ctx.state.get("code_fix_attempts", 0) + 1
    ctx.record_agent_call(states.REPAIR_CODE, "repair_code", "mock", _MOCK_USAGE)
    return states.VERIFY


def repair_pcc(ctx) -> str:
    # TODO(member1): edit sub-agent re-applies more conservatively (dtype/memcfg).
    ctx.state["pcc_fix_attempts"] = ctx.state.get("pcc_fix_attempts", 0) + 1
    ctx.record_agent_call(states.REPAIR_PCC, "repair_pcc", "mock", _MOCK_USAGE)
    return states.VERIFY


# ============================ Member 2 — evaluate ============================
def gate_pcc(ctx) -> str:
    v = _measure_pcc(ctx)  # leaf
    ctx.state["last_verdict"] = v
    if v["status"] == "ok":
        return states.REMEASURE
    if v["status"] == "pcc_low":
        if ctx.state.get("pcc_fix_attempts", 0) < states.MAX_PCC_FIX:
            return states.REPAIR_PCC
        ctx.state["last_decision"] = {"result": "discard", "reason": "pcc_failed", "pcc": v.get("pcc")}
        return states.REVERT
    # crash -> code repair
    if ctx.state.get("code_fix_attempts", 0) < states.MAX_CODE_FIX:
        return states.REPAIR_CODE
    ctx.state["last_decision"] = {"result": "discard", "reason": "edit_failed"}
    return states.REVERT


def _measure_pcc(ctx) -> dict:
    # TODO(member2): run e2e PCC test, parse number, compare manifest threshold
    # (manifest.pathmap.pcc.end_to_end.threshold). Exception -> {"status":"crash"}.
    return {"status": "ok", "pcc": 0.999}  # MOCK leaf: "ok" | "pcc_low" | "crash"


def remeasure(ctx) -> str:
    # TODO(member2): median-of-N tracy_tool runs + noise floor; a crash HERE is
    # infra (post-PCC) -> retry once, else discard reason="measure_failed".
    before = ctx.state["metric"]["current"]
    after = _measure_after(ctx, before)  # leaf
    profile_rel = _write_iter_profile(ctx)  # re-bucketed profile of the edited model
    ctx.state["last_decision"] = {
        "before": before,
        "after": after,
        "pcc": (ctx.state.get("last_verdict") or {}).get("pcc"),
        "profile": profile_rel,  # promoted to current_profile only on keep (COMMIT)
    }
    return states.DECIDE


def _write_iter_profile(ctx) -> str:
    # MOCK leaf: real REMEASURE gets full buckets from tracy_tool. Here we copy the
    # current profile and shrink the bucket we just attacked, so the bottleneck moves
    # and the NEXT ROUTE sees it (proving ROUTE reads current, not the frozen baseline).
    prof = json.loads(json.dumps(ctx.current_profile()))
    for b in prof.get("buckets", []):
        if b["id"] == ctx.state.get("current_bucket"):
            b["device_ms"] = round(b.get("device_ms", 0.0) * 0.5, 4)
    rel = f"profiles/iter_{ctx.state.get('iteration', 0):02d}_profile.json"
    (ctx.run.dir / rel).write_text(json.dumps(prof, indent=2, sort_keys=True))
    return rel


def _measure_after(ctx, before) -> float:
    # MOCK leaf: pretend each kept lever improves toward target so the skeleton
    # terminates at DONE. Real version returns the median device_ms.
    target = ctx.state["metric"].get("target")
    floor_target = target if target is not None else before - 1.0
    return round(max(floor_target, before - 1.0), 4)


def decide(ctx) -> str:
    # TODO(member2): pure keep/discard on (before, after, direction, floor).
    d = ctx.state.get("last_decision") or {}
    before, after = d.get("before"), d.get("after")
    direction = ctx.state["metric"].get("direction", "min")
    floor = 0.05
    improved = (
        before is not None
        and after is not None
        and ((direction == "min" and after <= before - floor) or (direction == "max" and after >= before + floor))
    )
    d["result"] = "keep" if improved else "discard"
    if not improved:
        d["reason"] = "no_gain"
    ctx.state["last_decision"] = d
    return states.COMMIT if improved else states.REVERT


def commit(ctx) -> str:
    # TODO(member2): git commit the kept edit; set git_sha_clean = new HEAD.
    # Promote this iteration's profile so the NEXT ROUTE routes on the new bottleneck.
    prof = (ctx.state.get("last_decision") or {}).get("profile")
    if prof:
        ctx.state["current_profile"] = prof
    return states.LOG


def revert(ctx) -> str:
    # TODO(member2): git reset --hard ctx.state["git_sha_clean"].
    return states.LOG
