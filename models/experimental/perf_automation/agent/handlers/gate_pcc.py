"""GATE_PCC handler (PLAN 8.6) — REAL routing, single-stage e2e.

The measurement is injectable (ctx.deps["pcc_runner"]) so this tests without
hardware; default runs the real e2e test. Routes the verdict:
  ok       -> REMEASURE
  pcc_low  -> REPAIR_PCC  (<=2), else discard (pcc_failed) -> REVERT
  crash    -> REPAIR_CODE (<=5), else discard (edit_failed) -> REVERT
"""

from __future__ import annotations

from .. import states


def gate_pcc(ctx) -> str:
    runner = ctx.deps.get("pcc_runner") or _default_runner()
    v = runner(ctx)
    ctx.state["last_verdict"] = v

    if v["status"] == "ok":
        return states.REMEASURE
    if v["status"] == "pcc_low":
        if ctx.state.get("pcc_fix_attempts", 0) < states.MAX_PCC_FIX:
            return states.REPAIR_PCC
        ctx.state["last_decision"] = {"result": "discard", "reason": "pcc_failed", "pcc": v.get("pcc")}
        return states.REVERT
    # crash
    if ctx.state.get("code_fix_attempts", 0) < states.code_fix_budget(ctx.state.get("selected_lever")):
        return states.REPAIR_CODE
    ctx.state["last_decision"] = {"result": "discard", "reason": "edit_failed"}
    return states.REVERT


def _default_runner():
    from ..pcc_runner import run_pcc

    return run_pcc
