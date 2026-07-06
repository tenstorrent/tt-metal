"""REPAIR_PCC handler (PLAN 8.5.2) — REAL accuracy recovery.

GATE_PCC ran the e2e PCC test and the edit was numerically too aggressive: it
parsed, it ran, but it dropped model accuracy below the manifest threshold.
Re-invoke the editor asking for a MORE CONSERVATIVE re-apply (higher-precision
dtype, safer memory config) that restores accuracy while keeping as much of the
speedup as possible. We reuse the editor's repair prompt by passing a PCC-shaped
`error`, so the lazy-fix guard ("do NOT delete the optimization") still applies.

Budget (MAX_PCC_FIX) enforced UPSTREAM by GATE_PCC; this handler does ONE repair
and bumps pcc_fix_attempts. Never crashes the loop.

-> VERIFY (syntax re-check) -> GATE_PCC (re-measures PCC).
"""

from __future__ import annotations

from .. import router, states


def _threshold(ctx):
    try:
        return ctx.manifest["pathmap"]["pcc"]["end_to_end"]["threshold"]
    except (KeyError, TypeError):
        return None


def _pcc_detail(ctx) -> str:
    pcc = (ctx.state.get("last_verdict") or {}).get("pcc")
    thr = _threshold(ctx)
    msg = "The optimization is syntactically correct and runs, but it lowered model accuracy"
    if pcc is not None:
        msg += f" (measured PCC={pcc}"
        msg += f", below the required threshold {thr})" if thr is not None else ")"
    msg += (
        ". Re-apply the SAME optimization MORE CONSERVATIVELY to restore accuracy: prefer "
        "a higher-precision dtype (e.g. bfloat16/float32 instead of bfp8) or a safer memory "
        "configuration, while keeping as much of the speedup as you can. Do NOT revert the "
        "optimization entirely."
    )
    return msg


def repair_pcc(ctx) -> str:
    lever = ctx.state.get("selected_lever")
    try:
        section = router.read_section(lever) if lever else ""
    except KeyError:
        section = ""

    runner = ctx.deps.get("edit_runner") or _default_runner()
    model, usage, prompt_text, response_text = "?", None, None, None
    try:
        result = runner(
            lever=lever,
            section=section,
            model_files=ctx.model_files(),
            error=_pcc_detail(ctx),
            cwd=str(ctx.model_root()),
        )
        model, usage = result.get("model", "?"), result.get("usage")
        prompt_text, response_text = result.get("prompt"), result.get("response")
    except Exception as exc:
        ctx.log_event(states.REPAIR_PCC, "warn", f"pcc-repair edit errored: {str(exc)[-300:]}")

    ctx.record_agent_call(states.REPAIR_PCC, "repair_pcc", model, usage, prompt=prompt_text, response=response_text)
    ctx.state["pcc_fix_attempts"] = ctx.state.get("pcc_fix_attempts", 0) + 1
    ctx.log_event(states.REPAIR_PCC, "info", f"pcc-fix attempt {ctx.state['pcc_fix_attempts']}/{states.MAX_PCC_FIX}")
    return states.VERIFY


def _default_runner():
    from ..edit_agent import make_edit_runner

    return make_edit_runner()
