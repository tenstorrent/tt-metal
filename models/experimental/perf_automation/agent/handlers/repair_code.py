"""REPAIR_CODE handler (PLAN 8.5.2) — REAL self-heal.

APPLY/VERIFY/GATE_PCC flagged a CODE problem (no edit landed, a syntax/parse
error, or a crash in the e2e test). Re-invoke the edit sub-agent WITH the error
so it fixes its own work. The repair prompt (edit_agent.build_repair_prompt)
explicitly tells the editor to KEEP the optimization, not delete it to dodge the
error (the lazy-fix guard).

Budget (MAX_CODE_FIX) is enforced UPSTREAM by whoever routes here; this handler
does ONE repair and bumps code_fix_attempts. Never crashes the loop: an editor
exception is logged and we still return to VERIFY (which re-checks ground truth).

-> VERIFY (re-validate the fixed edit).
"""

from __future__ import annotations

from .. import router, states


def _error_detail(ctx) -> str:
    v = ctx.state.get("last_verdict") or {}
    bits = [v.get("status"), v.get("error")]
    if v.get("file"):
        bits.append(f"(in {v['file']})")
    return " — ".join(str(b) for b in bits if b) or "the previous edit failed to apply"


def repair_code(ctx) -> str:
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
            error=_error_detail(ctx),
            cwd=str(ctx.model_root()),
        )
        model, usage = result.get("model", "?"), result.get("usage")
        prompt_text, response_text = result.get("prompt"), result.get("response")
    except Exception as exc:  # editor errored — edits may still be on disk; let VERIFY judge
        ctx.log_event(states.REPAIR_CODE, "warn", f"repair edit errored: {str(exc)[-300:]}")

    ctx.record_agent_call(states.REPAIR_CODE, "repair_code", model, usage, prompt=prompt_text, response=response_text)
    ctx.state["code_fix_attempts"] = ctx.state.get("code_fix_attempts", 0) + 1
    ctx.log_event(
        states.REPAIR_CODE, "info", f"code-fix attempt {ctx.state['code_fix_attempts']}/{states.MAX_CODE_FIX}"
    )
    return states.VERIFY


def _default_runner():
    from ..edit_agent import make_edit_runner

    return make_edit_runner()
