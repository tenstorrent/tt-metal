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

    # Accumulate the history of already-tried-and-failed approaches and FEED it to the repair
    # agent, so it can't blindly re-author the same fix that keeps failing identically (the
    # kernel-lever cycle). Same "feed accumulated context" pattern as the off-menu knob path; the
    # repair loop previously saw only the LATEST error. The PREVIOUS attempt's approach is its edit
    # summary: APPLY records it in ctx.state['last_edit']['summary']; later repairs overwrite that.
    error = _error_detail(ctx)
    prev_approach = (ctx.state.get("last_edit") or {}).get("summary", "")
    history = ctx.state.setdefault("repair_history", [])
    history.append(
        {
            "attempt": ctx.state.get("code_fix_attempts", 0),
            "approach": prev_approach,
            "error": error,
        }
    )

    # Route to the SAME agent APPLY used: a structural lever (sharding / FROM_PRINCIPLES / the
    # tt-lang-kernel author) must be repaired by the STRUCTURAL agent, not the generic edit agent —
    # otherwise a kernel repair loses the KERNEL_TEMPLATE (and re-does a sharding edit) and the
    # measure tool. APPLY recorded which kind this is in last_was_structural.
    is_structural = bool(ctx.state.get("last_was_structural"))
    import os as _os

    # on-device validation tool: let the repair editor TEST its fix before submitting, so it stops
    # re-submitting edits that crash the same way (reuses GATE_PCC's check). Disable AGENT_NO_EDIT_CHECK=1.
    _validate = None
    if _os.environ.get("AGENT_NO_EDIT_CHECK", "").lower() not in ("1", "true", "yes"):
        from ..pcc_runner import run_pcc as _run_pcc

        _pcc = ctx.deps.get("pcc_runner") or _run_pcc
        _validate = lambda: _pcc(ctx)  # noqa: E731

    # Climb the escalation ladder: this repair is one rung above the prior attempt
    # (repair 1 -> sonnet, repair 2+ -> opus). code_fix_attempts is bumped below.
    attempt = ctx.state.get("code_fix_attempts", 0) + 1
    call_kwargs = dict(
        lever=lever,
        section=section,
        model_files=ctx.model_files(),
        error=error,
        cwd=str(ctx.model_root()),
        validate=_validate,
        attempt=attempt,
    )
    if is_structural:
        from ..edit_agent import _format_prior_attempts

        runner = ctx.deps.get("structural_runner") or _default_structural_runner()
        # the structural runner has no prior_attempts arg — fold the history INTO the error so the
        # kernel author still sees what already failed and changes the approach, not just the latest.
        if len(history) > 1:
            call_kwargs["error"] = error + "\n" + _format_prior_attempts(history)
        scoped = ctx.state.get("exec_scoped_files")
        call_kwargs["model_files"] = scoped if scoped else ctx.model_files()
        call_kwargs["top_ops"] = ctx.state.get("top_ops") or []
        call_kwargs["hot_sources"] = ctx.state.get("hot_sources") or []
        if lever == "tt-lang-kernel" and _os.environ.get("AGENT_NO_PERF_CHECK", "").lower() not in (
            "1",
            "true",
            "yes",
        ):
            from .apply import _measure_device_ms

            call_kwargs["measure"] = lambda: _measure_device_ms(ctx)
            call_kwargs["baseline_ms"] = (ctx.state.get("metric") or {}).get("current")
    else:
        runner = ctx.deps.get("edit_runner") or _default_runner()
        call_kwargs["prior_attempts"] = history
    model, usage, prompt_text, response_text = "?", None, None, None
    try:
        result = runner(**call_kwargs)
        model, usage = result.get("model", "?"), result.get("usage")
        prompt_text, response_text = result.get("prompt"), result.get("response")
        # record THIS attempt's approach where the next repair reads it (same key APPLY writes)
        ctx.state.setdefault("last_edit", {})["summary"] = result.get("summary", "")
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


def _default_structural_runner():
    from ..structural_agent import make_structural_runner

    return make_structural_runner()
