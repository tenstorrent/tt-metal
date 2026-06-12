"""PLAN handler (PLAN 8.x) — REAL. Lead turns the chosen lever into a localized spec.

Reads the lever's playbook section + the op-class-filtered model map, lets the
lead (Read/Grep, no Edit) confirm the spot, and emits content-anchored edits.
NOOP (already applied) -> discard cheaply, no edit/gate/measure wasted.
Planner injectable via ctx.deps["plan_runner"].

In:  ctx.state["selected_lever"], ["current_bucket"]. Out: ctx.state["edit_spec"]. -> APPLY | REVERT
"""

from __future__ import annotations

from .. import model_map, router, states


def plan(ctx) -> str:
    lever = ctx.state.get("selected_lever")
    try:
        section = router.read_section(lever) if lever else ""
    except KeyError:
        section = ""

    subs = model_map.OP_CLASS_SUBSTRINGS.get(ctx.state.get("current_bucket") or "")
    try:
        mm = model_map.build_model_map(ctx.model_files(), root=ctx.model_root())
        skeleton = model_map.render_skeleton(mm, op_substrings=subs)
    except Exception:
        skeleton = ""

    runner = ctx.deps.get("plan_runner") or _default_runner()
    try:
        spec = runner(lever=lever, section=section, skeleton=skeleton, cwd=str(ctx.model_root()))
    except Exception as exc:  # planning failed -> let APPLY improvise from the lever
        ctx.state["edit_spec"] = None
        ctx.log_event(states.PLAN, "warn", f"plan failed; APPLY will improvise: {exc}")
        return states.APPLY

    ctx.record_agent_call(
        states.PLAN,
        "plan",
        spec.get("model", "?"),
        spec.get("usage"),
        prompt=spec.get("prompt"),
        response=spec.get("response"),
    )
    edits = spec.get("edits") or []
    summary = str(spec.get("summary", ""))
    ctx.state["edit_spec"] = {"summary": summary, "edits": edits}

    # No edits (or an explicit NOOP) = the optimization is already present.
    if not edits or summary.strip().upper().startswith("NOOP"):
        ctx.state["last_decision"] = {"result": "discard", "reason": "already_applied"}
        ctx.log_event(states.PLAN, "info", f"NOOP: {lever} already applied")
        return states.REVERT

    ctx.log_event(states.PLAN, "info", f"{len(edits)} edit(s): {summary[:80]}")
    return states.APPLY


def _default_runner():
    from ..plan_agent import make_plan_runner

    return make_plan_runner()
