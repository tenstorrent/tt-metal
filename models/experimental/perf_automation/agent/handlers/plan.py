"""PLAN handler (PLAN 8.x) — REAL. Lead turns the chosen lever into a localized spec.

Reads the lever's playbook section + the op-class-filtered model map, lets the
lead (Read/Grep, no Edit) confirm the spot, and emits content-anchored edits.
NOOP (already applied) -> discard cheaply, no edit/gate/measure wasted.
Planner injectable via ctx.deps["plan_runner"].

In:  ctx.state["selected_lever"], ["current_bucket"]. Out: ctx.state["edit_spec"]. -> APPLY | REVERT
"""

from __future__ import annotations

from pathlib import Path

from .. import model_map, router, states


def plan(ctx) -> str:
    lever = ctx.state.get("selected_lever")
    try:
        section = router.read_section(lever) if lever else ""
    except KeyError:
        section = ""

    # If the previous attempt was inert (op graph byte-identical), the edit
    # defined the optimization but did not redirect execution to it. Tell the
    # planner so it rewires the call site this time instead of repeating it.
    last = ctx.state.get("last_decision") or {}
    if last.get("reason") == "edit_inert":
        prev = ", ".join((ctx.state.get("last_edit") or {}).get("files") or []) or "the target file"
        section = (
            f"!! YOUR PREVIOUS EDIT TO {prev} CHANGED NOTHING — the op graph was byte-identical. "
            f"You defined the optimization but the model still ran the ORIGINAL operation. This time "
            f"you MUST replace the executed call so the optimized op is the one __call__ actually runs.\n\n"
            + (section or "")
        )

    subs = model_map.OP_CLASS_SUBSTRINGS.get(ctx.state.get("current_bucket") or "")
    # Steer away from files proven OFF the measured execution path: a prior edit
    # to them gave a byte-identical op graph (edit_inert). Excluding them from the
    # map the planner sees means it can only target code that actually runs.
    scoped = ctx.state.get("exec_scoped_files")
    if scoped:
        # Proactive: only files the profiled workload actually executes.
        files = [Path(f) for f in scoped]
    else:
        # Reactive fallback: drop files a prior edit proved off-path (edit_inert).
        inert = [i for i in (ctx.state.get("inert_files") or []) if i]
        files = [f for f in ctx.model_files() if not any(i in str(f) for i in inert)] or ctx.model_files()
    try:
        mm = model_map.build_model_map(files, root=ctx.model_root())
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
