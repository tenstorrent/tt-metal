"""APPLY handler (PLAN 8.5) — REAL, resilient.

Record a clean git checkpoint. If PLAN produced content-anchored edits, apply
them deterministically (no LLM). Otherwise the edit sub-agent applies the lever. We do
NOT blindly trust the agent's self-reported file list: edits land on disk before
the agent's final message, so if the report is empty or the agent errored, we
fall back to `git diff` for ground truth. If nothing actually changed -> route to
REPAIR_CODE (self-heal), never crash the loop.

In:  ctx.state["selected_lever"], manifest model_files, the model's git repo.
Out: ctx.state["git_sha_clean"], ["last_edit"]. -> VERIFY (or REPAIR_CODE/REVERT on failure)
"""

from __future__ import annotations

from .. import gitio, patch, router, states


def apply(ctx) -> str:
    lever = ctx.state.get("selected_lever")
    repo = gitio.repo_root(ctx.model_root())
    clean = gitio.head_sha(repo)
    ctx.state["git_sha_clean"] = clean  # REVERT target, recorded BEFORE editing

    spec = ctx.state.get("edit_spec")
    edits = spec.get("edits") if isinstance(spec, dict) else None

    # Fast path: PLAN gave content-anchored edits -> apply DETERMINISTICALLY, no LLM.
    # Self-validating: a missing / non-unique anchor fails loudly and self-heals.
    if edits:
        changed, failures = patch.apply_edits(ctx.model_root(), edits)
        if not failures:
            ctx.state["last_edit"] = {
                "files": changed,
                "summary": spec.get("summary", ""),
                "reported": changed,
                "error": None,
            }
            ctx.log_event(states.APPLY, "info", f"patch applied: {len(edits)} edit(s) -> {changed}")
            return states.VERIFY
        reason = "; ".join(f"{f['file']}: {f['reason']}" for f in failures)
        ctx.state["last_verdict"] = {"status": "patch_failed", "error": reason}
        ctx.log_event(states.APPLY, "warn", f"patch did not apply cleanly: {reason}")
        if ctx.state.get("code_fix_attempts", 0) < states.MAX_CODE_FIX:
            return states.REPAIR_CODE
        ctx.state["last_decision"] = {"result": "discard", "reason": "edit_failed"}
        return states.REVERT

    # Fallback: PLAN produced no structured edits (planning failed / improvise) -> LLM editor.
    try:
        section = router.read_section(lever) if lever else ""
    except KeyError:
        section = ""

    runner = ctx.deps.get("edit_runner") or _default_runner()
    reported, summary, model, usage, err = [], "", "?", None, None
    prompt_text, response_text = None, None
    try:
        result = runner(
            lever=lever,
            section=section,
            model_files=ctx.model_files(),
            spec=ctx.state.get("edit_spec"),
            cwd=str(ctx.model_root()),
        )
        reported = result.get("files") or []
        summary = result.get("summary", "")
        model, usage = result.get("model", "?"), result.get("usage")
        prompt_text, response_text = result.get("prompt"), result.get("response")
    except Exception as exc:  # editor errored — but its edits may already be on disk
        err = str(exc)[-500:]
    ctx.record_agent_call(states.APPLY, "edit", model, usage, prompt=prompt_text, response=response_text)

    # ground truth: what actually changed on disk since the clean checkpoint,
    # SCOPED to the model dir so unrelated repo edits aren't mistaken for the edit
    try:
        pathspec = ctx.model_root().relative_to(repo)
    except ValueError:
        pathspec = None
    changed = reported or gitio.changed_files(repo, clean, pathspec)
    if not changed:
        ctx.state["last_verdict"] = {"status": "edit_failed", "error": err or "edit produced no file changes"}
        ctx.log_event(states.APPLY, "warn", f"no edit landed: {err or 'empty report'}")
        if ctx.state.get("code_fix_attempts", 0) < states.MAX_CODE_FIX:
            return states.REPAIR_CODE
        ctx.state["last_decision"] = {"result": "discard", "reason": "edit_failed"}
        return states.REVERT

    ctx.state["last_edit"] = {"files": changed, "summary": summary, "reported": reported, "error": err}
    return states.VERIFY


def _default_runner():
    from ..edit_agent import make_edit_runner

    return make_edit_runner()
