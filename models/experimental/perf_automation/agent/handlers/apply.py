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
    # FIXER retry: an inert structural shard re-enters APPLY to be debugged. Keep the
    # ORIGINAL clean sha (the revert target + git-diff base) and the prior edit on disk
    # so the agent fixes its own edit; only record a fresh clean sha on the first apply.
    is_fixer = bool(ctx.state.get("inert_repair_error"))
    if not is_fixer:
        ctx.state["git_sha_clean"] = gitio.head_sha(repo)
    clean = ctx.state.get("git_sha_clean")

    spec = ctx.state.get("edit_spec")
    edits = spec.get("edits") if isinstance(spec, dict) else None
    # FROM_PRINCIPLES (a bucket with no playbook lever) routes to the THINKING structural
    # agent too — it optimizes the hottest op from first principles (roofline gap + menu),
    # no playbook section required. This is what makes the tool model-agnostic.
    is_from_principles = lever == states.FROM_PRINCIPLES
    is_structural = is_from_principles or _lever_type(ctx, lever) == "structural"
    ctx.state["last_was_structural"] = is_structural  # DECIDE's fixer gate reads this

    # Fast path: PLAN gave content-anchored edits -> apply DETERMINISTICALLY, no LLM.
    # Self-validating: a missing / non-unique anchor fails loudly and self-heals.
    # Structural levers SKIP this: a coordinated shard (tensor memory_config + program
    # config + consumer) is multi-site and needs the structural agent, not a one-anchor
    # find/replace — that is exactly the half-applied 'edit_inert' failure we saw.
    if edits and not is_structural:
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
        # Deterministic anchors didn't match the file verbatim (whitespace/anchor
        # drift). Instead of bailing, FALL BACK to the LLM editor below: it reads
        # the file and applies the lever/spec with the Edit tool, which matches
        # flexibly where the exact find-and-replace failed.
        reason = "; ".join(f"{f['file']}: {f['reason']}" for f in failures)
        ctx.state["last_verdict"] = {"status": "patch_failed", "error": reason}
        ctx.log_event(states.APPLY, "warn", f"deterministic patch missed ({reason}); falling back to LLM editor")
        # no return -> fall through to the LLM-editor block

    # LLM editor: PLAN produced no structured edits, OR the deterministic patch
    # above missed its anchors. The editor applies the lever/spec via Read+Edit.
    try:
        section = router.read_section(lever) if lever else ""
    except KeyError:
        section = ""

    if is_structural:
        runner = ctx.deps.get("structural_runner") or _default_structural_runner()
    else:
        runner = ctx.deps.get("edit_runner") or _default_runner()
    reported, summary, model, usage, err = [], "", "?", None, None
    prompt_text, response_text = None, None
    try:
        # SCOPE-FENCE: a structural edit is only meaningful on the executed path, so
        # restrict its editable files to exec_scoped_files (prevents editing dead /
        # off-path stubs like the speech path). Non-structural keeps the full list.
        scoped = ctx.state.get("exec_scoped_files")
        call_kwargs = dict(
            lever=lever,
            section=section,
            model_files=(scoped if (is_structural and scoped) else ctx.model_files()),
            spec=ctx.state.get("edit_spec"),
            cwd=str(ctx.model_root()),
            attempt=ctx.state.get("code_fix_attempts", 0),  # ladder rung 0 = haiku (first edit)
        )
        if is_structural:  # per-op targets + the op->source attribution (where the hot op lives)
            call_kwargs["top_ops"] = ctx.state.get("top_ops") or []
            call_kwargs["hot_sources"] = ctx.state.get("hot_sources") or []
            err = ctx.state.pop("inert_repair_error", None)  # FIXER: feed the op-graph-unchanged evidence back
            if err:
                call_kwargs["error"] = err
        # On-device validation tool: let the editor TEST its edit before submitting (the verified
        # authoring gap — it otherwise submits illegal configs that crash). Reuses the SAME check
        # GATE_PCC uses (pcc_runner). Disable with AGENT_NO_EDIT_CHECK=1.
        import os as _os

        if _os.environ.get("AGENT_NO_EDIT_CHECK", "").lower() not in ("1", "true", "yes"):
            from ..pcc_runner import run_pcc as _run_pcc

            _pcc = ctx.deps.get("pcc_runner") or _run_pcc
            call_kwargs["validate"] = lambda: _pcc(ctx)
        # Perf-feedback tool — kernel lever only: a kernel that is correct but not FASTER is
        # reverted at REMEASURE, so let the agent measure its kernel vs the baseline and iterate
        # toward a real win during authoring. Disable with AGENT_NO_PERF_CHECK=1.
        if lever == "tt-lang-kernel" and _os.environ.get("AGENT_NO_PERF_CHECK", "").lower() not in (
            "1",
            "true",
            "yes",
        ):
            call_kwargs["measure"] = lambda: _measure_device_ms(ctx)
            call_kwargs["baseline_ms"] = (ctx.state.get("metric") or {}).get("current")
        result = runner(**call_kwargs)
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
    diff_text = _edit_diff(repo, clean, pathspec)
    ctx.state["last_diff"] = diff_text  # GROUND TRUTH for the inert-repair agent (what it ACTUALLY wrote)
    ctx.state["edit_sig"] = _sig(diff_text)
    return states.VERIFY


def _edit_diff(repo, clean, pathspec):
    """The exact `git diff` the edit produced — fed back verbatim to the structural
    agent on an inert retry so it diagnoses from what it ACTUALLY wrote, not from a
    canned guess about what it wrote (the agent's own summary is unreliable)."""
    try:
        args = ["diff", clean] + (["--", str(pathspec)] if pathspec else [])
        out = gitio._git(args, repo)
        return out.stdout or ""
    except Exception:
        return ""


def _sig(diff_text):
    import hashlib

    if diff_text is None:
        return None
    return hashlib.sha256(diff_text.encode()).hexdigest()[:16]


def _measure_device_ms(ctx):
    """Profile the current edit and return {status, device_ms} for the kernel agent's
    measure_candidate tool. REUSES the SAME measure_runner REMEASURE uses (no duplication), so
    'faster here' means REMEASURE will agree. Never raises — a profiler crash returns status=crash."""
    runner = ctx.deps.get("measure_runner") or _default_measure_runner()
    try:
        profiles = runner(ctx)
    except Exception as exc:  # noqa: BLE001 — surface to the agent as a measure failure, don't kill APPLY
        return {"status": "crash", "error": str(exc)[-1000:]}
    if not profiles:
        return {"status": "crash", "error": "profiler returned no profile"}
    import statistics

    vals = [p.get("device_ms") for p in profiles if p.get("device_ms") is not None]
    if not vals:
        return {"status": "crash", "error": "profile had no device_ms"}
    return {"status": "ok", "device_ms": round(statistics.median(vals), 4)}


def _default_measure_runner():
    from ..measure import measure_runs

    return measure_runs


def _default_runner():
    from ..edit_agent import make_edit_runner

    return make_edit_runner()


def _default_structural_runner():
    from ..structural_agent import make_structural_runner

    return make_structural_runner()


def _lever_type(ctx, lever):
    """Look up the selected lever's lever_type from the playbook index.

    'structural' levers route to the structural-edit agent (coordinated multi-op
    shard) instead of the content-anchored patch / mechanical editor.
    """
    for e in getattr(ctx, "index", None) or []:
        if e.get("id") == lever:
            return e.get("lever_type")
    return None
