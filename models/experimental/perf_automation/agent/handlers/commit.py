"""COMMIT handler (PLAN 8.9) — REAL. Persist a KEPT edit.

DECIDE routes here only when REMEASURE showed a real, trusted improvement. We:
  1. promote this iteration's re-bucketed profile to `current_profile` so the NEXT
     ROUTE routes on the new bottleneck (not the frozen baseline);
  2. scoped `git commit` of the model dir so the win is durable AND becomes the new
     clean checkpoint (`git_sha_clean`) the next lever reverts to.

Scoped to the model dir via pathspec -> unrelated working-tree changes elsewhere in
the repo are never swept into the commit. Resilient: any git failure (e.g. the model
dir is untracked, so there is nothing to commit) is logged and the loop continues to
LOG rather than crashing.
"""

from __future__ import annotations

from .. import gitio, states


def commit(ctx) -> str:
    d = ctx.state.get("last_decision") or {}

    # Promote the edited model's profile so ROUTE sees the shifted bottleneck.
    prof = d.get("profile")
    if prof:
        ctx.state["current_profile"] = prof

    lever = ctx.state.get("selected_lever")
    it = ctx.state.get("iteration", 0)
    before, after = d.get("before"), d.get("after")
    unit = ctx.state.get("metric", {}).get("unit", "")
    msg = f"[perf_automation] iter {it}: keep '{lever}' ({before} -> {after} {unit})".rstrip()

    try:
        repo = gitio.repo_root(ctx.model_root())
        try:
            pathspec = ctx.model_root().relative_to(repo)
        except ValueError:
            pathspec = None
        new_sha = gitio.commit(repo, msg, pathspec)
        if new_sha:
            ctx.state["git_sha_clean"] = new_sha  # next lever's REVERT target is HERE
            ctx.log_event(states.COMMIT, "info", f"committed {new_sha[:10]} — keep '{lever}'")
        else:
            ctx.log_event(states.COMMIT, "warn", "nothing to commit under model dir (untracked?)")
    except gitio.GitError as exc:
        ctx.log_event(states.COMMIT, "warn", f"commit skipped: {exc}")

    if lever == states.FROM_PRINCIPLES:
        try:
            from ..promote import promote_win

            path = promote_win(ctx)
            if path:
                ctx.log_event(states.COMMIT, "info", f"promoted from-principles win -> provisional lever {path.name}")
        except Exception as exc:
            ctx.log_event(states.COMMIT, "warn", f"promote skipped: {exc}")

    # GRADUATION: a kept win that re-used a PROVISIONAL learned lever on a DIFFERENT model
    # cross-validates it -> graduate (rename LEARNED_ -> GRADUATED_, mark trusted, now committable).
    try:
        from ..promote import maybe_graduate

        gpath = maybe_graduate(ctx, lever)
        if gpath:
            ctx.log_event(states.COMMIT, "info", f"graduated learned lever -> {gpath.name} (cross-model validated)")
            # Auto-persist the graduated lever into the tool's playbook: a scoped commit of just
            # this file (GRADUATED_*.md is not gitignored, unlike provisional LEARNED_*.md), so
            # cross-validated knowledge accumulates in version control with no manual step.
            try:
                grepo = gitio.repo_root(gpath.parent)
                grel = gpath.relative_to(grepo)
                gsha = gitio.commit(grepo, f"[perf_automation] graduate learned lever '{gpath.stem}'", grel)
                if gsha:
                    ctx.log_event(states.COMMIT, "info", f"committed graduated lever {gsha[:10]} ({grel})")
            except (gitio.GitError, ValueError) as exc:
                ctx.log_event(states.COMMIT, "warn", f"graduated-lever commit skipped: {exc}")
    except Exception as exc:
        ctx.log_event(states.COMMIT, "warn", f"graduate skipped: {exc}")

    return states.LOG
