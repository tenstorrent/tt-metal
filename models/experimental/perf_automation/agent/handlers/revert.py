"""REVERT handler (PLAN 8.9) — REAL. Roll back a discarded or failed edit.

Reached when DECIDE discards (no gain / untrusted measurement) or an edit failed
past its repair budget. We restore the model dir to `git_sha_clean` — the clean HEAD
APPLY recorded BEFORE editing — via a scoped `git checkout <sha> -- <model dir>`.

Why scoped checkout and not `git reset --hard`:
  * a hard reset is repo-wide and would obliterate UNRELATED working-tree changes;
    the scoped checkout touches only the model dir.
  * checkout only rewrites tracked files and never deletes untracked content.

Resilient: if there is no recorded SHA, or the model dir is untracked at that SHA
(git matches nothing -> GitError), we log and continue to LOG rather than crash.
NOTE: when the model dir is untracked, git has no baseline to restore — the edit
persists on disk; per-file `.last_good_native` sidecars are the model's own fallback.
"""

from __future__ import annotations

from .. import gitio, states


def revert(ctx) -> str:
    sha = ctx.state.get("git_sha_clean")
    if not sha:
        ctx.log_event(states.REVERT, "warn", "no git_sha_clean recorded; nothing to revert")
        return states.LOG

    try:
        repo = gitio.repo_root(ctx.model_root())
        try:
            pathspec = ctx.model_root().relative_to(repo)
        except ValueError:
            pathspec = None
        gitio.checkout(repo, sha, pathspec)
        ctx.log_event(states.REVERT, "info", f"reverted model dir to {sha[:10]}")
    except gitio.GitError as exc:
        ctx.log_event(states.REVERT, "warn", f"revert skipped ({exc}); edit left on disk")

    return states.LOG
