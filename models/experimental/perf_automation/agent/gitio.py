"""Tiny git helpers for the Agent Loop (PLAN 8.5 / 8.9).

APPLY records a clean SHA before editing; REVERT resets to it. Path-scoped and
explicit — we never `git add -A`. Real subprocess calls (fast; tests run against
a throwaway temp repo).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(Exception):
    """A git command failed or the path is not inside a repo."""


def _git(args: list[str], cwd) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(["git", *args], returncode=124, stdout="", stderr="git timed out")


def repo_root(path) -> Path:
    """Toplevel of the repo containing `path` (walks up). Raises if not a repo."""
    r = _git(["rev-parse", "--show-toplevel"], path)
    if r.returncode != 0:
        raise GitError(f"not a git repo at {path}: {r.stderr.strip()}")
    return Path(r.stdout.strip())


def head_sha(repo) -> str:
    r = _git(["rev-parse", "HEAD"], repo)
    if r.returncode != 0:
        raise GitError(f"no HEAD at {repo}: {r.stderr.strip()}")
    return r.stdout.strip()


def is_clean(repo) -> bool:
    r = _git(["status", "--porcelain"], repo)
    if r.returncode != 0:
        raise GitError(f"git status failed at {repo}: {r.stderr.strip()}")
    return r.stdout.strip() == ""


def reset_hard(repo, sha: str) -> None:
    r = _git(["reset", "--hard", sha], repo)
    if r.returncode != 0:
        raise GitError(f"git reset --hard {sha} failed: {r.stderr.strip()}")


def changed_files(repo, sha: str, pathspec=None) -> list[str]:
    """Repo-relative paths changed in the working tree since `sha` (ground truth
    for what an edit actually touched, independent of the agent's self-report).
    `pathspec` scopes the diff (e.g. the model dir) so UNRELATED repo changes are
    never attributed to the edit."""
    args = ["diff", "--name-only", sha] + (["--", str(pathspec)] if pathspec else [])
    r = _git(args, repo)
    if r.returncode != 0:
        raise GitError(f"git diff failed at {repo}: {r.stderr.strip()}")
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def checkout(repo, sha: str, pathspec=None) -> None:
    """Restore tracked files to their state at `sha` (`git checkout <sha> -- <pathspec>`).

    This is REVERT's primitive. Deliberately SCOPED and non-destructive:
      * `pathspec` (the model dir) restricts the restore to the model — UNRELATED
        working-tree changes elsewhere in the repo are never touched.
      * `git checkout <sha> -- <path>` only rewrites tracked files; it never deletes
        untracked content. If `pathspec` was untracked at `sha`, git matches nothing
        and raises GitError — callers treat that as a safe no-op.
    """
    if pathspec is None:
        specs = ["."]
    elif isinstance(pathspec, (list, tuple)):
        specs = [str(p) for p in pathspec]
    else:
        specs = [str(pathspec)]
    args = ["checkout", sha, "--", *specs]
    r = _git(args, repo)
    if r.returncode != 0:
        raise GitError(f"git checkout {sha} -- {specs} failed: {r.stderr.strip()}")


def commit(repo, message: str, pathspec=None) -> str | None:
    """Stage + commit ONLY `pathspec` and return the new HEAD sha (COMMIT's primitive).

    Scoped on purpose: `git add -- <pathspec>` then `git commit -m <msg> -- <pathspec>`
    so a kept edit is persisted without sweeping in unrelated staged/working changes
    elsewhere in the repo. Returns None (no commit made) when nothing is staged under
    `pathspec`, so the caller keeps going without a spurious empty commit.
    """
    add = _git(["add", "--", str(pathspec) if pathspec else "."], repo)
    if add.returncode != 0:
        raise GitError(f"git add {pathspec} failed: {add.stderr.strip()}")
    staged_args = ["diff", "--cached", "--name-only"] + (["--", str(pathspec)] if pathspec else [])
    staged = _git(staged_args, repo)
    if staged.returncode == 0 and not staged.stdout.strip():
        return None
    cargs = ["commit", "--no-verify", "-m", message] + (["--", str(pathspec)] if pathspec else [])
    r = _git(cargs, repo)
    if r.returncode != 0:
        raise GitError(f"git commit failed: {r.stderr.strip()}")
    return head_sha(repo)
