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
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


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


def changed_files(repo, sha: str) -> list[str]:
    """Repo-relative paths changed in the working tree since `sha` (ground truth
    for what an edit actually touched, independent of the agent's self-report)."""
    r = _git(["diff", "--name-only", sha], repo)
    if r.returncode != 0:
        raise GitError(f"git diff failed at {repo}: {r.stderr.strip()}")
    return [ln for ln in r.stdout.splitlines() if ln.strip()]
