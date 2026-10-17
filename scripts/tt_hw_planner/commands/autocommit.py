# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Commit + push a model's artifacts to its repo after a stage finishes cleanly.

Opt-in. Each of the three pipeline stages (``auto-up``, ``emit-e2e``, ``optimize``)
can be told, with ``--commit-push`` (or the ``TT_HW_PLANNER_AUTOCOMMIT=1`` env var),
to — only when the stage returns 0 — stage the model's demo directory, commit it to
the local repo as ``apande-TT``'s configured identity, and push it to the model's
remote.

Design rules:
  * SCOPED to the model's own demo directory. Tool files (``scripts/tt_hw_planner/**``)
    and other models are never swept in — matching ``commit-tool``'s "one thing per
    commit" contract.
  * BEST-EFFORT. A stage must never fail because a commit or push did; every error is
    caught and reported, and the stage's own return code is preserved.
  * HOOK-FREE commit (``-c core.hooksPath=/dev/null`` + ``--no-verify``): this is a
    large repo whose pre-commit hooks (black/clang-format) are slow and routinely
    reject machine-generated files; the tool commits are mechanical and reviewed by
    the loop, not the hooks.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def _enabled(args) -> bool:
    if getattr(args, "commit_push", False):
        return True
    return os.environ.get("TT_HW_PLANNER_AUTOCOMMIT", "").strip().lower() in ("1", "true", "yes", "on")


def _resolve_demo_dir(model_id: str) -> Optional[Path]:
    """Accept either a planner model_id or a demo-directory path (the ``emit-e2e`` /
    ``optimize`` positional accepts both)."""
    if model_id:
        p = Path(model_id).expanduser()
        if p.is_dir() and (p / "tt").exists() or (p.is_dir() and any(p.glob("*.py"))):
            return p.resolve()
    try:
        from ..bringup_loop import find_demo_dir

        return find_demo_dir(model_id)
    except Exception:
        return None


def _repo_root(start: Path) -> Optional[Path]:
    r = _run(["git", "rev-parse", "--show-toplevel"], start)
    if r.returncode == 0 and r.stdout.strip():
        return Path(r.stdout.strip())
    return None


def _pick_remote(repo: Path, branch: str) -> Optional[str]:
    """The branch's upstream remote if set, else the fork (``apande``), else
    ``origin``, else the first remote."""
    up = _run(["git", "rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}"], repo)
    if up.returncode == 0 and "/" in up.stdout.strip():
        return up.stdout.strip().split("/", 1)[0]
    remotes = _run(["git", "remote"], repo).stdout.split()
    for pref in ("apande", "origin"):
        if pref in remotes:
            return pref
    return remotes[0] if remotes else None


#: Branches that hold the tt_hw_planner TOOL code, never model artifacts. Auto-commit
#: refuses to run here so a stage launched from a tool checkout can't push a model's
#: files onto the tool branch.
def _is_tool_branch(branch: str) -> bool:
    b = (branch or "").strip()
    return b == "feature/tt-hw-planner" or b.endswith("/feature/tt-hw-planner") or "tt-hw-planner" in b


def commit_and_push_stage(args, model_id: str, stage: str) -> None:
    """Public entry: no-op unless enabled; never raises."""
    if not _enabled(args):
        return
    try:
        _commit_and_push(args, model_id, stage)
    except Exception as exc:  # noqa: BLE001
        print(f"  [auto-commit] {stage}: skipped — {exc}", file=sys.stderr)


def _commit_and_push(args, model_id: str, stage: str) -> None:
    demo_dir = _resolve_demo_dir(model_id)
    if demo_dir is None or not Path(demo_dir).is_dir():
        print(f"  [auto-commit] {stage}: no demo dir for {model_id!r}; nothing to commit")
        return
    demo_dir = Path(demo_dir)
    repo = _repo_root(demo_dir)
    if repo is None:
        print(f"  [auto-commit] {stage}: {demo_dir} is not inside a git repo; skipping")
        return

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo).stdout.strip()
    if _is_tool_branch(branch):
        print(
            f"  [auto-commit] {stage}: current branch {branch!r} is the tt_hw_planner tool branch — "
            "refusing to commit model artifacts here (model work belongs on a model branch)"
        )
        return

    rel = os.path.relpath(demo_dir, repo)
    add = _run(["git", "add", "--", rel], repo)
    if add.returncode != 0:
        print(f"  [auto-commit] {stage}: git add failed — {add.stderr.strip()}")
        return

    # Anything actually staged under the model dir? (returncode 1 == differences exist)
    if _run(["git", "diff", "--cached", "--quiet", "--", rel], repo).returncode == 0:
        print(f"  [auto-commit] {stage}: no changes under {rel}; nothing to commit")
        return

    if branch == "HEAD":
        print(f"  [auto-commit] {stage}: detached HEAD; committing but will not push")
    msg = f"{model_id}: {stage} stage complete [tt_hw_planner auto-commit]"
    commit = _run(
        ["git", "-c", "core.hooksPath=/dev/null", "commit", "--no-verify", "-m", msg, "--", rel],
        repo,
    )
    if commit.returncode != 0:
        print(f"  [auto-commit] {stage}: git commit failed — {(commit.stderr or commit.stdout).strip()}")
        return
    sha = _run(["git", "rev-parse", "--short", "HEAD"], repo).stdout.strip()
    print(f"  [auto-commit] {stage}: committed {rel} as {sha} on {branch or 'HEAD'}")

    if branch == "HEAD":
        return
    remote = getattr(args, "commit_remote", None) or _pick_remote(repo, branch)
    if not remote:
        print(f"  [auto-commit] {stage}: no push remote; committed locally only")
        return
    push = _run(["git", "push", remote, f"HEAD:{branch}"], repo)
    if push.returncode != 0:
        print(f"  [auto-commit] {stage}: git push {remote} {branch} failed — {push.stderr.strip()}")
        return
    print(f"  [auto-commit] {stage}: pushed {branch} -> {remote}")


def wrap(real_cmd: Callable, stage: str) -> Callable:
    """Wrap a stage command so a clean (rc == 0) run auto-commits + pushes the model."""

    def _wrapped(args) -> int:
        rc = real_cmd(args)
        if rc == 0:
            commit_and_push_stage(args, getattr(args, "model_id", "") or "", stage)
        return rc

    _wrapped.__name__ = getattr(real_cmd, "__name__", "cmd") + "_autocommit"
    return _wrapped


def add_commit_push_args(parser) -> None:
    """Attach the opt-in flags to a stage subparser."""
    parser.add_argument(
        "--commit-push",
        action="store_true",
        help=(
            "When the stage finishes cleanly, commit the model's demo directory to the "
            "local repo and push it to its remote (also enabled by TT_HW_PLANNER_AUTOCOMMIT=1)."
        ),
    )
    parser.add_argument(
        "--commit-remote",
        default=None,
        help="Git remote to push to (default: the branch's upstream, else 'apande', else 'origin').",
    )
