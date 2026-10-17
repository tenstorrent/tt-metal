#!/usr/bin/env python3
"""Fail a pull request from a fork that changes files under .github/.

The self-hosted CIv2 runners refuse a job from a fork PR that changes .github/: the
job-started hook in tenstorrent/github-ci-infra (vm-sandbox/runner_hook_job_started.sh)
diffs ``<base>...<fork head> -- .github`` and exits 1. The contributor then sees only a
runner setup failure. This check runs in the pre-commit job, on a GitHub-hosted runner,
and gives the same verdict with a clear message.

The check uses the same inputs as the runner hook: ``pull_request.head.repo.fork`` from
``$GITHUB_EVENT_PATH``, and a three-dot diff of .github/ (deletions included) that ends at
the fork head, not at the PR merge commit. It passes when there is no pull_request event
(local runs, push, merge_group) or the PR is not from a fork. It is a usability check, not
a security control: the runner hook still enforces the rule, so any error here fails open.
all-static-checks.yaml runs only for PRs to main, so a PR to another base gets no early check.

The diff base is ``$PRE_COMMIT_FROM_REF`` (pre-commit sets it for ``--from-ref``), else
``pull_request.base.sha``. The diff head is ``pull_request.head.sha``, else
``$PRE_COMMIT_TO_REF``, else HEAD.

Usage::

    check_fork_github_changes.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

GITHUB_DIR = ".github"


def load_event(path: str | None) -> dict:
    """Return the GitHub event payload, or {} when there is none."""
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            event = json.load(f)
    except (OSError, ValueError):
        return {}
    return event if isinstance(event, dict) else {}


def _pr(event: dict, side: str) -> dict:
    return ((event.get("pull_request") or {}).get(side)) or {}


def is_fork_pull_request(event: dict) -> bool:
    """True when the event is a pull request whose head repo is a fork."""
    return (_pr(event, "head").get("repo") or {}).get("fork") is True


def diff_range(event: dict, env: dict) -> tuple[str, str] | None:
    """Return (base_ref, head_ref) for the diff, or None when no base is known."""
    base_ref = env.get("PRE_COMMIT_FROM_REF") or _pr(event, "base").get("sha")
    if not base_ref:
        return None
    head_ref = _pr(event, "head").get("sha") or env.get("PRE_COMMIT_TO_REF") or "HEAD"
    return base_ref, head_ref


def changed_github_files(base_ref: str, head_ref: str) -> list[str]:
    """Return the .github/ paths that differ between merge-base(base, head) and head."""
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...{head_ref}", "--", GITHUB_DIR],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def failure_message(files: list[str], event: dict) -> str:
    listing = "\n".join(f"  {f}" for f in files)
    base = _pr(event, "base")
    repo = (base.get("repo") or {}).get("full_name") or "tenstorrent/tt-metal"
    branch = base.get("ref") or "main"
    return (
        f"This check fails on purpose: this pull request comes from a fork and changes {GITHUB_DIR}/.\n\n"
        "Why: the tt-metal hardware tests run on Tenstorrent's own CI machines. For security,\n"
        "those machines do not run CI or workflow changes (.github/) from a fork. While this PR\n"
        "changes .github/, its hardware jobs fail at runner setup and its tests do not run.\n\n"
        f"Changed files under {GITHUB_DIR}/:\n"
        f"{listing}\n\n"
        "To fix:\n"
        "  1. Remove the .github/ changes from this PR:\n"
        f"       git fetch https://github.com/{repo}.git {branch}\n"
        '       git restore --staged --worktree --source="$(git merge-base FETCH_HEAD HEAD)" -- .github\n'
        '       git commit -m "Remove .github changes"\n'
        "  2. Push the branch again.\n"
        f"  3. If you need the .github/ changes, say so in the PR. A maintainer can land them from a branch in {repo}.\n"
    )


def main(argv: list[str] | None = None, env: dict | None = None) -> int:
    del argv  # pre-commit runs this hook with pass_filenames: false
    env = dict(os.environ) if env is None else env
    event = load_event(env.get("GITHUB_EVENT_PATH"))
    if not is_fork_pull_request(event):
        return 0

    refs = diff_range(event, env)
    if refs is None:
        print("check_fork_github_changes: no base ref for the diff, skipping", file=sys.stderr)
        return 0

    try:
        files = changed_github_files(*refs)
    except subprocess.CalledProcessError as e:
        print(f"check_fork_github_changes: git diff failed, skipping: {e.stderr.strip()}", file=sys.stderr)
        return 0
    if not files:
        return 0

    print(failure_message(files, event))
    if env.get("GITHUB_ACTIONS") == "true":
        print(
            f"::error::Fork PRs cannot change {GITHUB_DIR}/: Tenstorrent CI machines do not run workflow changes "
            f"from forks, so the hardware tests of this PR cannot run. Remove the {len(files)} {GITHUB_DIR}/ "
            "change(s); the steps are in the Run Pre-commit log."
        )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
