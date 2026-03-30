#!/usr/bin/env python3
"""Execute a restricted subset of gh commands.

This wrapper is designed for agent use where direct gh access is blocked.
It accepts a single command string, validates it against an allowlist, and
executes only safe commands.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence


ISSUE_REPO_TEST = "ebanerjeeTT/issue_dump"
PRIMARY_REPO = "tenstorrent/tt-metal"
ALLOWED_WORKFLOW_IDS = {"triage-ci", "triage-ci.yaml", "triage-ci.yml"}
READ_REPOS = {PRIMARY_REPO, ISSUE_REPO_TEST}

DENY_CHARS = {";", "&&", "||", "|", "`", "$("}


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str


def extract_option(tokens: Sequence[str], long_opt: str, short_opt: str | None = None) -> str | None:
    for i, tok in enumerate(tokens):
        if tok == long_opt or (short_opt and tok == short_opt):
            if i + 1 < len(tokens):
                return tokens[i + 1]
            return None
        if tok.startswith(f"{long_opt}="):
            return tok.split("=", 1)[1]
        if short_opt and tok.startswith(f"{short_opt}="):
            return tok.split("=", 1)[1]
    return None


def first_positional_after(tokens: Sequence[str], start_idx: int) -> str | None:
    for tok in tokens[start_idx:]:
        if not tok.startswith("-"):
            return tok
    return None


def has_option(tokens: Sequence[str], long_opt: str, short_opt: str | None = None) -> bool:
    return extract_option(tokens, long_opt, short_opt) is not None


def repo_for_command(tokens: Sequence[str]) -> str | None:
    return extract_option(tokens, "--repo", "-R")


def deny_for_suspicious_tokens(tokens: Sequence[str]) -> Decision | None:
    for tok in tokens:
        for marker in DENY_CHARS:
            if marker in tok:
                return Decision(False, f"Denied: suspicious token detected: {tok!r}")
    return None


def ensure_repo(tokens: Sequence[str], allowed_repos: set[str], *, required: bool = True) -> Decision | None:
    repo = repo_for_command(tokens)
    if repo is None:
        if required:
            return Decision(False, "Denied: command must specify --repo/-R explicitly.")
        return None
    if repo not in allowed_repos:
        return Decision(False, f"Denied: repo {repo!r} is not allowed.")
    return None


def validate(tokens: list[str]) -> Decision:
    if not tokens:
        return Decision(False, "Denied: empty command.")
    if tokens[0] != "gh":
        return Decision(False, "Denied: only commands starting with `gh` are allowed.")
    if len(tokens) < 2:
        return Decision(False, "Denied: missing gh subcommand.")

    suspicious = deny_for_suspicious_tokens(tokens)
    if suspicious:
        return suspicious

    root = tokens[1]
    sub = tokens[2] if len(tokens) > 2 else ""

    # Read-only auth status check is allowed.
    if root == "auth" and sub == "status":
        return Decision(True, "Allowed: gh auth status")

    # Issue commands.
    if root == "issue":
        if sub in {"list", "view"}:
            repo_check = ensure_repo(tokens, READ_REPOS, required=True)
            if repo_check:
                return repo_check
            return Decision(True, f"Allowed: gh issue {sub}")

        if sub in {"create", "edit", "close", "reopen", "comment"}:
            repo_check = ensure_repo(tokens, {ISSUE_REPO_TEST}, required=True)
            if repo_check:
                return repo_check
            return Decision(True, f"Allowed: gh issue {sub} on test issue repo only")

        return Decision(False, f"Denied: unsupported gh issue subcommand {sub!r}.")

    # PR read-only commands in primary repo.
    if root == "pr":
        if sub in {"list", "view"}:
            repo_check = ensure_repo(tokens, {PRIMARY_REPO}, required=True)
            if repo_check:
                return repo_check
            return Decision(True, f"Allowed: gh pr {sub}")
        if sub == "create":
            repo_check = ensure_repo(tokens, {PRIMARY_REPO}, required=True)
            if repo_check:
                return repo_check
            if extract_option(tokens, "--base") != "main":
                return Decision(False, "Denied: gh pr create must target --base main.")
            if not has_option(tokens, "--head"):
                return Decision(False, "Denied: gh pr create must include --head.")
            if not has_option(tokens, "--title"):
                return Decision(False, "Denied: gh pr create must include --title.")
            if not has_option(tokens, "--body"):
                return Decision(False, "Denied: gh pr create must include --body.")
            if "--draft" not in tokens:
                return Decision(False, "Denied: gh pr create must include --draft.")
            return Decision(True, "Allowed: gh pr create (draft to main in primary repo)")
        return Decision(False, f"Denied: unsupported gh pr subcommand {sub!r}.")

    # Run read-only commands in primary repo.
    if root == "run":
        if sub in {"list", "view", "download"}:
            repo_check = ensure_repo(tokens, {PRIMARY_REPO}, required=True)
            if repo_check:
                return repo_check
            return Decision(True, f"Allowed: gh run {sub}")
        return Decision(False, f"Denied: unsupported gh run subcommand {sub!r}.")

    # Workflow commands. Dispatch is tightly constrained.
    if root == "workflow":
        if sub in {"list", "view"}:
            repo_check = ensure_repo(tokens, {PRIMARY_REPO}, required=True)
            if repo_check:
                return repo_check
            return Decision(True, f"Allowed: gh workflow {sub}")

        if sub == "run":
            repo_check = ensure_repo(tokens, {PRIMARY_REPO}, required=True)
            if repo_check:
                return repo_check
            workflow_id = first_positional_after(tokens, 3)
            if workflow_id is None:
                return Decision(False, "Denied: gh workflow run requires workflow id/name argument.")
            if workflow_id not in ALLOWED_WORKFLOW_IDS:
                return Decision(
                    False,
                    f"Denied: workflow {workflow_id!r} not in allowlist {sorted(ALLOWED_WORKFLOW_IDS)}.",
                )
            return Decision(True, "Allowed: gh workflow run for triage-ci only")

        return Decision(False, f"Denied: unsupported gh workflow subcommand {sub!r}.")

    # Disallow gh api to prevent bypassing restrictions.
    if root == "api":
        return Decision(False, "Denied: gh api is not allowed by this wrapper.")

    return Decision(False, f"Denied: unsupported gh command group {root!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and run an allowlisted gh command string.",
    )
    parser.add_argument(
        "--command",
        required=True,
        help="Full gh command string to validate and execute.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only; do not execute command.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        tokens = shlex.split(args.command, posix=True)
    except ValueError as exc:
        print(f"Denied: invalid command quoting: {exc}", file=sys.stderr)
        return 2

    decision = validate(tokens)
    if not decision.allowed:
        print(decision.reason, file=sys.stderr)
        return 3

    print(decision.reason, file=sys.stderr)
    print("Command:", " ".join(shlex.quote(t) for t in tokens), file=sys.stderr)
    if args.dry_run:
        print("Dry-run mode: command not executed.", file=sys.stderr)
        return 0

    proc = subprocess.run(tokens, check=False)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
