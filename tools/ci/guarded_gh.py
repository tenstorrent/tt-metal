#!/usr/bin/env python3
"""Execute a restricted subset of gh/git commands.

This wrapper is designed for agent use where direct command access is blocked.
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
ALLOWED_WORKFLOW_IDS = {
    "triage-ci",
    "triage-ci.yaml",
    "triage-ci.yml",
    "all-static-checks",
    "all-static-checks.yaml",
    "all-static-checks.yml",
    "pr-gate",
    "pr-gate.yaml",
    "pr-gate.yml",
    "merge-gate",
    "merge-gate.yaml",
    "merge-gate.yml",
}
READ_REPOS = {PRIMARY_REPO, ISSUE_REPO_TEST}
ALLOWED_PUSH_REMOTE = "origin"
ALLOWED_PUSH_REFSPECS = {
    "ebanerjee/CI-maintenance",
    "HEAD:ebanerjee/CI-maintenance",
}
ALLOWED_COMMIT_OPTS = {"-m", "--message"}

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

    suspicious = deny_for_suspicious_tokens(tokens)
    if suspicious:
        return suspicious

    if tokens[0] == "git":
        if len(tokens) < 2:
            return Decision(False, "Denied: missing git subcommand.")
        if tokens[1] == "push":
            idx = 2
            while idx < len(tokens) and tokens[idx].startswith("-"):
                if tokens[idx] not in {"-u", "--set-upstream"}:
                    return Decision(False, f"Denied: unsupported git push option {tokens[idx]!r}.")
                idx += 1

            if idx >= len(tokens):
                return Decision(False, "Denied: git push requires remote.")
            remote = tokens[idx]
            idx += 1
            if remote != ALLOWED_PUSH_REMOTE:
                return Decision(False, f"Denied: git push remote must be {ALLOWED_PUSH_REMOTE!r}.")

            if idx >= len(tokens):
                return Decision(False, "Denied: git push requires explicit refspec.")
            refspec = tokens[idx]
            idx += 1
            if refspec not in ALLOWED_PUSH_REFSPECS:
                return Decision(False, f"Denied: refspec {refspec!r} is not allowed.")

            if idx != len(tokens):
                return Decision(False, "Denied: extra git push arguments are not allowed.")

            return Decision(True, "Allowed: git push to ebanerjee/CI-maintenance only")

        if tokens[1] == "commit":
            idx = 2
            has_message = False
            while idx < len(tokens):
                tok = tokens[idx]
                if tok.startswith("-"):
                    if tok in ALLOWED_COMMIT_OPTS:
                        if idx + 1 >= len(tokens):
                            return Decision(False, f"Denied: {tok} requires a value.")
                        has_message = True
                        idx += 2
                        continue
                    if any(tok.startswith(f"{opt}=") for opt in ALLOWED_COMMIT_OPTS):
                        has_message = True
                        idx += 1
                        continue
                    return Decision(False, f"Denied: unsupported git commit option {tok!r}.")
                return Decision(False, "Denied: positional arguments are not allowed for git commit.")

            if not has_message:
                return Decision(False, "Denied: git commit requires -m/--message.")
            return Decision(True, "Allowed: git commit with message only")

        return Decision(False, "Denied: only git push/commit are allowlisted.")

    if tokens[0] != "gh":
        return Decision(False, "Denied: only commands starting with `gh` or allowlisted `git` are allowed.")
    if len(tokens) < 2:
        return Decision(False, "Denied: missing gh subcommand.")

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
        if sub == "comment":
            repo_check = ensure_repo(tokens, {PRIMARY_REPO}, required=True)
            if repo_check:
                return repo_check
            if not has_option(tokens, "--body"):
                return Decision(False, "Denied: gh pr comment must include --body.")
            target = first_positional_after(tokens, 3)
            if target is None:
                return Decision(False, "Denied: gh pr comment requires PR number/url argument.")
            return Decision(True, "Allowed: gh pr comment in primary repo")
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
            return Decision(True, "Allowed: gh workflow run in allowlist")

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

    if len(tokens) >= 2 and tokens[0] == "git" and tokens[1] == "commit":
        first = subprocess.run(tokens, check=False)
        if first.returncode == 0:
            return 0
        print(
            "git commit failed; retrying once after staging hook-modified tracked files.",
            file=sys.stderr,
        )
        add = subprocess.run(["git", "add", "--update"], check=False)
        if add.returncode != 0:
            print("Retry aborted: git add --update failed.", file=sys.stderr)
            return first.returncode
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
        if staged.returncode == 0:
            print("Retry aborted: no staged changes after hook run.", file=sys.stderr)
            return first.returncode
        retry = subprocess.run(tokens, check=False)
        return retry.returncode

    proc = subprocess.run(tokens, check=False)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
