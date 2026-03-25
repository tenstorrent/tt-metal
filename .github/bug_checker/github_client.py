"""GitHub REST API client for fetching PR data and posting comments."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional

from bug_checker.logger import logger

REPO = "tenstorrent/tt-metal"
MAX_DIFF_LINES = 8000


@dataclass
class PRInfo:
    number: int
    title: str
    base_sha: str
    head_sha: str
    diff: str
    changed_files: list[str]
    labels: list[str]
    truncated_files: list[str] = field(default_factory=list)


def check_prerequisites(*, need_gh: bool = False, need_git: bool = False) -> None:
    """Validate that required external tools are installed and authenticated.

    Raises RuntimeError with a clear, actionable message on the first failure.
    Call this at process startup before making any API or subprocess calls.

    Args:
        need_gh: Require the gh CLI to be installed and authenticated.
        need_git: Require git to be installed.
    """
    if need_gh:
        _check_gh()
    if need_git:
        _check_git()


def _check_gh() -> None:
    result = subprocess.run(["gh", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "The 'gh' CLI is not installed or not on PATH. "
            "Install it from https://cli.github.com/ and make sure it is on your PATH."
        )
    result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "The 'gh' CLI is not authenticated. "
            "Run 'gh auth login' to authenticate before using --pr."
        )


def _check_git() -> None:
    result = subprocess.run(["git", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "git is not installed or not on PATH. "
            "Install git and ensure it is on your PATH."
        )


def _gh(*args: str, input_data: str | None = None) -> str:
    """Run a gh CLI command and return stdout."""
    cmd = ["gh", *args]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        input=input_data,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout


def fetch_pr_info(pr_number: int) -> PRInfo:
    """Fetch PR metadata, diff, changed files, and labels via gh CLI."""
    # Fetch PR metadata
    pr_json = _gh(
        "pr",
        "view",
        str(pr_number),
        "--repo",
        REPO,
        "--json",
        "title,baseRefOid,headRefOid,labels,files",
    )
    pr_data = json.loads(pr_json)

    # Fetch diff
    diff = _gh("pr", "diff", str(pr_number), "--repo", REPO)
    changed_files = [f["path"] for f in pr_data.get("files", [])]
    labels = [l["name"] for l in pr_data.get("labels", [])]

    diff, truncated_files = _truncate_diff(diff, changed_files)
    if truncated_files:
        logger.warning(
            f"PR diff truncated to {MAX_DIFF_LINES} lines. "
            f"{len(truncated_files)} file(s) were cut and will not be analyzed: "
            f"{', '.join(truncated_files)}"
        )

    return PRInfo(
        number=pr_number,
        title=pr_data.get("title", ""),
        base_sha=pr_data.get("baseRefOid", ""),
        head_sha=pr_data.get("headRefOid", ""),
        diff=diff,
        changed_files=changed_files,
        labels=labels,
        truncated_files=truncated_files,
    )


def fetch_branch_diff(base: str = "main") -> PRInfo:
    """Generate a PRInfo from the local branch diff against a base branch."""
    merge_base = subprocess.run(
        ["git", "merge-base", base, "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    diff = subprocess.run(
        ["git", "diff", merge_base],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    changed_files = (
        subprocess.run(
            ["git", "diff", "--name-only", merge_base],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()
    )

    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    diff, truncated_files = _truncate_diff(diff, changed_files)
    if truncated_files:
        logger.warning(
            f"Branch diff truncated to {MAX_DIFF_LINES} lines. "
            f"{len(truncated_files)} file(s) were cut and will not be analyzed: "
            f"{', '.join(truncated_files)}"
        )

    return PRInfo(
        number=0,
        title=f"Local branch: {branch}",
        base_sha=merge_base,
        head_sha="HEAD",
        diff=diff,
        changed_files=changed_files,
        labels=[],
        truncated_files=truncated_files,
    )


def fetch_file_content(path: str, ref: str) -> Optional[str]:
    """Fetch a file's content at a specific git ref via gh CLI."""
    try:
        content = _gh(
            "api",
            f"repos/{REPO}/contents/{path}",
            "--jq",
            ".content",
            "-H",
            "Accept: application/vnd.github.v3+json",
            "--method",
            "GET",
            "-f",
            f"ref={ref}",
        )
        import base64

        return base64.b64decode(content.strip()).decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Failed to fetch {path}@{ref}: {e}")
        return None


def diff_file_paths(diff: str) -> set[str]:
    """Return the set of file paths present in a unified diff.

    Scans `diff --git` headers only — does not parse hunk contents.
    Useful for determining which files survived a truncation.
    """
    paths: set[str] = set()
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"^diff --git a/\S+ b/(\S+)", line)
            if m:
                paths.add(m.group(1))
    return paths


def _truncate_diff(diff: str, changed_files: list[str]) -> tuple[str, list[str]]:
    """Truncate a diff to MAX_DIFF_LINES and return (truncated_diff, truncated_files).

    truncated_files lists the changed_files entries whose diff sections were cut.
    Returns the original diff and an empty list when no truncation is needed.
    """
    lines = diff.splitlines()
    if len(lines) <= MAX_DIFF_LINES:
        return diff, []

    truncated = (
        "\n".join(lines[:MAX_DIFF_LINES])
        + "\n\n# [diff truncated — too large for full analysis]"
    )
    files_in_truncated = diff_file_paths(truncated)
    truncated_files = [f for f in changed_files if f not in files_in_truncated]
    return truncated, truncated_files


def diff_line_numbers(diff: str) -> dict[str, set[int]]:
    """Parse a unified diff and return valid RIGHT-side line numbers per file.

    Valid lines are those present in the new version of each file: added lines
    (+) and context lines ( ). Removed lines (-) are excluded because they do
    not exist on the RIGHT side and cannot be targets for inline review comments.

    Returns a mapping of file_path -> set[int], suitable for pre-validating
    inline comment positions before making GitHub API calls.
    """
    result: dict[str, set[int]] = {}
    current_file: str | None = None
    new_line: int = 0

    for line in diff.splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"^diff --git a/\S+ b/(\S+)", line)
            current_file = m.group(1) if m else None
            new_line = 0
        elif line.startswith("@@ "):
            m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if m:
                new_line = int(m.group(1)) - 1  # incremented before first use
        elif line.startswith("+") and not line.startswith("+++"):
            new_line += 1
            if current_file is not None:
                result.setdefault(current_file, set()).add(new_line)
        elif line.startswith("-") and not line.startswith("---"):
            pass  # removed lines: don't exist on RIGHT side, don't increment
        elif line.startswith(" "):
            new_line += 1
            if current_file is not None:
                result.setdefault(current_file, set()).add(new_line)
        # Lines starting with \ (e.g. "\ No newline at end of file") are skipped

    return result


def post_pr_comment(
    pr_number: int,
    body: str,
    path: str | None = None,
    line: int | None = None,
    commit_sha: str | None = None,
) -> None:
    """Post a comment on a PR — either inline (review comment) or general."""
    if path and line and commit_sha:
        # Post inline review comment
        _gh(
            "api",
            f"repos/{REPO}/pulls/{pr_number}/comments",
            "--method",
            "POST",
            "-f",
            f"body={body}",
            "-f",
            f"path={path}",
            "-f",
            f"commit_id={commit_sha}",
            "-F",
            f"line={line}",
            "-f",
            "side=RIGHT",
        )
    else:
        # Post general PR comment
        _gh(
            "pr",
            "comment",
            str(pr_number),
            "--repo",
            REPO,
            "--body",
            body,
        )
