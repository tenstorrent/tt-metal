"""GitHub REST API client for fetching PR data and posting comments."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

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
    diff_lines = diff.splitlines()
    if len(diff_lines) > MAX_DIFF_LINES:
        logger.warning(
            f"PR diff is {len(diff_lines)} lines; truncating to {MAX_DIFF_LINES}. "
            "Findings may be incomplete for files beyond the truncation point."
        )
        diff = "\n".join(diff_lines[:MAX_DIFF_LINES]) + "\n\n# [diff truncated — too large for full analysis]"

    changed_files = [f["path"] for f in pr_data.get("files", [])]
    labels = [l["name"] for l in pr_data.get("labels", [])]

    return PRInfo(
        number=pr_number,
        title=pr_data.get("title", ""),
        base_sha=pr_data.get("baseRefOid", ""),
        head_sha=pr_data.get("headRefOid", ""),
        diff=diff,
        changed_files=changed_files,
        labels=labels,
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
