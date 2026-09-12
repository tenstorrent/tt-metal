"""GitHub issue access via the `gh` CLI."""

from __future__ import annotations

import json
import shutil
import subprocess

DEFAULT_REPO = "tenstorrent/tt-metal"
MAX_BODY_CHARS = 60_000


class IssueFetchFailed(RuntimeError):
    pass


def check_prerequisites() -> None:
    if shutil.which("gh") is None:
        raise IssueFetchFailed("`gh` CLI not found on PATH.")


def fetch_issue(number: int, repo: str = DEFAULT_REPO) -> dict:
    """Fetch the issue body and metadata.

    Comments are deliberately excluded. A maintainer may already have posted the
    right answer, and feeding that to the planner would let it copy a conclusion
    instead of deriving an experiment.
    """
    check_prerequisites()
    cmd = [
        "gh",
        "issue",
        "view",
        str(number),
        "--repo",
        repo,
        "--json",
        "number,title,body,author,labels,state,url,createdAt",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise IssueFetchFailed(f"gh issue view {number} failed: {completed.stderr.strip()[:500]}")

    issue = json.loads(completed.stdout)
    issue["author"] = (issue.get("author") or {}).get("login", "unknown")
    issue["labels"] = [label["name"] for label in issue.get("labels") or []]

    body = issue.get("body") or ""
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n\n[truncated by issue_verifier]"
    issue["body"] = body

    return issue
