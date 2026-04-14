#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Fetch Blackhole P2 issues from GitHub.

Uses the `gh` CLI to query tenstorrent/tt-llk for open issues labeled
"blackhole" + "P2". Outputs a JSON file with issue metadata (number, title,
body, labels, assignees, dates, URL) that downstream scripts can consume.

Usage:
    # Fetch all open Blackhole P2 issues (default)
    python scripts/fetch_bh_issues.py

    # Fetch and print summary table
    python scripts/fetch_bh_issues.py --summary

    # Save to custom path
    python scripts/fetch_bh_issues.py -o /tmp/bh_issues.json

    # Include closed issues too
    python scripts/fetch_bh_issues.py --state all

    # Filter by additional label
    python scripts/fetch_bh_issues.py --extra-label LLK

    # JSON output to stdout (for piping)
    python scripts/fetch_bh_issues.py --stdout
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = "tenstorrent/tt-llk"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "artifacts" / "bh_p2_issues.json"
)


def fetch_issues(
    state: str = "open",
    extra_labels: list[str] | None = None,
    limit: int = 200,
) -> list[dict]:
    """Fetch Blackhole P2 issues via gh CLI."""
    fields = "number,title,body,labels,assignees,createdAt,updatedAt,url,state,comments"

    cmd = [
        "gh",
        "issue",
        "list",
        "-R",
        REPO,
        "--label",
        "blackhole",
        "--label",
        "P2",
        "--state",
        state,
        "--limit",
        str(limit),
        "--json",
        fields,
    ]

    for label in extra_labels or []:
        cmd.extend(["--label", label])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"Error fetching issues: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    issues = json.loads(result.stdout)

    # Normalize structure for downstream consumers
    normalized = []
    for issue in issues:
        normalized.append(
            {
                "number": issue["number"],
                "title": issue["title"],
                "body": issue.get("body", ""),
                "url": issue.get(
                    "url", f"https://github.com/{REPO}/issues/{issue['number']}"
                ),
                "state": issue.get("state", state),
                "labels": [l["name"] for l in issue.get("labels", [])],
                "assignees": [
                    a.get("login", a.get("name", "unknown"))
                    for a in issue.get("assignees", [])
                ],
                "comment_count": (
                    issue.get("comments", {}).get("totalCount", 0)
                    if isinstance(issue.get("comments"), dict)
                    else len(issue.get("comments", []))
                ),
                "created_at": issue.get("createdAt", ""),
                "updated_at": issue.get("updatedAt", ""),
            }
        )

    # Sort by issue number (ascending)
    normalized.sort(key=lambda x: x["number"])
    return normalized


def print_summary(issues: list[dict]) -> None:
    """Print a formatted summary table of issues."""
    if not issues:
        print("No Blackhole P2 issues found.")
        return

    print(f"\n{'='*90}")
    print(f"  Blackhole P2 Issues  ({len(issues)} total)")
    print(f"{'='*90}")
    print(f"  {'#':<6} {'Title':<55} {'Assignee':<15} {'Labels'}")
    print(f"  {'-'*6} {'-'*55} {'-'*15} {'-'*20}")

    for issue in issues:
        num = f"#{issue['number']}"
        title = (
            issue["title"][:53] + ".." if len(issue["title"]) > 55 else issue["title"]
        )
        assignee = issue["assignees"][0] if issue["assignees"] else "(none)"
        if len(assignee) > 13:
            assignee = assignee[:11] + ".."
        extra_labels = [l for l in issue["labels"] if l not in ("blackhole", "P2")]
        labels_str = ", ".join(extra_labels[:3])
        print(f"  {num:<6} {title:<55} {assignee:<15} {labels_str}")

    print(f"{'='*90}")

    assigned = sum(1 for i in issues if i["assignees"])
    unassigned = len(issues) - assigned
    print(f"  Assigned: {assigned}  |  Unassigned: {unassigned}")
    print(f"{'='*90}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Blackhole P2 issues from tenstorrent/tt-llk"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--state",
        default="open",
        choices=["open", "closed", "all"],
        help="Issue state filter (default: open)",
    )
    parser.add_argument(
        "--extra-label",
        action="append",
        dest="extra_labels",
        help="Additional label filter (can be repeated)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max issues to fetch (default: 200)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a formatted summary table",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Output JSON to stdout instead of file",
    )
    args = parser.parse_args()

    issues = fetch_issues(
        state=args.state,
        extra_labels=args.extra_labels,
        limit=args.limit,
    )

    metadata = {
        "repo": REPO,
        "query": {
            "labels": ["blackhole", "P2"] + (args.extra_labels or []),
            "state": args.state,
        },
        "fetched_at": datetime.now().isoformat(),
        "count": len(issues),
        "issues": issues,
    }

    if args.stdout:
        json.dump(metadata, sys.stdout, indent=2)
        print()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {len(issues)} issues to {args.output}")

    if args.summary:
        print_summary(issues)

    return issues


if __name__ == "__main__":
    main()
