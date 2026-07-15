#!/usr/bin/env python3
"""
Render the test-owner escalation GitHub issue body from a report JSON.

Tests are grouped by the team lead they should move to (per .github/TESTOWNERS),
and each is linked to the exact start line of its entry. Tests whose team has no
lead set are grouped under a warning so they are not silently dropped.

Usage:
    python render_escalation_issue.py --report report.json --output body.md \
        --repo owner/name --ref <sha-or-branch>
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

BRAIN_FOOTER = (
    "@brAIn Please review all listed tests and the reassigned owners. Please "
    "update the `owner` field of all identified tests to match the slack "
    "credentials of the newly indicated owner. If an owner field does not "
    "exist, please create one. The owner entry should follow an "
    "<owner_slack_id> # <owner_name> pattern. For credentials of the new test "
    "owner, please consult .github/TESTOWNERS"
)


def _blob_url(repo, ref, file, line):
    """Permalink to a file line on GitHub (line falls back to the file top)."""
    anchor = f"#L{line}" if line else ""
    return f"https://github.com/{repo}/blob/{ref}/{file}{anchor}"


def render_issue(owners, repo, ref):
    """Render the escalation issue body (Markdown) for the flagged owners."""
    if not owners:
        return "No pipeline-reorg tests need owner reassignment. :tada:"

    lines = ["The following test owners are no longer valid test owners:"]
    for o in owners:
        if o["owner_id"]:
            who = o["name"] or "(unknown name)"
            lines.append(f"- {who} (`{o['owner_id']}`) — {o['test_count']} test(s)")
        else:
            lines.append(f"- No owner assigned — {o['test_count']} test(s)")
    lines.append("")

    # Group every flagged test by the GitHub handle of its new owner (the team
    # lead). Tests whose team has no lead set fall into a separate bucket keyed
    # by team so they are surfaced rather than dropped.
    by_lead = OrderedDict()
    no_lead = OrderedDict()
    for o in owners:
        for t in o["tests"]:
            entry = (t["name"], _blob_url(repo, ref, t["file"], t["line"]))
            handle = t.get("suggested_lead_github")
            if handle:
                by_lead.setdefault(handle, []).append(entry)
            else:
                no_lead.setdefault(t["team"], []).append(entry)

    for handle, tests in by_lead.items():
        lines.append(f"{handle} you have been assigned the following tests:")
        for name, url in tests:
            lines.append(f"- [{name}]({url})")
        lines.append("")

    for team, tests in no_lead.items():
        lines.append(
            f"⚠️ Team `{team}` has no lead set in `.github/TESTOWNERS` — assign "
            "one, then reassign the following tests:"
        )
        for name, url in tests:
            lines.append(f"- [{name}]({url})")
        lines.append("")

    lines.append(BRAIN_FOOTER)
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render the escalation issue body from a verify report.")

    parser.add_argument("--report", required=True, help="Path to the report JSON.")
    parser.add_argument("--output", help="File to write the body to (default: stdout).")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="owner/name for building test links (default: $GITHUB_REPOSITORY).",
    )
    parser.add_argument(
        "--ref",
        default=os.environ.get("GITHUB_SHA") or "main",
        help="Commit SHA or branch for test links (default: $GITHUB_SHA or main).",
    )
    args = parser.parse_args(argv)

    if not args.repo:
        print("::error::--repo (or $GITHUB_REPOSITORY) is required", file=sys.stderr)
        sys.exit(1)

    with open(args.report) as f:
        report = json.load(f)
    body = render_issue(report.get("deactivated_owners", []), args.repo, args.ref)

    if args.output:
        with open(args.output, "w") as f:
            f.write(body)
    else:
        print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
