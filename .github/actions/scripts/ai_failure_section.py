#!/usr/bin/env python3
"""Render the AI Summary's root causes as a Jira ticket section.

The Package and release run publishes an ``ai_run_summary`` artifact (the
ai_summary/run action); ci_digest already knows how to fetch it for an
arbitrary run id. This prints its failure rows as the ticket's
"Other information" section, RELEASE-7 style -- or nothing when the run
carries no summary, so the ticket is still filed without it.

Usage: ai_failure_section.py <owner/repo> <run_id>   (needs GH_TOKEN)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "scripts")))

from ci_digest import fetch_run_summary  # noqa: E402


def _one_line(text, limit=300):
    """Bullets render one line each; collapse whitespace and bound the length."""
    line = " ".join(str(text).split())
    return line if len(line) <= limit else line[: limit - 3] + "..."


def section(data):
    """The "Other information" section for a run-summary JSON, or ""."""
    rows = list((data or {}).get("failed") or []) + list((data or {}).get("infra_failure") or [])
    lines = []
    for row in rows:
        cause = row.get("root_cause") or row.get("error_message")
        if not cause:
            continue
        note = f"- {row.get('job_name') or 'unknown job'}: {_one_line(cause)}"
        if row.get("category"):
            note += f" [{row['category']}]"
        if row.get("log_complete") is False:
            note += " (log incomplete)"
        lines.append(note)
    if not lines:
        return ""
    header = ["### Other information", "Root cause analysis from the run's AI Summary:"]
    return "\n".join(header + lines) + "\n"


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: ai_failure_section.py <owner/repo> <run_id>")
    out = section(fetch_run_summary(sys.argv[1], int(sys.argv[2])))
    if out:
        print(out, end="")


if __name__ == "__main__":
    main()
