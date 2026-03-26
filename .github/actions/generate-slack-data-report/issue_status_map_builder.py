#!/usr/bin/env python3
"""Build a GitHub issue status map from Slack top-level messages."""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


ISSUE_URL_RE = re.compile(r"https://github\.com/tenstorrent/tt-metal/issues/(\d+)")


def fetch_issue(issue_number: int) -> dict:
    url = f"https://api.github.com/repos/tenstorrent/tt-metal/issues/{issue_number}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "tt-metal-triage-ci",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_issue_numbers(messages: list[dict]) -> set[int]:
    issue_numbers: set[int] = set()
    for msg in messages:
        text = str(msg.get("text", ""))
        for match in ISSUE_URL_RE.findall(text):
            issue_numbers.add(int(match))
    return issue_numbers


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: issue_status_map_builder.py <slack_json_path> <output_json_path>", file=sys.stderr)
        return 2

    slack_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    payload = json.loads(slack_path.read_text(encoding="utf-8"))
    messages = payload.get("messages", [])
    issue_numbers = sorted(extract_issue_numbers(messages))

    issues: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for n in issue_numbers:
        try:
            issue = fetch_issue(n)
            issues[str(n)] = {
                "url": issue.get("html_url"),
                "state": issue.get("state"),
                "closed_at": issue.get("closed_at"),
                "title": issue.get("title"),
            }
        except urllib.error.HTTPError as exc:
            failures[str(n)] = f"http_error_{exc.code}"
        except Exception as exc:  # noqa: BLE001
            failures[str(n)] = str(exc)

    out = {
        "source_slack_json": str(slack_path),
        "issue_count": len(issue_numbers),
        "issues": issues,
        "fetch_failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote issue status map to {output_path}")
    print(
        json.dumps(
            {
                "issue_count": len(issue_numbers),
                "fetched": len(issues),
                "failed": len(failures),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
