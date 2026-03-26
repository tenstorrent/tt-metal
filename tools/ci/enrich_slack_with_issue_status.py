#!/usr/bin/env python3
"""Enrich exported Slack thread JSON with GitHub issue closure booleans."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any


ISSUE_URL_RE = re.compile(r"https://github\.com/tenstorrent/tt-metal/issues/(\d+)")


def fetch_issue(issue_number: int, github_token: str | None) -> dict[str, Any]:
    url = f"https://api.github.com/repos/tenstorrent/tt-metal/issues/{issue_number}"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "tt-metal-triage-ci",
    }
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add issue_closed booleans to Slack export JSON.")
    parser.add_argument("--input", required=True, help="Input Slack export JSON path")
    parser.add_argument("--output", required=True, help="Output enriched JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    github_token = os.environ.get("GITHUB_TOKEN")

    with open(args.input, encoding="utf-8") as f:
        payload = json.load(f)

    messages = payload.get("messages", [])
    issue_numbers: set[int] = set()
    for msg in messages:
        for issue_num in ISSUE_URL_RE.findall(str(msg.get("text", ""))):
            issue_numbers.add(int(issue_num))

    issue_state_map: dict[int, dict[str, Any]] = {}
    failures: dict[int, str] = {}
    for n in sorted(issue_numbers):
        try:
            issue = fetch_issue(n, github_token)
            issue_state_map[n] = {
                "state": issue.get("state"),
                "closed": issue.get("state") == "closed",
                "url": issue.get("html_url"),
                "title": issue.get("title"),
            }
        except urllib.error.HTTPError as exc:
            failures[n] = f"http_error_{exc.code}"
        except Exception as exc:  # noqa: BLE001
            failures[n] = str(exc)

    for msg in messages:
        refs = sorted({int(x) for x in ISSUE_URL_RE.findall(str(msg.get("text", "")))})
        refs_detail = []
        for n in refs:
            state = issue_state_map.get(n)
            if state is None:
                refs_detail.append(
                    {
                        "number": n,
                        "url": f"https://github.com/tenstorrent/tt-metal/issues/{n}",
                        "state": "unknown",
                        "closed": False,
                    }
                )
            else:
                refs_detail.append(
                    {
                        "number": n,
                        "url": state["url"],
                        "state": state["state"],
                        "closed": state["closed"],
                    }
                )

        msg["referenced_issue_numbers"] = refs
        msg["referenced_issues"] = refs_detail
        msg["issue_closed"] = any(item.get("closed", False) for item in refs_detail)
        msg["all_referenced_issues_closed"] = bool(refs_detail) and all(
            item.get("closed", False) for item in refs_detail
        )
        msg["issue_status_lookup_failed"] = any(n in failures for n in refs)

    payload["issue_enrichment"] = {
        "github_lookup_count": len(issue_numbers),
        "github_lookup_failures": {str(k): v for k, v in failures.items()},
        "notes": "issue_closed=true means at least one referenced top-level GitHub issue is closed",
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        json.dumps(
            {
                "messages": len(messages),
                "issues_found": len(issue_numbers),
                "lookups_failed": len(failures),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
