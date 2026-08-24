#!/usr/bin/env python3
"""
Flag pipeline-reorg tests whose owner needs reassignment.

Given the set of Slack user IDs that are currently active, this scans
tests/pipeline_reorg/*.yaml and reports tests whose `owner_id` is NOT in
that active set, plus every test with an empty/missing owner. A test is flagged
when its owner is any of:
  - deactivated (Slack account marked DEACTIVATED / removed from the active set),
  - metal-infra, or
  - unassigned (owner_id blank or absent).

For each flagged owner the report also names the target team lead pulled from
.github/TESTOWNERS, keyed on each test's `team`, so the escalation renderer can
spell out the exact reassignment. Output is JSON; rendering the GitHub issue
body lives in render_escalation_issue.py.

Input (active IDs) accepts either form, auto-detected:
  - a JSON array of Slack ID strings: ["U123", "U456"]
  - a raw Slack users.list `members` array (objects with `id`/`deleted`);
    active = members where `deleted` is falsy.

Usage:
    python verify_test_owner.py --active-ids-file active.json
    python verify_test_owner.py --active-ids U123,U456
"""

import argparse
import json
import os
import re
import sys

from tests_by_owner import (
    DEFAULT_TESTS_DIR,
    REPO_ROOT,
    parse_all_tests,
    tests_for_owner,
)

DEFAULT_TESTOWNERS = os.path.join(REPO_ROOT, ".github", "TESTOWNERS")


def load_active_ids(args):
    """Resolve the set of active Slack IDs from CLI args."""
    if args.active_ids:
        return {i.strip() for i in args.active_ids.split(",") if i.strip()}

    with open(args.active_ids_file, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("::error::active-ids-file must contain a JSON array", file=sys.stderr)
        sys.exit(1)

    ids = set()
    for item in data:
        if isinstance(item, str):
            ids.add(item)
        elif isinstance(item, dict) and item.get("id") and not item.get("deleted"):
            # Raw Slack users.list member object; keep only non-deactivated.
            ids.add(item["id"])
    return ids


_LEAD_RE = re.compile(r"^\s+([\w-]+):\s*(\S+)\s*(?:#\s*(.*\S))?\s*$")


def load_team_leads(testowners_path):
    """Return {team: {"github": handle, "slack_id": id, "name": name}}.

    A lead is treated as unset (all fields None) when its handle is missing or
    still a REPLACE_ME placeholder, so escalation output can flag that
    explicitly.
    """
    leads = {}
    if not os.path.exists(testowners_path):
        return leads

    with open(testowners_path, "r") as f:
        for line in f:
            m = _LEAD_RE.match(line.rstrip("\n"))
            if not m:
                continue
            team, handle, comment = m.group(1), m.group(2), m.group(3)
            if handle == "REPLACE_ME":
                leads[team] = {"github": None, "slack_id": None, "name": None}
                continue
            slack_id, name = None, None
            if comment and "," in comment:
                slack_id, name = (p.strip() for p in comment.split(",", 1))
            leads[team] = {"github": handle, "slack_id": slack_id, "name": name}
    return leads


def find_deactivated_owners(active_ids, tests_dir, team_leads):
    """Group flagged tests by owner. Only owners with >=1 test are returned.

    An owner is flagged when its id is not in active_ids; the empty-owner
    sentinel ("") is never active, so unassigned tests are grouped under it.
    """
    all_tests = parse_all_tests(tests_dir)

    flagged_ids = sorted({t["owner_id"] for t in all_tests if t["owner_id"] not in active_ids})

    owners = []
    for owner_id in flagged_ids:
        owned = tests_for_owner(owner_id, all_tests=all_tests)
        if not owned:
            continue
        name = next((t["owner_name"] for t in owned if t["owner_name"]), None)
        tests = []
        for t in owned:
            lead = team_leads.get(t["team"], {})
            tests.append(
                {
                    "name": t["name"],
                    "file": t["file"],
                    "line": t["line"],
                    "team": t["team"],
                    "suggested_lead_github": lead.get("github"),
                    "suggested_lead_name": lead.get("name"),
                }
            )
        owners.append(
            {
                "owner_id": owner_id,
                "name": name,
                "test_count": len(tests),
                "tests": tests,
            }
        )

    owners.sort(key=lambda o: o["test_count"], reverse=True)
    return owners


def main(argv=None):
    parser = argparse.ArgumentParser(description="Flag pipeline-reorg tests whose owner needs reassignment (JSON).")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--active-ids-file",
        help="JSON file: array of active Slack IDs, or a Slack users.list members array.",
    )
    src.add_argument(
        "--active-ids",
        help="Comma-separated list of active Slack IDs.",
    )
    parser.add_argument("--tests-dir", default=DEFAULT_TESTS_DIR)
    parser.add_argument("--testowners", default=DEFAULT_TESTOWNERS)
    args = parser.parse_args(argv)

    active_ids = load_active_ids(args)
    if not active_ids:
        print(
            "::error::no active Slack IDs resolved; refusing to flag every owner",
            file=sys.stderr,
        )
        sys.exit(1)

    team_leads = load_team_leads(args.testowners)
    owners = find_deactivated_owners(active_ids, args.tests_dir, team_leads)

    print(json.dumps({"deactivated_owners": owners}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
