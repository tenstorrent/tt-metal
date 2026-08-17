#!/usr/bin/env python3
"""
List every pipeline-reorg test owned by a given Slack user, as JSON.

Scans tests/pipeline_reorg/*.yaml and returns, for each test whose `owner_id`
matches the requested Slack ID, the test's:
  - name
  - pipeline-reorg file (repo-relative path)
  - line number of the entry's "- name:" declaration
  - owning team

Usage:
    python tests_by_owner.py U03PUAKE719
    python tests_by_owner.py U03PUAKE719 --tests-dir tests/pipeline_reorg
"""

import argparse
import json
import os
import re
import sys

# .github/scripts/utils/tests_by_owner.py -> repo root is four levels up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_TESTS_DIR = os.path.join(REPO_ROOT, "tests", "pipeline_reorg")

# Entries are top-level YAML sequence items, so a new item begins with a dash in
# column 0. Keys (name/team/owner_id) are indented beneath it. We parse per
# item-block instead of using PyYAML so we can report exact line numbers.
_NAME_RE = re.compile(r"^\s*-?\s*name:\s*(.*\S)\s*$")
_TEAM_RE = re.compile(r"^\s*team:\s*([^#\s]+)")
_OWNER_RE = re.compile(r"^\s*owner_id:\s*(U\w+)\s*(?:#\s*(.*\S))?\s*$")


def _rel(path):
    """Repo-relative path when possible, else the path as given."""
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:
        return path


def parse_file(path):
    """Parse a single pipeline-reorg YAML file into a list of test dicts.

    Each returned dict has: name, team, owner_id, owner_name, file, line.
    A test entry is one that declares a `name`. Entries whose `owner_id` is
    missing or blank are still returned with owner_id set to "" (empty sentinel)
    so unowned tests can be flagged for reassignment; nameless sequence items
    (e.g. skip-list paths) are skipped.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    tests = []
    current = None
    rel_path = _rel(path)

    def flush():
        if current is None:
            return
        if current["owner_id"]:
            tests.append(current)
        elif current["name"]:
            # Named test with no resolvable owner: flag as empty-owner.
            current["owner_id"] = ""
            tests.append(current)

    for lineno, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")

        # A dash in column 0 starts a new sequence item. cmd block-scalar lines
        # are indented, so they never collide with this.
        if line.startswith("-"):
            flush()
            current = {
                "name": None,
                "team": None,
                "owner_id": None,
                "owner_name": None,
                "file": rel_path,
                "line": lineno,
            }
            m = _NAME_RE.match(line)
            if m:
                current["name"] = m.group(1)
            continue

        if current is None:
            continue

        if current["name"] is None:
            m = _NAME_RE.match(line)
            if m:
                current["name"] = m.group(1)
                current["line"] = lineno
                continue

        m = _TEAM_RE.match(line)
        if m:
            current["team"] = m.group(1)
            continue

        m = _OWNER_RE.match(line)
        if m:
            current["owner_id"] = m.group(1)
            current["owner_name"] = m.group(2)  # may be None if no "# Name"

    flush()
    return tests


def parse_all_tests(tests_dir=DEFAULT_TESTS_DIR):
    """Parse every *.yaml under tests_dir. Returns a flat list of test dicts."""
    if not os.path.isdir(tests_dir):
        print(f"::error::tests dir not found: {tests_dir}", file=sys.stderr)
        sys.exit(1)

    all_tests = []
    for fname in sorted(os.listdir(tests_dir)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        all_tests.extend(parse_file(os.path.join(tests_dir, fname)))
    return all_tests


def tests_for_owner(owner_id, tests_dir=DEFAULT_TESTS_DIR, all_tests=None):
    """Return the subset of tests owned by owner_id."""
    if all_tests is None:
        all_tests = parse_all_tests(tests_dir)
    return [t for t in all_tests if t["owner_id"] == owner_id]


def main(argv=None):
    parser = argparse.ArgumentParser(description="List pipeline-reorg tests owned by a Slack user ID (JSON).")

    parser.add_argument("owner_id", help="Slack user ID (e.g. U03PUAKE719)")
    parser.add_argument(
        "--tests-dir",
        default=DEFAULT_TESTS_DIR,
        help="Directory of pipeline-reorg test YAMLs (default: tests/pipeline_reorg)",
    )
    args = parser.parse_args(argv)

    tests = tests_for_owner(args.owner_id, tests_dir=args.tests_dir)
    print(json.dumps(tests, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
