#!/usr/bin/env python3
"""Resolve a failed CI job to its owner via tests/pipeline_reorg.

Every pipeline_reorg entry carries `owner_id: U... # Display Name` and `team:`
(100% coverage, enforced by models/MIGRATING_TO_TIERED_CI.md). The display
name lives in the comment, which yaml.safe_load drops, so entries are
line-scanned with the same regex approach as .github/scripts/utils/
tests_by_owner.py. An entry whose owner comment is missing is treated as
unresolvable rather than surfacing a raw Slack id in a ticket.

The join key is the job's leaf name minus the runner tag:
"Llama 3.1-8B e2e tests [bh_p150]" matches the entry named
"Llama 3.1-8B e2e tests".

CLI: job leaf names on stdin, one per line; writes
"name<TAB>owner<TAB>team" for each resolvable input (unresolvable inputs
produce no line). Library: lookup(job_leaf) -> {"owner", "team"} | None.
"""

from __future__ import annotations

import glob
import os
import re
import sys

from failure_title import _RUNNER_TAG

PIPELINE_REORG = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tests", "pipeline_reorg")
)

# Entry fields sit at two-space indent (or on the "- " line itself); the
# deeper indent of cmd block scalars keeps their content from matching.
_ENTRY_RE = re.compile(r"^- ")
_NAME_RE = re.compile(r"^(?:- |  )name:\s*(.+?)\s*$")
_OWNER_RE = re.compile(r"^(?:- |  )owner_id:\s*\S+\s*#\s*(.*\S)\s*$")
_TEAM_RE = re.compile(r"^(?:- |  )team:\s*(\S+)\s*$")


def _load(root=None):
    """name -> {"owner", "team"} across every pipeline_reorg file.

    Sorted file order and first-hit-wins keep duplicate names deterministic.
    """
    table = {}
    for path in sorted(glob.glob(os.path.join(root or PIPELINE_REORG, "*.yaml"))):
        current = {}

        def commit():
            if current.get("name") and current.get("owner") and current.get("team"):
                table.setdefault(current["name"], {"owner": current["owner"], "team": current["team"]})

        with open(path, encoding="utf-8") as f:
            for line in f:
                if _ENTRY_RE.match(line):
                    commit()
                    current = {}
                for key, rx in (("name", _NAME_RE), ("owner", _OWNER_RE), ("team", _TEAM_RE)):
                    m = rx.match(line)
                    if m:
                        current[key] = m.group(1)
                        break
            commit()
    return table


_TABLE = None


def lookup(job_leaf, root=None):
    """Owner record for one job leaf name, or None when unmapped."""
    global _TABLE
    if root is not None:
        return _load(root).get(_RUNNER_TAG.sub("", job_leaf).strip())
    if _TABLE is None:
        _TABLE = _load()
    return _TABLE.get(_RUNNER_TAG.sub("", job_leaf).strip())


def main():
    for line in sys.stdin:
        leaf = line.strip()
        if not leaf:
            continue
        hit = lookup(leaf)
        if hit:
            print(f"{leaf}\t{hit['owner']}\t{hit['team']}")


if __name__ == "__main__":
    main()
