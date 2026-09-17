#!/usr/bin/env python3
"""Resolve a failed CI job to its owner via tests/pipeline_reorg.

Every pipeline_reorg entry carries `owner_id: U... # Display Name` and `team:`
(100% coverage, enforced by models/MIGRATING_TO_TIERED_CI.md). The display
name lives in the comment, which yaml.safe_load drops, so a line regex
recovers owner_id -> display name and the rest of the entry is read as
plain YAML. An entry whose owner comment is missing is treated as
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

import yaml

from failure_title import _RUNNER_TAG

PIPELINE_REORG = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tests", "pipeline_reorg")
)

# Owner comments sit at entry indent; the deeper indent of cmd block scalars
# keeps their content from matching.
_OWNER_COMMENT_RE = re.compile(r"^(?:- |  )owner_id:\s*(\S+)\s*#\s*(.*\S)", re.M)


def _load(root=None):
    """name -> {"owner", "team"} across every pipeline_reorg file.

    Sorted file order and first-hit-wins keep duplicate names deterministic.
    """
    table = {}
    for path in sorted(glob.glob(os.path.join(root or PIPELINE_REORG, "*.yaml"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        owners = dict(_OWNER_COMMENT_RE.findall(text))
        entries = yaml.safe_load(text)
        # Not every file is a test matrix (ttsim-skip-list.yaml is a mapping).
        for entry in entries if isinstance(entries, list) else []:
            owner = owners.get(entry.get("owner_id"))
            if entry.get("name") and owner and entry.get("team"):
                table.setdefault(entry["name"], {"owner": owner, "team": entry["team"]})
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
