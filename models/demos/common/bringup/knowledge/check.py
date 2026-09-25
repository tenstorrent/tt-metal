# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Format check for the shared knowledge files (run by the orchestrator after every agent step).

    python -m models.demos.common.bringup.knowledge.check [--known-issues PATH] [--repo-map PATH]

known_issues.md: the five sections plus "Proposed", every bullet has Symptom:, Cause:, Fix:, Found:.
repo_map.md: a table with a Need and a Where column; every backticked repo path in it exists.
Records known_issue_errors, repo_map_errors, proposed_entries.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.core.spec import CODE_ROOT

HERE = Path(__file__).resolve().parent
SECTIONS = ["API behavior", "Accuracy", "Performance", "Infrastructure", "Serving contract", "Proposed"]
FIELDS = ("Symptom:", "Cause:", "Fix:", "Found:")


def entries(text: str) -> dict[str, list[str]]:
    out, cur = {}, None
    for line in text.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            out[cur] = []
        elif line.startswith("- ") and cur:
            out[cur].append(line)
    return out


def check_known_issues(path: Path) -> tuple[list[str], int]:
    ent = entries(path.read_text())
    errs = [f"missing section '{s}'" for s in SECTIONS if s not in ent]
    errs += [f"unknown section '{s}'" for s in ent if s not in SECTIONS]
    for sec, items in ent.items():
        for it in items:
            miss = [f for f in FIELDS if f not in it]
            if miss:
                errs.append(f"{sec}: entry {it[:60]!r} lacks {miss}")
    return errs, len(ent.get("Proposed", []))


def check_repo_map(path: Path, root: Path = CODE_ROOT) -> list[str]:
    text = path.read_text()
    errs = [] if re.search(r"^\| *Need *\| *Where", text, re.M) else ["no '| Need | Where' table"]
    for p in re.findall(r"`((?:models|ttnn|tt_metal|scripts|tests)/[^`\s:]+)", text):
        if not (root / p).exists():
            errs.append(f"path does not exist: {p}")
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-issues", default=str(HERE / "known_issues.md"))
    ap.add_argument("--repo-map", default=str(HERE / "repo_map.md"))
    a = ap.parse_args(argv)
    ki, proposed = check_known_issues(Path(a.known_issues))
    rm = check_repo_map(Path(a.repo_map))
    for e in ki:
        print(f"KNOWN ISSUES {e}")
    for e in rm:
        print(f"REPO MAP {e}")
    metrics.record("known_issue_errors", len(ki))
    metrics.record("repo_map_errors", len(rm))
    metrics.record("proposed_entries", proposed)
    return 1 if ki or rm else 0


if __name__ == "__main__":
    raise SystemExit(main())
