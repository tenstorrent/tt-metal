# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Format check for the knowledge files (run by the orchestrator after every agent step).

    python -m models.demos.common.bringup.knowledge.check [--known-issues PATH] [--repo-map PATH] [--model-dir DIR]

Shared known_issues.md: the six sections; every bullet has Symptom:, Cause:, Fix:, Found:, the anchors Applies to: and
Seen at:, and stays within SHARED_MAX_CHARS. A model's bringup/known_issues.md (--model-dir): bullets with Symptom:,
Cause:, Fix:, Found: (no sections, no anchors required). repo_map.md (shared and the model's): a table with a Need and
a Where column; every backticked repo path in it exists.
Records known_issue_errors, repo_map_errors, model_entries (the model's known issues, for the run-end curation).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.core.spec import CODE_ROOT

HERE = Path(__file__).resolve().parent
SECTIONS = ["API behavior", "Accuracy", "Testing", "Performance", "Infrastructure", "Serving contract"]
FIELDS = ("Symptom:", "Cause:", "Fix:", "Found:")
ANCHORS = ("Applies to:", "Seen at:")
# Every agent step reads the shared file in full: one lesson is one short bullet (detail goes to the model's file).
SHARED_MAX_CHARS = 800
# A model's agents read its file in full too: one short bullet per lesson, the full text in known_issues_detail.md
# (a `## <title>` section, pointed at by `Detail: known_issues_detail.md.` in the bullet).
MODEL_MAX_CHARS = 600


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
    """The shared file: (errors, number of entries)."""
    ent = entries(path.read_text())
    errs = [f"missing section '{s}'" for s in SECTIONS if s not in ent]
    errs += [f"unknown section '{s}'" for s in ent if s not in SECTIONS]
    for sec, items in ent.items():
        for it in items:
            miss = [f for f in FIELDS + ANCHORS if f not in it]
            if miss:
                errs.append(f"{sec}: entry {it[:60]!r} lacks {miss}")
            if len(it) > SHARED_MAX_CHARS:
                errs.append(f"{sec}: entry {it[:60]!r} is {len(it)} chars (max {SHARED_MAX_CHARS})")
    return errs, sum(len(v) for v in ent.values())


def check_model_known_issues(path: Path) -> tuple[list[str], int]:
    """A model's bringup/known_issues.md (absent is fine): (errors, number of entries)."""
    if not path.exists():
        return [], 0
    items = [line for line in path.read_text().splitlines() if line.startswith("- ")]
    errs = [f"entry {it[:60]!r} lacks {miss}" for it in items if (miss := [f for f in FIELDS if f not in it])]
    errs += [
        f"entry {it[:60]!r} is {len(it)} chars (max {MODEL_MAX_CHARS}; put the rest in known_issues_detail.md)"
        for it in items
        if len(it) > MODEL_MAX_CHARS
    ]
    detail = path.with_name("known_issues_detail.md")
    sections = (
        {s.strip().rstrip(".") for s in re.findall(r"^## (.+)$", detail.read_text(), re.M)}
        if detail.exists()
        else set()
    )
    for it in items:
        if "Detail: known_issues_detail.md" in it:
            m = re.match(r"- \*\*(.+?)\*\*", it)
            title = m.group(1).strip().rstrip(".") if m else ""
            if title not in sections:
                errs.append(f"entry {it[:60]!r} points at known_issues_detail.md, which has no '## {title}' section")
    return errs, len(items)


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
    ap.add_argument("--model-dir", help="also check <dir>/bringup/known_issues.md and repo_map.md")
    a = ap.parse_args(argv)
    ki, _ = check_known_issues(Path(a.known_issues))
    rm = check_repo_map(Path(a.repo_map))
    model_entries = 0
    if a.model_dir:
        b = Path(a.model_dir) / "bringup"
        mki, model_entries = check_model_known_issues(b / "known_issues.md")
        ki += [f"{b / 'known_issues.md'}: {e}" for e in mki]
        if (b / "repo_map.md").exists():
            rm += [f"{b / 'repo_map.md'}: {e}" for e in check_repo_map(b / "repo_map.md")]
    for e in ki:
        print(f"KNOWN ISSUES {e}")
    for e in rm:
        print(f"REPO MAP {e}")
    metrics.record("known_issue_errors", len(ki))
    metrics.record("repo_map_errors", len(rm))
    metrics.record("model_entries", model_entries)
    return 1 if ki or rm else 0


if __name__ == "__main__":
    raise SystemExit(main())
