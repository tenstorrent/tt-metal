# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Format check for the knowledge files (run by the orchestrator after every agent step).

    python -m models.demos.common.bringup.knowledge.check [--known-issues PATH] [--repo-map PATH] [--model-dir DIR]

Shared known_issues.md: the six sections; every bullet has Symptom:, Cause:, Fix:, Found:, the anchors Applies to: and
Seen at:, and stays within SHARED_MAX_CHARS. A model's bringup/known_issues.md (--model-dir): bullets with Symptom:,
Cause:, Fix:, Found: (no sections, no anchors required). repo_map.md (shared and the model's): a table with a Need and
a Where column; every backticked repo path in it exists.
Every entry starts with its id, `- **[K-12] Title.**`: unique over the shared file and every model's file, never
reused or renumbered (the knowledge-use reports count by id). A new shared entry takes `K-<next>`, a model's the
prefix of that model's file (GLM-, HY4-, ...) and its next number; a promoted or moved entry keeps its id. The check
prints the next free id of each file it reads.
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
ID_RE = re.compile(r"^- \*\*\[([A-Z][A-Z0-9]*-\d+)\] ")
TITLE_RE = re.compile(r"^- \*\*(?:\[[A-Z][A-Z0-9]*-\d+\] )?(.+?)\*\*")
SHARED_PREFIX = "K"
SHARED_FILE = "models/demos/common/bringup/knowledge/known_issues.md"


def entry_id(line: str) -> str:
    m = ID_RE.match(line)
    return m.group(1) if m else ""


def entry_title(line: str) -> str:
    """The bold title without its id."""
    m = TITLE_RE.match(line)
    return m.group(1) if m else ""


def body_len(line: str) -> int:
    """An entry's length without its id (the limits predate the ids)."""
    m = ID_RE.match(line)
    return len(line) - (len(m.group(0)) - len("- **") if m else 0)


def knowledge_files(root: Path = CODE_ROOT) -> list[Path]:
    """The shared known_issues.md, then every model's bringup/known_issues.md."""
    return [root / SHARED_FILE] + sorted(root.glob("models/demos/*/bringup/known_issues.md"))


def check_ids(files: list[Path]) -> tuple[list[str], dict[str, str]]:
    """Every entry has an id, unique over all files. Returns (errors, file -> next free id of its own prefix)."""
    errs, seen, nxt = [], {}, {}
    for f in files:
        if not f.exists():
            continue
        prefixes = []
        for line in f.read_text().splitlines():
            if not line.startswith("- "):
                continue
            i = entry_id(line)
            if not i:
                errs.append(f"{f}: entry {line[:60]!r} has no id `- **[<PREFIX>-<n>] ...`")
                continue
            if i in seen:
                errs.append(f"id {i} is used twice ({seen[i]} and {f})")
            seen[i] = str(f)
            prefixes.append(i.split("-")[0])
        own = SHARED_PREFIX if str(f).endswith(SHARED_FILE) else max(set(prefixes), key=prefixes.count, default="")
        if own:
            nxt[str(f)] = f"{own}-{max((int(i.split('-')[1]) for i in seen if i.split('-')[0] == own), default=0) + 1}"
    return errs, nxt


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
            if body_len(it) > SHARED_MAX_CHARS:
                errs.append(f"{sec}: entry {it[:60]!r} is {body_len(it)} chars (max {SHARED_MAX_CHARS})")
    return errs, sum(len(v) for v in ent.values())


def check_model_known_issues(path: Path) -> tuple[list[str], int]:
    """A model's bringup/known_issues.md (absent is fine): (errors, number of entries)."""
    if not path.exists():
        return [], 0
    items = [line for line in path.read_text().splitlines() if line.startswith("- ")]
    errs = [f"entry {it[:60]!r} lacks {miss}" for it in items if (miss := [f for f in FIELDS if f not in it])]
    errs += [
        f"entry {it[:60]!r} is {body_len(it)} chars (max {MODEL_MAX_CHARS}; put the rest in known_issues_detail.md)"
        for it in items
        if body_len(it) > MODEL_MAX_CHARS
    ]
    detail = path.with_name("known_issues_detail.md")
    sections = (
        {s.strip().rstrip(".") for s in re.findall(r"^## (.+)$", detail.read_text(), re.M)}
        if detail.exists()
        else set()
    )
    for it in items:
        if "Detail: known_issues_detail.md" in it:
            title = entry_title(it).strip().rstrip(".")
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
    id_errs, nxt = check_ids(knowledge_files())
    ki += id_errs
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
    for f, i in nxt.items():
        if f.endswith(SHARED_FILE) or (a.model_dir and f.startswith(str(Path(a.model_dir).resolve()) + "/")):
            print(f"NEXT ID {Path(f).relative_to(CODE_ROOT)}: {i}")
    for e in rm:
        print(f"REPO MAP {e}")
    metrics.record("known_issue_errors", len(ki))
    metrics.record("repo_map_errors", len(rm))
    metrics.record("model_entries", model_entries)
    return 1 if ki or rm else 0


if __name__ == "__main__":
    raise SystemExit(main())
