#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Open-risks gate for a Quasar codegen run (see references/logging.md § Open risks).

Every agent self-log ($LOG_DIR/agent_*.md) has an `## Open risks` section whose
entries are one bullet each:

    - R1 CLOSED: <risk> — evidence: <test id, file:line or ticket>
    - R2 DEFERRED: <risk> — PR: <one line that goes into the PR body>

or the single word `none`. An `OPEN` entry, a malformed entry or a missing
section blocks the run. Outside that section, a line that uses a waiver word
(WAIVER_RE) must cite an entry of the same log as `R<n>`, so a risk an agent
noticed cannot be waved off in prose.

Only the pipeline self-logs of the run's state are read (state.json `CYCLE`,
`SKIP_WRITER`): analyzer, writer and tester of cycle CYCLE, the refiner version
that preceded it, optimizer and prettifier. Each one that is absent is MISSING.

Usage:
    open_risks.py check    --log-dir DIR   # exit 0 = clear, 1 = blocked
    open_risks.py deferred --log-dir DIR   # print the DEFERRED PR lines
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

WAIVER_RE = re.compile(
    r"\b(untested|unexercised|not exercised|not tested|unverified|not verified|negligible|conservative(?:ly)?"
    r"|may be wrong|assume|assumes|assumed|assuming)\b",
    re.I,
)
ENTRY_RE = re.compile(r"^R(\d+)\s+(CLOSED|DEFERRED|OPEN)\s*:\s*(.*)$", re.I)
REF_RE = re.compile(r"\bR(\d+)\b")
BULLET_RE = re.compile(r"^[-*]\s+")
SUB_BULLET_RE = re.compile(r"^\s+[-*]\s+")
NONE_WORDS = ("none", "n/a")


def _state(log_dir: str) -> dict:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import state

    return state._load(Path(log_dir) / "state.json")


def expected_logs(log_dir: str) -> list[str]:
    """Self-log names the run's state says must exist."""
    st = _state(log_dir)
    cycle = int(st.get("CYCLE") or 1)
    names = ["agent_analyzer.md"]
    if cycle > 1:
        names.append(f"agent_analysis_refiner_v{cycle - 1}.md")
    if str(st.get("SKIP_WRITER")).lower() != "true":
        names.append(f"agent_writer_cycle{cycle}.md")
    names += [
        f"agent_tester_cycle{cycle}.md",
        "agent_optimizer.md",
        "agent_prettifier.md",
    ]
    return names


def _clean(s: str) -> str:
    return s.replace("`", "").replace("**", "").strip()


def parse(text: str):
    """Return (section_found, entries, malformed, prose) for one self-log.

    entries: list of (id, status, body); malformed: offending entry lines;
    prose: (lineno, line) pairs outside the Open risks section and code fences.
    """
    lines = text.splitlines()
    in_fence = in_section = found = False
    raw_entries: list[str] = []
    prose: list[tuple[int, str]] = []
    for no, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if re.match(r"^#{1,3}\s", s):
            in_section = bool(re.match(r"^#{2,3}\s*Open risks\b", s, re.I))
            found = found or in_section
            continue
        if in_section:
            if not s:
                continue
            if raw_entries and (SUB_BULLET_RE.match(line) or not BULLET_RE.match(line)):
                # Indented bullets and wrapped lines continue the entry above.
                raw_entries[-1] += " " + _clean(BULLET_RE.sub("", s))
                continue
            text = _clean(BULLET_RE.sub("", s))
            if text.lower().rstrip(".") not in NONE_WORDS:
                raw_entries.append(text)
        elif s:
            prose.append((no, s))
    entries, malformed = [], []
    for e in raw_entries:
        m = ENTRY_RE.match(e)
        if not m:
            malformed.append(e)
            continue
        rid, status, body = m.group(1), m.group(2).upper(), m.group(3)
        if status == "CLOSED" and not re.search(r"evidence:\s*\S", body, re.I):
            malformed.append(e)
        elif status == "DEFERRED" and not re.search(r"\bPR:\s*\S", body):
            malformed.append(e)
        else:
            entries.append((rid, status, body))
    return found, entries, malformed, prose


def check(log_dir: str) -> tuple[list[str], int, int]:
    """Return (blocking findings, closed count, deferred count)."""
    findings: list[str] = []
    closed = deferred = 0
    for name in expected_logs(log_dir):
        path = os.path.join(log_dir, name)
        if not os.path.isfile(path):
            findings.append(f"MISSING {name}: self-log not written")
            continue
        found, entries, malformed, prose = parse(
            open(path, encoding="utf-8", errors="replace").read()
        )
        if not found:
            findings.append(f"MISSING {name}: no '## Open risks' section")
        ids = {rid for rid, _, _ in entries}
        for rid, status, body in entries:
            if status == "OPEN":
                findings.append(f"OPEN {name} R{rid}: {body}")
            closed += status == "CLOSED"
            deferred += status == "DEFERRED"
        findings += [f"MALFORMED {name}: {e}" for e in malformed]
        for no, line in prose:
            if WAIVER_RE.search(line) and not (set(REF_RE.findall(line)) & ids):
                findings.append(f"WAIVER {name}:{no}: {line[:200]}")
    return findings, closed, deferred


def deferred_lines(log_dir: str) -> list[str]:
    out = []
    for name in expected_logs(log_dir):
        path = os.path.join(log_dir, name)
        if not os.path.isfile(path):
            continue
        _, entries, _, _ = parse(open(path, encoding="utf-8", errors="replace").read())
        for _, status, body in entries:
            if status == "DEFERRED":
                line = re.split(r"\bPR:\s*", body, maxsplit=1)[1].strip()
                if line not in out:
                    out.append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["check", "deferred"])
    ap.add_argument("--log-dir", required=True)
    args = ap.parse_args()
    if args.cmd == "deferred":
        print("\n".join(deferred_lines(args.log_dir)))
        return 0
    findings, closed, deferred = check(args.log_dir)
    for f in findings:
        print(f)
    if findings:
        print(
            f"OPEN_RISKS: BLOCKED ({len(findings)} item(s)) — close (evidence) or defer (PR text) each one"
        )
        return 1
    print(f"OPEN_RISKS: CLEAR ({closed} closed, {deferred} deferred)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
