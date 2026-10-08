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

Only the latest writer / tester cycle and the latest refiner version are read;
earlier cycles were superseded.

Usage:
    open_risks.py check    --log-dir DIR   # exit 0 = clear, 1 = blocked
    open_risks.py deferred --log-dir DIR   # print the DEFERRED PR lines
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

WAIVER_RE = re.compile(
    r"\b(untested|unexercised|not exercised|negligible|conservative(?:ly)?|may be wrong|assum(?:e|es|ed|ing))\b",
    re.I,
)
ENTRY_RE = re.compile(r"^R(\d+)\s+(CLOSED|DEFERRED|OPEN)\s*:\s*(.*)$", re.I)
REF_RE = re.compile(r"\bR(\d+)\b")
BULLET_RE = re.compile(r"^\s*[-*]\s+")
_CYCLED = re.compile(r"^agent_(writer_cycle|tester_cycle|analysis_refiner_v)(\d+)$")


def latest_logs(log_dir: str) -> list[str]:
    """agent_*.md paths, keeping only the highest cycle/version per cycled role."""
    best: dict[str, tuple[int, str]] = {}
    plain = []
    for path in sorted(glob.glob(os.path.join(log_dir, "agent_*.md"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        m = _CYCLED.match(stem)
        if not m:
            plain.append(path)
            continue
        n = int(m.group(2))
        if m.group(1) not in best or n > best[m.group(1)][0]:
            best[m.group(1)] = (n, path)
    return sorted(plain + [p for _, p in best.values()])


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
            if BULLET_RE.match(line):
                raw_entries.append(_clean(BULLET_RE.sub("", line)))
            elif s and raw_entries:
                raw_entries[-1] += " " + _clean(s)
            elif s and _clean(s).lower().rstrip(".") not in ("none", "n/a"):
                raw_entries.append(_clean(s))
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
    paths = latest_logs(log_dir)
    if not paths:
        return [f"MISSING {log_dir}: no agent_*.md self-logs"], 0, 0
    for path in paths:
        name = os.path.basename(path)
        found, entries, malformed, prose = parse(open(path, encoding="utf-8", errors="replace").read())
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
    for path in latest_logs(log_dir):
        _, entries, _, _ = parse(open(path, encoding="utf-8", errors="replace").read())
        for _, status, body in entries:
            if status == "DEFERRED":
                out.append(re.split(r"\bPR:\s*", body, maxsplit=1)[1].strip())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        print(f"OPEN_RISKS: BLOCKED ({len(findings)} item(s)) — close (evidence) or defer (PR text) each one")
        return 1
    print(f"OPEN_RISKS: CLEAR ({closed} closed, {deferred} deferred)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
