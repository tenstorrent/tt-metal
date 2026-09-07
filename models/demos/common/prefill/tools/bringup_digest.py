#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Aggregate prefill bring-up logs into a per-stage difficulty digest.

Reads the append-only ``bringup_log.jsonl`` files written during bring-up (see
``docs/MODEL_BRINGUP_RECIPE.md`` §7) and reports, per stage: how many models reached it, how many
reiterations it cost, which test failed most often, and what kinds of judgment calls it forced.

Nothing in the log stores a reiteration count -- it is derived here as the number of ``verify``
events that failed, so a stage whose Testing table went green on the first run scores 0.

Usage:
    bringup_digest.py                      # glob models/demos/*/bringup_log.jsonl
    bringup_digest.py path/to/log.jsonl ...  # explicit files
    bringup_digest.py --lint               # only report malformed records
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

STAGES = ["D1", "D2", "D3", "M1", "M2", "M3", "P1", "P2", "P3", "P4"]
EVENTS = {"start", "enter", "verify", "judgment", "fallback", "skip"}
KINDS = {"spec_gap", "recipe_gap", "donor_wrong", "ttnn_gap", "model_quirk", "env"}
TEXT_FIELDS = ("issue", "fix", "why")
TEXT_CAP = 160  # characters, per field -- keep entries comparable across models

DEFAULT_GLOB = "models/demos/*/bringup_log.jsonl"


def model_name(path: str) -> str:
    return os.path.basename(os.path.dirname(os.path.abspath(path)))


def parse_time(rec: dict):
    raw = rec.get("t")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def lint(rec: dict, where: str) -> list[str]:
    """Return human-readable complaints about one record. Empty list means well-formed."""
    out = []
    ev = rec.get("ev")
    if ev not in EVENTS:
        out.append(f"{where}: unknown ev {ev!r}")
        return out  # every other check depends on ev
    if ev != "start":
        if rec.get("stage") not in STAGES:
            out.append(f"{where}: unknown stage {rec.get('stage')!r}")
        if parse_time(rec) is None:
            out.append(f"{where}: missing or unparseable t {rec.get('t')!r}")
    if ev == "verify":
        if rec.get("result") not in ("pass", "fail"):
            out.append(f"{where}: verify result must be pass|fail, got {rec.get('result')!r}")
        if rec.get("result") == "fail" and not rec.get("failed"):
            out.append(f"{where}: failing verify must list the failed test ids")
    if ev == "judgment":
        if rec.get("kind") not in KINDS:
            out.append(f"{where}: judgment kind must be one of {sorted(KINDS)}, got {rec.get('kind')!r}")
        for field in ("issue", "fix"):
            if not rec.get(field):
                out.append(f"{where}: judgment needs a {field}")
    if ev == "fallback" and not rec.get("why"):
        out.append(f"{where}: fallback needs a why")
    if ev == "skip" and not (rec.get("row") and rec.get("why")):
        out.append(f"{where}: skip needs a row and a why")
    for field in TEXT_FIELDS:
        val = rec.get(field)
        if isinstance(val, str):
            if len(val) > TEXT_CAP:
                out.append(f"{where}: {field} is {len(val)} chars, cap is {TEXT_CAP}")
            if "\n" in val:
                out.append(f"{where}: {field} contains a newline; one sentence on one line")
    return out


def lint_sequence(model: str, records: list[dict]) -> list[str]:
    """Complaints that need more than one record: a stage worked on without being entered, or
    entered and never verified. Both cost the digest a metric, so say so."""
    out = []
    seen: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        stage, ev = rec.get("stage"), rec.get("ev")
        if stage in STAGES and ev in EVENTS:
            seen[stage].add(ev)
    for stage, events in seen.items():
        if "enter" not in events:
            out.append(f"{model}: {stage} has {sorted(events)} but no enter; elapsed time unmeasurable")
        elif "verify" not in events:
            out.append(f"{model}: {stage} entered but never verified; reiterations unmeasurable")
    return out


def load(paths: list[str]) -> tuple[dict[str, list[dict]], list[str]]:
    """Return {model: [records...]} plus lint complaints, records in file order."""
    logs: dict[str, list[dict]] = {}
    complaints: list[str] = []
    for path in paths:
        model = model_name(path)
        records = []
        with open(path) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                where = f"{model}:{lineno}"
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    complaints.append(f"{where}: not valid JSON ({exc.msg})")
                    continue
                if not isinstance(rec, dict):
                    complaints.append(f"{where}: expected an object, got {type(rec).__name__}")
                    continue
                complaints.extend(lint(rec, where))
                records.append(rec)
        complaints.extend(lint_sequence(model, records))
        logs[model] = records
    return logs, complaints


def stage_rows(logs: dict[str, list[dict]]) -> list[dict]:
    """One row per stage, aggregated over models."""
    entered: dict[str, set[str]] = defaultdict(set)
    green: dict[str, set[str]] = defaultdict(set)
    reiters: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    failures: dict[str, Counter] = defaultdict(Counter)
    hours: dict[str, list[float]] = defaultdict(list)

    for model, records in logs.items():
        opened: dict[str, datetime] = {}
        for rec in records:
            stage, ev = rec.get("stage"), rec.get("ev")
            if stage not in STAGES:
                continue
            when = parse_time(rec)
            entered[stage].add(model)
            reiters[stage].setdefault(model, 0)
            if ev == "enter":
                if when and stage not in opened:
                    opened[stage] = when
            elif ev == "verify":
                if rec.get("result") == "fail":
                    reiters[stage][model] += 1
                    for test in rec.get("failed") or []:
                        failures[stage][test] += 1
                elif rec.get("result") == "pass":
                    reiters[stage].setdefault(model, 0)
                    if model not in green[stage]:
                        green[stage].add(model)
                        if when and stage in opened:
                            hours[stage].append((when - opened[stage]).total_seconds() / 3600.0)

    rows = []
    for stage in STAGES:
        if stage not in entered:
            continue
        counts = reiters[stage]
        values = list(counts.values()) or [0]
        top = failures[stage].most_common(1)
        rows.append(
            {
                "stage": stage,
                "models": len(entered[stage]),
                "green": len(green[stage]),
                "avg": sum(values) / len(values),
                "worst": max(values),
                "hours": (sum(hours[stage]) / len(hours[stage])) if hours[stage] else None,
                "top_test": f"{short_test(top[0][0])} ({top[0][1]})" if top else "-",
            }
        )
    return rows


def short_test(node_id: str) -> str:
    """Trim a pytest node id to the part a human scans for."""
    tail = node_id.split("/")[-1]
    return tail if len(tail) <= 52 else tail[:49] + "..."


def render(rows: list[dict], logs: dict[str, list[dict]]) -> None:
    if not rows:
        print("no stage activity logged")
        return

    header = f"{'stage':<6}{'models':>7}{'green':>7}{'reiters':>9}{'worst':>7}{'hours':>7}  top failing test"
    print(header)
    print("-" * len(header))
    for row in rows:
        hours = f"{row['hours']:.1f}" if row["hours"] is not None else "-"
        print(
            f"{row['stage']:<6}{row['models']:>7}{row['green']:>7}"
            f"{row['avg']:>9.1f}{row['worst']:>7}{hours:>7}  {row['top_test']}"
        )

    kinds = Counter()
    fallbacks = Counter()
    skips = Counter()
    for records in logs.values():
        for rec in records:
            if rec.get("ev") == "judgment":
                kinds[rec.get("kind")] += 1
            elif rec.get("ev") == "fallback":
                fallbacks[rec.get("stage")] += 1
            elif rec.get("ev") == "skip":
                skips[rec.get("stage")] += 1

    if kinds:
        print("\njudgment calls by kind")
        for kind, count in kinds.most_common():
            print(f"  {str(kind):<12} {count}")
    if fallbacks:
        print("\ntorch CPU fallbacks: " + ", ".join(f"{s}={c}" for s, c in sorted(fallbacks.items())))
    if skips:
        print("skipped test rows: " + ", ".join(f"{s}={c}" for s, c in sorted(skips.items())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help=f"bringup_log.jsonl paths (default: {DEFAULT_GLOB})")
    ap.add_argument("--lint", action="store_true", help="report malformed records only")
    args = ap.parse_args()

    paths = args.logs or sorted(glob.glob(DEFAULT_GLOB))
    if not paths:
        print(f"no logs found (looked for {DEFAULT_GLOB})", file=sys.stderr)
        return 1

    logs, complaints = load(paths)
    if complaints:
        print(f"{len(complaints)} malformed record(s):", file=sys.stderr)
        for complaint in complaints:
            print(f"  {complaint}", file=sys.stderr)
        print("", file=sys.stderr)
    if args.lint:
        return 1 if complaints else 0

    print(f"{len(logs)} model(s): {', '.join(sorted(logs))}\n")
    render(stage_rows(logs), logs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
