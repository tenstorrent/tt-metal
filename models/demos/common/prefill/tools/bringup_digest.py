#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Read every model's ``bringup_log.jsonl`` and report how the bring-up recipe is actually going.

Specified by ``docs/MODEL_BRINGUP_RECIPE.md`` §7.5-7.6, which names this tool but did not ship it.

    tools/bringup_digest.py            # per-stage summary across all models
    tools/bringup_digest.py --lint     # malformed records only (exit 1 if any)
    tools/bringup_digest.py --model llama3_1_8b_d_p   # one model, per-stage detail

Three things are DERIVED here rather than tracked by the agent, which is the point of the log:

* **reiterations** — failing ``verify`` events in a stage. A Testing table green on its first run
  scores 0. This is the headline: it separates a stage that is understood from one being guessed at.
* **elapsed** — ``enter`` to the first passing ``verify``.
* **bottleneck test** — the pytest node id appearing most often across a stage's ``failed`` lists.

A stage with a high reiteration average across models is a stage the recipe does not explain well
enough. A high ``spec_gap`` / ``recipe_gap`` count is the same signal aimed at the inputs instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

STAGES = ["D1", "D2", "D3", "M1", "M2", "M3", "P1", "P2", "P3", "P4"]
KINDS = ["spec_gap", "recipe_gap", "donor_wrong", "ttnn_gap", "model_quirk", "env"]
EVENTS = {"start", "enter", "verify", "judgment", "fallback", "skip"}
MAX_FIELD = 160  # §7.4: issue / fix / why are one sentence, one line, at most 160 characters

REQUIRED = {
    "start": ["model", "mesh", "recipe_sha"],
    "enter": ["t", "stage"],
    "verify": ["t", "stage", "result", "failed"],
    "judgment": ["t", "stage", "kind", "issue", "fix", "failed"],
    "fallback": ["t", "stage", "block", "why"],
    "skip": ["t", "stage", "row", "why"],
}


def find_logs(root: Path, model: str | None) -> list[Path]:
    logs = sorted(root.glob("models/demos/*/bringup_log.jsonl"))
    if model:
        logs = [p for p in logs if p.parent.name == model]
    return logs


def lint_record(rec: dict, lineno: int, path: Path) -> list[str]:
    """Every rule §7 states, checked mechanically. Returns human-readable problems."""
    problems = []
    ev = rec.get("ev")
    if ev not in EVENTS:
        return [f"{path}:{lineno}: unknown ev {ev!r} (expected one of {sorted(EVENTS)})"]
    for field in REQUIRED[ev]:
        if field not in rec:
            problems.append(f"{path}:{lineno}: {ev} record missing required field {field!r}")
    if ev != "start" and rec.get("stage") not in STAGES:
        problems.append(f"{path}:{lineno}: stage {rec.get('stage')!r} not in {STAGES}")
    if ev == "verify":
        if rec.get("result") not in ("pass", "fail"):
            problems.append(f"{path}:{lineno}: verify result must be 'pass' or 'fail'")
        # §7.2: `failed` must be non-empty when result is fail — a failure with no node id is
        # unattributable, and the bottleneck-test derivation silently loses it.
        if rec.get("result") == "fail" and not rec.get("failed"):
            problems.append(f"{path}:{lineno}: verify result=fail must list the failing node ids")
    if ev == "judgment" and rec.get("kind") not in KINDS:
        problems.append(f"{path}:{lineno}: judgment kind {rec.get('kind')!r} not in the closed vocabulary {KINDS}")
    for field in ("issue", "fix", "why"):
        val = rec.get(field)
        if isinstance(val, str) and len(val) > MAX_FIELD:
            problems.append(f"{path}:{lineno}: {field!r} is {len(val)} chars (cap {MAX_FIELD})")
        if isinstance(val, str) and "\n" in val:
            problems.append(f"{path}:{lineno}: {field!r} must be one line")
    return problems


def parse_time(value: str):
    for fmt in ("%Y-%m-%dT%H:%MZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue
    return None


def load(path: Path):
    records, problems = [], []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"{path}:{lineno}: not valid JSON ({e.msg})")
            continue
        problems.extend(lint_record(rec, lineno, path))
        records.append(rec)
    return records, problems


def stage_stats(records):
    """Per-stage: reiterations, hours, green, and the most-often-failing node id."""
    entered, first_pass, reiters = {}, {}, Counter()
    failed_nodes = defaultdict(Counter)
    for rec in records:
        stage, ev = rec.get("stage"), rec.get("ev")
        if ev == "enter" and stage not in entered:
            entered[stage] = parse_time(rec.get("t"))
        elif ev == "verify":
            if rec.get("result") == "fail":
                reiters[stage] += 1
                for node in rec.get("failed") or []:
                    failed_nodes[stage][node] += 1
            elif stage not in first_pass:
                first_pass[stage] = parse_time(rec.get("t"))
    out = {}
    for stage in STAGES:
        if stage not in entered and stage not in first_pass:
            continue
        start, end = entered.get(stage), first_pass.get(stage)
        hours = (end - start).total_seconds() / 3600 if start and end else None
        worst = failed_nodes[stage].most_common(1)
        out[stage] = {
            "green": stage in first_pass,
            "reiters": reiters[stage],
            "hours": hours,
            "worst": f"{worst[0][0]} ({worst[0][1]})" if worst else "-",
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lint", action="store_true", help="report malformed records only; exit 1 if any")
    ap.add_argument("--model", help="restrict to one model package name")
    ap.add_argument("--root", default=None, help="repo root (default: inferred from this file)")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parents[5]
    logs = find_logs(root, args.model)
    if not logs:
        print(f"no bringup_log.jsonl found under {root}/models/demos/", file=sys.stderr)
        return 1

    all_problems, per_model = [], {}
    for path in logs:
        records, problems = load(path)
        all_problems.extend(problems)
        per_model[path.parent.name] = records

    if args.lint:
        for problem in all_problems:
            print(problem)
        print(f"\n{len(all_problems)} malformed record(s) across {len(logs)} log(s)")
        return 1 if all_problems else 0

    if all_problems:
        print(f"warning: {len(all_problems)} malformed record(s); run with --lint for detail\n", file=sys.stderr)

    stats = {model: stage_stats(records) for model, records in per_model.items()}
    print(f"{'stage':<6} {'models':>6} {'green':>5} {'reiters':>7} {'worst':>5} {'hours':>5}  top failing test")
    print("-" * 78)
    for stage in STAGES:
        rows = [s[stage] for s in stats.values() if stage in s]
        if not rows:
            continue
        green = sum(1 for r in rows if r["green"])
        reit = [r["reiters"] for r in rows]
        hours = [r["hours"] for r in rows if r["hours"] is not None]
        worst_counter = Counter(r["worst"] for r in rows if r["worst"] != "-")
        worst_test = worst_counter.most_common(1)[0][0] if worst_counter else "-"
        print(
            f"{stage:<6} {len(rows):>6} {green:>5} {sum(reit) / len(reit):>7.1f} {max(reit):>5} "
            f"{(sum(hours) / len(hours) if hours else 0):>5.1f}  {worst_test}"
        )

    kinds = Counter(r["kind"] for recs in per_model.values() for r in recs if r.get("ev") == "judgment")
    if kinds:
        print("\njudgments by kind: " + ", ".join(f"{k}={kinds[k]}" for k in KINDS if kinds[k]))
    fallbacks = [r for recs in per_model.values() for r in recs if r.get("ev") == "fallback"]
    if fallbacks:
        print(f"torch CPU fallbacks: {len(fallbacks)} — " + ", ".join(sorted({r["block"] for r in fallbacks})))
    skips = Counter(r["row"] for recs in per_model.values() for r in recs if r.get("ev") == "skip")
    if skips:
        print(f"dropped Testing-table rows: {sum(skips.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
