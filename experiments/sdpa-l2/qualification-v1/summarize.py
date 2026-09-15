# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read-only summary of qualification JSONL, keeping failures per case/head."""

import argparse
import collections
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
parser.add_argument("--results", default="results.jsonl")
args = parser.parse_args()
rows = {}
for line in (args.directory / args.results).read_text().splitlines():
    row = json.loads(line)
    rows[row["id"], row["mode"]] = row
manifest = json.loads((args.directory / "manifest-final.json").read_text())
summary = {"planned_inputs": len(manifest), "planned_mode_cases": 2 * len(manifest), "modes": {}}
for mode in ("fast", "accurate"):
    selected = [r for r in rows.values() if r["mode"] == mode]
    counts = collections.Counter(r["status"] for r in selected)
    counts["MISSING"] = len(manifest) - len(selected)
    groups = {}
    for group in ("normal", "stress", "structural", "boundary"):
        subset = [r for r in selected if r["group"] == group]
        groups[group] = dict(collections.Counter(r["status"] for r in subset))
    failures = collections.Counter(f.split(":")[-1] for r in selected for f in r.get("failed_gates", []))
    distributions = {}
    for dist in sorted({r["distribution"] for r in selected}):
        subset = [r for r in selected if r["distribution"] == dist and "per_head" in r]
        heads = [(r, h) for r in subset for h in r["per_head"] if h["l2_pct"] is not None]
        record = dict(collections.Counter(r["status"] for r in subset))
        if heads:
            worst_r, worst_h = max(heads, key=lambda rh: rh[1]["l2_pct"])
            record.update(
                worst_head_l2_pct=worst_h["l2_pct"],
                worst_case=worst_r["id"],
                worst_head=worst_h["head"],
                max_row_p99_pct=max(h["row_p99_pct"] for _, h in heads),
                max_row_pct=max(h["row_max_pct"] for _, h in heads),
            )
        distributions[dist] = record
    summary["modes"][mode] = dict(
        counts=dict(counts), groups=groups, failing_head_gates=dict(failures), distributions=distributions
    )
print(json.dumps(summary, indent=2))
