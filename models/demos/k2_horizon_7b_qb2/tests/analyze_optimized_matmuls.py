"""Compact search table with every attempted geometry and per-role winners."""

import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("input", type=Path)
a = p.parse_args()
data = json.loads(a.input.read_text())
rows = data["records"]
fields = [
    "role",
    "k",
    "n",
    "nphysical",
    "dtype",
    "fidelity",
    "cores",
    "input_shard_tiles",
    "block_w",
    "readers",
    "fp32",
    "per_core_N",
    "reader_row_bytes",
    "traced_host_us",
    "pcc",
    "passed",
    "error",
]
with a.input.with_suffix(".csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
summary = {}
for role in dict.fromkeys(r["role"] for r in rows):
    good = sorted(
        [r for r in rows if r["role"] == role and r.get("passed")], key=lambda r: r.get("traced_host_us", 1e9)
    )
    summary[role] = {"fastest": good[:8], "attempted": sum(r["role"] == role for r in rows), "passed": len(good)}
a.input.with_name(a.input.stem + "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
