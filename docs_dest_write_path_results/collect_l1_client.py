#!/usr/bin/env python3
"""Collects the l1_client_* columns of an L1 client sweep into one table.

Usage: collect_l1_client.py <logs/l1_client/<label>> [--csv out.csv]
"""

import csv
import json
from statistics import median
import sys
from pathlib import Path

CASE_KEYS = ("dwp_kernel", "dwp_path", "dwp_tiles_per_bank", "dwp_kt_dim", "math_fidelity")


def case_name(row):
    # Keep every experimental axis: old short names silently merged distinct runs.
    return json.dumps(
        {k: v for k, v in row.items() if k.startswith(("dwp_", "formats.", "slot_")) or k == "math_fidelity"},
        sort_keys=True,
        separators=(",", ":"),
    )


def collect(root):
    table = {}
    for sel_dir in sorted(root.glob("sel_*"), key=lambda p: int(p.name[4:])):
        for f in sel_dir.glob("perf_dest_*_quasar.csv"):
            for row in csv.DictReader(f.open()):
                if row.get("marker") != "TILE_LOOP":
                    continue
                for k, v in row.items():
                    if "l1_client_" in k and v not in ("", None):
                        metric = k[k.index("l1_client_") + len("l1_client_") :]
                        table.setdefault(case_name(row), {}).setdefault(metric, []).append(float(v))
    return {case: {metric: median(values) for metric, values in metrics.items()} for case, metrics in table.items()}


def main():
    root = Path(sys.argv[1])
    table = collect(root)
    metrics = sorted({m for v in table.values() for m in v})
    cases = sorted(table)
    width = max(len(m) for m in metrics)
    print(f"{'metric':<{width}} " + " ".join(f"{c:>22}" for c in cases))
    for m in metrics:
        cells = [table[c].get(m) for c in cases]
        print(f"{m:<{width}} " + " ".join("".rjust(22) if x is None else f"{x:22.3f}" for x in cells))
    if "--csv" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--csv") + 1])
        with out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["metric", *cases])
            for m in metrics:
                w.writerow([m, *[table[c].get(m, "") for c in cases]])


if __name__ == "__main__":
    main()
