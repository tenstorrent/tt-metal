#!/usr/bin/env python3
"""Extract the cross-model MoE parameter comparison table from moe_plan.md and write it as CSV."""

import csv
import os
import re
import sys


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plan_path = os.path.join(script_dir, "..", "moe_plan.md")
    out_path = os.path.join(script_dir, "..", "moe_comparison.csv")

    if len(sys.argv) > 1:
        out_path = sys.argv[1]

    with open(plan_path) as f:
        content = f.read()

    # Find the comparison table by its header row
    marker = "| **Model** | **hidden_size**"
    start = content.find(marker)
    if start < 0:
        print("ERROR: comparison table not found in moe_plan.md", file=sys.stderr)
        sys.exit(1)

    # Table ends at the next blank line or notes block
    end = content.find("\n\n", start)
    table_text = content[start:end].strip()

    rows = []
    for line in table_text.split("\n"):
        if not line.startswith("|") or "---" in line:
            continue
        cells = [c.strip().replace("**", "") for c in line.split("|")[1:-1]]
        rows.append(cells)

    if len(rows) < 2:
        print("ERROR: table has fewer than 2 rows", file=sys.stderr)
        sys.exit(1)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows) - 1} model rows to {out_path}")


if __name__ == "__main__":
    main()
