#!/usr/bin/env python3
"""Did pinning the pack loop's alignment remove the bistability, and what did it cost?

Each pass is a full matmul sweep. One is unmodified; the others force the pack loop to
a fixed position inside its 16-byte block. We count how many configs still show a jump
between runs, and how much the pinning slowed the kernel down.

Usage:  perf_align_fix_report.py [dir]      (default ~/alignfix)
"""

import glob
import os
import sys

import pandas as pd

# Same detector as the loop-factor sweep, so counts compare directly with the 96 configs
# flagged there. It is more sensitive than the 2% gate: a 2% jump seen even once in 20
# runs gives cv 0.44%, well above the 0.2% threshold.
STD_FLOOR = 5
CV_FLOOR = 0.002
KEY_SKIP = {"marker", "cv", "bad"}


def load(root, name):
    files = [
        f
        for f in glob.glob(os.path.join(root, name, "**", "*.csv"), recursive=True)
        if not f.endswith((".post.csv", ".counters.csv"))
    ]
    if not files:
        return None
    t = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    t = t[t["marker"] == "TILE_LOOP"].copy()
    t["std(L1_TO_L1)"] = t["std(L1_TO_L1)"].fillna(0)
    t["cv"] = t["std(L1_TO_L1)"] / t["mean(L1_TO_L1)"]
    t["bad"] = (t["std(L1_TO_L1)"] > STD_FLOOR) & (t["cv"] > CV_FLOOR)
    return t


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/alignfix")
    names = [
        d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))
    ]
    tables, rows = {}, []
    for name in names:
        t = load(root, name)
        if t is None or t.empty:
            print(f"{name}: no CSVs")
            continue
        tables[name] = t
        rows.append(
            {
                "pass": name,
                "configs": len(t),
                "flagged": int(t.bad.sum()),
                "flagged_%": round(t.bad.mean() * 100, 3),
                "median_cycles": round(t["mean(L1_TO_L1)"].median()),
                "worst_cv_%": round(t["cv"].max() * 100, 3),
            }
        )
    if not rows:
        print(f"nothing to report under {root}")
        return 1

    r = pd.DataFrame(rows)
    if "baseline" in tables:
        base = r.loc[r["pass"] == "baseline", "median_cycles"].iloc[0]
        r["cost_cycles"] = r["median_cycles"] - base
        r["cost_%"] = (r["cost_cycles"] / base * 100).round(3)
    print(r.to_string(index=False))

    # Compare per config, not just by count: a pass could fix 96 and break 96 others
    # while the totals look unchanged.
    if "baseline" in tables:
        keycols = [
            c
            for c in tables["baseline"].columns
            if "(" not in c and c not in KEY_SKIP and "loop" not in c.lower()
        ]

        def flagged_set(t):
            k = t[keycols].astype(str).agg("|".join, axis=1)
            return set(k[t.bad])

        b = flagged_set(tables["baseline"])
        print(f"\nbaseline flagged {len(b)} configs")
        for name, t in tables.items():
            if name == "baseline":
                continue
            s = flagged_set(t)
            print(
                f"  {name:10s} flagged {len(s):4d}   "
                f"fixed {len(b - s):4d} of baseline's   "
                f"newly broken {len(s - b):4d}"
            )
    print(
        "\nA pass with 0 flagged means pinning the alignment removed the bistability"
        "\nfor every config. 'newly broken' above 0 means the susceptible position is"
        "\nconfig-dependent, and pinning cannot work as a blanket fix."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
