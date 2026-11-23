#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compare two LLK perf reports (baseline vs candidate), config by config.

usage: perf_compare.py <baseline-run-dir> <candidate-run-dir> [--csv out.csv]
  A run dir is perf_data/runs/<tag> (the RUN_LLK_TESTS_PERF_REPORT line of run_test.sh / chip_queue.py).
Prints, per zone (INIT, TILE_LOOP, KERNEL, OUTSIDE = KERNEL - INIT - TILE_LOOP) and run type, the geometric-mean change
(candidate / baseline - 1; negative = candidate faster), median cycles, worst/best config and how many configs moved
by more than 2%. --csv writes the TILE_LOOP rows for the tracker.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

META = {
    "test_name",
    "commit_sha",
    "arch",
    "run_id",
    "timestamp",
    "pipeline",
    "pr_number",
}


def load(run_dir):
    files = glob.glob(f"{run_dir}/*.parquet")
    if not files:
        sys.exit(f"no parquet in {run_dir}")
    d = pd.concat(pd.read_parquet(f) for f in files)
    mets = [c for c in d.columns if c.startswith(("mean(", "std("))]
    keys = [c for c in d.columns if c not in META and c not in mets and c != "marker"]
    for k in keys:
        d[k] = d[k].astype(str)
    return d, keys, [c[5:-1] for c in mets if c.startswith("mean(")]


def main():
    args = sys.argv[1:]
    csv = None
    if "--csv" in args:
        i = args.index("--csv")
        csv = args[i + 1]
        del args[i : i + 2]
    base, cand = args
    a, keys, rts = load(base)
    b, _, _ = load(cand)
    rows = []
    for rt in rts:
        col = f"mean({rt})"
        pa = a.pivot_table(index=keys, columns="marker", values=col, aggfunc="first")
        pb = b.pivot_table(index=keys, columns="marker", values=col, aggfunc="first")
        for p in (pa, pb):
            if {"KERNEL", "INIT", "TILE_LOOP"} <= set(p.columns):
                p["OUTSIDE"] = p["KERNEL"] - p["INIT"] - p["TILE_LOOP"]
        j = pa.join(pb, lsuffix="_a", rsuffix="_b", how="inner")
        for zone in ("INIT", "TILE_LOOP", "OUTSIDE", "KERNEL"):
            if f"{zone}_a" not in j:
                continue
            x, y = j[f"{zone}_a"].astype(float), j[f"{zone}_b"].astype(float)
            ok = (x > 0) & (y > 0)
            if not ok.any():
                continue
            r = (y[ok] / x[ok]).values
            rows.append(
                dict(
                    zone=zone,
                    run_type=rt,
                    configs=int(ok.sum()),
                    change_pct=round(100 * (np.exp(np.log(r).mean()) - 1), 2),
                    median_base=float(x[ok].median()),
                    median_cand=float(y[ok].median()),
                    median_delta=float((y[ok] - x[ok]).median()),
                    best_pct=round(100 * (r.min() - 1), 1),
                    worst_pct=round(100 * (r.max() - 1), 1),
                    faster=int((r < 0.98).sum()),
                    slower=int((r > 1.02).sum()),
                )
            )
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(
        f"{Path(cand).name} vs {Path(base).name}  (change: negative = candidate faster)"
    )
    print(df.to_string(index=False))
    if csv:
        df[df.zone == "TILE_LOOP"].to_csv(csv, index=False)


if __name__ == "__main__":
    main()
