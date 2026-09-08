#!/usr/bin/env python3
"""Would a median-vs-median gate fire on unchanged code?

Two identical sweeps, per-run L1_TO_L1 cycles for every config. For each config we
compare the median of sweep 1 with the median of sweep 2. A bistable config lands in
its slow state in 70% to 93% of runs, so the median is the slow state and the 4% flip
between states never reaches it. What remains is the true run-to-run noise plus the
0.23% that alignment moves the slow state.

Usage:  perf_median_gate_report.py [dir]     (default ~/mediangate)
        expects <dir>/baseline_runs.csv and <dir>/baseline2_runs.csv
"""

import os
import sys

import pandas as pd

GATE = 0.02


def per_config(path):
    d = pd.read_csv(path)
    g = d.groupby("variant_id")["cycles"]
    return pd.DataFrame(
        {
            "n": g.size(),
            "median": g.median(),
            "mean": g.mean(),
            "min": g.min(),
            "max": g.max(),
        }
    )


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/mediangate")
    a = per_config(os.path.join(root, "baseline_runs.csv"))
    b = per_config(os.path.join(root, "baseline2_runs.csv"))
    j = a.join(b, lsuffix="_1", rsuffix="_2", how="inner")
    print(f"configs: sweep1 {len(a)}  sweep2 {len(b)}  joined {len(j)}")
    print(f"runs per config: {int(j.n_1.median())} / {int(j.n_2.median())}\n")

    j["median_move"] = (j.median_2 - j.median_1).abs() / j.median_1
    j["mean_move"] = (j.mean_2 - j.mean_1).abs() / j.mean_1
    # the detector the gate uses today, evaluated inside each sweep
    j["within_move_1"] = (j.max_1 - j.min_1) / j.median_1
    j["within_move_2"] = (j.max_2 - j.min_2) / j.median_2

    rows = []
    for name, col in [
        ("median vs median (proposed)", "median_move"),
        ("mean vs mean", "mean_move"),
        ("(max-min)/median inside sweep 1 (today)", "within_move_1"),
        ("(max-min)/median inside sweep 2 (today)", "within_move_2"),
    ]:
        rows.append(
            {
                "metric": name,
                ">0.5%": int((j[col] > 0.005).sum()),
                ">1%": int((j[col] > 0.01).sum()),
                ">2%": int((j[col] > GATE).sum()),
                "worst_%": round(j[col].max() * 100, 3),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\nworst configs by median move (gate = {GATE*100:.0f}%):")
    w = j.sort_values("median_move", ascending=False).head(12)
    print(
        w[["median_1", "median_2", "median_move", "within_move_1", "within_move_2"]]
        .assign(
            median_move=lambda x: (x.median_move * 100).round(3),
            within_move_1=lambda x: (x.within_move_1 * 100).round(2),
            within_move_2=lambda x: (x.within_move_2 * 100).round(2),
        )
        .to_string()
    )
    fired = int((j.median_move > GATE).sum())
    print(
        f"\nA median-vs-median gate at 2% fires on {fired} of {len(j)} configs of unchanged code."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
