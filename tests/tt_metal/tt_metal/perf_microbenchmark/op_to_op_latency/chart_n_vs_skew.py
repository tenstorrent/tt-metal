#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dual-axis chart: at a fixed (saturated) core count, aggregate read BW falls with N while
starvation rises -- showing the BW drop is a skew tax, not a throughput collapse. The 'fair'
line (median per-core BW x cores) rises with N; the gap to actual aggregate is the skew tax."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--csv", type=Path, required=True)
ap.add_argument("--cores", type=int, nargs="+", default=[56])
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

df = pd.read_csv(args.csv)
fig, axes = plt.subplots(1, len(args.cores), figsize=(7 * len(args.cores), 5), squeeze=False)
for ax, c in zip(axes[0], args.cores):
    s = df[df.cores == c].sort_values("N")
    fair = c * s["per_core_read_med"]
    ax.plot(
        s["N"], s["agg_read_gbps"], "o-", color="C0", lw=2, label="aggregate read BW (actual, gated by slowest core)"
    )
    ax.plot(s["N"], fair, "s--", color="C2", lw=1.5, label="'fair' BW = median per-core × cores (no skew)")
    ax.fill_between(s["N"], s["agg_read_gbps"], fair, color="C3", alpha=0.12)
    ax.set_xlabel("N (reader trid in-flight)")
    ax.set_ylabel("read BW (GB/s)", color="C0")
    ax.set_title(f"cores={c}: BW drop with N is skew, not throughput loss")
    ax.set_xscale("log", base=2)
    ax.set_xticks(s["N"])
    ax.set_xticklabels(s["N"])
    ax.set_ylim(0, max(fair.max(), s["agg_read_gbps"].max()) * 1.1)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(s["N"], s["starvation_ratio"], "^-", color="C1", lw=2, label="starvation ratio (fastest/slowest core)")
    ax2.set_ylabel("starvation ratio (fastest ÷ slowest core)", color="C1")
    ax2.set_ylim(1.0, s["starvation_ratio"].max() * 1.1)

    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="lower center", fontsize=8, framealpha=0.9)

fig.tight_layout()
fig.savefig(args.out, dpi=130)
print(f"wrote {args.out}")
