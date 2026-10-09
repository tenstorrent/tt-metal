# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize a tracy run of test_fused_rms_norm_prefill.py: per-shape device kernel time of
DitFusedDistributedRmsnormDeviceOperation, warmup calls dropped (same metric as the rmsnorm-prefill campaign)."""

import glob
import sys

import pandas as pd

OP_CODE = "DitFusedDistributedRmsnormDeviceOperation"
KERNEL = "DEVICE KERNEL DURATION [ns]"
WIDTH = "INPUT_0_X_PAD[LOGICAL]"
WARMUP = 3
SHAPES = {896: "h3584", 1024: "h4096", 1536: "h6144", 1792: "h7168"}


def main(report_dir):
    hits = sorted(glob.glob(f"{report_dir}/reports/*/ops_perf_results_*.csv"))
    if not hits:
        print(f"PROBE_SUMMARY no ops_perf_results csv under {report_dir}")
        return 1
    df = pd.read_csv(hits[-1])
    d = df[df["OP CODE"] == OP_CODE].copy()
    d["W"] = d[WIDTH].astype(str).str.split("[").str[0].astype(int)
    d = d.sort_values(["DEVICE ID", "GLOBAL CALL COUNT"])
    d["call"] = d.groupby(["W", "DEVICE ID"]).cumcount()
    print(f"PROBE_SUMMARY csv={hits[-1]}")
    print("PROBE_SUMMARY shape | n | us_chip_mean | us_max_over_chips | us_min_chip_mean | per-device means | cores")
    for w, name in SHAPES.items():
        g = d[(d["W"] == w) & (d["call"] >= WARMUP)]
        if g.empty:
            print(f"PROBE_SUMMARY {name} | 0 rows")
            continue
        per_dev = g.groupby("DEVICE ID")[KERNEL].mean() / 1e3
        per_call_max = g.groupby("call")[KERNEL].max() / 1e3
        cores = int(g["CORE COUNT"].iloc[0]) if "CORE COUNT" in g else -1
        print(
            f"PROBE_SUMMARY {name} | {len(g)} | {g[KERNEL].mean() / 1e3:.3f} | {per_call_max.mean():.3f} | "
            f"{per_dev.min():.3f} | {' '.join(f'{v:.2f}' for v in per_dev)} | {cores}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
