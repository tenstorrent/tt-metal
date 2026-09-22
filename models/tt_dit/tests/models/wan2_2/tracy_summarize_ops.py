# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Summarise a Tracy `ops_perf_results_*.csv` for the TI2V-5B: device time per step and top ops.

Validates a reported per-step time against the device: the sum of `DEVICE KERNEL DURATION`
over one device's ops, divided by the number of denoise steps in the capture, is the device
kernel time per step. The pipeline is sequence- and tensor-parallel, so every device runs the
same op stream and the per-device totals should agree; the report shows min / mean / max over
devices. `OP TO OP LATENCY` and wall-clock are not trustworthy under Tracy (see
`Wan2_2_TI2V_nadim_opt.md` section 5) and are not used.

    python models/tt_dit/tests/models/wan2_2/tracy_summarize_ops.py \\
        generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv --steps 2 --top 25
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

KERNEL = "DEVICE KERNEL DURATION [ns]"


def summarise(csv_path: str, *, steps: int, top: int, op_filter: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df[KERNEL].notna() & (df[KERNEL] > 0)]
    if op_filter:
        df = df[df["OP CODE"].str.contains(op_filter, case=False, regex=True)]

    per_dev = df.groupby("DEVICE ID")[KERNEL].sum() / 1e6  # ms
    print(f"ops with device time: {len(df)}  devices: {len(per_dev)}  steps in capture: {steps}")
    print(
        f"device kernel time, total per device: min {per_dev.min():.1f}  mean {per_dev.mean():.1f}  "
        f"max {per_dev.max():.1f} ms"
    )
    if steps:
        print(
            f"device kernel time per step:        min {per_dev.min() / steps:.2f}  mean {per_dev.mean() / steps:.2f}  "
            f"max {per_dev.max() / steps:.2f} ms/step"
        )

    # Rank by mean-over-devices total so a chip with an outlier does not dominate.
    n_dev = max(len(per_dev), 1)
    by_op = (
        df.groupby("OP CODE")
        .agg(total_ms=(KERNEL, lambda s: s.sum() / 1e6 / n_dev), count=(KERNEL, lambda s: len(s) / n_dev))
        .sort_values("total_ms", ascending=False)
    )
    by_op["share_%"] = 100 * by_op["total_ms"] / by_op["total_ms"].sum()
    by_op["mean_us"] = 1e3 * by_op["total_ms"] / by_op["count"]
    if steps:
        by_op["ms_per_step"] = by_op["total_ms"] / steps
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)
    print(f"\ntop {top} ops by device kernel time (per device, mean over devices):")
    print(by_op.head(top).to_string(float_format=lambda x: f"{x:,.2f}"))
    return by_op


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--steps", type=int, default=0, help="denoise steps in the capture (0: skip per-step numbers)")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--filter", default=None, help="regex on OP CODE")
    a = p.parse_args(argv)
    summarise(a.csv, steps=a.steps, top=a.top, op_filter=a.filter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
