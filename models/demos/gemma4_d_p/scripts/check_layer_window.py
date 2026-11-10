# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Count ops between a layer-perf start/stop signpost pair per device; fail if any op lost its device duration.

    python3 -I check_layer_window.py <ops_perf_results.csv> gemma4-layer-local-sz4096-chunk1
"""
import sys

import pandas as pd

csv, base = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv)
sp = df[df["OP TYPE"] == "signpost"]
t0 = sp[sp["OP CODE"] == base + "-start"]["HOST START TS"].min()
t1 = sp[sp["OP CODE"] == base + "-stop"]["HOST START TS"].min()
if pd.isna(t0) or pd.isna(t1):
    sys.exit(f"signposts {base}-start/-stop not found in {csv}")
ops = df[(df["OP TYPE"] != "signpost") & (df["HOST START TS"] > t0) & (df["HOST START TS"] < t1)]
per_dev = ops.groupby("DEVICE ID").size() if "DEVICE ID" in ops.columns else pd.Series({0: len(ops)})
print(f"{base}: ops in window per device: min {per_dev.min()} max {per_dev.max()} ({len(per_dev)} devices)")
dur = "DEVICE KERNEL DURATION [ns]"
if dur in ops.columns:
    lost = int(ops[dur].isna().sum() + (ops[dur] == 0).sum())
    print(f"{base}: ops with no device duration (dropped): {lost}")
    if lost:
        sys.exit("FAIL: ops lost device data")
