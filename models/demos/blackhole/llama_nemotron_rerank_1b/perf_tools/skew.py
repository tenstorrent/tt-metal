# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Per op type in the signposted window: median kernel duration, per-core min/max duration and first-to-last start skew (us).
import csv
import statistics
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
st = max(i for i, r in enumerate(rows) if r["OP CODE"] == "start")
en = min(i for i, r in enumerate(rows) if r["OP CODE"] == "stop" and i > st)
by = defaultdict(list)
f = lambda r, k: float(r[k] or 0) / 1e3
for r in rows[st + 1 : en]:
    if r["OP TYPE"] == "signpost":
        continue
    k = r["OP CODE"][:26] + " " + r["INPUT_0_X_PAD[LOGICAL]"].split("[")[0]
    by[k].append(
        (
            f(r, "DEVICE FW DURATION [ns]"),
            f(r, "DEVICE KERNEL DURATION [ns]"),
            f(r, "DEVICE KERNEL DURATION PER CORE MIN [ns]"),
            f(r, "DEVICE KERNEL DURATION PER CORE MAX [ns]"),
            f(r, "DEVICE KERNEL FIRST TO LAST START [ns]"),
        )
    )
print(f"{'op':34s} {'fw':>8s} {'kernel':>8s} {'coreMin':>8s} {'coreMax':>8s} {'startSkew':>9s}")
for k, v in by.items():
    m = [statistics.median(x[i] for x in v) for i in range(5)]
    print(f"{k:34s} " + " ".join(f"{x:8.1f}" for x in m))
