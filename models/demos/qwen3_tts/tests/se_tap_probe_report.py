# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Score the Speaker-Encoder tap probe against the newest profiler CSV.

Deliberately does NOT align rows positionally. The probe's arms emit different op
counts per launch — a matmul tap is 1 op, `ttnn.gather` on dim=1 is 3 (it needs a
transpose either side), and `slice + concat` is 12-20 (a non-tile-aligned slice of a
tiled tensor round-trips through untilize/tilize) — so counting spans by hand
desynchronises after the first case and silently reports one arm's time under
another's name. That happened, and the giveaway was a chunk whose CORE COUNT held
two different values.

Instead it groups by the columns the CSV already carries: OP CODE, MATH FIDELITY and
CORE COUNT. The fidelity arms separate cleanly that way, which is the comparison the
probe exists to make.
"""

from __future__ import annotations

import collections
import csv
import glob
import os
import statistics
import sys


def main() -> int:
    csvs = sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"), key=os.path.getmtime)
    if not csvs:
        print("no profiler CSV — run under `python -m tracy -p -v -r`")
        return 1
    path = csvs[-1]
    man = "generated/se_tap_manifest.json"
    if os.path.exists(man) and os.path.getmtime(path) < os.path.getmtime(man) - 5:
        print(f"REFUSING: newest CSV {path} predates the manifest.")
        return 1
    print(f"csv: {path}\n")

    g = collections.defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            code = (r.get("OP CODE") or "").strip()
            dur = (r.get("DEVICE KERNEL DURATION [ns]") or "").strip()
            if not code or not dur.isdigit():
                continue
            g[(code, (r.get("MATH FIDELITY") or "-").strip(), int(float(r.get("CORE COUNT") or 0)))].append(int(dur))

    print("| op code | fidelity | cores | n | median us | min | max |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for k in sorted(g, key=lambda k: -statistics.median(g[k]) * len(g[k])):
        v = g[k]
        print(
            f"| {k[0].replace('DeviceOperation','')} | {k[1]} | {k[2]} | {len(v)} "
            f"| {statistics.median(v)/1000:.1f} | {min(v)/1000:.1f} | {max(v)/1000:.1f} |"
        )

    mm = {k: v for k, v in g.items() if k[0] == "MatmulDeviceOperation"}
    if mm:
        print("\n--- the question the probe asks ---")
        by_cores = collections.defaultdict(dict)
        for (_c, fid, cores), v in mm.items():
            by_cores[cores][fid] = statistics.median(v) / 1000
        for cores in sorted(by_cores):
            row = by_cores[cores]
            span = max(row.values()) - min(row.values())
            print(
                f"  {cores:2d} cores: "
                + "  ".join(f"{f}={row[f]:.1f}us" for f in sorted(row))
                + f"   spread {span:.1f} us"
                + ("  -> fidelity is FREE here (not math-bound)" if span < 1.0 else "")
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
