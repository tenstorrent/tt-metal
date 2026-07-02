# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Map Conv2dDeviceOperation rows of a tracy ops CSV (in execution order) to the sweep manifest.

Usage: python _parse_ct_sweep.py <ct_sweep.json> <ops_perf_results.csv>
Prints a per-config table of conv device time (sum over the config's DRAM slices) sorted within
each F so the fastest fitting strategy is obvious.
"""

import csv
import json
import sys


def main(manifest_path: str, csv_path: str) -> None:
    manifest = json.loads(open(manifest_path).read())

    conv_us = []  # device kernel duration (ns) of each Conv2dDeviceOperation row, in order
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["OP CODE"].strip() == "Conv2dDeviceOperation":
                conv_us.append(float(row["DEVICE KERNEL DURATION [ns]"]) / 1000.0)

    # Walk manifest in order; each "ok" config consumed n_dev_ops conv rows.
    i = 0
    rows = []
    for m in manifest:
        if m["status"] != "ok":
            rows.append((m, None))
            continue
        n = m["n_dev_ops"]
        dt = sum(conv_us[i : i + n]) if i + n <= len(conv_us) else None
        i += n
        rows.append((m, dt))

    print(f"matched {i}/{len(conv_us)} conv rows\n")
    print(f"{'F':>7} {'shard':>7} {'abh':>5} {'slices':>7} {'status':>8} {'dev_us':>9} {'pcc':>9}")
    print("-" * 60)
    by_f = {}
    for m, dt in rows:
        by_f.setdefault(m["F"], []).append((m, dt))
    for F in sorted(by_f):
        for m, dt in sorted(by_f[F], key=lambda x: (x[1] is None, x[1] or 0)):
            dts = f"{dt:9.1f}" if dt is not None else f"{'-':>9}"
            pcc = f"{m['pcc']:.5f}" if m["pcc"] is not None else "-"
            print(
                f"{m['F']:>7} {m['shard']:>7} {str(m['abh']):>5} {str(m['slices']):>7} {m['status']:>8} {dts} {pcc:>9}"
            )
        print()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
