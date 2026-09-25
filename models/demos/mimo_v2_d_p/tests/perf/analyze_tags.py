# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generic signpost summary: per tag, mean over iterations of the summed device-kernel time (max over chips),
plus the per-op breakdown with --ops.

    python models/demos/mimo_v2_d_p/tests/perf/analyze_tags.py <csv> [--ops] [--filter substr]
"""

import argparse
import collections
import csv


def load(path):
    data = collections.OrderedDict()
    cur, it = None, collections.Counter()
    for x in csv.DictReader(open(path)):
        if x["OP TYPE"] == "signpost":
            c = x["OP CODE"]
            if c.endswith("_start"):
                cur = c[: -len("_start")]
                it[cur] += 1
                data.setdefault(cur, collections.defaultdict(lambda: collections.defaultdict(list)))
            elif c.endswith("_end"):
                cur = None
            continue
        if cur is not None:
            data[cur][it[cur]][int(x["DEVICE ID"])].append((x["OP CODE"], float(x["DEVICE KERNEL DURATION [ns]"] or 0) / 1e3))
    return data


def summarize(iters):
    per_op = collections.defaultdict(list)
    for devs in iters.values():
        n = min(len(v) for v in devs.values())
        for j in range(n):
            per_op[f"{j:02d} {next(iter(devs.values()))[j][0]}"].append(max(v[j][1] for v in devs.values()))
    ops = {k: sum(v) / len(v) for k, v in per_op.items()}
    return sum(ops.values()), ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--ops", action="store_true")
    ap.add_argument("--filter", default="")
    a = ap.parse_args()
    for tag, iters in load(a.csv).items():
        if a.filter not in tag:
            continue
        total, ops = summarize(iters)
        print(f"{tag:<50s} {total:10.1f} us")
        if a.ops:
            for k, v in ops.items():
                print(f"    {k:<56s} {v:9.1f}")


if __name__ == "__main__":
    main()
