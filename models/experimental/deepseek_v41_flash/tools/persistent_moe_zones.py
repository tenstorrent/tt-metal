# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Per-stage times of the persistent TP12 MoE, from a streaming-profiler zone CSV.

Capture with the one-shot runs left out (their builds share the persistent kernels' zone ids):

    TT_METAL_STREAMING_PROFILER=1 TT_METAL_STREAMING_PROFILER_ZONE_CSV=/tmp/fe_zones.csv \\
    MOE_PERF_PERSISTENT_ONLY=1 MOE_PERF_WARMUP=5 MOE_PERF_STEPS=40 \\
    pytest models/experimental/deepseek_v41_flash/tests/decode/test_moe_persistent.py::test_moe_persistent_tp12_perf

    python models/experimental/deepseek_v41_flash/tools/persistent_moe_zones.py /tmp/fe_zones.csv --warmup 5

The launch runs its rows from eager requests, then as many from traced ones. Device clocks are not
aligned across chips, so every time is measured within one chip, from that iteration's go (the end of the
control core's FE_CTRL_MCAST):

- router: the last router core's FE_R_TOPK end (gate matmul, scores, local top-k);
- ids out: the leader's FE_ROUTE_WEIGHTS_MCAST end (merge, weights, the ids broadcast) -- stage 1 done;
- partials: the reduce senders' last FE_RED_PARTIALS_WAIT end. Off the root that is every output core's
  partial staged -- stage 2 done; on the root it is every output core's cross-chip sum done -- stage 3;
- row out (root): FE_OUT_PUSH end;
- period: go to the next go; ready wait: the control core's FE_CTRL_READY_WAIT, its idle wait for the
  previous row's stragglers.
"""

import argparse
import csv
import statistics
from collections import defaultdict


def _load(path):
    """``(mhz, zones)``: zones[launch][chip][(core, risc, name)] = [(start, end), ...] in issue order."""
    zones = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(path) as f:
        mhz = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)
        pending = {}
        for row in reader:
            chip, risc, cycles, launch, name, kind = int(row[0]), row[3], int(row[5]), int(row[7]), row[10], row[11]
            if not name.startswith("FE_"):
                continue
            key = ((int(row[15]), int(row[16])), risc, name)
            if kind == "ZONE_START":
                pending[(chip, key)] = cycles
            elif kind == "ZONE_END":
                zones[launch][chip][key].append((pending.pop((chip, key)), cycles))
    return mhz, zones


def _per_iter(chip_zones, name, end=True):
    """Iteration k -> the latest end (or start) of zone ``name`` across the cores that run it."""
    out = defaultdict(lambda: None)
    for (_, _, n), spans in chip_zones.items():
        if n != name:
            continue
        for k, (s, e) in enumerate(spans):
            t = e if end else s
            out[k] = t if out[k] is None else max(out[k], t)
    return out


def _durations(chip_zones, name):
    """Iteration k -> the longest duration of zone ``name`` across the cores that run it."""
    out = defaultdict(int)
    for (_, _, n), spans in chip_zones.items():
        if n == name:
            for k, (s, e) in enumerate(spans):
                out[k] = max(out[k], e - s)
    return out


def _median(values):
    return statistics.median(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("--warmup", type=int, default=5, help="rows skipped at the start of each loop")
    args = parser.parse_args()

    mhz, zones = _load(args.csv)
    us = lambda cycles: cycles / mhz
    launches = sorted(l for l in zones if any(_per_iter(c, "FE_CTRL_MCAST") for c in zones[l].values()))
    for launch in launches:
        chips = zones[launch]
        root = next(c for c, z in chips.items() if any(n == "FE_OUT_PUSH" for (_, _, n) in z))
        rows = {}
        for chip, z in chips.items():
            go = _per_iter(z, "FE_CTRL_MCAST")
            marks = {
                "router": _per_iter(z, "FE_R_TOPK"),
                "ids out": _per_iter(z, "FE_ROUTE_WEIGHTS_MCAST"),
                "partials": _per_iter(z, "FE_RED_PARTIALS_WAIT"),
            }
            if chip == root:
                marks["row out"] = _per_iter(z, "FE_OUT_PUSH")
            # The control core waits for stragglers from its second row on.
            ready = {k + 1: d for k, d in _durations(z, "FE_CTRL_READY_WAIT").items()}
            n = len(go)
            rows[chip] = (n, go, marks, ready)

        n = min(r[0] for r in rows.values())
        half = n // 2
        print(f"launch {launch}: {n} rows ({half} eager, then {n - half} traced), root chip {root}, {mhz:.0f} MHz")
        for loop, lo, hi in (("eager", args.warmup, half), ("traced", half + args.warmup, n - 1)):
            print(f"  {loop} (rows {lo}..{hi - 1}), median us from the row's go:")
            print(
                f"    {'chips':<10}"
                + "".join(f"{c:>10}" for c in ("router", "ids out", "partials", "row out"))
                + f"{'period':>10}{'ready wait':>12}"
            )
            for label, members in (("root", [root]), ("non-root", [c for c in rows if c != root])):
                cols = []
                for mark in ("router", "ids out", "partials", "row out"):
                    vals = [
                        us(rows[c][2][mark][k] - rows[c][1][k])
                        for c in members
                        if mark in rows[c][2]
                        for k in range(lo, hi)
                        if rows[c][2][mark][k] is not None
                    ]
                    cols.append(_median(vals))
                period = _median([us(rows[c][1][k + 1] - rows[c][1][k]) for c in members for k in range(lo, hi)])
                ready = _median([us(rows[c][3][k]) for c in members for k in range(lo, hi) if k in rows[c][3]])
                print(f"    {label:<10}" + "".join(f"{v:>10.1f}" for v in cols) + f"{period:>10.1f}{ready:>12.1f}")


if __name__ == "__main__":
    main()
