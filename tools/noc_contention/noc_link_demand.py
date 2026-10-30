#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-link NoC demand over time from a tt-npe timeline: which links are oversubscribed, in which rows, when.

    noc_link_demand.py npe_timeline.json [--bin 4] [--top 12] [--rows 2,3] [--kinds NOC0_EAST,NOC1_WEST]
                       [--threshold 100] [--link-bw 60.9] [--clock-mhz 1350] [--csv links.csv]

The timeline is what `tt_npe.py -e --timeline-json-filepath ...` emits when it replays a tt-metal NoC trace
(see README.md). Its timestep_data[i].link_demand lists, for every link with traffic in timestep i,
[chip, row, col, kind, demand], where demand is the OFFERED rate in bytes per cycle (every active transfer
contributes its injection rate; a contended transfer keeps offering while it is stretched, so demand above the
link bandwidth means transfers are queueing on that link) and kind is one of
  NOC0_EAST / NOC0_SOUTH   the +x / +y hop out of (col, row) on NOC_0
  NOC1_WEST / NOC1_NORTH   the -x / -y hop out of (col, row) on NOC_1
  NOC0_IN / NOC1_IN        injection from the core's NIU into the NoC
  NOC0_OUT / NOC1_OUT      ejection from the NoC into the core
This script converts demand to percent of the link bandwidth (--link-bw in bytes per cycle; the device model's
value by default: blackhole 60.9, wormhole_b0 30.0, quasar 243.6, taken from common_info.arch) and also reports
utilization = min(demand, bandwidth), the delivered rate. Coordinates are the trace's physical NoC coordinates
(on Blackhole: workers at x = 1-7 / 10-16, y = 2-11; DRAM ports at x = 0 and x = 9).

Read this, not tt-npe's summary line: the summary averages utilization over the whole run and every link of the
chip, which hides a hand-off link that is contended for half the run. This script reports, per link, the mean
demand and utilization over the run, the number of timesteps above the threshold and the peak, then per physical
row the oversubscribed link-timesteps per time window and by link kind, and for the rows asked for a per-column
heatmap of each kind.
"""
import argparse
import collections
import csv
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("timeline")
    ap.add_argument("--bin", type=float, default=4.0, help="window in us for the per-row tables")
    ap.add_argument("--top", type=int, default=12, help="busiest links to list")
    ap.add_argument("--rows", default="", help="physical rows to print per-column heatmaps for (comma list)")
    ap.add_argument("--kinds", default="NOC0_EAST,NOC1_WEST,NOC0_OUT,NOC1_OUT", help="link kinds for the heatmaps")
    ap.add_argument(
        "--threshold", type=float, default=100.0, help="demand %% of link bandwidth counted as oversubscribed"
    )
    ap.add_argument("--link-bw", type=float, default=0.0, help="link bandwidth in bytes/cycle (default: the model's)")
    ap.add_argument("--clock-mhz", type=float, default=1350.0, help="core clock (Blackhole 1350, Wormhole 1000)")
    ap.add_argument("--csv", default="", help="write per-link, per-window mean demand and oversubscribed counts")
    a = ap.parse_args()
    clk = a.clock_mhz  # cycles per us

    tl = json.load(open(a.timeline))
    steps = tl["timestep_data"]
    cyc = tl["common_info"]["cycles_per_timestep"]
    arch = tl["common_info"].get("arch", "")
    bw = a.link_bw or {"blackhole": 60.9, "wormhole_b0": 30.0, "quasar": 243.6}.get(arch, 0.0)
    if bw <= 0:
        raise SystemExit(f"unknown arch {arch!r}: pass --link-bw (bytes per cycle)")
    scale = 100.0 / bw  # bytes/cycle -> % of link bandwidth
    t_end = max(st["end_cycle"] for st in steps)
    nbin = int(t_end / clk / a.bin) + 1
    steps_per_bin = a.bin * clk / cyc
    # link -> per window [sum of demand %, oversubscribed timesteps, peak demand %, sum of utilization %]
    acc = collections.defaultdict(lambda: [[0.0, 0, 0.0, 0.0] for _ in range(nbin)])
    for st in steps:
        w = int(st["start_cycle"] / clk / a.bin)
        for _chip, row, col, kind, dem in st["link_demand"]:
            d = dem * scale
            e = acc[(row, col, kind)][w]
            e[0] += d
            e[1] += d > a.threshold
            e[2] = max(e[2], d)
            e[3] += min(d, 100.0)
    print(
        f"timeline: {len(steps)} timesteps x {cyc} cycles = {t_end / clk:.1f} us at {clk:.0f} MHz ({arch}, link "
        f"{bw:g} B/cycle = {bw * clk / 1e3:.1f} GB/s); windows of {a.bin} us; oversubscribed = demand > {a.threshold:.0f} %"
    )

    # (a) the busiest links: mean demand over the run, oversubscribed timesteps, peak, and when
    ranked = []
    for link, wins in acc.items():
        mean = sum(w[0] for w in wins) / len(steps)
        util = sum(w[3] for w in wins) / len(steps)
        over = sum(w[1] for w in wins)
        peak = max(w[2] for w in wins)
        ranked.append((over, mean, util, peak, link))
    ranked.sort(key=lambda r: (-r[0], -r[1]))
    print(
        f"\n== busiest links (timesteps > {a.threshold:.0f} % | mean offered demand % | mean utilization % | peak %):"
    )
    for over, mean, util, peak, (row, col, kind) in ranked[: a.top]:
        wins = acc[(row, col, kind)]
        hot = [i for i, w in enumerate(wins) if w[1] > 0]
        span = f"{hot[0] * a.bin:.0f}-{(hot[-1] + 1) * a.bin:.0f} us" if hot else "-"
        print(
            f"  ({col:2d},{row:2d}) {kind:<11} over {over:5d}  demand {mean:6.1f}  util {util:5.1f}  peak {peak:6.0f}"
            f"  oversubscribed in {span}"
        )

    # (b) per physical row: oversubscribed link-timesteps per window (all kinds), and by kind over the run
    byrow = collections.defaultdict(lambda: [0] * nbin)
    bykind = collections.defaultdict(int)
    for (row, col, kind), wins in acc.items():
        for i, w in enumerate(wins):
            byrow[row][i] += w[1]
            bykind[(row, kind)] += w[1]
    print(f"\n== oversubscribed link-timesteps per physical row and {a.bin:g} us window (all kinds):")
    print("  row | " + " ".join(f"{int(i * a.bin):4d}" for i in range(nbin)))
    for row in sorted(byrow):
        print(f"  {row:3d} | " + " ".join(f"{v:4d}" if v else "   ." for v in byrow[row]))
    kinds = sorted({k for _, k in bykind})
    print("\n== oversubscribed link-timesteps per row by link kind (whole run):")
    print("  row | " + " ".join(f"{k:>11}" for k in kinds))
    for row in sorted({r for r, _ in bykind}):
        print(f"  {row:3d} | " + " ".join(f"{bykind[(row, k)]:11d}" for k in kinds))

    # (c) per requested row and kind: mean demand per window, one line per column
    for row in [int(r) for r in a.rows.split(",") if r != ""]:
        for kind in [k for k in a.kinds.split(",") if k]:
            cols = sorted({c for (r, c, k) in acc if r == row and k == kind})
            if not cols:
                continue
            print(f"\n== row {row} {kind}: mean demand % per window, one line per column x")
            print("   x | " + " ".join(f"{int(i * a.bin):4d}" for i in range(nbin)))
            for c in cols:
                wins = acc[(row, c, kind)]
                print(f"  {c:2d} | " + " ".join(f"{w[0] / steps_per_bin:4.0f}" for w in wins))

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(
                [
                    "row",
                    "col",
                    "kind",
                    "window_start_us",
                    "mean_demand_pct",
                    "mean_util_pct",
                    "oversubscribed_timesteps",
                    "peak_pct",
                ]
            )
            for (row, col, kind), wins in sorted(acc.items()):
                for i, w in enumerate(wins):
                    if w[0]:
                        wr.writerow(
                            [
                                row,
                                col,
                                kind,
                                i * a.bin,
                                round(w[0] / steps_per_bin, 2),
                                round(w[3] / steps_per_bin, 2),
                                w[1],
                                round(w[2], 1),
                            ]
                        )
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
