#!/usr/bin/env python3
"""
Analyze AGMM kernel zone profile to find per-zone time distribution.

Reads profile_log_device.csv produced by the device profiler and aggregates:
  - Custom zone events (ZONE_START / ZONE_END pairs)
  - Sum-zone totals (ZONE_TOTAL events from DeviceZoneScopedSumN1/N2)
grouped by zone name and RISC processor type.

Usage:
    python tools/agmm_profile_analyze.py <profile_log_device.csv>
                                          [--run-host-id N]
                                          [--top-zones K]

If --run-host-id is omitted, the script picks the LAST distinct run host id in
the log (assumed to be the measured execute_trace pass, after compile+capture
warmups).

Output:
  Per zone (sorted by total time, descending):
    name, processor, count, mean_ns, p50_ns, p90_ns, max_ns, total_ns
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional, Tuple

CHIP_FREQ_MHZ = 1000  # wormhole_b0; first line of CSV says "CHIP_FREQ[MHz]: 1000"


def cycles_to_ns(cycles: int) -> float:
    return cycles / CHIP_FREQ_MHZ * 1000  # cycles @ 1 GHz -> ns


def percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = pct / 100.0 * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def parse_log(path: str, want_run_host_id: Optional[int] = None):
    """Return (events, run_host_ids).

    events: list of dicts. Each row of the CSV (post header).
    """
    events = []
    run_host_ids = []
    with open(path, "r") as f:
        # First line is metadata: "ARCH: wormhole_b0, CHIP_FREQ[MHz]: 1000, ..."
        meta = f.readline()
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            # Trim whitespace from values
            row = {k.strip(): v.strip() if isinstance(v, str) else v for k, v in row.items()}
            try:
                run_host_id = int(row.get("run host ID") or row.get("run_host_id") or 0)
            except ValueError:
                continue
            if run_host_id not in run_host_ids:
                run_host_ids.append(run_host_id)
            events.append(row)
    return events, run_host_ids, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_log", help="path to profile_log_device.csv")
    ap.add_argument(
        "--run-host-id", type=int, default=None, help="run-host-id of the AGMM op invocation to analyze (default: last)"
    )
    ap.add_argument("--top-zones", type=int, default=30)
    ap.add_argument("--include-system", action="store_true", help="include FW/KERNEL framework zones in output")
    args = ap.parse_args()

    events, run_host_ids, meta = parse_log(args.profile_log)
    print(f"# {meta.strip()}")
    print(
        f"# Found {len(events)} events across {len(run_host_ids)} run-host-id(s):"
        f" {run_host_ids[:10]}{'...' if len(run_host_ids) > 10 else ''}"
    )
    if args.run_host_id is None:
        # heuristic: pick last (largest) run-host-id
        target = max(run_host_ids) if run_host_ids else 0
        print(f"# Auto-selected last run-host-id={target}")
    else:
        target = args.run_host_id
        print(f"# Filtering to run-host-id={target}")
    filtered = [e for e in events if int(e.get("run host ID", 0)) == target]
    print(f"# Filtered: {len(filtered)} events for run-host-id={target}")
    print()

    # Group by (core_x, core_y, risc, timer_id) -> [(type, cycles), ...]
    # zone_name is the user-given name (e.g., "send", "cb_wait_in0"), or
    # framework name like "BRISC-FW" / "BRISC-KERNEL".
    paired_zones: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    # (zone_name, processor) -> list of durations in ns

    # Track pending starts: (core, risc, timer_id) -> start_cycles
    pending: Dict[Tuple[str, str, str, str], int] = {}

    # Sum totals: ZONE_TOTAL events carry the accumulated cycles in the 'data'
    # column (not paired; one event per sum-zone per run).
    sum_totals: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for e in filtered:
        zone = e.get("zone name", "")
        etype = e.get("type", "")
        proc = e.get("RISC processor type", "")
        cx, cy = e.get("core_x", ""), e.get("core_y", "")
        timer = e.get("timer_id", "")
        try:
            cycles = int(e.get("time[cycles since reset]", "0"))
        except ValueError:
            continue
        try:
            data = int(e.get("data", "0") or "0")
        except ValueError:
            data = 0
        key = (cx, cy, proc, timer)
        if etype == "ZONE_START":
            pending[key] = cycles
        elif etype == "ZONE_END":
            if key in pending:
                dur_cycles = cycles - pending[key]
                paired_zones[(zone, proc)].append(cycles_to_ns(dur_cycles))
                del pending[key]
        elif etype == "ZONE_TOTAL":
            # accumulated cycles in 'data'
            sum_totals[(zone, proc)].append(cycles_to_ns(data))

    if pending:
        print(f"# WARN: {len(pending)} unpaired ZONE_START events")

    # Build summary table
    SystemZones = {
        "BRISC-FW",
        "BRISC-KERNEL",
        "NCRISC-FW",
        "NCRISC-KERNEL",
        "TRISC-FW",
        "TRISC-KERNEL",
        "ERISC-FW",
        "ERISC-KERNEL",
        "BRISC",
        "NCRISC",
        "TRISC0",
        "TRISC1",
        "TRISC2",
        "ERISC",
    }

    rows = []
    for (zone, proc), durs in paired_zones.items():
        if not args.include_system and zone in SystemZones:
            continue
        durs_sorted = sorted(durs)
        rows.append(
            {
                "zone": zone,
                "proc": proc,
                "kind": "scoped",
                "count": len(durs),
                "mean_ns": mean(durs),
                "p50_ns": percentile(durs_sorted, 50),
                "p90_ns": percentile(durs_sorted, 90),
                "max_ns": max(durs),
                "total_ns": sum(durs),
            }
        )
    for (zone, proc), totals in sum_totals.items():
        if not args.include_system and zone in SystemZones:
            continue
        totals_sorted = sorted(totals)
        rows.append(
            {
                "zone": zone,
                "proc": proc,
                "kind": "sum",
                "count": len(totals),
                "mean_ns": mean(totals),
                "p50_ns": percentile(totals_sorted, 50),
                "p90_ns": percentile(totals_sorted, 90),
                "max_ns": max(totals),
                "total_ns": sum(totals),
            }
        )

    rows.sort(key=lambda r: r["total_ns"], reverse=True)

    # Print table
    print(
        f"{'zone':<22} {'proc':<8} {'kind':<6} {'count':>6} "
        f"{'mean_ns':>10} {'p50_ns':>10} {'p90_ns':>10} {'max_ns':>10} {'total_ns':>12}"
    )
    print("-" * 110)
    for r in rows[: args.top_zones]:
        print(
            f"{r['zone']:<22} {r['proc']:<8} {r['kind']:<6} {r['count']:>6} "
            f"{r['mean_ns']:>10.0f} {r['p50_ns']:>10.0f} {r['p90_ns']:>10.0f} "
            f"{r['max_ns']:>10.0f} {r['total_ns']:>12.0f}"
        )

    # Per-processor summary: how much time spent in custom zones vs total
    print()
    print("# Per-processor total zone time (custom zones only):")
    proc_totals: Dict[str, float] = defaultdict(float)
    proc_counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        proc_totals[r["proc"]] += r["total_ns"]
        proc_counts[r["proc"]] += r["count"]
    for proc in sorted(proc_totals.keys()):
        print(f"  {proc:<8}  total={proc_totals[proc]:>12,.0f} ns   events={proc_counts[proc]}")


if __name__ == "__main__":
    main()
