# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reduce a `profile_log_device.csv` into a per-stage rms_norm breakdown.

Host-only (no device). Pairs with `perf_zone_harness.py`, which produces the log.

    python3 tests/ttnn/unit_tests/operations/rms_norm/perf_zone_report.py <log.csv> [--all-cores]

Prints, per (RISC, zone name): the number of executions, and the summed
per-execution duration in ns aggregated over the cores that ran it — `max` is the
CRITICAL-PATH core (the one the wall is set by), `p50` the typical core. Also
prints, per RISC, the marker count against the 250/RISC cap and the fraction of
the `*-KERNEL` span the zones cover — both required checks before ranking
anything off these numbers (device-zone-scope-attribution.md §7).
"""

from __future__ import annotations

import csv
import statistics
import sys

CLK_MHZ_DEFAULT = 1350.0
MARKER_CAP = 250


def load(path):
    rows = list(csv.reader(open(path)))
    clk = CLK_MHZ_DEFAULT
    for tok in rows[0]:
        if "CHIP_FREQ" in tok:
            clk = float(tok.split(":")[1])
    return rows[2:], clk


def analyse(path, all_cores=False, runtime_id=None):
    rows, clk = load(path)
    # `ReadDeviceProfiler` APPENDS: one log can hold several programs, told apart
    # by "run host ID" (col 8). Default to the LAST program in the file.
    ids = sorted({int(r[7]) for r in rows if len(r) > 7 and r[7].strip().isdigit()})
    rid = runtime_id if runtime_id is not None else (ids[-1] if ids else None)
    if rid is not None:
        rows = [r for r in rows if len(r) > 7 and r[7].strip().isdigit() and int(r[7]) == rid]
    print(f"# program run-host-ids in log: {ids} -> analysing {rid}")
    # (core, risc) -> list of (zone, type, cycles)
    per_core = {}
    markers = {}
    prog_lo = prog_hi = None
    for r in rows:
        if len(r) < 12:
            continue
        if r[10].strip().endswith("-KERNEL"):
            t_ = int(r[5])
            prog_lo = t_ if prog_lo is None else min(prog_lo, t_)
            prog_hi = t_ if prog_hi is None else max(prog_hi, t_)
        core = (int(r[1]), int(r[2]))
        risc = r[3].strip()
        t = int(r[5])
        zone = r[10].strip()
        typ = r[11].strip()
        per_core.setdefault((core, risc), []).append((zone, typ, t))
        markers[(core, risc)] = markers.get((core, risc), 0) + 1

    # per (risc, zone) -> {core: (total_ns, count)}
    agg = {}
    spans = {}
    coverage = {}
    for (core, risc), evs in per_core.items():
        open_stack = {}
        totals = {}
        counts = {}
        kernel_lo = kernel_hi = None
        zone_lo = zone_hi = None
        for zone, typ, t in evs:
            if typ == "ZONE_START":
                open_stack.setdefault(zone, []).append(t)
                if zone.endswith("-KERNEL"):
                    kernel_lo = t if kernel_lo is None else min(kernel_lo, t)
                elif not zone.endswith("-FW"):
                    zone_lo = t if zone_lo is None else min(zone_lo, t)
            elif typ == "ZONE_END":
                st = open_stack.get(zone)
                if not st:
                    continue
                t0 = st.pop()
                d = (t - t0) * 1000.0 / clk  # ns (cycles / MHz = us)
                totals[zone] = totals.get(zone, 0.0) + d
                counts[zone] = counts.get(zone, 0) + 1
                if zone.endswith("-KERNEL"):
                    kernel_hi = t if kernel_hi is None else max(kernel_hi, t)
                elif not zone.endswith("-FW"):
                    zone_hi = t if zone_hi is None else max(zone_hi, t)
        for zone, tot in totals.items():
            agg.setdefault((risc, zone), {})[core] = (tot, counts[zone])
        if kernel_lo is not None and kernel_hi is not None:
            spans[(core, risc)] = (kernel_hi - kernel_lo) * 1000.0 / clk
            if zone_lo is not None:
                coverage[(core, risc)] = (zone_hi - zone_lo) / max(1e-9, (kernel_hi - kernel_lo))

    print(f"# {path}   clock={clk} MHz")
    if prog_lo is not None:
        print(
            f"# DEVICE KERNEL span (max end - min start over all cores/RISCs) = {(prog_hi - prog_lo) * 1000.0 / clk:.0f} ns"
        )
    print("\n## marker budget / zone coverage (silent truncation check)")
    for risc in sorted({r for (_, r) in markers}):
        ms = [m for (c, r), m in markers.items() if r == risc]
        cov = [v for (c, r), v in coverage.items() if r == risc]
        sp = [v for (c, r), v in spans.items() if r == risc]
        print(
            f"  {risc:8s} cores={len(ms):4d} markers/RISC max={max(ms):4d} (cap {MARKER_CAP})"
            f"  KERNEL span ns max={max(sp):9.0f} p50={statistics.median(sp):9.0f}"
            f"  zone coverage of span p50={statistics.median(cov) * 100 if cov else 0:5.1f}%"
        )

    print("\n## per-stage ns (summed over a core's executions)")
    print(f"  {'RISC':8s} {'zone':22s} {'cores':>5s} {'exec':>5s} {'max_ns':>9s} {'p50_ns':>9s} {'min_ns':>9s}")
    order = sorted(agg.items(), key=lambda kv: (kv[0][0], -max(v[0] for v in kv[1].values())))
    for (risc, zone), percore in order:
        tots = [v[0] for v in percore.values()]
        cnts = {v[1] for v in percore.values()}
        print(
            f"  {risc:8s} {zone:22s} {len(percore):5d} {max(cnts):5d} "
            f"{max(tots):9.0f} {statistics.median(tots):9.0f} {min(tots):9.0f}"
        )

    if all_cores:
        print("\n## critical-path core timeline (the core with the longest BRISC-KERNEL span)")
        worst = max(((c, r) for (c, r) in spans if r == "BRISC"), key=lambda k: spans[k])
        base = None
        for (core, risc), evs in sorted(per_core.items()):
            if core != worst[0]:
                continue
            for zone, typ, t in evs:
                if base is None:
                    base = t
                if typ == "ZONE_START":
                    print(f"  {risc:8s} {zone:22s} start +{(t - base) * 1000.0 / clk:9.0f} ns")
                else:
                    print(f"  {risc:8s} {zone:22s} end   +{(t - base) * 1000.0 / clk:9.0f} ns")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rid = None
    for a in sys.argv[1:]:
        if a.startswith("--rid="):
            rid = int(a.split("=")[1])
    analyse(args[0], all_cores="--all-cores" in sys.argv, runtime_id=rid)
