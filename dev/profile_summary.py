#!/usr/bin/env python3
"""Text summary of a device-profiler CSV, so a profiled run is readable without copying the multi-GB CSV back.

    python3 dev/profile_summary.py generated/profiler/.logs/profile_log_device.csv [per_core.txt] [> summary.txt]

Per program run (run host ID): core count and longest kernel span. For the longest run (the attention op), across
every (device, core): the span, the compute busy time (RING_STEP, summed on the TRISC that has the most), the idle
rest, and the reader / writer zones, as min / p10 / median / p90 / max in microseconds; the busy spread per device
(max / mean, the imbalance a work split could remove); the slowest and fastest cores; and the cores that ran the
group producer zone (RD_GROUP_TX). Hardware perf counters (TT_METAL_PROFILE_PERF_COUNTERS) are summarized if present.
Streams the CSV once; memory is per open zone, not per row.

The zones exist only in a build with SPARSE_STEP_PROFILE defined (sparse_ring_reader.cpp / sparse_ring_writer.cpp
SPARSE_ZONE, sparse_ring_streaming.hpp MaybeDeviceZoneScopedN); without it only the kernel spans are reported.
"""

import collections
import re
import sys

COMPUTE_ZONE = "RING_STEP"
WINDOW_NOTE = (
    "note: per-step zones cover only the TT_SPARSE_PROF_* window (default pass 0, ring iteration 1, steps 0-99), so busy /"
    " idle / step counts below describe that window, not the whole op; use the sum5/10/15 steps for whole-op totals."
)
SUM_ZONE = "RING_STEP_SUM"
READER_ZONES = [  # sparse_ring_reader.cpp
    "RD_RES_PIECES",
    "RD_GROUP_RX",
    "RD_GROUP_TX",
    "RD_CHAIN_RX",
    "RD_CHAIN_TX",
    "RD_FETCH_K",
    "RD_SYNC",
    "RD_RES_PLAN",
    "RD_RES_BITS",
    "RD_RES_Q",
    "RD_RES_POOLED",
]
OTHER_ZONES = ["RES_NORM", "NORM_MATMUL_RECIP", "NORM_MUL_BCAST", "WR_RES_OUT"]  # compute + sparse_ring_writer.cpp
KERNEL_ZONES = ["BRISC-KERNEL", "NCRISC-KERNEL", "TRISC-KERNEL"]
WANTED = set([COMPUTE_ZONE] + READER_ZONES + OTHER_ZONES + KERNEL_ZONES)


def pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))] if s else 0.0


def stats(vals):
    return "min %8.1f  p10 %8.1f  med %8.1f  p90 %8.1f  max %8.1f" % (
        min(vals),
        pct(vals, 0.1),
        pct(vals, 0.5),
        pct(vals, 0.9),
        max(vals),
    )


def main(path):
    freq_mhz = 1350.0
    open_starts = collections.defaultdict(list)  # (run, dev, x, y, risc, zone) -> start stack
    total = collections.defaultdict(int)  # same key -> summed cycles
    count = collections.defaultdict(int)
    span = {}  # (run, dev, x, y) -> [first kernel start, last kernel end]
    zspan = {}  # (run, dev, x, y, risc, zone) -> [first start, last end] (compute steps and pooled reads)
    TIMELINE_ZONES = ("RD_SYNC", "RD_RES_PIECES", "RD_RES_POOLED")
    events = collections.defaultdict(list)  # (run, dev, x, y) -> [(start, end, label)]: reader phases, step gaps
    last_step_end = {}  # (run, dev, x, y) -> end of the previous TRISC_0 step
    # Hardware perf counters (TT_METAL_PROFILE_PERF_COUNTERS; timer id 9090): (run, dev, x, y) -> counter -> (value,
    # ref cycles). The meta data column carries them as {'counter type'; 'FPU_COUNTER'; 'value'; n; 'ref cnt'; m}.
    counters = collections.defaultdict(dict)
    re_type = re.compile(r"counter type['\"]?\s*[:;]\s*['\"]?([A-Za-z0-9_]+)")
    re_val = re.compile(r"['\"]value['\"]\s*[:;]\s*(\d+)")
    re_ref = re.compile(r"ref cnt['\"]?\s*[:;]\s*(\d+)")
    # Sum zones (TT_METAL_PROFILER_SUM=1, SPARSE_PROF_SUM in sparse_prof_window.hpp): one ZONE_TOTAL row per RISC with
    # the whole op's summed step cycles in the data column. (run, dev, x, y, risc) -> cycles.
    step_sums = {}
    with open(path) as f:
        head = f.readline()
        if "CHIP_FREQ[MHz]:" in head:
            freq_mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        f.readline()
        for line in f:
            c = line.split(",", 12)
            if len(c) > 12 and "counter type" in c[12]:
                mt, mv, mr = re_type.search(c[12]), re_val.search(c[12]), re_ref.search(c[12])
                if mt and mv and mr:
                    counters[(c[7], c[0], c[1], c[2])][mt.group(1)] = (int(mv.group(1)), int(mr.group(1)))
                continue
            if len(c) >= 12 and c[11] == "ZONE_TOTAL" and c[10] == SUM_ZONE:
                step_sums[(c[7], c[0], c[1], c[2], c[3])] = int(c[6])
                continue
            if len(c) < 12 or c[10] not in WANTED:
                continue
            dev, x, y, risc, t, run, zone, typ = c[0], c[1], c[2], c[3], int(c[5]), c[7], c[10], c[11]
            key = (run, dev, x, y, risc, zone)
            if typ == "ZONE_START":
                open_starts[key].append(t)
                if zone == COMPUTE_ZONE and risc == "TRISC_0":
                    core = (run, dev, x, y)
                    prev = last_step_end.get(core)
                    if prev is not None and t - prev > 20 * freq_mhz:
                        events[core].append((prev, t, "gap"))
            elif typ == "ZONE_END" and open_starts[key]:
                t0 = open_starts[key].pop()
                total[key] += t - t0
                count[key] += 1
                if zone == COMPUTE_ZONE and risc == "TRISC_0":
                    last_step_end[(run, dev, x, y)] = t
                if zone in TIMELINE_ZONES and risc in ("NCRISC", "BRISC"):
                    events[(run, dev, x, y)].append((t0, t, zone))
                if zone in (COMPUTE_ZONE, "RD_RES_POOLED"):
                    s = zspan.setdefault(key, [t0, t])
                    s[0], s[1] = min(s[0], t0), max(s[1], t)
                if zone in KERNEL_ZONES:
                    s = span.setdefault((run, dev, x, y), [t0, t])
                    s[0], s[1] = min(s[0], t0), max(s[1], t)
    us = lambda cyc: cyc / freq_mhz
    report_step_sums(step_sums, us)

    # Run host IDs are per device: the attention op is each device's longest run.
    runs = collections.defaultdict(list)
    for (run, dev, x, y), (a, b) in span.items():
        runs[(dev, run)].append(us(b - a))
    print("frequency %.0f MHz" % freq_mhz)
    if not runs:
        print("no kernel zones found")
        return
    print("\nlongest program runs (device, run host ID: cores, longest kernel span us):")
    for (dev, run), v in sorted(runs.items(), key=lambda kv: -max(kv[1]))[:16]:
        print("  dev %3s run %8s: %5d cores, %9.1f us" % (dev, run, len(v), max(v)))
    main_runs = {}
    for (dev, run), v in runs.items():
        if dev not in main_runs or max(v) > max(runs[(dev, main_runs[dev])]):
            main_runs[dev] = run
    print("\nthe attention op is taken as each device's longest run")
    print(WINDOW_NOTE)
    if counters:
        # Busy fraction of each engine over the compute kernel (value / ref cycles), across the op's cores: how the
        # unpacker, FPU, SFPU and packer share the time (the dense kernel keeps them overlapped).
        per_counter = collections.defaultdict(list)
        for (run, dev, x, y), cs in counters.items():
            if main_runs.get(dev) != run:
                continue
            for name, (v, r) in cs.items():
                if r > 0:
                    per_counter[name].append(v / r)
        print("\nperf counters, busy fraction (value / ref cycles) across the op's cores:")
        for name in sorted(per_counter):
            vals = per_counter[name]
            print(
                "  %-30s n %5d  min %.3f  p10 %.3f  med %.3f  p90 %.3f  max %.3f"
                % (name, len(vals), min(vals), pct(vals, 0.1), pct(vals, 0.5), pct(vals, 0.9), max(vals))
            )

    cores = sorted(k for k in span if main_runs.get(k[1]) == k[0])
    per = {}
    for k in cores:
        main_run, dev, x, y = k
        busy_by_risc = {
            r: total.get((main_run, dev, x, y, r, COMPUTE_ZONE), 0) for r in ("TRISC_0", "TRISC_1", "TRISC_2")
        }
        busy = max(busy_by_risc.values())
        row = {"span": us(span[k][1] - span[k][0]), "busy": us(busy)}
        row["idle"] = row["span"] - row["busy"]
        for z in READER_ZONES + OTHER_ZONES:
            row[z] = us(
                max(
                    total.get((main_run, dev, x, y, r, z), 0)
                    for r in ("BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2")
                )
            )
        row["steps"] = max(count.get((main_run, dev, x, y, r, COMPUTE_ZONE), 0) for r in busy_by_risc)
        # Where the idle sits: before the first step (prologue), between steps (stalls), after the last (tail).
        risc = max(busy_by_risc, key=busy_by_risc.get)
        a, b = zspan.get((main_run, dev, x, y, risc, COMPUTE_ZONE), span[k])
        row["prologue"] = us(a - span[k][0])
        row["stalls"] = us(b - a - busy)
        row["tail"] = us(span[k][1] - b)
        p = zspan.get((main_run, dev, x, y, "NCRISC", "RD_RES_POOLED")) or zspan.get(
            (main_run, dev, x, y, "BRISC", "RD_RES_POOLED")
        )
        row["pooled_start"] = us(p[0] - span[k][0]) if p else 0.0
        per[k] = row

    compute_cores = [k for k in cores if per[k]["busy"] > 0]
    if not compute_cores:
        print("  %d cores with kernels, no RING_STEP zones (build without SPARSE_STEP_PROFILE?)" % len(cores))
        return
    print(
        "  %d cores with kernels, %d with compute steps, %d devices"
        % (len(cores), len(compute_cores), len({k[1] for k in cores}))
    )

    print("\nacross compute cores (us):")
    for name in (
        ["span", "busy", "idle", "prologue", "stalls", "tail", "pooled_start", "steps"] + READER_ZONES + OTHER_ZONES
    ):
        vals = [per[k][name] for k in compute_cores]
        if any(vals):
            print("  %-14s %s" % (name, stats(vals)))

    print("\nper device: span max, busy max / mean (imbalance), idle mean (us):")
    for dev in sorted({k[1] for k in compute_cores}, key=lambda d: int(d) if d.isdigit() else d):
        ks = [k for k in compute_cores if k[1] == dev]
        b = [per[k]["busy"] for k in ks]
        mean_b = sum(b) / len(b)
        print(
            "  dev %3s: span %8.1f  busy max %8.1f mean %8.1f (%.2f)  idle mean %8.1f"
            % (
                dev,
                max(per[k]["span"] for k in ks),
                max(b),
                mean_b,
                max(b) / mean_b if mean_b else 0,
                sum(per[k]["idle"] for k in ks) / len(ks),
            )
        )

    def show(ks, title):
        print(
            "\n"
            + title
            + " (dev, x, y: span, busy, steps, prologue / stalls / tail, group rx / tx, chain rx / tx, fetch):"
        )
        for k in ks:
            r = per[k]
            print(
                "  %3s %2s %2s: %8.1f %8.1f %5d  %7.1f %7.1f %7.1f  %8.1f %8.1f  %8.1f %8.1f  %8.1f"
                % (
                    k[1],
                    k[2],
                    k[3],
                    r["span"],
                    r["busy"],
                    r["steps"],
                    r["prologue"],
                    r["stalls"],
                    r["tail"],
                    r["RD_GROUP_RX"],
                    r["RD_GROUP_TX"],
                    r["RD_CHAIN_RX"],
                    r["RD_CHAIN_TX"],
                    r["RD_FETCH_K"],
                )
            )

    # Optional second argument: every compute core's row, for offline work-split studies.
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as out:
            out.write("dev x y span busy steps prologue stalls tail fetch group_rx group_tx\n")
            for k in compute_cores:
                r = per[k]
                out.write(
                    "%s %s %s %.1f %.1f %d %.1f %.1f %.1f %.1f %.1f %.1f\n"
                    % (
                        k[1],
                        k[2],
                        k[3],
                        r["span"],
                        r["busy"],
                        r["steps"],
                        r["prologue"],
                        r["stalls"],
                        r["tail"],
                        r["RD_FETCH_K"],
                        r["RD_GROUP_RX"],
                        r["RD_GROUP_TX"],
                    )
                )
    by_busy = sorted(compute_cores, key=lambda k: -per[k]["busy"])
    show(by_busy[:12], "busiest compute cores")
    show(by_busy[-12:], "least busy compute cores")
    # Timelines of the busiest cores: when each ring iteration's arrival wait ended, the piece streams, the pooled
    # phase, and every compute gap over 20 us (us from the core's kernel start).
    for k in by_busy[:3]:
        print(
            "\ntimeline, dev %s core (%s, %s), us from kernel start (gaps: compute idle between steps):"
            % (k[1], k[2], k[3])
        )
        for a, b, label in sorted(events.get(k, [])):
            print("  %8.1f - %8.1f  %7.1f  %s" % (us(a - span[k][0]), us(b - span[k][0]), us(b - a), label))
    producers = [k for k in compute_cores if per[k]["RD_GROUP_TX"] > 0]
    if producers:
        dev0 = producers[0][1]
        show(
            sorted((k for k in producers if k[1] == dev0), key=lambda k: (int(k[3]), int(k[2]))),
            "group producers, device " + dev0,
        )


def report_step_sums(step_sums, us):
    """Whole-op compute busy time per core from the sum zones: per device, min / median / max over cores of the
    largest TRISC total (the TRISC that is busiest bounds the core), and the five busiest cores."""
    if not step_sums:
        return
    per_core = collections.defaultdict(int)  # (run, dev, x, y) -> max over TRISCs
    for (run, dev, x, y, risc), cyc in step_sums.items():
        if risc.startswith("TRISC"):
            per_core[(run, dev, x, y)] = max(per_core[(run, dev, x, y)], cyc)
    # Per device keep the run with the largest total (the attention op).
    run_total = collections.defaultdict(int)
    for (run, dev, x, y), cyc in per_core.items():
        run_total[(run, dev)] += cyc
    main_run = {}
    for (run, dev), cyc in run_total.items():
        if cyc > run_total.get((main_run.get(dev), dev), -1):
            main_run[dev] = run
    print("\nwhole-op step time per core from sum zones (us; max over TRISCs; RING_STEP_SUM):")
    print("  %-6s %6s %9s %9s %9s" % ("device", "cores", "min", "median", "max"))
    busiest = []
    for dev in sorted(main_run, key=lambda d: int(d) if d.isdigit() else d):
        vals = sorted(cyc for (run, d, x, y), cyc in per_core.items() if d == dev and run == main_run[dev])
        busiest += [(cyc, dev, x, y) for (run, d, x, y), cyc in per_core.items() if d == dev and run == main_run[dev]]
        print("  %-6s %6d %9.1f %9.1f %9.1f" % (dev, len(vals), us(vals[0]), us(vals[len(vals) // 2]), us(vals[-1])))
    for cyc, dev, x, y in sorted(busiest, reverse=True)[:5]:
        print("  busiest: dev %s core (%s, %s) %.1f us" % (dev, x, y, us(cyc)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv")
