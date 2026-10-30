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
# Per-stage zones of one compute step (SPARSE_STAGE in sparse_ring_streaming.hpp, windowed like RING_STEP).
STAGE_ZONES = [
    "ST_S1_QK",
    "ST_WAIT_QK",
    "ST_S2_MAX",
    "ST_S2B_TEST",
    "ST_S3_ALPHA",
    "ST_S4_RESCALE",
    "ST_S56_EXP_PV",
    "ST_S6B_ROWSUM",
    "ST_END_SYNC",
]
WANTED = set([COMPUTE_ZONE] + READER_ZONES + OTHER_ZONES + KERNEL_ZONES + STAGE_ZONES)
# Whole-op totals (TT_SPARSE_PROF_TOTALS=1): SPARSE_TOTALS TS_DATA records, data = index << 32 | value, the index into
# this list, which must equal sparse_prof::kTotalNames in sparse_prof_window.hpp (checked at startup); cycles for the
# timed names and plain counts for the N_* names.
TOTAL_NAMES = [
    "KERNEL",
    "RING_STEP_SUM",
    "RING_STEP",
    "ST_S1_QK",
    "ST_WAIT_QK",
    "ST_S2_MAX",
    "ST_S2B_TEST",
    "ST_S3_ALPHA",
    "ST_S4_RESCALE",
    "ST_S56_EXP_PV",
    "ST_S6B_ROWSUM",
    "ST_END_SYNC",
    "C_K_WAIT",
    "C_POP_KV",
    "C_PLAN_WAIT",
    "C_V_WAIT",
    "C_ITER",
    "RES_NORM",
    "N_STEPS",
    "N_SCORE_TILES",
    "N_RESCALE",
    "N_MEMBERS",
    "RD_SYNC",
    "RD_RES_Q",
    "RD_RES_BITS",
    "RD_RES_PLAN",
    "RD_RES_PIECES",
    "RD_GROUP_RX",
    "RD_GROUP_TX",
    "RD_FETCH_K",
    "RD_CHAIN_RX",
    "RD_CHAIN_TX",
    "RD_RES_POOLED",
    "WR_RES_OUT",
]


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


def check_total_names():
    """TOTAL_NAMES must equal the kernel's kTotalNames: the records carry an index into it."""
    import os

    hdr = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_prof_window.hpp",
    )
    try:
        text = open(hdr).read()
    except OSError:
        return
    table = text[text.index("kTotalNames[] = {") : text.index("};", text.index("kTotalNames[] = {"))]
    names = re.findall(r'"([A-Z0-9_]+)"', table)
    if names != TOTAL_NAMES:
        sys.exit("profile_summary.py TOTAL_NAMES differs from kTotalNames in sparse_prof_window.hpp")


def main(path):
    check_total_names()
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
    totals = {}  # (run, dev, x, y, risc, name) -> whole-op total (TT_SPARSE_PROF_TOTALS)
    unknown_ts = collections.Counter()
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
            if len(c) >= 12 and c[11].startswith("TS_DATA"):
                if c[10] == "SPARSE_TOTALS":  # data = index into TOTAL_NAMES << 32 | value
                    data = int(c[6])
                    idx, value = data >> 32, data & 0xFFFFFFFF
                    if idx < len(TOTAL_NAMES):
                        totals[(c[7], c[0], c[1], c[2], c[3], TOTAL_NAMES[idx])] = value
                    else:
                        unknown_ts["SPARSE_TOTALS index %d" % idx] += 1
                else:
                    unknown_ts[c[10]] += 1
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
    report_totals(totals, us)
    if unknown_ts:
        print("\nTS_DATA records with other names: %s" % dict(unknown_ts.most_common(8)))

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

    # Stage breakdown of the windowed steps: per thread (TRISC_0 unpack, TRISC_1 math, TRISC_2 pack), the median
    # over compute cores of each stage's time per step. A stage's time on a thread includes its waits on the others,
    # so the thread whose time matches the stage's length is the one that gates it.
    if any(total.get((k[0], k[1], k[2], k[3], "TRISC_1", z), 0) for k in compute_cores for z in STAGE_ZONES):
        print("\nstages per step, median over compute cores (us): unpack / math / pack, and share of the step on math")
        step_us = {}
        for k in compute_cores:
            n = max(count.get((k[0], k[1], k[2], k[3], "TRISC_1", COMPUTE_ZONE), 0), 1)
            step_us[k] = us(total.get((k[0], k[1], k[2], k[3], "TRISC_1", COMPUTE_ZONE), 0)) / n
        med = lambda v: sorted(v)[len(v) // 2]
        step_med = med([step_us[k] for k in compute_cores])
        for z in STAGE_ZONES:
            cols = []
            for r in ("TRISC_0", "TRISC_1", "TRISC_2"):
                v = []
                for k in compute_cores:
                    n = max(count.get((k[0], k[1], k[2], k[3], "TRISC_1", COMPUTE_ZONE), 0), 1)
                    v.append(us(total.get((k[0], k[1], k[2], k[3], r, z), 0)) / n)
                cols.append(med(v))
            share = cols[1] / step_med if step_med else 0.0
            print("  %-14s %8.2f %8.2f %8.2f   %5.1f%%" % (z, cols[0], cols[1], cols[2], 100 * share))
        print("  %-14s %8s %8.2f %8s" % ("RING_STEP", "", step_med, ""))

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


def report_totals(totals, us):
    """Whole-op totals per RISC (TT_SPARSE_PROF_TOTALS=1): every timed zone and wait summed over the whole op, the
    step / score-tile / rescale counts, and per-core derived numbers (time per step and per score tile, time outside
    steps). Per device the attention op is the run with the largest compute KERNEL total."""
    if not totals:
        return
    risc_order = ["TRISC_0", "TRISC_1", "TRISC_2", "NCRISC", "BRISC"]
    run_total = collections.defaultdict(int)
    for (run, dev, x, y, risc, name), v in totals.items():
        if name == "KERNEL" and risc.startswith("TRISC"):
            run_total[(run, dev)] += v
    main_run = {}
    for (run, dev), v in run_total.items():
        if v > run_total.get((main_run.get(dev), dev), -1):
            main_run[dev] = run
    core = collections.defaultdict(dict)  # (dev, x, y) -> {(risc, name): value}
    for (run, dev, x, y, risc, name), v in totals.items():
        if main_run.get(dev) == run:
            core[(dev, x, y)][(risc, name)] = v
    print(
        "\nwhole-op totals (TT_SPARSE_PROF_TOTALS), %d cores on %d devices; us (N_* are counts):"
        % (len(core), len(main_run))
    )
    print("  %-14s %-8s %5s %10s %10s %10s %10s" % ("name", "risc", "cores", "min", "median", "p90", "max"))
    for name in TOTAL_NAMES:
        for risc in risc_order:
            vals = [c[(risc, name)] for c in core.values() if (risc, name) in c]
            if not vals:
                continue
            f = (lambda v: float(v)) if name.startswith("N_") else us
            vals = [f(v) for v in vals]
            print(
                "  %-14s %-8s %5d %10.1f %10.1f %10.1f %10.1f"
                % (name, risc, len(vals), min(vals), pct(vals, 0.5), pct(vals, 0.9), max(vals))
            )
    # Per compute core: steps, tiles, time per step / per tile on each TRISC, and the time outside steps.
    rows = []
    for key, c in core.items():
        steps = c.get(("TRISC_1", "N_STEPS"), 0)
        tiles = c.get(("TRISC_1", "N_SCORE_TILES"), 0)
        if not steps:
            continue
        step_t = {r: c.get((r, "RING_STEP_SUM"), 0) for r in risc_order[:3]}
        kern = max(c.get((r, "KERNEL"), 0) for r in risc_order[:3])
        rows.append((key, steps, tiles, step_t, kern, c))
    if not rows:
        return
    print("\nper compute core (median / max over cores):")

    def line(label, vals, unit=""):
        print("  %-44s %10.3f %10.3f %s" % (label, pct(vals, 0.5), max(vals), unit))

    line("steps", [r[1] for r in rows])
    line("score tiles", [r[2] for r in rows])
    line("score tiles per step", [r[2] / r[1] for r in rows])
    line("rescale steps / steps", [r[5].get(("TRISC_1", "N_RESCALE"), 0) / r[1] for r in rows])
    line("members per step", [r[5].get(("TRISC_1", "N_MEMBERS"), 0) / r[1] for r in rows])
    for risc in risc_order[:3]:
        line("step time %s (ms)" % risc, [us(r[3][risc]) / 1e3 for r in rows])
        line("  per step %s (us)" % risc, [us(r[3][risc]) / r[1] for r in rows])
        line("  per score tile %s (us)" % risc, [us(r[3][risc]) / r[2] for r in rows])
    line("compute kernel (max TRISC, ms)", [us(r[4]) / 1e3 for r in rows])
    line("outside steps = kernel - max TRISC steps (ms)", [us(r[4] - max(r[3].values())) / 1e3 for r in rows])
    print("\nwhole-op stage split per thread (sum over cores of the stage / sum of RING_STEP_SUM):")
    stages = [n for n in TOTAL_NAMES if n.startswith("ST_")] + ["C_V_WAIT"]
    for risc in risc_order[:3]:
        denom = sum(r[3][risc] for r in rows) or 1
        print(
            "  %-8s %s"
            % (
                risc,
                "  ".join(
                    "%s %.1f%%"
                    % (n[3:] if n.startswith("ST_") else n, 100.0 * sum(r[5].get((risc, n), 0) for r in rows) / denom)
                    for n in stages
                ),
            )
        )
    # The slowest compute core's breakdown (its op-critical path), and the median core's.
    rows.sort(key=lambda r: r[4])
    for label, r in (("median", rows[len(rows) // 2]), ("slowest", rows[-1])):
        (dev, x, y), c = r[0], r[5]
        print("\n%s compute core: dev %s (%s, %s), %d steps, %d score tiles" % (label, dev, x, y, r[1], r[2]))
        for risc in risc_order:
            items = [(n, c[(risc, n)]) for n in TOTAL_NAMES if (risc, n) in c and not n.startswith("N_")]
            if items:
                print("  %-8s %s" % (risc, "  ".join("%s %.1f" % (n, us(v)) for n, v in items)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv")
