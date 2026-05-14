#!/usr/bin/env python3
"""Aggregate SDPA fw zone durations from profile_log_device.csv.

Each device zone has a ZONE_START and ZONE_END row sharing (core_x, core_y,
RISC, run_host_id, trace_id, zone_name). Pair them up, compute duration in
cycles, then print per-zone stats.

Note: zones are recorded on whichever TRISC reaches the macro. The compute
kernel zones get reported on all three TRISCs (UNPACK=TRISC_0, MATH=TRISC_1,
PACK=TRISC_2); the duration is the same on all three (kernel runs as one
program). We aggregate per RISC type.

Also: the profile tests run SDPA twice (warmup + timed). We keep all
instances and just rely on count/avg.

Usage:
    python3 analyze_sdpa_profile.py <report_dir> [<report_dir> ...]

Where each <report_dir> is a folder under generated/profiler/reports/<timestamp>/
that contains profile_log_device.csv.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path


def analyze_report(report_dir: Path, label: str) -> None:
    csv_path = report_dir / "profile_log_device.csv"
    if not csv_path.is_file():
        print(f"!! missing {csv_path}")
        return

    # header line is row 1: "ARCH: wormhole_b0, CHIP_FREQ[MHz]: 1000, ..."
    # column header is row 2
    with csv_path.open() as f:
        meta = f.readline()
        # extract freq
        freq_mhz = 1000.0
        for tok in meta.split(","):
            tok = tok.strip()
            if tok.startswith("CHIP_FREQ"):
                try:
                    freq_mhz = float(tok.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass
        # column header
        reader = csv.reader(f)
        headers = [h.strip() for h in next(reader)]
        col_core_x = headers.index("core_x")
        col_core_y = headers.index("core_y")
        col_risc = headers.index("RISC processor type")
        col_time = headers.index("time[cycles since reset]")
        col_run_id = headers.index("run host ID")
        col_trace_id = headers.index("trace id")
        col_trace_ctr = headers.index("trace id counter")
        col_zone = headers.index("zone name")
        col_type = headers.index("type")

        # key -> list of (timestamp, type)
        events = defaultdict(list)
        for row in reader:
            if len(row) <= col_type:
                continue
            zone = row[col_zone].strip()
            if not zone.startswith(("fw-", "sdpa-fw-")):
                continue
            key = (
                row[col_core_x].strip(),
                row[col_core_y].strip(),
                row[col_risc].strip(),
                row[col_run_id].strip(),
                row[col_trace_id].strip(),
                row[col_trace_ctr].strip(),
                zone,
            )
            events[key].append((int(row[col_time]), row[col_type].strip()))

    # Within each (key) group: events are interleaved START/END pairs (if the
    # zone appears N times in the loop, we get N starts then N ends, or
    # interleaved -- we sort and pair by order: 1st START with 1st END, etc.).
    durations = defaultdict(list)  # zone_name -> [(risc, duration_cycles), ...]
    for key, ev in events.items():
        zone = key[-1]
        risc = key[2]
        starts = sorted([t for t, kind in ev if kind == "ZONE_START"])
        ends = sorted([t for t, kind in ev if kind == "ZONE_END"])
        for s, e in zip(starts, ends):
            durations[zone].append((risc, e - s))

    print()
    print(f"========== {label} :: {report_dir.name} ==========")
    print(f"chip_freq = {freq_mhz} MHz   (1 cycle = {1000.0 / freq_mhz:.2f} ns)")
    print()

    # Group by zone, then by RISC type. Show: count, min, mean, max in ns.
    cyc_to_ns = 1000.0 / freq_mhz

    # Preferred display order. Older legacy zone names (`fw-softmax`,
    # `fw-sm-sub-exp`) are listed for backward compat with older reports.
    zone_order = [
        "sdpa-fw-compute",
        "fw-qk-mm",
        "fw-mask",
        "fw-softmax",  # legacy (pass 1)
        "fw-sm-max",
        "fw-sm-sub-exp",  # legacy (pass 2, fused sub+exp)
        "fw-sm-sub",
        "fw-sm-exp",
        "fw-sm-pack-scores",
        "fw-sm-pack-sum",
        "fw-pv-mm",
        "fw-online",
        "fw-final",
        "sdpa-fw-reader",
        "sdpa-fw-writer",
    ]
    seen = set(durations.keys())
    extra = sorted(seen - set(zone_order))

    fmt = "{:<20s}{:<10s}{:>7s}{:>14s}{:>14s}{:>14s}{:>14s}"
    print(fmt.format("zone", "RISC", "count", "min[ns]", "mean[ns]", "max[ns]", "sum[ns]"))
    print("-" * 95)

    for zone in zone_order + extra:
        if zone not in durations:
            continue
        risc_durations = defaultdict(list)
        for risc, dur in durations[zone]:
            risc_durations[risc].append(dur)
        # Pick "representative" RISC: MATH (TRISC_1) for compute zones,
        # BRISC for sdpa-fw-writer, NCRISC for sdpa-fw-reader, but show all.
        for risc in sorted(risc_durations.keys()):
            ds = risc_durations[risc]
            n = len(ds)
            mn = min(ds)
            mx = max(ds)
            sm = sum(ds)
            mean = sm / n
            print(
                fmt.format(
                    zone,
                    risc,
                    f"{n}",
                    f"{mn * cyc_to_ns:,.0f}",
                    f"{mean * cyc_to_ns:,.0f}",
                    f"{mx * cyc_to_ns:,.0f}",
                    f"{sm * cyc_to_ns:,.0f}",
                )
            )
        print()


def main():
    if len(sys.argv) <= 1:
        print(
            "usage: analyze_sdpa_profile.py <report_dir> [<report_dir> ...]\n"
            "  report_dir = generated/profiler/reports/<timestamp>/",
            file=sys.stderr,
        )
        sys.exit(1)
    for raw in sys.argv[1:]:
        path = Path(raw)
        analyze_report(path, path.name)


if __name__ == "__main__":
    main()
