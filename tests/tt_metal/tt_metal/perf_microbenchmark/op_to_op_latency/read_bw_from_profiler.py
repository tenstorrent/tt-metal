#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Compute reader DRAM read bandwidth from device-profiler kernel timestamps.

The buffer-tune `dram_pipeline_gbps` number is host wall-clock (enqueue + Finish),
so for short single-program runs it is dominated by ~60-125 us of fixed dispatch
overhead and badly understates real NoC bandwidth. This script instead uses the
on-device reader markers so we time *only the kernel's read phase*:

    READ_BEFORE_BARRIER   emitted right before the first noc_async_read_tile
    READ_LAST_BARRIER     emitted right after the final read barrier completes

Per core, per program:
    duration = READ_LAST_BARRIER - READ_BEFORE_BARRIER   (device cycles)
    bytes    = --pages-per-core * --tile-bytes
    BW       = bytes / duration

This span still pays the pipeline fill latency at the start and the drain latency
at the end (a fixed "tax" we accept); for a ~10 us read phase that tax is a small
fraction. Reads and writes contend on the NoC/DRAM, so for an isolated read number
run with --read-only; otherwise this reports read BW *under* that contention.

The whole-kernel span (NCRISC_GO -> NCRISC_DONE) is available via --span go_to_done.

Requires the run to have produced generated/profiler/.logs/profile_log_device.csv,
i.e. TT_METAL_DEVICE_PROFILER=1 in the environment and the test invoked with
--use-device-profiler.

Example:
  python3 tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/read_bw_from_profiler.py \\
    --pages-per-core 2048
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Reuse the canonical CSV header/freq parsing from the export script (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_op_to_op_profiler_csv import (  # noqa: E402
    DEFAULT_LOG,
    READ_RISC_TYPES,
    load_profiler_csv,
    parse_chip_freq_mhz,
)

# Default DRAM page / CB tile size: Float16_b 32x32 tile = 2048 bytes (matches the
# benchmark's kDataFormat). Override with --tile-bytes if the format changes.
DEFAULT_TILE_BYTES = 2048

START_ZONE = "READ_BEFORE_BARRIER"
END_ZONES = {
    "first_read_to_last_barrier": "READ_LAST_BARRIER",
    "go_to_done": "NCRISC_DONE",
}
START_ZONES = {
    "first_read_to_last_barrier": "READ_BEFORE_BARRIER",
    "go_to_done": "NCRISC_GO",
}


def walk_read_spans(df: pd.DataFrame, start_zone: str, end_zone: str, min_prog_id: int) -> pd.DataFrame:
    """Per (chip, core, prog_id) reader start/end cycle timestamps for the chosen span."""
    zones = {"PROG_ID", start_zone, end_zone}
    markers = df[(df["zone name"].isin(zones)) & (df["RISC processor type"].isin(READ_RISC_TYPES))].copy()
    markers = markers.sort_values(["PCIe slot", "core_x", "core_y", "RISC processor type", "time[cycles since reset]"])

    rows: list[dict] = []
    group_cols = ["PCIe slot", "core_x", "core_y", "RISC processor type"]
    for (chip, core_x, core_y, _risc), group in markers.groupby(group_cols, sort=False):
        current_prog: int | None = None
        start_cycles: int | None = None
        for _, row in group.iterrows():
            zone = row["zone name"]
            time_cycles = int(row["time[cycles since reset]"])
            if zone == "PROG_ID":
                current_prog = int(row["data"])
                start_cycles = None
            elif current_prog is None:
                continue
            elif zone == start_zone:
                start_cycles = time_cycles
            elif zone == end_zone and start_cycles is not None:
                if current_prog >= min_prog_id:
                    rows.append(
                        {
                            "chip": chip,
                            "core_x": core_x,
                            "core_y": core_y,
                            "prog_id": current_prog,
                            "start_cycles": start_cycles,
                            "end_cycles": time_cycles,
                            "duration_cycles": time_cycles - start_cycles,
                        }
                    )
                start_cycles = None
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reader DRAM read bandwidth from device profiler markers.")
    parser.add_argument("--input-file", type=Path, default=None, help=f"default: $TT_METAL_HOME/{DEFAULT_LOG}")
    parser.add_argument("--output-csv", type=Path, default=None, help="write per-(prog,core) rows to this CSV")
    parser.add_argument(
        "--pages-per-core",
        type=int,
        required=True,
        help="logical tiles read per core per program (= benchmark --num-pages-per-core / "
        "--buffer-tune-pages-per-core)",
    )
    parser.add_argument("--tile-bytes", type=int, default=DEFAULT_TILE_BYTES, help="bytes per tile (default 2048)")
    parser.add_argument(
        "--span",
        choices=sorted(END_ZONES.keys()),
        default="first_read_to_last_barrier",
        help="first_read_to_last_barrier (default; read phase) or go_to_done (whole kernel)",
    )
    parser.add_argument(
        "--min-prog-id", type=int, default=1, help="skip programs below this id (default 1 skips warmup PROG_ID=0)"
    )
    args = parser.parse_args()

    log_path = (args.input_file or Path(os.environ.get("TT_METAL_HOME", ".")) / DEFAULT_LOG).resolve()
    if not log_path.is_file():
        print(f"Input not found: {log_path}", file=sys.stderr)
        return 1

    freq_mhz = parse_chip_freq_mhz(log_path)
    df = load_profiler_csv(log_path)

    start_zone = START_ZONES[args.span]
    end_zone = END_ZONES[args.span]
    spans = walk_read_spans(df, start_zone, end_zone, args.min_prog_id)
    if spans.empty:
        print(
            f"No reader spans found ({start_zone} -> {end_zone}). "
            "Re-run the test with --use-device-profiler (and TT_METAL_DEVICE_PROFILER=1), "
            "and ensure the kernel emits READ_LAST_BARRIER (reader_mode 2).",
            file=sys.stderr,
        )
        return 1

    bytes_per_core = args.pages_per_core * args.tile_bytes
    # 1 byte / ns == 1 GB/s; duration_ns = cycles / freq_ghz; GB/s = bytes / duration_ns.
    freq_ghz = freq_mhz / 1000.0
    spans["duration_us"] = spans["duration_cycles"] / freq_mhz
    spans["per_core_gbps"] = bytes_per_core / (spans["duration_cycles"] / freq_ghz)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        spans.to_csv(args.output_csv, index=False)
        print(f"Wrote {args.output_csv} ({len(spans)} per-(prog,core) rows)")

    num_cores = spans.groupby(["chip", "core_x", "core_y"]).ngroups
    progs = sorted(spans["prog_id"].unique())
    pc = spans["per_core_gbps"]
    dur = spans["duration_us"]

    print(f"Chip frequency: {freq_mhz:.1f} MHz")
    print(f"Span: {start_zone} -> {end_zone}")
    print(f"Programs (>= id {args.min_prog_id}): {progs}   active reader cores: {num_cores}")
    print(f"Bytes/core/program: {bytes_per_core} ({args.pages_per_core} tiles x {args.tile_bytes} B)")
    print()
    print("Per-core read BW (one value per core per program):")
    print(
        f"  per_core_gbps: median={pc.median():.2f}  mean={pc.mean():.2f}  "
        f"min={pc.min():.2f}  max={pc.max():.2f}  n={len(pc)}"
    )
    print(f"  read-phase duration_us: median={dur.median():.2f}  min={dur.min():.2f}  max={dur.max():.2f}")
    print()

    # Aggregate device read BW: total bytes moved by all cores divided by the wall span
    # from the first core starting to the last core finishing (captures cross-core skew;
    # reads contend, so this is the realized shared BW). A single program id can appear
    # in several execution *instances* (trace capture + timed replay run it ~ms apart),
    # so split each prog id into instances by start-time gaps before aggregating.
    gap_threshold = max(5.0 * float(spans["duration_cycles"].median()), 1.0)

    spans = spans.sort_values(["prog_id", "start_cycles"]).reset_index(drop=True)
    instance_ids: list[int] = []
    inst = 0
    prev_prog = None
    prev_start = None
    for _, row in spans.iterrows():
        prog = int(row["prog_id"])
        start = int(row["start_cycles"])
        if prog != prev_prog:
            inst = 0
        elif prev_start is not None and (start - prev_start) > gap_threshold:
            inst += 1
        instance_ids.append(inst)
        prev_prog, prev_start = prog, start
    spans["instance"] = instance_ids

    print("Aggregate device read BW per (program, execution instance) — all cores, union span:")
    agg_gbps_vals = []
    for (prog, inst), g in spans.groupby(["prog_id", "instance"]):
        cores = g.groupby(["chip", "core_x", "core_y"]).ngroups
        union_cycles = int(g["end_cycles"].max() - g["start_cycles"].min())
        union_us = union_cycles / freq_mhz
        total_bytes = cores * bytes_per_core
        agg_gbps = total_bytes / (union_cycles / freq_ghz) if union_cycles > 0 else 0.0
        agg_gbps_vals.append(agg_gbps)
        print(
            f"  prog {int(prog)} inst {int(inst)}: cores={cores}  union_span={union_us:.2f} us  "
            f"aggregate={agg_gbps:.1f} GB/s"
        )

    if len(agg_gbps_vals) > 1:
        print(f"  aggregate across instances: median={pd.Series(agg_gbps_vals).median():.1f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
