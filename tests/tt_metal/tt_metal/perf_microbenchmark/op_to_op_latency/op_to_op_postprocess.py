#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Consolidated post-processing for the op-to-op latency CI microbenchmark.

The test binary (test_op_to_op_latency) runs one pinned steady-state config to
completion and dumps two raw profiler CSVs:

  generated/profiler/.logs/profile_log_device.csv      device profiler (KERNEL zones)
  generated/profiler/.logs/profile_log_device_rt.csv   realtime profiler (per-program go/done)

This module turns those into a small set of scalar metrics for the CI gate. Two
kinds of op-to-op number are produced:

  official_op2op_us   (GATED) per-core, adjacent-op gap from the standard
                      auto-emitted {BRISC,NCRISC,TRISC}-KERNEL zones:
                      last KERNEL end(k) -> first DM-KERNEL start(k+1). This is
                      the canonical tools/tracy device number (matches
                      device_post_proc_config.py 'op2op' / process_device_log.py),
                      works on every platform, and needs no custom markers. It is
                      the metric we gate regressions on.

  rt_gap_to_next_go_ns (TRACKED) chip-dispatcher done->go gap from the realtime
                      profiler. Cleaner/absolute, but the RT profiler is only
                      active on some setups (not T3K/remote/ETH dispatch), so it
                      is recorded where available and NOT gated.

Also emitted for context: device_kernel_dur_us (per-op kernel span) and the
per-core pack-finish -> next-unpack-start op2op (the definition the research
benchmark used).

The metric extraction functions (official_kernel_metrics etc.) are copied from
the research benchmark's decompose_latency_bw.py / export_op_to_op_profiler_csv.py
so this file is self-contained.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

DEFAULT_DEVICE_LOG = "generated/profiler/.logs/profile_log_device.csv"
DEFAULT_RT_LOG = "generated/profiler/.logs/profile_log_device_rt.csv"

UNPACK_RISC = "TRISC_0"
PACK_RISC = "TRISC_2"
# Standard firmware-emitted whole-kernel envelope zones (no custom markers).
KERNEL_ZONES = ("BRISC-KERNEL", "NCRISC-KERNEL", "TRISC-KERNEL")
DM_KERNEL_ZONES = ("BRISC-KERNEL", "NCRISC-KERNEL")


# --------------------------------------------------------------------------- #
# CSV loading (device profiler log has a metadata first line with CHIP_FREQ)
# --------------------------------------------------------------------------- #
def parse_chip_freq_mhz(log_path: Path) -> float:
    with log_path.open() as f:
        header = f.readline()
    match = re.search(r"CHIP_FREQ\[MHz\]:\s*([0-9.]+)", header)
    if not match:
        raise ValueError(f"Could not parse CHIP_FREQ from first line of {log_path}")
    return float(match.group(1))


def load_device_csv(log_path: Path) -> pd.DataFrame:
    df = pd.read_csv(log_path, skiprows=1)
    df.columns = df.columns.str.strip()
    return df


def _median(xs) -> float:
    s = pd.Series([float(x) for x in xs if x is not None and x == x])
    return float(s.median()) if len(s) else float("nan")


def _minimum(xs) -> float:
    s = pd.Series([float(x) for x in xs if x is not None and x == x])
    return float(s.min()) if len(s) else float("nan")


# --------------------------------------------------------------------------- #
# GATED metric: official KERNEL-zone op2op (tools/tracy methodology)
# --------------------------------------------------------------------------- #
def official_kernel_metrics(df: pd.DataFrame, freq_mhz: float, max_gap_us: float = 50.0):
    """Reproduce the tools/tracy methodology from the standard auto-emitted KERNEL zones.

      op2op             : per core, adjacent ops -> first DM-KERNEL start(k+1) - last KERNEL end(k)
      device_kernel_dur : per op, last KERNEL end - first KERNEL start across riscs

    Segments ops by the start->end->start transition per core (no PROG_ID needed, so robust to
    trace capture/replay duplicate program ids). The max_gap_us cap drops trace-instance
    boundaries. Returns (op2op_us list, kernel_dur_us list).
    """
    z = df[df["zone name"].isin(KERNEL_ZONES) & df["type"].isin(["ZONE_START", "ZONE_END"])]
    op2op, kdur = [], []
    cap = max_gap_us * freq_mhz
    for _key, g in z.groupby(["PCIe slot", "core_x", "core_y"], sort=False):
        evs = sorted(
            (
                int(r["time[cycles since reset]"]),
                r["type"] == "ZONE_START",
                r["zone name"] in DM_KERNEL_ZONES,
            )
            for _, r in g.iterrows()
        )
        ops = []  # each: {start, dm, end}
        cur = None
        prev_was_end = False
        for t, is_start, is_dm in evs:
            if is_start:
                if cur is None or prev_was_end:  # first start after a run of ends = new op
                    if cur is not None:
                        ops.append(cur)
                    cur = {"start": t, "dm": t if is_dm else None, "end": None}
                elif is_dm and cur["dm"] is None:
                    cur["dm"] = t
                prev_was_end = False
            else:
                if cur is not None:
                    cur["end"] = t
                prev_was_end = True
        if cur is not None:
            ops.append(cur)
        for o in ops:
            if o["end"] is not None:
                kdur.append((o["end"] - o["start"]) / freq_mhz)
        for a, b in zip(ops, ops[1:]):
            if a["end"] is not None and b["dm"] is not None:
                gap = b["dm"] - a["end"]
                if 0 < gap < cap:
                    op2op.append(gap / freq_mhz)
    return op2op, kdur


# --------------------------------------------------------------------------- #
# Secondary device metric: per-core pack-finish -> next-unpack-start op2op
# (the definition the research benchmark used; kept for context/trending)
# --------------------------------------------------------------------------- #
def pack_to_unpack_op2op_us(df: pd.DataFrame, freq_mhz: float, min_prog_id: int):
    markers = df[
        (df["zone name"].isin(["PROG_ID", "TILE_IDX", "FINISH_LAST_PUSH"]))
        & (df["RISC processor type"].isin([UNPACK_RISC, PACK_RISC]))
    ].copy()
    markers = markers.sort_values(["PCIe slot", "core_x", "core_y", "RISC processor type", "time[cycles since reset]"])
    pack_finish: dict[tuple, int] = {}
    unpack_tile0: dict[tuple, int] = {}
    for key, group in markers.groupby(["PCIe slot", "core_x", "core_y", "RISC processor type"], sort=False):
        chip, cx, cy, risc = key
        cur_prog = None
        for _, row in group.iterrows():
            zone = row["zone name"]
            t = int(row["time[cycles since reset]"])
            data = int(row["data"])
            if zone == "PROG_ID":
                cur_prog = data
            elif zone == "TILE_IDX" and risc == UNPACK_RISC and data == 0 and cur_prog is not None:
                unpack_tile0[(chip, cx, cy, cur_prog)] = t
            elif zone == "FINISH_LAST_PUSH" and risc == PACK_RISC:
                if cur_prog is None:
                    cur_prog = data
                pack_finish[(chip, cx, cy, cur_prog)] = t

    gaps = []
    for (chip, cx, cy, prog), finish_t in pack_finish.items():
        if prog < min_prog_id:
            continue
        nxt = unpack_tile0.get((chip, cx, cy, prog + 1))
        if nxt is not None and nxt > finish_t:
            gaps.append((nxt - finish_t) / freq_mhz)
    return gaps


# --------------------------------------------------------------------------- #
# TRACKED metric: realtime profiler chip-dispatcher gap_to_next_go
# --------------------------------------------------------------------------- #
def rt_gap_to_next_go_ns(rt_path: Path, min_prog_id: int):
    """Parse profile_log_device_rt.csv (written by the test) and return the list of
    per-program gap_to_next_go values in ns. Empty if the file is missing/inactive."""
    if not rt_path.exists():
        return []
    df = pd.read_csv(rt_path)
    df.columns = df.columns.str.strip()
    gaps = []
    for _, row in df.iterrows():
        prog = int(row["program_id"])
        gap_cycles = float(row["gap_to_next_go_cycles"])
        freq_ghz = float(row["frequency_ghz"])  # cycles per ns
        if prog < min_prog_id or gap_cycles <= 0 or freq_ghz <= 0:
            continue
        gaps.append(gap_cycles / freq_ghz)
    return gaps


# --------------------------------------------------------------------------- #
# Top-level metric computation
# --------------------------------------------------------------------------- #
def compute_metrics(device_csv: Path, rt_csv: Path | None, min_prog_id: int) -> dict:
    freq_mhz = parse_chip_freq_mhz(device_csv)
    df = load_device_csv(device_csv)

    op2op_us, kdur_us = official_kernel_metrics(df, freq_mhz)
    pack_unpack_us = pack_to_unpack_op2op_us(df, freq_mhz, min_prog_id)
    rt_gaps_ns = rt_gap_to_next_go_ns(rt_csv, min_prog_id) if rt_csv is not None else []

    return {
        "chip_freq_mhz": freq_mhz,
        "min_prog_id": min_prog_id,
        # GATED metric
        "official_op2op_us": _median(op2op_us),
        "official_op2op_us_min": _minimum(op2op_us),
        "official_op2op_n": len(op2op_us),
        # context
        "device_kernel_dur_us": _median(kdur_us),
        "pack_to_unpack_op2op_us": _median(pack_unpack_us),
        "pack_to_unpack_op2op_n": len(pack_unpack_us),
        # TRACKED metric (may be empty if RT profiler inactive on this platform)
        "rt_gap_to_next_go_ns": _median(rt_gaps_ns),
        "rt_gap_to_next_go_n": len(rt_gaps_ns),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-process one op_to_op_latency run into CI metrics.")
    ap.add_argument(
        "--input-file", type=Path, default=None, help=f"device CSV (default $TT_METAL_HOME/{DEFAULT_DEVICE_LOG})"
    )
    ap.add_argument(
        "--rt-file", type=Path, default=None, help=f"realtime CSV (default $TT_METAL_HOME/{DEFAULT_RT_LOG})"
    )
    ap.add_argument(
        "--min-prog-id",
        type=int,
        default=3,
        help="drop transitions from program ids below this (skip cold trace transitions)",
    )
    ap.add_argument("--out-json", type=Path, default=None, help="write metrics JSON to this path")
    args = ap.parse_args()

    home = os.environ.get("TT_METAL_HOME", ".")
    device_csv = args.input_file or Path(home) / DEFAULT_DEVICE_LOG
    rt_csv = args.rt_file or Path(home) / DEFAULT_RT_LOG

    if not device_csv.exists():
        print(f"ERROR: device profiler CSV not found: {device_csv}", file=sys.stderr)
        return 1

    metrics = compute_metrics(device_csv, rt_csv, args.min_prog_id)

    width = max(len(k) for k in metrics)
    print("op_to_op_latency metrics:")
    for k, v in metrics.items():
        print(f"  {k:<{width}} : {v}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(metrics, indent=2))
        print(f"wrote {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
