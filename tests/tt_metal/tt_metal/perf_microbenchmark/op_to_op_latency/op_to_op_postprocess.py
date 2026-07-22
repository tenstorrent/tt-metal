#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Consolidated post-processing for the op-to-op latency CI microbenchmark.

The test binary (test_op_to_op_latency) runs one pinned steady-state config to
completion and dumps two raw profiler CSVs:

  generated/profiler/.logs/profile_log_device.csv      device profiler (KERNEL zones, via process_device_log)
  generated/profiler/.logs/profile_log_device_rt.csv   realtime profiler (optional; only if the
                                                        binary ran with --use-realtime-profiler)

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

The device log is loaded through the official parser (tools/tracy/process_device_log.py),
not by reading the CSV columns directly: that module owns the on-disk schema and is
updated in lockstep with it, so consuming its structured output keeps this test robust
to CSV layout changes (profiler-team review). We normalize its output to our own stable
column names and do the op-to-op math on top of that.
"""

from __future__ import annotations

import argparse
import json
import os
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

# DeviceRecordEvent ids -> marker names, kept in sync with the EV_* constants in the compute
# kernel (kernels/compute_copy_with_nops.cpp). recordEvent is lighter than DeviceTimestampedData
# (event id only, no data payload = less L1 buffer pressure, so less op2op perturbation). The event
# id is the low 16 bits of the marker's timer id, and events carry no payload (data == 0); we
# backfill the name so the name-keyed metric walks below work unchanged.
EVENT_NAMES = {
    12: "TILE_IDX",  # lean-mode first-math (tile 0)
    13: "FINISH_LAST_PUSH",  # pack finish
}

# Program id is emitted as an event id EV_PROG_BASE + program_id (not a DeviceTimestampedData
# "PROG_ID", which TT_METAL_PROFILER_ACCUMULATE=1 compiles out). The pack_to_unpack metric needs
# PROG_ID to survive accumulate, so we recover it from the event id: an event in
# [EV_PROG_BASE, EV_PROG_MAX) maps to zone "PROG_ID" with data = id - EV_PROG_BASE. Keep
# EV_PROG_BASE in sync with the compute kernel's EV_PROG_BASE.
EV_PROG_BASE = 64
EV_PROG_MAX = EV_PROG_BASE + 4096


# --------------------------------------------------------------------------- #
# Device log loading via the official parser (owns the CSV schema for us)
# --------------------------------------------------------------------------- #
def _to_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return 0


def load_device_events(log_path: Path) -> tuple[pd.DataFrame, float]:
    """Load the device profiler log through tools/tracy/process_device_log.py.

    We deliberately do NOT read the CSV columns directly: that module owns the on-disk
    schema and is updated in lockstep with it, so consuming its structured output keeps
    us robust to CSV layout changes. Returns a normalized event frame with our own
    stable column names (chip, core_x, core_y, risc, zone, type, t, data,
    trace_id_count) plus the chip frequency in MHz.
    """
    try:
        from tracy.process_device_log import import_device_profile_log
    except ImportError as exc:  # pragma: no cover - env guard
        raise SystemExit(
            "Could not import tracy.process_device_log; run under the tt-metal profiler "
            f"environment (PYTHONPATH must include the tools/ dir). Original error: {exc}"
        )

    devices = import_device_profile_log(str(log_path))
    freq_mhz = float(devices["deviceInfo"]["freq"])

    rows = []
    for chip, ddata in devices["devices"].items():
        for core, cdata in ddata["cores"].items():
            core_x, core_y = core
            for risc, rdata in cdata["riscs"].items():
                for timer_id, t, data in rdata["timeseries"]:
                    zone = timer_id["zone_name"]
                    dval = _to_int(data)
                    # DeviceRecordEvent markers land as TS_EVENT rows with an empty zone name; recover
                    # the marker name from the event id (low 16 bits of the timer id). Named events
                    # (TILE_IDX / FINISH_LAST_PUSH) carry no payload (data 0); PROG_ID events encode
                    # the program in the id itself (data = id - EV_PROG_BASE).
                    if timer_id["type"] == "TS_EVENT":
                        ev = _to_int(timer_id.get("id", 0)) & 0xFFFF
                        if ev in EVENT_NAMES:
                            zone = EVENT_NAMES[ev]
                            dval = 0
                        elif EV_PROG_BASE <= ev < EV_PROG_MAX:
                            zone = "PROG_ID"
                            dval = ev - EV_PROG_BASE
                    rows.append(
                        {
                            "chip": chip,
                            "core_x": core_x,
                            "core_y": core_y,
                            "risc": risc,
                            "zone": zone,
                            "type": timer_id["type"],
                            "t": int(t),
                            "data": dval,
                            "trace_id_count": _to_int(timer_id.get("trace_id_count", -1)),
                        }
                    )
    return pd.DataFrame(rows), freq_mhz


def select_measured_trace_replay(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the measured trace replay.

    With trace warmup replays the device profiler accumulates every replay's
    markers into one log, tagged by trace_id_count (1..R, incrementing per replay);
    the final drain dumps them all at once. Warmup replays reuse the same PROG_IDs as
    the measured replay, so they can't be separated by program id -- but the measured
    replay is always the last one, i.e. the max trace_id_count. No-op for non-trace
    (FD back-to-back) runs, where the parser reports trace_id_count == -1.
    """
    if df.empty or "trace_id_count" not in df.columns:
        return df
    valid = df["trace_id_count"][df["trace_id_count"] >= 0]
    if valid.empty:
        return df
    return df[df["trace_id_count"] == valid.max()]


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
    z = df[df["zone"].isin(KERNEL_ZONES) & df["type"].isin(["ZONE_START", "ZONE_END"])]
    op2op, kdur = [], []
    cap = max_gap_us * freq_mhz
    for _key, g in z.groupby(["chip", "core_x", "core_y"], sort=False):
        evs = sorted(
            (
                int(r["t"]),
                r["type"] == "ZONE_START",
                r["zone"] in DM_KERNEL_ZONES,
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
        (df["zone"].isin(["PROG_ID", "TILE_IDX", "FINISH_LAST_PUSH"])) & (df["risc"].isin([UNPACK_RISC, PACK_RISC]))
    ].copy()
    markers = markers.sort_values(["chip", "core_x", "core_y", "risc", "t"])
    pack_finish: dict[tuple, int] = {}
    unpack_tile0: dict[tuple, int] = {}
    for key, group in markers.groupby(["chip", "core_x", "core_y", "risc"], sort=False):
        chip, cx, cy, risc = key
        cur_prog = None
        for _, row in group.iterrows():
            zone = row["zone"]
            t = int(row["t"])
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
    df, freq_mhz = load_device_events(device_csv)
    df = select_measured_trace_replay(df)

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
        "--rt-file",
        type=Path,
        default=None,
        help="optional realtime-profiler CSV (canonical path: $TT_METAL_HOME/"
        f"{DEFAULT_RT_LOG}). The CI flow is device-profiler-only (portable across WH/BH), so "
        "RT is off by default; pass this only when you ran the binary with "
        "--use-realtime-profiler, to also report the (ungated) rt_gap_to_next_go metric.",
    )
    ap.add_argument(
        "--min-prog-id",
        type=int,
        default=3,
        help="drop transitions from program ids below this (skip cold trace transitions)",
    )
    ap.add_argument("--out-json", type=Path, default=None, help="write metrics JSON to this path")
    ap.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="golden JSON to gate against. If its gated value is null the run is in "
        "record mode (prints the measured value and passes without gating).",
    )
    ap.add_argument(
        "--gate-metric",
        default=None,
        help="restrict gating to a single metric key. Default: gate every non-null metric in "
        "the golden's 'golden' block (e.g. official_op2op_us and pack_to_unpack_op2op_us).",
    )
    ap.add_argument(
        "--tolerance-pct",
        type=float,
        default=None,
        help="regression tolerance in percent (overrides golden's tolerance_pct)",
    )
    args = ap.parse_args()

    home = os.environ.get("TT_METAL_HOME", ".")
    device_csv = args.input_file or Path(home) / DEFAULT_DEVICE_LOG
    # RT is opt-in: only read it when explicitly pointed at a file, so a stale RT CSV from an
    # earlier run can never leak into a device-only (CI) run.
    rt_csv = args.rt_file

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

    if args.golden is not None:
        gate_metrics = [args.gate_metric] if args.gate_metric else None
        return gate_against_golden(metrics, args.golden, gate_metrics, args.tolerance_pct)

    return 0


def _fmt(v) -> str:
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def gate_against_golden(metrics: dict, golden_path: Path, gate_metrics, tolerance_override) -> int:
    """Compare measured metrics against a golden JSON, mirroring the runtime-perf
    compare_*.py convention. Gates every non-null value in the golden's 'golden' block
    (or only `gate_metrics` if provided). A metric whose golden value is null is in record
    mode: printed and skipped. Returns a process exit code:

      0  every populated gate passes (or all are null -> record mode)
      1  any gated metric regresses, or a gated measurement is missing/NaN
    """
    if not golden_path.exists():
        print(f"ERROR: golden file not found: {golden_path}", file=sys.stderr)
        return 1

    golden = json.loads(golden_path.read_text())
    golden_block = golden.get("golden", {})
    tol = tolerance_override if tolerance_override is not None else golden_block.get("tolerance_pct", 15.0)

    keys = list(gate_metrics) if gate_metrics else [k for k in golden_block if k != "tolerance_pct"]
    if not keys:
        print(f"ERROR: golden {golden_path.name} defines no gate metrics", file=sys.stderr)
        return 1

    failed = False
    for key in keys:
        golden_value = golden_block.get(key)
        measured = metrics.get(key)
        if golden_value is None:
            print(
                f"[record mode] golden '{key}' not populated in {golden_path.name}; "
                f"measured={_fmt(measured)}. Populate it to enable this gate. Passing."
            )
            continue
        if measured is None or measured != measured:  # None or NaN
            print(f"gate: FAIL — measured {key} is missing/NaN (no samples extracted)", file=sys.stderr)
            failed = True
            continue
        lo = golden_value * (1.0 - tol / 100.0)
        hi = golden_value * (1.0 + tol / 100.0)
        status = "PASS" if lo <= measured <= hi else "FAIL"
        line = (
            f"gate: {key} measured={measured:.4f} golden={golden_value:.4f} "
            f"allowed=[{lo:.4f}, {hi:.4f}] (+/-{tol}%) -> {status}"
        )
        if status == "PASS":
            print(line)
        else:
            print(line, file=sys.stderr)
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
