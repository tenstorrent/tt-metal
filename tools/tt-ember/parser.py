#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt


RE_TS = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)")
RE_DEV = re.compile(r"\b(?:Device|Chip)\s+ID\s+(?P<dev>\d+)\b")
# Newer tt-umd telemetry reports TDP as "<value>/<limit> W" (e.g. "TDP 16/150 W"); the limit is
# a static budget, not a measurement, so it is matched but discarded. TDC/VCORE remain bare values.
RE_METRIC = re.compile(
    r"\b(?P<key>TDP|TDC|VCORE)\s+(?P<val>\d+(?:\.\d+)?)(?:\s*/\s*\d+(?:\.\d+)?)?\s*(?P<unit>W|A|mV)\b"
)

RE_PROGRAM_ROW = re.compile(
    r"^\s*"
    r"(?P<grid>\d+x\d+)\s+"
    r"(?P<cores>\d+)\s+"
    r"(?:(?P<tiles_per_core>\d+(?:\s*/\s*\d+)?)\s+)?"
    r"(?P<algo_time_s>\d+(?:\.\d+)?)\s+"
    r"(?P<tflops>\d+(?:\.\d+)?)\s+"
    r"(?P<per_iter_ms>\d+(?:\.\d+)?)\s+"
    r"(?P<start>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+"
    r"(?P<end>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)"
    r"\s*$"
)

# Matches the device-side profiler "zone name" markers emitted by DeviceTimestampedData()
# in the reader/compute/writer kernels (see high_power_matmul), e.g. "READER_KERNEL_START".
# The initial version of this analysis only looks at whole-kernel start/end, not the
# per-iteration TRANSFER_START/END or SHMEM_READ/WRITE markers.
RE_KERNEL_ZONE = re.compile(r"^(?P<kernel>READER|COMPUTE|WRITER)_KERNEL_(?P<edge>START|END)$")

# Matches the aggregate wait-vs-active cycle counters emitted once per kernel invocation (see
# high_power_matmul's reader/compute/writer kernels), e.g. "READER_WAIT_CYCLES",
# "READER_TRANSFER_CYCLES", "COMPUTE_WAIT_CYCLES", "COMPUTE_COMPUTE_CYCLES". WAIT = blocked on
# the circular buffer; TRANSFER/COMPUTE (grouped as "active" below) = actual NoC transfer or
# matmul/pack work. There's no on-device OVERHEAD marker (see compute_kernel_utilization_rows
# for why) -- it's derived on the host instead, as lifetime_s - wait_s - active_s.
RE_KERNEL_CYCLES = re.compile(r"^(?P<kernel>READER|COMPUTE|WRITER)_(?P<metric>WAIT|TRANSFER|COMPUTE)_CYCLES$")


@dataclass(frozen=True)
class Sample:
    ts: datetime
    device_id: int
    vcore_mv: float
    tdc_a: float
    tdp_w: float

    @property
    def vcore_v(self) -> float:
        return self.vcore_mv / 1000.0


@dataclass(frozen=True)
class ProgramInterval:
    grid: str
    cores: int
    tiles_per_core: Optional[str]
    algo_time_s: float
    tflops: float
    per_iter_ms: float
    start_time: datetime
    end_time: datetime


@dataclass(frozen=True)
class DeviceMarker:
    core_x: int
    core_y: int
    risc: str
    cycle: int
    zone_name: str
    data: int = 0


def validate_subdir_name(subdir: str) -> str:
    """
    Validate output subdirectory name.

    Only allow simple directory names made of letters, numbers, '_', '-', and '.'.
    """
    if not subdir:
        raise argparse.ArgumentTypeError("Subdirectory name must not be empty.")

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", subdir):
        raise argparse.ArgumentTypeError(
            "Invalid --subdir value. Allowed characters: letters, numbers, '_', '-', '.'."
        )

    if subdir in {".", ".."}:
        raise argparse.ArgumentTypeError("Invalid --subdir value.")

    return subdir


def parse_line(line: str) -> Optional[Sample]:
    m_ts = RE_TS.search(line)
    m_dev = RE_DEV.search(line)
    if not m_ts or not m_dev:
        return None

    ts = datetime.fromisoformat(m_ts.group("ts"))
    dev = int(m_dev.group("dev"))

    metrics: Dict[str, Tuple[float, str]] = {}
    for m in RE_METRIC.finditer(line):
        metrics[m.group("key")] = (float(m.group("val")), m.group("unit"))

    if "TDP" not in metrics or "TDC" not in metrics or "VCORE" not in metrics:
        return None

    tdp, u1 = metrics["TDP"]
    tdc, u2 = metrics["TDC"]
    vcore, u3 = metrics["VCORE"]

    if (u1, u2, u3) != ("W", "A", "mV"):
        return None

    return Sample(ts=ts, device_id=dev, vcore_mv=vcore, tdc_a=tdc, tdp_w=tdp)


def slot_time(ts: datetime, t0: datetime, slot_ms: int) -> datetime:
    slot_s = slot_ms / 1000.0
    dt = (ts - t0).total_seconds()
    k = int(dt // slot_s)
    return t0 + timedelta(seconds=k * slot_s)


def parse_file_binned(
    path: Path,
    slot_ms: int,
    device_id_filter: Optional[int] = None,
) -> Tuple[Dict[datetime, Dict[int, Sample]], Set[int], Dict[str, int], List[Sample]]:
    raw_samples: List[Sample] = []
    stats = {
        "lines_total": 0,
        "lines_matched": 0,
        "lines_unmatched": 0,
        "lines_filtered_out": 0,
    }

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stats["lines_total"] += 1
            line = line.strip()
            if not line:
                continue

            s = parse_line(line)
            if s is None:
                stats["lines_unmatched"] += 1
                continue

            stats["lines_matched"] += 1

            if device_id_filter is not None and s.device_id != device_id_filter:
                stats["lines_filtered_out"] += 1
                continue

            raw_samples.append(s)

    if not raw_samples:
        return {}, set(), stats, []

    raw_samples.sort(key=lambda x: x.ts)
    t0 = raw_samples[0].ts

    slots: Dict[datetime, Dict[int, Sample]] = defaultdict(dict)
    devices: Set[int] = set()

    for s in raw_samples:
        st = slot_time(s.ts, t0, slot_ms)
        slots[st][s.device_id] = s
        devices.add(s.device_id)

    return slots, devices, stats, raw_samples


def build_timeseries(
    slots: Dict[datetime, Dict[int, Sample]],
    devices: Set[int],
) -> Tuple[np.ndarray, List[int], Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    ts_sorted = sorted(slots.keys())
    if not ts_sorted:
        return np.array([]), [], {}, {}, {}

    t0 = ts_sorted[0]
    t = np.array([(ts - t0).total_seconds() for ts in ts_sorted], dtype=float)

    dev_ids = sorted(devices)
    v_v: Dict[int, np.ndarray] = {}
    i_a: Dict[int, np.ndarray] = {}
    p_w: Dict[int, np.ndarray] = {}

    for d in dev_ids:
        vv = np.full(len(ts_sorted), np.nan, dtype=float)
        ia = np.full(len(ts_sorted), np.nan, dtype=float)
        pw = np.full(len(ts_sorted), np.nan, dtype=float)

        for idx, ts in enumerate(ts_sorted):
            s = slots[ts].get(d)
            if s is None:
                continue
            vv[idx] = s.vcore_v
            ia[idx] = s.tdc_a
            pw[idx] = s.tdp_w

        v_v[d] = vv
        i_a[d] = ia
        p_w[d] = pw

    return t, dev_ids, v_v, i_a, p_w


def plot_png(
    t: np.ndarray,
    dev_ids: List[int],
    v_v: Dict[int, np.ndarray],
    i_a: Dict[int, np.ndarray],
    p_w: Dict[int, np.ndarray],
    out_png: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 9))
    ax_v, ax_i, ax_p = axes

    for d in dev_ids:
        ax_v.plot(t, v_v[d], label=f"dev{d}")
        ax_i.plot(t, i_a[d], label=f"dev{d}")
        ax_p.plot(t, p_w[d], label=f"dev{d}")

    ax_v.set_ylabel("VCORE [V]")
    ax_i.set_ylabel("TDC [A]")
    ax_p.set_ylabel("TDP [W]")
    ax_p.set_xlabel("Time [s] (binned slots)")

    for ax in axes:
        ax.grid(True)
        ax.legend(ncols=min(4, max(1, len(dev_ids))), fontsize=9)

    fig.suptitle("UMD Telemetry: Voltage / Current / Power per Device")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def parse_program_log(path: Path) -> List[ProgramInterval]:
    rows: List[ProgramInterval] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = RE_PROGRAM_ROW.match(line)
            if not m:
                continue

            rows.append(
                ProgramInterval(
                    grid=m.group("grid"),
                    cores=int(m.group("cores")),
                    tiles_per_core=m.group("tiles_per_core"),
                    algo_time_s=float(m.group("algo_time_s")),
                    tflops=float(m.group("tflops")),
                    per_iter_ms=float(m.group("per_iter_ms")),
                    start_time=datetime.fromisoformat(m.group("start")),
                    end_time=datetime.fromisoformat(m.group("end")),
                )
            )

    rows.sort(key=lambda r: r.start_time)
    return rows


def load_device_kernel_markers(path: Path) -> List[DeviceMarker]:
    """
    Load READER/COMPUTE/WRITER *_KERNEL_START / *_KERNEL_END rows plus the aggregate
    *_WAIT_CYCLES / *_TRANSFER_CYCLES / *_COMPUTE_CYCLES rows from a tt-metal device profiler
    CSV (generated/profiler/.logs/profile_log_device.csv). Everything else (firmware/dispatch
    zones) is ignored.
    """
    markers: List[DeviceMarker] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        f.readline()  # skip the "ARCH: ..., CHIP_FREQ[MHz]: ..." line before the CSV header
        # The header row has ", " (comma-space) separators, e.g. "PCIe slot, core_x, ...".
        # Without skipinitialspace, DictReader keys would be " core_x", " zone name", etc.
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            zone = (row.get("zone name") or "").strip()
            if not (RE_KERNEL_ZONE.match(zone) or RE_KERNEL_CYCLES.match(zone)):
                continue

            try:
                cycle = int(row["time[cycles since reset]"])
                core_x = int(row["core_x"])
                core_y = int(row["core_y"])
                data = int(row["data"]) if row.get("data") not in (None, "") else 0
            except (KeyError, ValueError):
                continue

            markers.append(
                DeviceMarker(
                    core_x=core_x,
                    core_y=core_y,
                    risc=row.get("RISC processor type", ""),
                    cycle=cycle,
                    zone_name=zone,
                    data=data,
                )
            )

    markers.sort(key=lambda m: m.cycle)
    return markers


def compute_kernel_intervals(
    markers: List[DeviceMarker],
    intervals: List[ProgramInterval],
) -> List[Dict[str, object]]:
    """
    Convert device-cycle KERNEL_START/KERNEL_END markers into absolute host wall-clock
    start/end times, one row per kernel (reader/compute/writer) per grid run.

    Each core emits its own marker independently: reader/writer once per core, compute once
    per TRISC (x3, since TRISC0/1/2 all execute the same compute kernel source). So a run on
    N cores has N READER_KERNEL_START, N WRITER_KERNEL_START and 3*N COMPUTE_KERNEL_START
    markers -- not a fixed count. Runs execute strictly sequentially on one device, so sorting
    each zone's markers by cycle and slicing off `cores * multiplier` at a time (using each
    run's own core count from the program log) recovers the per-run boundaries without an
    explicit run id in the CSV.

    Anchors each run to that run's Start/End Time from the program log (RE_PROGRAM_ROW), using
    the run's global min(*_KERNEL_START cycle) / max(*_KERNEL_END cycle) across all three
    kernels as the two anchor points -- NOT each kernel's own min/max, which would trivially
    reproduce the run's start/end time and hide real inter-kernel timing differences (e.g.
    writer starting later than reader because it waits on the first compute result).
    """
    PER_CORE_MULTIPLIER = {"READER": 1, "COMPUTE": 3, "WRITER": 1}

    by_kernel_edge: Dict[tuple, List[DeviceMarker]] = defaultdict(list)
    for m in markers:
        match = RE_KERNEL_ZONE.match(m.zone_name)
        if match:
            by_kernel_edge[(match.group("kernel"), match.group("edge"))].append(m)
    for marker_list in by_kernel_edge.values():
        marker_list.sort(key=lambda m: m.cycle)

    cursor = {key: 0 for key in by_kernel_edge}
    results: List[Dict[str, object]] = []

    for run in intervals:
        run_slices: Dict[str, tuple] = {}
        for kernel_name, multiplier in PER_CORE_MULTIPLIER.items():
            count = run.cores * multiplier
            starts = by_kernel_edge.get((kernel_name, "START"), [])
            ends = by_kernel_edge.get((kernel_name, "END"), [])
            s0 = cursor.get((kernel_name, "START"), 0)
            e0 = cursor.get((kernel_name, "END"), 0)
            run_slices[kernel_name] = (starts[s0 : s0 + count], ends[e0 : e0 + count])
            cursor[(kernel_name, "START")] = s0 + count
            cursor[(kernel_name, "END")] = e0 + count

        all_starts = [m for s, _ in run_slices.values() for m in s]
        all_ends = [m for _, e in run_slices.values() for m in e]
        if not all_starts or not all_ends:
            continue

        anchor_cycle_a = min(m.cycle for m in all_starts)
        anchor_cycle_b = max(m.cycle for m in all_ends)
        duration_s = (run.end_time - run.start_time).total_seconds()

        if anchor_cycle_b <= anchor_cycle_a or duration_s <= 0:
            continue

        freq_hz = (anchor_cycle_b - anchor_cycle_a) / duration_s

        def to_host_time(cycle: int, _start=run.start_time, _a=anchor_cycle_a, _f=freq_hz) -> datetime:
            return _start + timedelta(seconds=(cycle - _a) / _f)

        for kernel_name, (kernel_starts, kernel_ends) in run_slices.items():
            if not kernel_starts or not kernel_ends:
                continue

            k_start = to_host_time(min(m.cycle for m in kernel_starts))
            k_end = to_host_time(max(m.cycle for m in kernel_ends))

            results.append(
                {
                    "grid": run.grid,
                    "cores": run.cores,
                    "kernel": kernel_name,
                    "start_time": k_start,
                    "end_time": k_end,
                    "duration_s": (k_end - k_start).total_seconds(),
                }
            )

    return results


def print_kernel_interval_report(kernel_intervals: List[Dict[str, object]]) -> None:
    print("\nKERNEL START/END INTERVAL ANALYSIS (device profiler)", file=sys.stderr)
    print("=" * 130, file=sys.stderr)
    print(
        f"{'Grid':>6} {'Kernel':>8} {'Duration[s]':>12} {'Start':>26} {'End':>26}",
        file=sys.stderr,
    )
    print("=" * 130, file=sys.stderr)

    for r in kernel_intervals:
        print(
            f"{str(r['grid']):>6} "
            f"{str(r['kernel']):>8} "
            f"{format_num(r['duration_s'], '.6f'):>12} "
            f"{str(r['start_time']):>26} "
            f"{str(r['end_time']):>26}",
            file=sys.stderr,
        )


def save_kernel_intervals_csv(kernel_intervals: List[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["grid", "cores", "kernel", "start_time", "end_time", "duration_s"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in kernel_intervals:
            out = dict(row)
            out["start_time"] = str(out["start_time"])
            out["end_time"] = str(out["end_time"])
            writer.writerow(out)


_KERNEL_CORE_MULTIPLIER = {"READER": 1, "COMPUTE": 3, "WRITER": 1}


def compute_run_windows(
    markers: List[DeviceMarker],
    intervals: List[ProgramInterval],
) -> List[Optional[Tuple[int, int, float]]]:
    """
    For each run (in the same order as `intervals`), return (anchor_cycle_a, anchor_cycle_b,
    freq_hz) -- the run's overall device-cycle window and device-cycles-per-host-second rate,
    anchored to that run's Start/End Time. Anchors use the GLOBAL min(*_KERNEL_START cycle) /
    max(*_KERNEL_END cycle) across all three kernels (see compute_kernel_intervals for why: a
    kernel's own min/max would trivially reproduce the run's start/end time).

    Returns None for a run whose markers couldn't be resolved (e.g. missing/misaligned data).
    """
    by_kernel_edge: Dict[Tuple[str, str], List[DeviceMarker]] = defaultdict(list)
    for m in markers:
        match = RE_KERNEL_ZONE.match(m.zone_name)
        if match:
            by_kernel_edge[(match.group("kernel"), match.group("edge"))].append(m)
    for marker_list in by_kernel_edge.values():
        marker_list.sort(key=lambda m: m.cycle)

    cursor = {key: 0 for key in by_kernel_edge}
    windows: List[Optional[Tuple[int, int, float]]] = []

    for run in intervals:
        all_starts: List[DeviceMarker] = []
        all_ends: List[DeviceMarker] = []
        for kernel_name, multiplier in _KERNEL_CORE_MULTIPLIER.items():
            count = run.cores * multiplier
            starts = by_kernel_edge.get((kernel_name, "START"), [])
            ends = by_kernel_edge.get((kernel_name, "END"), [])
            s0 = cursor.get((kernel_name, "START"), 0)
            e0 = cursor.get((kernel_name, "END"), 0)
            all_starts.extend(starts[s0 : s0 + count])
            all_ends.extend(ends[e0 : e0 + count])
            cursor[(kernel_name, "START")] = s0 + count
            cursor[(kernel_name, "END")] = e0 + count

        if not all_starts or not all_ends:
            windows.append(None)
            continue

        anchor_cycle_a = min(m.cycle for m in all_starts)
        anchor_cycle_b = max(m.cycle for m in all_ends)
        duration_s = (run.end_time - run.start_time).total_seconds()

        if anchor_cycle_b <= anchor_cycle_a or duration_s <= 0:
            windows.append(None)
            continue

        freq_hz = (anchor_cycle_b - anchor_cycle_a) / duration_s
        windows.append((anchor_cycle_a, anchor_cycle_b, freq_hz))

    return windows


def find_run_index(cycle: int, windows: List[Optional[Tuple[int, int, float]]]) -> Optional[int]:
    for i, w in enumerate(windows):
        if w is None:
            continue
        anchor_a, anchor_b, _ = w
        if anchor_a <= cycle <= anchor_b:
            return i
    return None


def compute_kernel_utilization_rows(
    markers: List[DeviceMarker],
    intervals: List[ProgramInterval],
) -> List[Dict[str, object]]:
    """
    Build one row per (grid run, kernel, core, RISC) with that core's own kernel lifetime
    (start/end, converted to host wall-clock time) and its wait/active/overhead breakdown:
      wait_s / wait_pct       -- blocked on the circular buffer (see kernel READMEs)
      active_s / active_pct   -- actual NoC transfer (reader/writer) or matmul/pack (compute)
      overhead_s / overhead_pct -- everything else (index arithmetic, tile-register handshake,
                                   CB bookkeeping)

    overhead_s is *derived*, not measured on-device: lifetime_s - wait_s - active_s. An earlier
    version measured it directly with its own wall-clock reads, but those spans are bounded
    only by two read_wall_clock_cycles() calls with no real hardware operation (and thus no
    genuine barrier) in between -- on-device timing artifacts (compiler scheduling,
    cycle-counter granularity) made it read far too small even after adding explicit compiler
    barriers. Deriving it by subtraction is exact by construction, since lifetime/wait/active
    are all independently measured across real hardware operations (cb_reserve_back/
    cb_wait_front, NoC read/write + barrier, matmul_tiles/pack_tile).

    Each marker already carries its own core_x/core_y/risc identity, so cores are matched by
    that identity directly (no positional/coordinate assumptions). Which RUN a given marker
    belongs to is resolved via cycle-window membership (compute_run_windows/find_run_index)
    rather than core-coordinate-range guessing, since core_x/core_y in the CSV are physical
    NOC coordinates, not necessarily the logical 0-based grid coordinates used to launch the
    kernels.
    """
    windows = compute_run_windows(markers, intervals)

    starts_by_core: Dict[Tuple[str, int, int, str], List[DeviceMarker]] = defaultdict(list)
    ends_by_core: Dict[Tuple[str, int, int, str], List[DeviceMarker]] = defaultdict(list)
    wait_by_core: Dict[Tuple[str, int, int, str], List[DeviceMarker]] = defaultdict(list)
    active_by_core: Dict[Tuple[str, int, int, str], List[DeviceMarker]] = defaultdict(list)

    for m in markers:
        zmatch = RE_KERNEL_ZONE.match(m.zone_name)
        if zmatch:
            key = (zmatch.group("kernel"), m.core_x, m.core_y, m.risc)
            if zmatch.group("edge") == "START":
                starts_by_core[key].append(m)
            else:
                ends_by_core[key].append(m)
            continue

        cmatch = RE_KERNEL_CYCLES.match(m.zone_name)
        if cmatch:
            key = (cmatch.group("kernel"), m.core_x, m.core_y, m.risc)
            if cmatch.group("metric") == "WAIT":
                wait_by_core[key].append(m)
            else:
                active_by_core[key].append(m)

    all_keys = set(starts_by_core) | set(ends_by_core) | set(wait_by_core) | set(active_by_core)

    rows: List[Dict[str, object]] = []

    for key in all_keys:
        kernel_name, core_x, core_y, risc = key

        per_run: Dict[int, Dict[str, DeviceMarker]] = defaultdict(dict)
        for slot, marker_list in (
            ("start", starts_by_core.get(key, [])),
            ("end", ends_by_core.get(key, [])),
            ("wait", wait_by_core.get(key, [])),
            ("active", active_by_core.get(key, [])),
        ):
            for m in marker_list:
                ridx = find_run_index(m.cycle, windows)
                if ridx is not None:
                    per_run[ridx][slot] = m

        for ridx, parts in per_run.items():
            if "start" not in parts or "end" not in parts:
                continue

            window = windows[ridx]
            if window is None:
                continue

            run = intervals[ridx]
            anchor_a, _anchor_b, freq_hz = window

            def to_host_time(cycle: int, _start=run.start_time, _a=anchor_a, _f=freq_hz) -> datetime:
                return _start + timedelta(seconds=(cycle - _a) / _f)

            k_start = to_host_time(parts["start"].cycle)
            k_end = to_host_time(parts["end"].cycle)
            lifetime_s = (k_end - k_start).total_seconds()

            wait_s = (parts["wait"].data / freq_hz) if "wait" in parts and freq_hz > 0 else float("nan")
            active_s = (parts["active"].data / freq_hz) if "active" in parts and freq_hz > 0 else float("nan")

            # Sanity check: a sub-interval accumulated *within* one kernel invocation can never
            # exceed that invocation's own lifetime. If it does, the reading is corrupted (rare
            # profiler marker export artifacts near 2^32/2^64 boundaries have been observed) --
            # discard it as invalid rather than let one bad row skew an average across cores.
            if lifetime_s <= 0 or wait_s > lifetime_s * 1.01:
                wait_s = float("nan")
            if lifetime_s <= 0 or active_s > lifetime_s * 1.01:
                active_s = float("nan")

            # overhead_s is derived, not measured -- see the docstring above. Only valid when
            # both wait_s and active_s are themselves valid; clamped at 0 to absorb tiny
            # floating-point/measurement noise rather than reporting a nonsensical negative.
            if lifetime_s > 0 and np.isfinite(wait_s) and np.isfinite(active_s):
                overhead_s = max(0.0, lifetime_s - wait_s - active_s)
            else:
                overhead_s = float("nan")

            wait_pct = (wait_s / lifetime_s * 100.0) if lifetime_s > 0 and np.isfinite(wait_s) else float("nan")
            active_pct = (
                (active_s / lifetime_s * 100.0) if lifetime_s > 0 and np.isfinite(active_s) else float("nan")
            )
            overhead_pct = (
                (overhead_s / lifetime_s * 100.0) if lifetime_s > 0 and np.isfinite(overhead_s) else float("nan")
            )

            rows.append(
                {
                    "grid": run.grid,
                    "cores": run.cores,
                    "kernel": kernel_name,
                    "core_x": core_x,
                    "core_y": core_y,
                    "risc": risc,
                    "kernel_start": k_start,
                    "kernel_end": k_end,
                    "lifetime_s": lifetime_s,
                    "wait_s": wait_s,
                    "active_s": active_s,
                    "overhead_s": overhead_s,
                    "wait_pct": wait_pct,
                    "active_pct": active_pct,
                    "overhead_pct": overhead_pct,
                }
            )

    rows.sort(key=lambda r: (str(r["grid"]), str(r["kernel"]), r["core_x"], r["core_y"], str(r["risc"])))
    return rows


def save_kernel_utilization_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "grid",
        "cores",
        "kernel",
        "core_x",
        "core_y",
        "risc",
        "kernel_start",
        "kernel_end",
        "lifetime_s",
        "wait_s",
        "active_s",
        "overhead_s",
        "wait_pct",
        "active_pct",
        "overhead_pct",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["kernel_start"] = str(out["kernel_start"])
            out["kernel_end"] = str(out["kernel_end"])
            writer.writerow(out)


_COMPUTE_RISC_LABELS = {"TRISC_0": "Unpack", "TRISC_1": "Math", "TRISC_2": "Pack"}


def aggregate_kernel_utilization(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Average wait_pct/active_pct across cores, per (grid, kernel, risc), preserving grid and
    series order. Grouping includes `risc` (not just kernel) because the compute kernel runs
    independently on TRISC0/1/2 (unpack/math/pack) -- these do fundamentally different work
    (only TRISC1/math actually calls matmul_tiles; TRISC0 mostly waits on input tiles; TRISC2
    mostly waits on the FPU), so averaging all three together would blend three unrelated
    profiles into one number that represents none of them. Reader/writer only ever have one
    risc each, so this is a no-op change for those two.
    """
    grid_order: List[str] = []
    seen_grids = set()
    for r in rows:
        grid = str(r["grid"])
        if grid not in seen_grids:
            seen_grids.add(grid)
            grid_order.append(grid)

    series_keys: List[Tuple[str, str]] = []
    seen_series = set()
    groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        kernel = str(r["kernel"])
        risc = str(r["risc"])
        groups[(str(r["grid"]), kernel, risc)].append(r)
        if (kernel, risc) not in seen_series:
            seen_series.add((kernel, risc))
            series_keys.append((kernel, risc))

    agg: List[Dict[str, object]] = []
    for grid in grid_order:
        for kernel, risc in series_keys:
            group_rows = groups.get((grid, kernel, risc), [])
            wait_vals = [r["wait_pct"] for r in group_rows if np.isfinite(r["wait_pct"])]
            active_vals = [r["active_pct"] for r in group_rows if np.isfinite(r["active_pct"])]
            overhead_vals = [r["overhead_pct"] for r in group_rows if np.isfinite(r["overhead_pct"])]
            agg.append(
                {
                    "grid": grid,
                    "kernel": kernel,
                    "risc": risc,
                    "core_count": len(group_rows),
                    "avg_wait_pct": safe_mean(wait_vals),
                    "avg_active_pct": safe_mean(active_vals),
                    "avg_overhead_pct": safe_mean(overhead_vals),
                }
            )
    return agg


def plot_kernel_utilization(agg: List[Dict[str, object]], fig_dir: Path, dpi: int) -> List[Path]:
    """
    One stacked bar chart per (kernel, risc) series: avg active% / overhead% / wait% per grid.
    For reader/writer this is one file each (single risc). For compute this produces THREE
    separate files (one per TRISC role) rather than one blended file -- see
    aggregate_kernel_utilization for why blending TRISC0/1/2 together is misleading. Overhead
    is derived (lifetime - wait - active, see compute_kernel_utilization_rows), so it's always
    present as long as wait/active are both valid for that row.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_files: List[Path] = []

    active_label = {"READER": "Transfer", "COMPUTE": "Compute", "WRITER": "Transfer"}
    blocked_on = {
        "READER": "input CB full",
        "COMPUTE": "waiting on input tiles",
        "WRITER": "output tile not ready",
    }

    series_keys: List[Tuple[str, str]] = []
    seen = set()
    for r in agg:
        key = (str(r["kernel"]), str(r["risc"]))
        if key not in seen:
            seen.add(key)
            series_keys.append(key)

    for kernel, risc in series_keys:
        series_rows = [
            r for r in agg if r["kernel"] == kernel and r["risc"] == risc and r["core_count"] > 0
        ]
        if not series_rows:
            continue

        grids = [str(r["grid"]) for r in series_rows]
        wait_pct = [float(r["avg_wait_pct"]) if np.isfinite(r["avg_wait_pct"]) else 0.0 for r in series_rows]
        active_pct = [
            float(r["avg_active_pct"]) if np.isfinite(r["avg_active_pct"]) else 0.0 for r in series_rows
        ]
        overhead_pct = [
            float(r["avg_overhead_pct"]) if np.isfinite(r["avg_overhead_pct"]) else 0.0 for r in series_rows
        ]

        x = np.arange(len(grids), dtype=float)
        overhead_base = list(active_pct)
        wait_base = [a + o for a, o in zip(active_pct, overhead_pct)]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x, active_pct, label=active_label[kernel], color="tab:green")
        ax.bar(x, overhead_pct, bottom=overhead_base, label="Overhead (index/CB bookkeeping)", color="tab:olive")
        ax.bar(x, wait_pct, bottom=wait_base, label=f"Wait ({blocked_on[kernel]})", color="tab:red")

        ax.set_xticks(x)
        ax.set_xticklabels(grids, rotation=45, ha="right")
        ax.set_xlabel("Grid size")
        ax.set_ylabel("% of kernel lifetime (avg over cores)")
        ax.set_ylim(0, 105)

        if kernel == "COMPUTE":
            role_label = _COMPUTE_RISC_LABELS.get(risc, risc)
            ax.set_title(f"Compute kernel ({role_label}/{risc}) utilization vs grid size")
            out_name = f"compute_{role_label.lower()}_utilization.png"
        else:
            ax.set_title(f"{kernel.title()} kernel utilization vs grid size")
            out_name = f"{kernel.lower()}_utilization.png"

        ax.legend(loc="lower right")
        ax.grid(True, axis="y")

        fig.tight_layout()
        out_path = fig_dir / out_name
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        out_files.append(out_path)

    return out_files


_COMBINED_RESOURCES = [
    ("READER", None, "Reader"),
    ("COMPUTE", "TRISC_1", "Compute"),
    ("WRITER", None, "Writer"),
]


def aggregate_combined_utilization(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Average active_s/wait_s/active_pct/wait_pct across cores, per (grid, resource), for the
    three pipeline stages that make up "where did the time go": Reader (NoC read), Compute
    (TRISC1/math -- the only role that actually runs matmul_tiles), Writer (NoC write).
    """
    grid_order: List[str] = []
    seen_grids = set()
    for r in rows:
        grid = str(r["grid"])
        if grid not in seen_grids:
            seen_grids.add(grid)
            grid_order.append(grid)

    agg: List[Dict[str, object]] = []
    for grid in grid_order:
        for kernel, risc, label in _COMBINED_RESOURCES:
            subset = [
                r
                for r in rows
                if str(r["grid"]) == grid
                and str(r["kernel"]) == kernel
                and (risc is None or str(r["risc"]) == risc)
            ]
            active_s_vals = [r["active_s"] for r in subset if np.isfinite(r["active_s"])]
            wait_s_vals = [r["wait_s"] for r in subset if np.isfinite(r["wait_s"])]
            overhead_s_vals = [r["overhead_s"] for r in subset if np.isfinite(r["overhead_s"])]
            active_pct_vals = [r["active_pct"] for r in subset if np.isfinite(r["active_pct"])]
            wait_pct_vals = [r["wait_pct"] for r in subset if np.isfinite(r["wait_pct"])]
            overhead_pct_vals = [r["overhead_pct"] for r in subset if np.isfinite(r["overhead_pct"])]
            lifetime_s_vals = [r["lifetime_s"] for r in subset if np.isfinite(r["lifetime_s"])]

            agg.append(
                {
                    "grid": grid,
                    "resource": label,
                    "core_count": len(subset),
                    "avg_active_s": safe_mean(active_s_vals),
                    "avg_wait_s": safe_mean(wait_s_vals),
                    "avg_overhead_s": safe_mean(overhead_s_vals),
                    "avg_active_pct": safe_mean(active_pct_vals),
                    "avg_wait_pct": safe_mean(wait_pct_vals),
                    "avg_overhead_pct": safe_mean(overhead_pct_vals),
                    "avg_lifetime_s": safe_mean(lifetime_s_vals),
                }
            )
    return agg


def plot_combined_utilization(agg: List[Dict[str, object]], fig_dir: Path, dpi: int) -> List[Path]:
    """
    Two grouped+stacked bar charts (percentage and absolute seconds), each showing all three
    pipeline stages (Reader / Compute[TRISC1-math] / Writer) side by side per grid, active vs
    wait. "Wait" here means idle, blocked on the *neighboring* pipeline stage (reader waiting
    for compute to free up circular-buffer space, compute waiting on reader, writer waiting on
    compute) -- NOT wasted/lost time. A resource with high wait% just means it's faster than
    whichever stage is the real bottleneck for this workload; a resource with active% close to
    100% (little to no wait) is the bottleneck.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_files: List[Path] = []

    grid_order: List[str] = []
    seen_grids = set()
    for r in agg:
        grid = str(r["grid"])
        if grid not in seen_grids:
            seen_grids.add(grid)
            grid_order.append(grid)

    resource_labels = [label for _, _, label in _COMBINED_RESOURCES]
    active_colors = {"Reader": "tab:blue", "Compute": "tab:green", "Writer": "tab:orange"}
    wait_color = "0.75"
    duration_color = "0.25"

    n_grids = len(grid_order)
    x_base = np.arange(n_grids, dtype=float)

    def find_row(grid: str, label: str) -> Optional[Dict[str, object]]:
        for r in agg:
            if str(r["grid"]) == grid and r["resource"] == label:
                return r
        return None

    for active_key, overhead_key, wait_key, ylabel, suffix, y_max, show_duration_ref in (
        (
            "avg_active_pct",
            "avg_overhead_pct",
            "avg_wait_pct",
            "% of kernel lifetime (avg over cores)",
            "pct",
            105.0,
            False,
        ),
        ("avg_active_s", "avg_overhead_s", "avg_wait_s", "Seconds (avg over cores)", "seconds", None, True),
    ):
        # The seconds chart adds a 4th "Run duration" reference bar per grid (the actual
        # wall-clock length of that run) -- since reader/compute/writer run concurrently, none
        # of their active+wait stacks should exceed it. The percentage chart doesn't need this:
        # each resource's own stack is already normalized to its own 100%.
        series_labels = resource_labels + (["Run duration"] if show_duration_ref else [])
        n_series = len(series_labels)
        bar_width = 0.8 / n_series

        fig, ax = plt.subplots(figsize=(14, 7))

        for i, label in enumerate(series_labels):
            offset = (i - (n_series - 1) / 2.0) * bar_width
            xs = x_base + offset

            if label == "Run duration":
                duration_vals = []
                for grid in grid_order:
                    lifetimes = [
                        float(find_row(grid, r)["avg_lifetime_s"])
                        for r in resource_labels
                        if find_row(grid, r) and np.isfinite(find_row(grid, r)["avg_lifetime_s"])
                    ]
                    duration_vals.append(max(lifetimes) if lifetimes else 0.0)
                ax.bar(xs, duration_vals, width=bar_width * 0.95, color=duration_color, label="Run duration")
                continue

            active_vals = []
            overhead_vals = []
            wait_vals = []
            for grid in grid_order:
                row = find_row(grid, label)
                a = float(row[active_key]) if row and np.isfinite(row[active_key]) else 0.0
                o = float(row[overhead_key]) if row and np.isfinite(row[overhead_key]) else 0.0
                w = float(row[wait_key]) if row and np.isfinite(row[wait_key]) else 0.0
                active_vals.append(a)
                overhead_vals.append(o)
                wait_vals.append(w)

            overhead_base = list(active_vals)
            wait_base = [a + o for a, o in zip(active_vals, overhead_vals)]

            ax.bar(xs, active_vals, width=bar_width * 0.95, color=active_colors[label], label=f"{label} active")
            ax.bar(
                xs,
                overhead_vals,
                width=bar_width * 0.95,
                bottom=overhead_base,
                color="tab:olive",
                label=f"{label} overhead",
            )
            ax.bar(
                xs,
                wait_vals,
                width=bar_width * 0.95,
                bottom=wait_base,
                color=wait_color,
                hatch="//",
                edgecolor="0.4",
                label=f"{label} wait",
            )

        ax.set_xticks(x_base)
        ax.set_xticklabels(grid_order, rotation=45, ha="right")
        ax.set_xlabel("Grid size")
        ax.set_ylabel(ylabel)
        if y_max is not None:
            ax.set_ylim(0, y_max)
        ax.set_title("Reader / Compute (math) / Writer: active vs wait, per grid")
        ax.legend(loc="upper right", ncol=5 if show_duration_ref else 3, fontsize=8)
        ax.grid(True, axis="y")

        fig.tight_layout()
        out_path = fig_dir / f"combined_utilization_{suffix}.png"
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        out_files.append(out_path)

    return out_files


def drop_startup_transient(
    samples: List[Sample],
    settle_max_ms: float,
) -> Tuple[List[Sample], float]:
    """Drop a leading current transient from an interval, if there is one.

    The device can draw a short inrush spike at the very start of an interval, before the
    workload reaches steady state -- observed on p100a as ~8 ms at 78 A in front of a 3 s
    window that then sits flat at 70 A. That is 0.27% of the samples, so it barely moves the
    mean, but peak_current_a is a plain max over the window and is set entirely by it: the
    interval reports a 9.20 A dynamic peak where the steady value is 1.18 A.

    A fixed larger --trim-ms would also remove it, but the right amount depends on the board,
    the host and the workload, so it is detected per interval instead. The steady level is
    taken as the median of the back half of the window (robust to a leading spike, and to the
    ramp-down at the far end). Leading samples are dropped while they sit outside a tolerance
    around that level, stopping at the first point that stays inside it.

    Deliberately conservative, so that a genuinely rising workload is never mistaken for a
    transient and trimmed away:

      * never removes more than settle_max_ms from the front, or 10% of the window;
      * requires the signal to stay settled for a short confirmation run, so one in-tolerance
        sample in the middle of a spike does not end the trim early;
      * if no settled point is found inside the cap, drops nothing at all.

    Returns the surviving samples and how many milliseconds were removed.
    """
    if settle_max_ms <= 0.0 or len(samples) < 20:
        return samples, 0.0

    span_s = (samples[-1].ts - samples[0].ts).total_seconds()
    if span_s <= 0.0:
        return samples, 0.0

    # Never eat more than a tenth of the window, however the cap is set.
    budget_ms = min(settle_max_ms, span_s * 1000.0 * 0.10)
    if budget_ms <= 0.0:
        return samples, 0.0

    back_half = [s.tdc_a for s in samples[len(samples) // 2:]]
    steady = float(np.median(back_half))
    mad = float(np.median([abs(c - steady) for c in back_half]))
    # TDC is reported as whole amps, so MAD collapses to 0 on a flat window; the floor keeps
    # the tolerance meaningful there and stops quantisation noise from looking like a spike.
    tol = max(2.0, 3.0 * mad)

    t0 = samples[0].ts
    confirm = timedelta(milliseconds=2.0)
    cutoff = t0 + timedelta(milliseconds=budget_ms)

    settled_at = None
    for i, s in enumerate(samples):
        if s.ts > cutoff:
            break
        if abs(s.tdc_a - steady) > tol:
            continue
        # Candidate: require every sample within the confirmation run to be settled too.
        ok = True
        for t in samples[i:]:
            if t.ts - s.ts > confirm:
                break
            if abs(t.tdc_a - steady) > tol:
                ok = False
                break
        if ok:
            settled_at = i
            break

    if settled_at is None or settled_at == 0:
        return samples, 0.0

    removed_ms = (samples[settled_at].ts - t0).total_seconds() * 1000.0
    return samples[settled_at:], removed_ms


def samples_in_interval(
    raw_samples: List[Sample],
    start_ts: datetime,
    end_ts: datetime,
) -> List[Sample]:
    return [s for s in raw_samples if start_ts <= s.ts <= end_ts]


def safe_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def safe_max(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.max(values))


def compute_pause_base_currents(
    raw_samples: List[Sample],
    intervals: List[ProgramInterval],
    trim_ms: float,
) -> Tuple[List[float], List[int]]:
    trim_delta = timedelta(milliseconds=trim_ms)
    pause_base_currents: List[float] = []
    pause_sample_counts: List[int] = []

    for i in range(len(intervals) - 1):
        prev_end = intervals[i].end_time - trim_delta
        next_start = intervals[i + 1].start_time + trim_delta

        if next_start <= prev_end:
            pause_base_currents.append(float("nan"))
            pause_sample_counts.append(0)
            continue

        pause_samples = samples_in_interval(raw_samples, prev_end, next_start)
        pause_currents = [s.tdc_a for s in pause_samples]

        pause_base_currents.append(safe_mean(pause_currents))
        pause_sample_counts.append(len(pause_samples))

    return pause_base_currents, pause_sample_counts


def attach_dynamic_baseline(
    results: List[Dict[str, object]],
    pause_base_currents: List[float],
) -> List[Dict[str, object]]:
    """Baseline is the current measured in the pause right after the *first* interval, used as
    one fixed reference for every interval in the run -- it is not re-estimated per interval.

    Previously each interval got its own local estimate (the mean of the pause immediately
    before and after it), which tracks the board's own warm-up drift within a run -- but that
    same drift means two separately-run cases (e.g. run 5 minutes apart, one further into a
    multi-case sweep than the other) get baselines from different points on that drift curve,
    which can make whichever case ran later look artificially lower in dynamic current even
    when its raw current was actually higher. A single fixed reference per run removes that
    per-interval tracking (and reintroduces later-interval drift as apparent "dynamic" current
    within the run instead) in exchange for not itself drifting.
    """
    if not results:
        return results

    reference_base_current_a = (
        pause_base_currents[0]
        if pause_base_currents and np.isfinite(pause_base_currents[0])
        else float("nan")
    )

    out: List[Dict[str, object]] = []

    for i, r in enumerate(results):
        rr = dict(r)

        base_current_a = reference_base_current_a

        avg_current_a = rr["avg_current_a"]
        peak_current_a = rr["peak_current_a"]
        window_s = rr["window_s"]

        dynamic_avg_current_a = (
            max(0.0, avg_current_a - base_current_a)
            if np.isfinite(avg_current_a) and np.isfinite(base_current_a)
            else float("nan")
        )
        dynamic_peak_current_a = (
            max(0.0, peak_current_a - base_current_a)
            if np.isfinite(peak_current_a) and np.isfinite(base_current_a)
            else float("nan")
        )

        dynamic_charge_avg_as = (
            dynamic_avg_current_a * window_s
            if np.isfinite(dynamic_avg_current_a) and np.isfinite(window_s)
            else float("nan")
        )
        dynamic_charge_avg_mah = (
            dynamic_charge_avg_as / 3.6 if np.isfinite(dynamic_charge_avg_as) else float("nan")
        )

        dynamic_charge_peak_as = (
            dynamic_peak_current_a * window_s
            if np.isfinite(dynamic_peak_current_a) and np.isfinite(window_s)
            else float("nan")
        )
        dynamic_charge_peak_mah = (
            dynamic_charge_peak_as / 3.6 if np.isfinite(dynamic_charge_peak_as) else float("nan")
        )

        rr["base_current_a"] = base_current_a
        rr["dynamic_avg_current_a"] = dynamic_avg_current_a
        rr["dynamic_peak_current_a"] = dynamic_peak_current_a
        rr["dynamic_charge_avg_as"] = dynamic_charge_avg_as
        rr["dynamic_charge_avg_mah"] = dynamic_charge_avg_mah
        rr["dynamic_charge_peak_as"] = dynamic_charge_peak_as
        rr["dynamic_charge_peak_mah"] = dynamic_charge_peak_mah

        out.append(rr)

    return out


def compute_interval_metrics(
    raw_samples: List[Sample],
    intervals: List[ProgramInterval],
    trim_ms: float,
    settle_max_ms: float = 0.0,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    trim_delta = timedelta(milliseconds=trim_ms)

    for row in intervals:
        adj_start = row.start_time + trim_delta
        adj_end = row.end_time - trim_delta

        if adj_end <= adj_start:
            results.append(
                {
                    "grid": row.grid,
                    "cores": row.cores,
                    "algo_time_s_reported": row.algo_time_s,
                    "interval_start": adj_start,
                    "interval_end": adj_end,
                    "settle_trim_ms": 0.0,
                    "window_s": 0.0,
                    "sample_count": 0,
                    "avg_current_a": float("nan"),
                    "peak_current_a": float("nan"),
                    "avg_vcore_v": float("nan"),
                    "avg_tdp_w": float("nan"),
                    "avg_power_vi_w": float("nan"),
                    "charge_avg_as": float("nan"),
                    "charge_avg_mah": float("nan"),
                    "charge_peak_as": float("nan"),
                    "charge_peak_mah": float("nan"),
                    "energy_j_tdp": float("nan"),
                    "energy_mwh_tdp": float("nan"),
                    "energy_j_vi": float("nan"),
                    "energy_mwh_vi": float("nan"),
                }
            )
            continue

        interval_samples = samples_in_interval(raw_samples, adj_start, adj_end)

        interval_samples, settle_trim_ms = drop_startup_transient(
            interval_samples, settle_max_ms
        )
        if settle_trim_ms > 0.0:
            adj_start = interval_samples[0].ts

        currents = [s.tdc_a for s in interval_samples]
        voltages = [s.vcore_v for s in interval_samples]
        tdps = [s.tdp_w for s in interval_samples]
        powers_vi = [s.tdc_a * s.vcore_v for s in interval_samples]

        window_s = (adj_end - adj_start).total_seconds()
        avg_current_a = safe_mean(currents)
        peak_current_a = safe_max(currents)
        avg_vcore_v = safe_mean(voltages)
        avg_tdp_w = safe_mean(tdps)
        avg_power_vi_w = safe_mean(powers_vi)

        charge_avg_as = avg_current_a * window_s if np.isfinite(avg_current_a) else float("nan")
        charge_avg_mah = charge_avg_as / 3.6 if np.isfinite(charge_avg_as) else float("nan")

        charge_peak_as = peak_current_a * window_s if np.isfinite(peak_current_a) else float("nan")
        charge_peak_mah = charge_peak_as / 3.6 if np.isfinite(charge_peak_as) else float("nan")

        energy_j_tdp = avg_tdp_w * window_s if np.isfinite(avg_tdp_w) else float("nan")
        energy_mwh_tdp = energy_j_tdp / 3.6 if np.isfinite(energy_j_tdp) else float("nan")

        energy_j_vi = avg_power_vi_w * window_s if np.isfinite(avg_power_vi_w) else float("nan")
        energy_mwh_vi = energy_j_vi / 3.6 if np.isfinite(energy_j_vi) else float("nan")

        results.append(
            {
                "grid": row.grid,
                "cores": row.cores,
                "algo_time_s_reported": row.algo_time_s,
                "interval_start": adj_start,
                "interval_end": adj_end,
                "settle_trim_ms": settle_trim_ms,
                "window_s": window_s,
                "sample_count": len(interval_samples),
                "avg_current_a": avg_current_a,
                "peak_current_a": peak_current_a,
                "avg_vcore_v": avg_vcore_v,
                "avg_tdp_w": avg_tdp_w,
                "avg_power_vi_w": avg_power_vi_w,
                "charge_avg_as": charge_avg_as,
                "charge_avg_mah": charge_avg_mah,
                "charge_peak_as": charge_peak_as,
                "charge_peak_mah": charge_peak_mah,
                "energy_j_tdp": energy_j_tdp,
                "energy_mwh_tdp": energy_mwh_tdp,
                "energy_j_vi": energy_j_vi,
                "energy_mwh_vi": energy_mwh_vi,
            }
        )

    pause_base_currents, pause_sample_counts = compute_pause_base_currents(
        raw_samples=raw_samples,
        intervals=intervals,
        trim_ms=trim_ms,
    )

    results = attach_dynamic_baseline(results, pause_base_currents)

    # base_current_a now comes from a single reference pause (the one right after the first
    # interval -- see attach_dynamic_baseline), so that is the only pause any row's baseline
    # quality depends on. base_pause_left_samples is kept at 0 throughout (no "left" pause is
    # used anymore); base_pause_right_samples reports that one reference pause's sample count
    # for every row, so a thin reference pause is visible as a quality flag across the whole run.
    reference_pause_samples = pause_sample_counts[0] if pause_sample_counts else 0
    for r in results:
        r["base_pause_left_samples"] = 0
        r["base_pause_right_samples"] = reference_pause_samples

    return results


def format_num(x: object, fmt: str = ".6f") -> str:
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "nan"
        return format(float(x), fmt)
    return str(x)


def print_interval_report(results: List[Dict[str, object]], trim_ms: float) -> None:
    print("\nPROGRAM INTERVAL ANALYSIS", file=sys.stderr)
    print("=" * 220, file=sys.stderr)
    print(
        f"{'Grid':>6} "
        f"{'Cores':>6} "
        f"{'Win[s]':>10} "
        f"{'Samples':>8} "
        f"{'Base I[A]':>12} "
        f"{'Avg I[A]':>12} "
        f"{'Dyn Avg[A]':>12} "
        f"{'Peak I[A]':>12} "
        f"{'Dyn Peak[A]':>12} "
        f"{'Qavg[mAh]':>12} "
        f"{'QdynAvg[mAh]':>14} "
        f"{'Qpeak[mAh]':>12} "
        f"{'QdynPk[mAh]':>12} "
        f"{'End(-trim)':>26}",
        file=sys.stderr,
    )
    print("=" * 220, file=sys.stderr)

    for r in results:
        print(
            f"{str(r['grid']):>6} "
            f"{int(r['cores']):>6} "
            f"{format_num(r['window_s'], '.6f'):>10} "
            f"{int(r['sample_count']):>8} "
            f"{format_num(r.get('base_current_a', float('nan')), '.6f'):>12} "
            f"{format_num(r['avg_current_a'], '.6f'):>12} "
            f"{format_num(r.get('dynamic_avg_current_a', float('nan')), '.6f'):>12} "
            f"{format_num(r['peak_current_a'], '.6f'):>12} "
            f"{format_num(r.get('dynamic_peak_current_a', float('nan')), '.6f'):>12} "
            f"{format_num(r['charge_avg_mah'], '.6f'):>12} "
            f"{format_num(r.get('dynamic_charge_avg_mah', float('nan')), '.6f'):>14} "
            f"{format_num(r['charge_peak_mah'], '.6f'):>12} "
            f"{format_num(r.get('dynamic_charge_peak_mah', float('nan')), '.6f'):>12} "
            f"{str(r['interval_end']):>26}",
            file=sys.stderr,
        )

    print(f"\nTrim applied to each side: {trim_ms:.3f} ms", file=sys.stderr)
    print("Base current is estimated from pause intervals between adjacent runs.", file=sys.stderr)


def save_interval_csv(results: List[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "grid",
        "cores",
        "algo_time_s_reported",
        "interval_start",
        "interval_end",
        "window_s",
        "sample_count",
        "avg_current_a",
        "peak_current_a",
        "avg_vcore_v",
        "avg_tdp_w",
        "avg_power_vi_w",
        "charge_avg_as",
        "charge_avg_mah",
        "charge_peak_as",
        "charge_peak_mah",
        "energy_j_tdp",
        "energy_mwh_tdp",
        "energy_j_vi",
        "energy_mwh_vi",
        "base_current_a",
        "dynamic_avg_current_a",
        "dynamic_peak_current_a",
        "dynamic_charge_avg_as",
        "dynamic_charge_avg_mah",
        "dynamic_charge_peak_as",
        "dynamic_charge_peak_mah",
        "base_pause_left_samples",
        "base_pause_right_samples",
        "settle_trim_ms",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            out = dict(row)
            out["interval_start"] = str(out["interval_start"])
            out["interval_end"] = str(out["interval_end"])
            writer.writerow(out)


def plot_duration_and_metric(
    grids: List[str],
    durations: List[float],
    metric: List[float],
    metric_label: str,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    x = np.arange(len(grids), dtype=float)

    color_left = "tab:blue"
    color_right = "tab:red"

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    line1 = ax1.plot(x, durations, marker="o", color=color_left, label="Duration [s]")
    line2 = ax2.plot(x, metric, marker="s", color=color_right, label=metric_label)

    ax1.set_xticks(x)
    ax1.set_xticklabels(grids, rotation=45, ha="right")
    ax1.set_xlabel("Grid size")

    ax1.set_ylabel("Duration [s]", color=color_left)
    ax2.set_ylabel(metric_label, color=color_right)

    ax1.tick_params(axis="y", colors=color_left)
    ax2.tick_params(axis="y", colors=color_right)

    ax1.grid(True)
    ax1.set_title(title)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_single_metric(
    grids: List[str],
    values: List[float],
    ylabel: str,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    x = np.arange(len(grids), dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, values, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(grids, rotation=45, ha="right")
    ax.set_xlabel("Grid size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_dual_metric(
    grids: List[str],
    values_left: List[float],
    values_right: List[float],
    ylabel_left: str,
    ylabel_right: str,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    x = np.arange(len(grids), dtype=float)

    color_left = "tab:blue"
    color_right = "tab:red"

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    line1 = ax1.plot(x, values_left, marker="o", color=color_left, label=ylabel_left)
    line2 = ax2.plot(x, values_right, marker="s", color=color_right, label=ylabel_right)

    ax1.set_xticks(x)
    ax1.set_xticklabels(grids, rotation=45, ha="right")
    ax1.set_xlabel("Grid size")

    ax1.set_ylabel(ylabel_left, color=color_left)
    ax2.set_ylabel(ylabel_right, color=color_right)

    ax1.tick_params(axis="y", colors=color_left)
    ax2.tick_params(axis="y", colors=color_right)

    ax1.grid(True)
    ax1.set_title(title)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def plot_compute_charge_vs_cores(
    cores: List[int],
    grids: List[str],
    values: List[float],
    ylabel: str,
    title: str,
    out_path: Path,
    dpi: int,
    show_ideal: bool = True,
) -> None:
    """
    Plot a metric versus grid size with an ideal linear scaling reference.

    X-axis labels are grid strings (e.g. "3x2").
    The ideal reference line scales proportionally to core count.
    """
    if not cores or not values or len(cores) != len(values):
        return

    x_pos = np.arange(len(cores), dtype=float)
    x_cores = np.array(cores, dtype=float)
    y = np.array(values, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(x_pos, y, marker="o", label="Measured")

    # Ideal linear scaling: reference from first finite point, scaled by core count
    if show_ideal:
        finite_idx = np.where(np.isfinite(y))[0]
        if finite_idx.size > 0:
            i0 = int(finite_idx[0])
            c0 = x_cores[i0]
            y0 = y[i0]

            if c0 > 0 and np.isfinite(y0):
                y_ref = y0 * (x_cores / c0)
                ax.plot(
                    x_pos,
                    y_ref,
                    linestyle="--",
                    marker=None,
                    color="orange",
                    label="Ideal linear scaling",
                )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(grids, rotation=45, ha="right")
    ax.set_xlabel("Grid size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_program_interval_figures(
    results: List[Dict[str, object]],
    fig_dir: Path,
    dpi: int,
) -> List[Path]:
    out_files: List[Path] = []

    valid = [r for r in results if str(r["grid"]) != ""]
    if not valid:
        return out_files

    fig_dir.mkdir(parents=True, exist_ok=True)

    grids = [str(r["grid"]) for r in valid]
    cores = [int(r["cores"]) for r in valid]

    durations = [float(r["window_s"]) if np.isfinite(r["window_s"]) else np.nan for r in valid]
    peak_currents = [float(r["peak_current_a"]) if np.isfinite(r["peak_current_a"]) else np.nan for r in valid]
    avg_currents = [float(r["avg_current_a"]) if np.isfinite(r["avg_current_a"]) else np.nan for r in valid]
    charge_avg = [float(r["charge_avg_mah"]) if np.isfinite(r["charge_avg_mah"]) else np.nan for r in valid]
    charge_peak = [float(r["charge_peak_mah"]) if np.isfinite(r["charge_peak_mah"]) else np.nan for r in valid]

    dynamic_avg_currents = [float(r.get("dynamic_avg_current_a", np.nan)) for r in valid]
    dynamic_peak_currents = [float(r.get("dynamic_peak_current_a", np.nan)) for r in valid]
    dynamic_charge_avg = [float(r.get("dynamic_charge_avg_mah", np.nan)) for r in valid]
    dynamic_charge_peak = [float(r.get("dynamic_charge_peak_mah", np.nan)) for r in valid]
    base_currents = [float(r.get("base_current_a", np.nan)) for r in valid]

    figure_specs = [
        (
            fig_dir / "duration_vs_peak_current.png",
            lambda p: plot_duration_and_metric(
                grids=grids,
                durations=durations,
                metric=peak_currents,
                metric_label="Peak current [A]",
                title="Duration and Peak Current vs Grid Size",
                out_path=p,
                dpi=dpi,
            ),
        ),
        (
            fig_dir / "duration_vs_avg_current.png",
            lambda p: plot_duration_and_metric(
                grids=grids,
                durations=durations,
                metric=avg_currents,
                metric_label="Average current [A]",
                title="Duration and Average Current vs Grid Size",
                out_path=p,
                dpi=dpi,
            ),
        ),
        (
            fig_dir / "charge_avg_current.png",
            lambda p: plot_single_metric(
                grids=grids,
                values=charge_avg,
                ylabel="Charge consumed from average current [mAh]",
                title="Charge Consumed from Average Current vs Grid Size",
                out_path=p,
                dpi=dpi,
            ),
        ),
        (
            fig_dir / "charge_peak_current.png",
            lambda p: plot_single_metric(
                grids=grids,
                values=charge_peak,
                ylabel="Charge consumed from peak current [mAh]",
                title="Charge Consumed from Peak Current vs Grid Size",
                out_path=p,
                dpi=dpi,
            ),
        ),
        (
            fig_dir / "duration_vs_dynamic_peak_current.png",
            lambda p: plot_duration_and_metric(
                grids=grids,
                durations=durations,
                metric=dynamic_peak_currents,
                metric_label="Dynamic peak current [A]",
                title="Duration and Dynamic Peak Current vs Grid Size",
                out_path=p,
                dpi=dpi,
            ),
        ),
        (
            fig_dir / "duration_vs_dynamic_avg_current.png",
            lambda p: plot_duration_and_metric(
                grids=grids,
                durations=durations,
                metric=dynamic_avg_currents,
                metric_label="Dynamic average current [A]",
                title="Duration and Dynamic Average Current vs Grid Size",
                out_path=p,
                dpi=dpi,
            ),
        ),
        (
            fig_dir / "dynamic_charge_avg_current.png",
            lambda p: plot_single_metric(
                grids=grids,
                values=dynamic_charge_avg,
                ylabel="Dynamic charge from average current [mAh]",
                title="Dynamic Charge from Average Current vs Grid Size",
                out_path=p,
                dpi=dpi,
            ),
        ),
        (
            fig_dir / "dynamic_charge_peak_current.png",
            lambda p: plot_compute_charge_vs_cores(
                cores=cores,
                grids=grids,
                values=dynamic_charge_peak,
                ylabel="Dynamic charge from peak current [mAh]",
                title="Dynamic Charge (Peak Current) vs Grid Size",
                out_path=p,
                dpi=dpi,
                show_ideal=False,
            ),
        ),
        (
            fig_dir / "q_compute_vs_cores.png",
            lambda p: plot_compute_charge_vs_cores(
                cores=cores,
                grids=grids,
                values=dynamic_charge_peak,
                ylabel="Q_compute from peak current [mAh]",
                title="Q_compute vs Grid Size",
                out_path=p,
                dpi=dpi,
            ),
        ),
        (
            fig_dir / "base_current.png",
            lambda p: plot_single_metric(
                grids=grids,
                values=base_currents,
                ylabel="Base current [A]",
                title="Estimated Base Current vs Grid Size",
                out_path=p,
                dpi=dpi,
            ),
        ),
        (
            fig_dir / "charge_total_vs_compute_avg.png",
            lambda p: plot_dual_metric(
                grids=grids,
                values_left=charge_avg,
                values_right=dynamic_charge_avg,
                ylabel_left="Q Total from average current [mAh]",
                ylabel_right="Q Compute from average current [mAh]",
                title="Q Total and Q Compute (Average Current) vs Grid Size",
                out_path=p,
                dpi=dpi,
            ),
        ),
        (
            fig_dir / "charge_total_vs_compute_peak.png",
            lambda p: plot_dual_metric(
                grids=grids,
                values_left=charge_peak,
                values_right=dynamic_charge_peak,
                ylabel_left="Q Total from peak current [mAh]",
                ylabel_right="Q Compute from peak current [mAh]",
                title="Q Total and Q Compute (Peak Current) vs Grid Size",
                out_path=p,
                dpi=dpi,
            ),
        ),
        (
            fig_dir / "execution_time_vs_dynamic_peak_current.png",
            lambda p: plot_duration_and_metric(
                grids=grids,
                durations=durations,
                metric=dynamic_peak_currents,
                metric_label="Dynamic peak current [A]",
                title="Execution Time vs Dynamic Peak Current",
                out_path=p,
                dpi=dpi,
            ),
        ),
        (
            fig_dir / "execution_time_vs_dynamic_avg_current.png",
            lambda p: plot_duration_and_metric(
                grids=grids,
                durations=durations,
                metric=dynamic_avg_currents,
                metric_label="Dynamic average current [A]",
                title="Execution Time vs Dynamic Average Current",
                out_path=p,
                dpi=dpi,
            ),
        ),
    ]

    for out_path, plot_fn in figure_specs:
        plot_fn(out_path)
        out_files.append(out_path)

    return out_files


def copy_input_file(src: Path, dst_dir: Path) -> Path:
    """
    Copy input file into destination directory, preserving original file name.

    If the source file is already in the destination path, do nothing.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    src = src.resolve()
    dst_path = (dst_dir / src.name).resolve()

    if src == dst_path:
        return dst_path


    shutil.copy2(src, dst_path)
    return dst_path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, type=Path)
    ap.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Root output directory. Results are written to <output>/<subdir>/",
    )
    ap.add_argument(
        "--subdir",
        required=True,
        type=validate_subdir_name,
        help="Name of the run subdirectory inside the output root.",
    )
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--slot-ms",
        type=int,
        default=1000,
        help="Bin width for slots in milliseconds (default 1000 ms)",
    )
    ap.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="Keep only telemetry samples for the specified device/chip ID",
    )
    ap.add_argument(
        "--program-log",
        type=Path,
        default=None,
        help="Optional TT host summary log file. When provided, compute interval-based metrics.",
    )
    ap.add_argument(
        "--trim-ms",
        type=float,
        default=6.0,
        help="Trim each interval by this many ms on both sides when --program-log is used (default 6.0)",
    )
    ap.add_argument(
        "--settle-max-ms",
        type=float,
        default=25.0,
        help=(
            "Additionally drop a leading startup transient from each interval, searching at "
            "most this many ms in from the start (and never more than 10%% of the window). "
            "The amount actually removed is reported per interval as settle_trim_ms. "
            "0 disables (default 25.0)"
        ),
    )
    ap.add_argument(
        "--program-csv",
        type=Path,
        default=None,
        help=(
            "Optional output CSV file path. If omitted, default is "
            "<output>/<subdir>/program_intervals.csv"
        ),
    )
    ap.add_argument(
        "--device-profiler-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to tt-metal's device profiler CSV "
            "(generated/profiler/.logs/profile_log_device.csv), produced by "
            "ReadMeshDeviceProfilerResults() in the app. When provided (together with "
            "--program-log), computes per-kernel (reader/compute/writer) start/end "
            "intervals, converted to host wall-clock time, for each grid run."
        ),
    )
    ap.add_argument(
        "--kernel-csv",
        type=Path,
        default=None,
        help=(
            "Optional output CSV file path for kernel start/end intervals. If omitted, "
            "default is <output>/<subdir>/kernel_intervals.csv"
        ),
    )

    args = ap.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2

    if args.program_log is not None and not args.program_log.exists():
        print(f"ERROR: program log file not found: {args.program_log}", file=sys.stderr)
        return 6

    if args.device_profiler_csv is not None:
        if not args.device_profiler_csv.exists():
            print(f"ERROR: device profiler csv file not found: {args.device_profiler_csv}", file=sys.stderr)
            return 8
        if args.program_log is None:
            print("ERROR: --device-profiler-csv requires --program-log.", file=sys.stderr)
            return 9

    run_dir = args.output / args.subdir
    fig_dir = run_dir / "Figures"

    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    copied_input = copy_input_file(args.input, run_dir)
    copied_program_log: Optional[Path] = None
    if args.program_log is not None:
        copied_program_log = copy_input_file(args.program_log, run_dir)

    telemetry_png = fig_dir / "telemetry_overview.png"

    slots, devices, stats, raw_samples = parse_file_binned(
        args.input,
        args.slot_ms,
        device_id_filter=args.device_id,
    )

    if stats["lines_matched"] == 0:
        print("ERROR: No telemetry lines matched. Check log format.", file=sys.stderr)
        return 3

    if not devices:
        if args.device_id is not None:
            print(f"ERROR: No telemetry samples found for device ID {args.device_id}.", file=sys.stderr)
        else:
            print("ERROR: Parsed samples, but no devices remained after filtering.", file=sys.stderr)
        return 4

    t, dev_ids, v_v, i_a, p_w = build_timeseries(slots, devices)
    if t.size == 0:
        print("ERROR: Parsed samples, but no slots produced.", file=sys.stderr)
        return 5

    plot_png(t, dev_ids, v_v, i_a, p_w, telemetry_png, args.dpi)

    filter_info = f" device_filter={args.device_id}" if args.device_id is not None else ""

    print(
        f"OK: matched={stats['lines_matched']}/{stats['lines_total']} "
        f"unmatched={stats['lines_unmatched']} "
        f"filtered_out={stats['lines_filtered_out']} "
        f"devices={len(dev_ids)} "
        f"slots={len(t)} "
        f"slot_ms={args.slot_ms}"
        f"{filter_info} "
        f"run_dir={run_dir} "
        f"telemetry_png={telemetry_png}",
        file=sys.stderr,
    )

    print(f"Copied telemetry input to: {copied_input}", file=sys.stderr)
    if copied_program_log is not None:
        print(f"Copied program log to: {copied_program_log}", file=sys.stderr)

    if args.program_log is not None:
        intervals = parse_program_log(args.program_log)
        if not intervals:
            print("ERROR: --program-log was provided, but no valid summary rows were parsed.", file=sys.stderr)
            return 7

        interval_results = compute_interval_metrics(
            raw_samples=raw_samples,
            intervals=intervals,
            trim_ms=args.trim_ms,
            settle_max_ms=args.settle_max_ms,
        )

        print_interval_report(interval_results, args.trim_ms)

        program_csv = args.program_csv
        if program_csv is None:
            program_csv = run_dir / "program_intervals.csv"

        save_interval_csv(interval_results, program_csv)
        print(f"Program interval CSV written to: {program_csv}", file=sys.stderr)

        figure_paths = plot_program_interval_figures(
            results=interval_results,
            fig_dir=fig_dir,
            dpi=args.dpi,
        )

        for fig_path in figure_paths:
            print(f"Program interval figure written to: {fig_path}", file=sys.stderr)

        if args.device_profiler_csv is not None:
            copied_device_csv = copy_input_file(args.device_profiler_csv, run_dir)
            print(f"Copied device profiler csv to: {copied_device_csv}", file=sys.stderr)

            device_markers = load_device_kernel_markers(args.device_profiler_csv)
            kernel_intervals = compute_kernel_intervals(device_markers, intervals)

            if not kernel_intervals:
                print(
                    "WARNING: no kernel start/end intervals could be computed from the "
                    "device profiler csv (check that the app ran with "
                    "TT_METAL_DEVICE_PROFILER=1).",
                    file=sys.stderr,
                )
            else:
                print_kernel_interval_report(kernel_intervals)

                kernel_csv = args.kernel_csv
                if kernel_csv is None:
                    kernel_csv = run_dir / "kernel_intervals.csv"

                save_kernel_intervals_csv(kernel_intervals, kernel_csv)
                print(f"Kernel interval CSV written to: {kernel_csv}", file=sys.stderr)

            utilization_rows = compute_kernel_utilization_rows(device_markers, intervals)

            if not utilization_rows:
                print(
                    "WARNING: no per-core wait/active utilization rows could be computed from "
                    "the device profiler csv.",
                    file=sys.stderr,
                )
            else:
                utilization_csv = run_dir / "kernel_utilization.csv"
                save_kernel_utilization_csv(utilization_rows, utilization_csv)
                print(f"Kernel utilization CSV written to: {utilization_csv}", file=sys.stderr)

                utilization_agg = aggregate_kernel_utilization(utilization_rows)
                new_fig_dir = run_dir / "FiguresNew"
                utilization_figures = plot_kernel_utilization(utilization_agg, new_fig_dir, args.dpi)

                for fig_path in utilization_figures:
                    print(f"Kernel utilization figure written to: {fig_path}", file=sys.stderr)

                combined_agg = aggregate_combined_utilization(utilization_rows)
                combined_figures = plot_combined_utilization(combined_agg, new_fig_dir, args.dpi)

                for fig_path in combined_figures:
                    print(f"Combined utilization figure written to: {fig_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
