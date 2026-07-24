#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Export op-to-op latency tables from profile_log_device.csv.

Per-core op-to-op gap (program k -> k+1):
  TRISC_2 FINISH_LAST_PUSH at end of pack TRISC kernel (program k)
  to TRISC_0 TILE_IDX for tile 0 when program k+1's compute starts.

Example:
  python3 tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/export_op_to_op_profiler_csv.py \\
    --input-file "$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

DEFAULT_LOG = "generated/profiler/.logs/profile_log_device.csv"
DEFAULT_RT_LOG = "generated/profiler/.logs/profile_log_device_rt.csv"
UNPACK_RISC = "TRISC_0"
PACK_RISC = "TRISC_2"
# Dataflow kernels: Blackhole uses BRISC (writer) / NCRISC (reader); other arches may use RISCV_*.
READ_RISC_TYPES = frozenset({"NCRISC", "RISCV_1"})
WRITE_RISC_TYPES = frozenset({"BRISC", "RISCV_0"})
DATAFLOW_RISC_TYPES = READ_RISC_TYPES | WRITE_RISC_TYPES


def parse_chip_freq_mhz(log_path: Path) -> float:
    with log_path.open() as f:
        header = f.readline()
    match = re.search(r"CHIP_FREQ\[MHz\]:\s*([0-9.]+)", header)
    if not match:
        raise ValueError(f"Could not parse CHIP_FREQ from first line of {log_path}")
    return float(match.group(1))


# DeviceRecordEvent ids -> original marker names. Kept in sync with the EV_* constants in the
# op_to_op kernels (reader/writer/compute). recordEvent is lighter than DeviceTimestampedData
# (no data payload = less L1 buffer pressure); the id is the low 16 bits of timer_id.
EVENT_NAMES = {
    1: "NCRISC_GO",
    2: "READ_BEFORE_BARRIER",
    3: "READ_AFTER_BARRIER",
    4: "READ_LAST_BARRIER",
    5: "NCRISC_DONE",
    6: "BRISC_GO",
    7: "WRITE_BEFORE_BARRIER",
    8: "WRITE_LAST_ISSUED",
    9: "WRITE_LAST_FLUSHED",
    10: "WRITE_AFTER_BARRIER",
    11: "BRISC_DONE",
    12: "TILE_IDX",  # lean-mode first-math (tile 0)
    13: "FINISH_LAST_PUSH",  # last-math / pack finish
}


def load_profiler_csv(log_path: Path) -> pd.DataFrame:
    df = pd.read_csv(log_path, skiprows=1)
    df.columns = df.columns.str.strip()
    # Backfill original marker names for DeviceRecordEvent markers (type == TS_EVENT): they have an
    # empty zone name; the event id is timer_id & 0xFFFF (get_id = (id & 0xFFFF) | (type << 16)).
    # Events carry no payload, so set data = 0 (the TILE_IDX event is tile 0). This keeps every
    # name-keyed walk (and decompose / event_timeline) working unchanged.
    if "type" in df.columns and "timer_id" in df.columns:
        is_ev = df["type"].astype(str).str.strip() == "TS_EVENT"
        if is_ev.any():
            ids = df.loc[is_ev, "timer_id"].astype("int64") & 0xFFFF
            df.loc[is_ev, "zone name"] = ids.map(EVENT_NAMES).values
            df.loc[is_ev, "data"] = 0
    return df


def walk_core_markers(df: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    """Return per-core program finish (pack) and first-tile start (unpack) times."""
    markers = df[
        (df["zone name"].isin(["PROG_ID", "TILE_IDX", "FINISH_LAST_PUSH"]))
        & (df["RISC processor type"].isin([UNPACK_RISC, PACK_RISC]))
    ].copy()
    markers = markers.sort_values(["PCIe slot", "core_x", "core_y", "RISC processor type", "time[cycles since reset]"])

    # (chip, core_x, core_y, prog_id) -> cycles
    pack_finish: dict[tuple, int] = {}
    unpack_tile0_start: dict[tuple, int] = {}
    tile_rows: list[dict] = []

    group_cols = ["PCIe slot", "core_x", "core_y", "RISC processor type"]
    for key, group in markers.groupby(group_cols, sort=False):
        chip, core_x, core_y, risc = key
        current_prog: int | None = None

        for _, row in group.iterrows():
            zone = row["zone name"]
            time_cycles = int(row["time[cycles since reset]"])
            data = int(row["data"])

            if zone == "PROG_ID":
                current_prog = data
            elif zone == "TILE_IDX" and risc == UNPACK_RISC:
                if current_prog is None:
                    continue
                tile_rows.append(
                    {
                        "chip": chip,
                        "core_x": core_x,
                        "core_y": core_y,
                        "prog_id": current_prog,
                        "tile_idx": data,
                        "unpack_start_cycles": time_cycles,
                    }
                )
                if data == 0:
                    unpack_tile0_start[(chip, core_x, core_y, current_prog)] = time_cycles
            elif zone == "FINISH_LAST_PUSH" and risc == PACK_RISC:
                if current_prog is None:
                    current_prog = data
                pack_finish[(chip, core_x, core_y, current_prog)] = time_cycles

    return pack_finish, unpack_tile0_start, pd.DataFrame(tile_rows)


def walk_done_to_go(df: pd.DataFrame, freq_mhz: float, min_prog_id: int) -> pd.DataFrame:
    """Done/go decomposition: "MCAST GO issue cost + done counting in dispatch
    core". On BH this materialises as the NoC-propagation interval "dispatch
    sees workers done -> a worker observes the new GO multicast".

    We have three event markers (all on the same chip clock domain):
      - DISP_DONE_OBSERVED (dispatch_s):   wait_for_workers spin exits.
      - DISP_GO_ISSUED    (dispatch_s):    MCAST GO write fired to NoC
                                            (registers programmed, returns
                                             before propagation completes).
      - WORKER_GO_OBSERVED (per worker):   GO spin exits + scope start +
                                            host_assigned_id read.

    Reported per (chip, dispatch DISP_DONE timestamp):
      dg_issue_ns    = DISP_GO_ISSUED - DISP_DONE_OBSERVED on dispatch_s.
                        Pure single-write fire cost (~30 cycles on BH).
      dg_first_ns    = min over workers of
                       (WORKER_GO_OBSERVED - DISP_DONE_OBSERVED).
                        Nearest worker observes GO.
      dg_median_ns   = median over workers (typical worker)
      dg_last_ns     = max over workers (farthest worker; ~= our old
                        brisc_fw_done_to_go_us upper bound)

    Requires TT_METAL_DEVICE_PROFILER_DISPATCH=1 at runtime so dispatch_s
    markers land in profile_log_device.csv.

    `min_prog_id` filters the dispatch program-id tag (popped_pid). Trace
    boundaries (>3 us idle) are dropped automatically.
    """
    disp_done = df[df["zone name"] == "DISP_DONE_OBSERVED"].copy()
    disp_go = df[df["zone name"] == "DISP_GO_ISSUED"].copy()
    worker_go = df[df["zone name"] == "WORKER_GO_OBSERVED"].copy()
    if disp_done.empty or worker_go.empty:
        return pd.DataFrame()
    # The dispatch_s core appears in both DISP_* and (sometimes, on startup)
    # WORKER_GO_OBSERVED if FW happens to log it. Identify dispatch cores per chip
    # and exclude them from the worker pool.
    worker_go = worker_go.copy()
    # Drop rows where worker is actually a dispatch core for the same chip.
    dispatch_set: set[tuple] = set()
    for _, r in disp_done[["PCIe slot", "core_x", "core_y"]].drop_duplicates().iterrows():
        dispatch_set.add((int(r["PCIe slot"]), int(r["core_x"]), int(r["core_y"])))
    worker_go = worker_go[
        ~worker_go.apply(lambda r: (int(r["PCIe slot"]), int(r["core_x"]), int(r["core_y"])) in dispatch_set, axis=1)
    ]

    # 10 us window cap (in cycles). Anything beyond is a trace boundary, not a real transition.
    window_cycles = int(3000.0 / 1000.0 * freq_mhz)  # 3 us = ~4050 cycles @ 1.35 GHz

    rows: list[dict] = []
    for chip, dd_chip in disp_done.groupby("PCIe slot"):
        dd_chip = dd_chip.sort_values("time[cycles since reset]").reset_index(drop=True)
        dg_chip = disp_go[disp_go["PCIe slot"] == chip].sort_values("time[cycles since reset]").reset_index(drop=True)
        wg_chip = worker_go[worker_go["PCIe slot"] == chip].copy()
        wg_t = wg_chip["time[cycles since reset]"].astype("int64").values
        for i, drow in dd_chip.iterrows():
            d_t = int(drow["time[cycles since reset]"])
            d_pid = int(drow["data"])
            if d_pid < min_prog_id:
                continue
            # Worker events RIGHT AFTER this DISP_DONE, within window.
            lo, hi = d_t, d_t + window_cycles
            mask = (wg_t > lo) & (wg_t < hi)
            deltas = wg_t[mask] - d_t
            if len(deltas) == 0:
                continue
            # Find DISP_GO_ISSUED that immediately follows.
            dg_after = dg_chip[dg_chip["time[cycles since reset]"] > d_t]
            if len(dg_after) > 0:
                issue_cycles = int(dg_after.iloc[0]["time[cycles since reset]"]) - d_t
            else:
                issue_cycles = pd.NA
            rows.append(
                {
                    "chip": chip,
                    "program_id": d_pid,
                    "done_observed_cycles": d_t,
                    "n_workers_responded": int(len(deltas)),
                    "dg_issue_cycles": issue_cycles,
                    "dg_issue_ns": (issue_cycles / freq_mhz * 1000.0) if issue_cycles is not pd.NA else pd.NA,
                    "dg_first_cycles": int(deltas.min()),
                    "dg_first_ns": float(deltas.min()) / freq_mhz * 1000.0,
                    "dg_median_cycles": int(pd.Series(deltas).median()),
                    "dg_median_ns": float(pd.Series(deltas).median()) / freq_mhz * 1000.0,
                    "dg_last_cycles": int(deltas.max()),
                    "dg_last_ns": float(deltas.max()) / freq_mhz * 1000.0,
                }
            )
    return pd.DataFrame(rows)


def walk_barrier_markers(df: pd.DataFrame) -> dict[str, dict[tuple, int]]:
    """Per (chip, core_x, core_y, prog_id) timestamps for reader/writer barriers and go/done.

    BRISC-FW zone (firmware scope around each program's kernel launch path) is also
    captured, separately from the user BRISC_GO/BRISC_DONE markers emitted from
    inside our writer kernel. The FW.start fires before PROG_ID is read, so we
    forward-tag pending FW.start with the next PROG_ID we see.
    """
    user_zones = [
        "PROG_ID",
        "READ_BEFORE_BARRIER",
        "READ_AFTER_BARRIER",
        "WRITE_BEFORE_BARRIER",
        "WRITE_LAST_ISSUED",
        "WRITE_LAST_FLUSHED",
        "WRITE_AFTER_BARRIER",
        "BRISC_GO",
        "BRISC_DONE",
        "NCRISC_GO",
        "NCRISC_DONE",
    ]
    fw_zones = ["BRISC-FW", "NCRISC-FW"]
    markers = df[
        (df["zone name"].isin(user_zones + fw_zones)) & (df["RISC processor type"].isin(DATAFLOW_RISC_TYPES))
    ].copy()
    markers = markers.sort_values(["PCIe slot", "core_x", "core_y", "RISC processor type", "time[cycles since reset]"])

    keys = [
        "read_before",
        "read_after",
        "write_before",
        "write_last_issued",
        "write_last_flushed",
        "write_after",
        "brisc_go",
        "brisc_done",
        "ncrisc_go",
        "ncrisc_done",
        "brisc_fw_start",
        "brisc_fw_end",
        "ncrisc_fw_start",
        "ncrisc_fw_end",
    ]
    out: dict[str, dict[tuple, int]] = {k: {} for k in keys}
    group_cols = ["PCIe slot", "core_x", "core_y", "RISC processor type"]

    for key, group in markers.groupby(group_cols, sort=False):
        chip, core_x, core_y, risc = key
        is_reader = risc in READ_RISC_TYPES
        is_writer = risc in WRITE_RISC_TYPES
        current_prog: int | None = None
        pending_fw_start: int | None = None
        for _, row in group.iterrows():
            zone = row["zone name"]
            time_cycles = int(row["time[cycles since reset]"])
            zone_type = row.get("type", "")
            if zone == "BRISC-FW" and zone_type == "ZONE_START" and is_writer:
                pending_fw_start = time_cycles
                current_prog = None
                continue
            if zone == "NCRISC-FW" and zone_type == "ZONE_START" and is_reader:
                pending_fw_start = time_cycles
                current_prog = None
                continue
            if zone == "PROG_ID":
                current_prog = int(row["data"])
                if pending_fw_start is not None:
                    tkey = (chip, core_x, core_y, current_prog)
                    if is_writer:
                        out["brisc_fw_start"][tkey] = pending_fw_start
                    elif is_reader:
                        out["ncrisc_fw_start"][tkey] = pending_fw_start
                    pending_fw_start = None
                continue
            if current_prog is None:
                continue
            tkey = (chip, core_x, core_y, current_prog)
            if zone == "READ_BEFORE_BARRIER" and is_reader:
                out["read_before"][tkey] = time_cycles
            elif zone == "READ_AFTER_BARRIER" and is_reader:
                out["read_after"][tkey] = time_cycles
            elif zone == "WRITE_BEFORE_BARRIER" and is_writer:
                out["write_before"][tkey] = time_cycles
            elif zone == "WRITE_LAST_ISSUED" and is_writer:
                out["write_last_issued"][tkey] = time_cycles
            elif zone == "WRITE_LAST_FLUSHED" and is_writer:
                out["write_last_flushed"][tkey] = time_cycles
            elif zone == "WRITE_AFTER_BARRIER" and is_writer:
                out["write_after"][tkey] = time_cycles
            elif zone == "BRISC_GO" and is_writer:
                out["brisc_go"][tkey] = time_cycles
            elif zone == "BRISC_DONE" and is_writer:
                out["brisc_done"][tkey] = time_cycles
            elif zone == "NCRISC_GO" and is_reader:
                out["ncrisc_go"][tkey] = time_cycles
            elif zone == "NCRISC_DONE" and is_reader:
                out["ncrisc_done"][tkey] = time_cycles
            elif zone == "BRISC-FW" and zone_type == "ZONE_END" and is_writer:
                out["brisc_fw_end"][tkey] = time_cycles
                current_prog = None
            elif zone == "NCRISC-FW" and zone_type == "ZONE_END" and is_reader:
                out["ncrisc_fw_end"][tkey] = time_cycles
                current_prog = None

    return out


def load_rt_profiler_csv(rt_path: Path) -> pd.DataFrame:
    if not rt_path.is_file():
        return pd.DataFrame()
    return pd.read_csv(rt_path)


def dedupe_rt_records(rt_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Collapse consecutive RT records with the same program_id (common in trace replay)."""
    if rt_df.empty:
        return rt_df, 0
    deduped_rows: list[pd.Series] = []
    collapsed = 0
    for _, group in rt_df.groupby("chip_id"):
        records = group.sort_values("go_cycles")
        kept: list[pd.Series] = []
        for _, row in records.iterrows():
            if kept and int(row["program_id"]) == int(kept[-1]["program_id"]):
                kept[-1] = row
                collapsed += 1
            else:
                kept.append(row)
        deduped_rows.extend(kept)
    if not deduped_rows:
        return pd.DataFrame(), collapsed
    return pd.DataFrame(deduped_rows), collapsed


def build_rt_dispatch_gaps(rt_df: pd.DataFrame, freq_mhz: float, min_prog_id: int) -> tuple[pd.DataFrame, int]:
    """Chip-level done (end) -> go (next start) from real-time profiler records."""
    if rt_df.empty:
        return pd.DataFrame(), 0
    rt_df, collapsed = dedupe_rt_records(rt_df)
    rows: list[dict] = []
    skipped_same_pid = 0
    for chip_id, group in rt_df.groupby("chip_id"):
        records = group.sort_values("go_cycles")
        records = records[records["program_id"] >= min_prog_id]
        go = records["go_cycles"].astype(int).tolist()
        done = records["done_cycles"].astype(int).tolist()
        pids = records["program_id"].astype(int).tolist()
        for i in range(len(pids) - 1):
            if pids[i] == pids[i + 1]:
                skipped_same_pid += 1
                continue
            gap_cycles = go[i + 1] - done[i]
            if gap_cycles < 0:
                continue
            # Steady-state only: consecutive program ids (skip trace wrap e.g. 8→3).
            if pids[i + 1] != pids[i] + 1:
                continue
            ghz = rt_freq_ghz(rt_df, freq_mhz)
            gap_ns = rt_cycles_to_ns(gap_cycles, ghz)
            rows.append(
                {
                    "chip": chip_id,
                    "from_prog_id": pids[i],
                    "to_prog_id": pids[i + 1],
                    "done_cycles": done[i],
                    "go_cycles": go[i + 1],
                    "dispatch_gap_cycles": gap_cycles,
                    "dispatch_gap_ns": gap_ns,
                    "dispatch_gap_us": gap_ns / 1000.0,
                }
            )
    return pd.DataFrame(rows), collapsed + skipped_same_pid


def rt_freq_ghz(rt_df: pd.DataFrame, fallback_mhz: float) -> float:
    if rt_df.empty or "frequency_ghz" not in rt_df.columns:
        return fallback_mhz / 1000.0
    ghz = float(rt_df["frequency_ghz"].iloc[0])
    return ghz if ghz > 0.0 else fallback_mhz / 1000.0


def rt_freq_mhz(rt_df: pd.DataFrame, fallback_mhz: float) -> float:
    return rt_freq_ghz(rt_df, fallback_mhz) * 1000.0


def rt_cycles_to_ns(cycles: int, ghz: float) -> float:
    """RT profiler: duration_ns = cycles / frequency_ghz (see test RT CSV writer)."""
    return float(cycles) / ghz if ghz > 0.0 else 0.0


def build_rt_program_table(rt_df: pd.DataFrame, min_prog_id: int, freq_mhz: float) -> pd.DataFrame:
    """Per-program chip dispatch timestamps: go (start) and done (end)."""
    if rt_df.empty:
        return pd.DataFrame()
    rt_df, _ = dedupe_rt_records(rt_df)
    ghz = rt_freq_ghz(rt_df, freq_mhz)
    rows: list[dict] = []
    for chip_id, group in rt_df.groupby("chip_id"):
        records = group.sort_values("go_cycles")
        records = records[records["program_id"] >= min_prog_id]
        for _, row in records.iterrows():
            go = int(row["go_cycles"])
            done = int(row["done_cycles"])
            duration_cycles = done - go if done >= go else 0
            if "duration_ns" in row and pd.notna(row["duration_ns"]):
                duration_ns = float(row["duration_ns"])
            else:
                duration_ns = rt_cycles_to_ns(duration_cycles, ghz)
            gap_next = (
                int(row["gap_to_next_go_cycles"])
                if "gap_to_next_go_cycles" in row.index and pd.notna(row["gap_to_next_go_cycles"])
                else 0
            )
            gap_next_ns = rt_cycles_to_ns(gap_next, ghz) if gap_next > 0 else 0.0
            rows.append(
                {
                    "chip": chip_id,
                    "program_id": int(row["program_id"]),
                    "go_cycles": go,
                    "done_cycles": done,
                    "program_duration_cycles": duration_cycles,
                    "program_duration_ns": duration_ns,
                    "program_duration_us": duration_ns / 1000.0,
                    "gap_to_next_go_cycles": gap_next,
                    "gap_to_next_go_ns": gap_next_ns,
                    "frequency_ghz": ghz,
                }
            )
    return pd.DataFrame(rows)


def merge_dispatch_into_timeline(
    timeline: pd.DataFrame, rt_gaps: pd.DataFrame, rt_df: pd.DataFrame, device_freq_mhz: float
) -> pd.DataFrame:
    """Attach chip-level dispatch done/go stamps to each per-core transition row."""
    if timeline.empty:
        return timeline
    out = timeline.copy()
    for col in (
        "chip_dispatch_done_cycles",
        "chip_dispatch_go_cycles",
        "chip_dispatch_gap_cycles",
        "chip_dispatch_gap_ns",
        "chip_dispatch_gap_us",
    ):
        out[col] = pd.NA
    if rt_gaps.empty:
        return out
    lookup = {(int(r.chip), int(r.from_prog_id), int(r.to_prog_id)): r for _, r in rt_gaps.iterrows()}
    for idx, row in out.iterrows():
        key = (int(row["chip"]), int(row["from_prog_id"]), int(row["to_prog_id"]))
        if key not in lookup:
            continue
        d = lookup[key]
        done_c = int(d["done_cycles"])
        go_c = int(d["go_cycles"])
        gap_c = int(d["dispatch_gap_cycles"])
        out.at[idx, "chip_dispatch_done_cycles"] = done_c
        out.at[idx, "chip_dispatch_go_cycles"] = go_c
        out.at[idx, "chip_dispatch_gap_cycles"] = gap_c
        gap_ns = float(d["dispatch_gap_ns"]) if "dispatch_gap_ns" in d else float(d["dispatch_gap_us"]) * 1000.0
        out.at[idx, "chip_dispatch_gap_ns"] = gap_ns
        out.at[idx, "chip_dispatch_gap_us"] = gap_ns / 1000.0

    # NOTE on BRISC firmware prelude:
    #   RT-profiler chip_dispatch_go_cycles and device-profiler brisc_go_cycles
    #   are on different cycle counters / epochs and CANNOT be subtracted to
    #   get a per-core "FW prelude" without an explicit sync offset (which
    #   the current CSVs don't expose). Use `brisc_done_to_go_us` instead —
    #   that is per-core BRISC kernel-exit(k) → BRISC kernel-entry(k+1) on
    #   the same device-cycle counter, which is the per-core firmware +
    #   dispatch transition window we actually want.
    return out


def build_timeline(gaps: pd.DataFrame, barriers: dict[str, dict[tuple, int]], freq_mhz: float) -> pd.DataFrame:
    """Merge pack→unpack gaps with reader/writer barrier timestamps per transition."""
    if gaps.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for _, row in gaps.iterrows():
        chip, core_x, core_y = int(row.chip), int(row.core_x), int(row.core_y)
        from_prog, to_prog = int(row.from_prog_id), int(row.to_prog_id)
        k_end = (chip, core_x, core_y, from_prog)
        k_start = (chip, core_x, core_y, to_prog)
        rec = row.to_dict()
        rec["write_before_barrier_cycles"] = barriers["write_before"].get(k_end)
        rec["write_after_barrier_cycles"] = barriers["write_after"].get(k_end)
        rec["read_before_barrier_cycles"] = barriers["read_before"].get(k_start)
        rec["read_after_barrier_cycles"] = barriers["read_after"].get(k_start)
        rec["brisc_done_cycles"] = barriers["brisc_done"].get(k_end)
        rec["brisc_go_cycles"] = barriers["brisc_go"].get(k_start)
        rec["ncrisc_done_cycles"] = barriers["ncrisc_done"].get(k_end)
        rec["ncrisc_go_cycles"] = barriers["ncrisc_go"].get(k_start)
        # FW-level zone bounds: FW.start fires BEFORE our user BRISC_GO marker
        # (FW kernel-launch dispatch path runs in between). FW done->go is a
        # tighter "done/go" interval: prev FW signaled done -> next FW received
        # GO and entered scope. Excludes our pre-marker user code (~75 ns) and
        # the FW kernel-launch prelude (~0.5-1 us) that inflates user done->go.
        rec["brisc_fw_start_cycles"] = barriers["brisc_fw_start"].get(k_start)
        rec["brisc_fw_end_cycles"] = barriers["brisc_fw_end"].get(k_end)
        rec["ncrisc_fw_start_cycles"] = barriers["ncrisc_fw_start"].get(k_start)
        rec["ncrisc_fw_end_cycles"] = barriers["ncrisc_fw_end"].get(k_end)
        bd = rec.get("brisc_done_cycles")
        bg = rec.get("brisc_go_cycles")
        if bd is not None and bg is not None and not (pd.isna(bd) or pd.isna(bg)):
            rec["brisc_done_to_go_cycles"] = int(bg) - int(bd)
            rec["brisc_done_to_go_us"] = rec["brisc_done_to_go_cycles"] / freq_mhz
        bfs = rec.get("brisc_fw_start_cycles")
        bfe = rec.get("brisc_fw_end_cycles")
        if bfs is not None and bfe is not None and not (pd.isna(bfs) or pd.isna(bfe)):
            rec["brisc_fw_done_to_go_cycles"] = int(bfs) - int(bfe)
            rec["brisc_fw_done_to_go_us"] = rec["brisc_fw_done_to_go_cycles"] / freq_mhz
        rb = rec.get("read_before_barrier_cycles")
        ra = rec.get("read_after_barrier_cycles")
        if rb is not None and ra is not None and not (pd.isna(rb) or pd.isna(ra)):
            rb_i, ra_i = int(rb), int(ra)
            rec["read_barrier_cycles"] = ra_i - rb_i
            rec["read_barrier_us"] = rec["read_barrier_cycles"] / freq_mhz
        pf = rec.get("pack_finish_cycles")
        if pf is not None and ra is not None and not (pd.isna(pf) or pd.isna(ra)):
            rec["pack_finish_to_read_after_cycles"] = int(ra) - int(pf)
            rec["pack_finish_to_read_after_us"] = rec["pack_finish_to_read_after_cycles"] / freq_mhz
        wb = rec.get("write_before_barrier_cycles")
        wa = rec.get("write_after_barrier_cycles")
        if wb is not None and wa is not None and not (pd.isna(wb) or pd.isna(wa)):
            rec["write_barrier_cycles"] = int(wa) - int(wb)
            rec["write_barrier_us"] = rec["write_barrier_cycles"] / freq_mhz
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty and "gap_us" in out.columns:
        for col in out.columns:
            if col.endswith("_cycles") and col != "gap_cycles":
                us_col = col.replace("_cycles", "_us")
                if us_col not in out.columns and out[col].notna().any():
                    out[us_col] = out[col] / freq_mhz
    return out


def add_buffering_context_columns(
    timeline: pd.DataFrame,
    freq_mhz: float,
    *,
    input_cb_depth_tiles: int,
    tiles_per_core: int,
    reader_push_tile_count: int,
    reader_batch_push: bool,
) -> pd.DataFrame:
    """Annotate timeline for HW review: CB config + derived buffering vs full op-to-op gap."""
    if timeline.empty:
        return timeline
    out = timeline.copy()
    out["input_cb_depth_tiles"] = input_cb_depth_tiles
    out["tiles_per_core"] = tiles_per_core
    out["reader_push_tiles_per_chunk"] = reader_push_tile_count
    out["reader_batch_push"] = reader_batch_push
    out["input_cb_double_buffered"] = input_cb_depth_tiles >= 2
    out["per_tile_dram_read_barrier"] = True

    # Tile 0 of program k+1: DRAM read + noc_async_read_barrier (READ_BEFORE → READ_AFTER).
    if "read_barrier_us" in out.columns:
        out["tile0_dram_buffering_latency_us"] = out["read_barrier_us"]
    else:
        out["tile0_dram_buffering_latency_us"] = pd.NA

    if "read_after_barrier_cycles" in out.columns and "unpack_tile0_start_cycles" in out.columns:
        ra = out["read_after_barrier_cycles"]
        u0 = out["unpack_tile0_start_cycles"]
        mask = ra.notna() & u0.notna()
        out["read_after_to_unpack_tile0_us"] = pd.NA
        out.loc[mask, "read_after_to_unpack_tile0_us"] = (u0[mask].astype(int) - ra[mask].astype(int)) / freq_mhz

    if "gap_us" in out.columns and "tile0_dram_buffering_latency_us" in out.columns:
        buf = out["tile0_dram_buffering_latency_us"]
        gap = out["gap_us"]
        mask = buf.notna() & gap.notna()
        out["device_gap_excluding_tile0_dram_buffer_us"] = pd.NA
        out.loc[mask, "device_gap_excluding_tile0_dram_buffer_us"] = gap[mask] - buf[mask]
        out["tile0_dram_buffer_fraction_of_device_gap"] = pd.NA
        out.loc[mask & (gap[mask] > 0), "tile0_dram_buffer_fraction_of_device_gap"] = buf[mask] / gap[mask]

    return out


def print_buffering_summary(timeline: pd.DataFrame) -> None:
    if timeline.empty or "tile0_dram_buffering_latency_us" not in timeline.columns:
        return
    buf = timeline["tile0_dram_buffering_latency_us"].dropna()
    if buf.empty:
        return
    depth = int(timeline["input_cb_depth_tiles"].iloc[0])
    tiles = int(timeline["tiles_per_core"].iloc[0])
    push = (
        int(timeline["reader_push_tiles_per_chunk"].iloc[0]) if "reader_push_tiles_per_chunk" in timeline.columns else 2
    )
    batch = bool(timeline["reader_batch_push"].iloc[0]) if "reader_batch_push" in timeline.columns else False
    print(
        f"Buffering context: input CB depth={depth} tiles, {tiles} tiles/core, reader_push={push}, "
        f"reader_batch_push={batch}, per-tile DRAM read barrier=True"
    )
    print(
        f"  tile0_dram_buffering_latency_us (prog k+1 tile 0 read+barrier): "
        f"median={buf.median():.3f} mean={buf.mean():.3f} min={buf.min():.3f} max={buf.max():.3f}"
    )
    if "device_gap_excluding_tile0_dram_buffer_us" in timeline.columns:
        rest = timeline["device_gap_excluding_tile0_dram_buffer_us"].dropna()
        if not rest.empty:
            print(
                f"  device_gap_excluding_tile0_dram_buffer_us (pack finish→unpack tile0 minus above): "
                f"median={rest.median():.3f} mean={rest.mean():.3f}"
            )
    if "read_after_to_unpack_tile0_us" in timeline.columns:
        handoff = timeline["read_after_to_unpack_tile0_us"].dropna()
        if not handoff.empty:
            print(
                f"  read_after_to_unpack_tile0_us (buffer ready→compute TILE_IDX): "
                f"median={handoff.median():.3f} mean={handoff.mean():.3f}"
            )


def compute_gaps(
    pack_finish: dict[tuple, int],
    unpack_tile0_start: dict[tuple, int],
    min_prog_id: int,
) -> pd.DataFrame:
    cores = {(k[0], k[1], k[2]) for k in pack_finish} | {(k[0], k[1], k[2]) for k in unpack_tile0_start}
    gap_rows: list[dict] = []

    for chip, core_x, core_y in sorted(cores):
        prog_ids = sorted(
            {p for (c, x, y, p) in pack_finish if (c, x, y) == (chip, core_x, core_y) and p >= min_prog_id}
            | {p for (c, x, y, p) in unpack_tile0_start if (c, x, y) == (chip, core_x, core_y) and p >= min_prog_id}
        )
        for from_prog, to_prog in zip(prog_ids, prog_ids[1:]):
            end_key = (chip, core_x, core_y, from_prog)
            start_key = (chip, core_x, core_y, to_prog)
            if end_key not in pack_finish or start_key not in unpack_tile0_start:
                continue
            end_cycles = pack_finish[end_key]
            start_cycles = unpack_tile0_start[start_key]
            gap_rows.append(
                {
                    "chip": chip,
                    "core_x": core_x,
                    "core_y": core_y,
                    "from_prog_id": from_prog,
                    "to_prog_id": to_prog,
                    "pack_finish_cycles": end_cycles,
                    "unpack_tile0_start_cycles": start_cycles,
                    "gap_cycles": start_cycles - end_cycles,
                }
            )

    return pd.DataFrame(gap_rows)


def build_prog_table(df: pd.DataFrame) -> pd.DataFrame:
    progs = df[(df["zone name"] == "PROG_ID") & (df["RISC processor type"] == UNPACK_RISC)].copy()
    if progs.empty:
        progs = df[df["zone name"] == "PROG_ID"].copy()
    rows = []
    for _, row in progs.iterrows():
        rows.append(
            {
                "chip": row["PCIe slot"],
                "core_x": row["core_x"],
                "core_y": row["core_y"],
                "risc": row["RISC processor type"],
                "prog_id": int(row["data"]),
                "time_cycles": int(row["time[cycles since reset]"]),
            }
        )
    return pd.DataFrame(rows)


def add_gap_us(gaps: pd.DataFrame, freq_mhz: float) -> pd.DataFrame:
    out = gaps.copy()
    if not out.empty:
        out["gap_us"] = out["gap_cycles"] / freq_mhz
    return out


def build_per_core_summary(gaps: pd.DataFrame) -> pd.DataFrame:
    """Mean/min/max gap per (core_x, core_y) across program transitions."""
    if gaps.empty:
        return pd.DataFrame()
    summary = (
        gaps.groupby(["chip", "core_x", "core_y"], as_index=False)
        .agg(
            num_transitions=("gap_us", "count"),
            mean_gap_us=("gap_us", "mean"),
            min_gap_us=("gap_us", "min"),
            max_gap_us=("gap_us", "max"),
            mean_gap_cycles=("gap_cycles", "mean"),
        )
        .sort_values(["core_x", "core_y"])
    )
    for col in ("mean_gap_us", "min_gap_us", "max_gap_us", "mean_gap_cycles"):
        summary[col] = summary[col].round(3)
    return summary


def build_chip_summary(gaps: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across all cores and all gap rows."""
    if gaps.empty:
        return pd.DataFrame()
    us = gaps["gap_us"]
    summary = pd.DataFrame(
        [
            {
                "num_cores": int(gaps.groupby(["core_x", "core_y"]).ngroups),
                "num_gap_rows": len(gaps),
                "mean_gap_us": round(float(us.mean()), 3),
                "median_gap_us": round(float(us.median()), 3),
                "min_gap_us": round(float(us.min()), 3),
                "max_gap_us": round(float(us.max()), 3),
                "std_gap_us": round(float(us.std()), 3) if len(us) > 1 else 0.0,
            }
        ]
    )
    return summary


def print_done_go_validation_summary(
    timeline: pd.DataFrame, rt_gaps: pd.DataFrame, min_prog_id: int, freq_mhz: float
) -> None:
    """Summarize done/go metrics: chip dispatch (WH ~500-800ns) vs per-core BRISC FW."""
    print(f"Done/go validation (min_prog_id>={min_prog_id}, freq={freq_mhz:.1f} MHz):")
    if not rt_gaps.empty and "dispatch_gap_us" in rt_gaps.columns:
        dg = rt_gaps["dispatch_gap_us"]
        dg_ns = dg * 1000.0
        print(
            f"  chip_dispatch done→go (RT): median={dg.median():.3f} us ({dg_ns.median():.1f} ns) "
            f"min={dg.min():.3f} max={dg.max():.3f} n={len(dg)}"
        )
    else:
        print("  chip_dispatch done→go (RT): no data")
    if not timeline.empty and "brisc_done_to_go_us" in timeline.columns:
        bg = pd.to_numeric(timeline["brisc_done_to_go_us"], errors="coerce").dropna()
        if not bg.empty:
            bg_ns = bg * 1000.0
            print(
                f"  brisc_done_to_go (per-core USER markers, includes ~0.5-1us FW prelude): "
                f"median={bg.median():.3f} us ({bg_ns.median():.1f} ns) "
                f"min={bg.min():.3f} max={bg.max():.3f} n={len(bg)}"
            )
    if not timeline.empty and "brisc_fw_done_to_go_us" in timeline.columns:
        fbg = pd.to_numeric(timeline["brisc_fw_done_to_go_us"], errors="coerce").dropna()
        if not fbg.empty:
            fbg_ns = fbg * 1000.0
            print(
                f"  brisc_FW_done_to_go (per-core FW worker round-trip): "
                f"median={fbg.median():.3f} us ({fbg_ns.median():.1f} ns) "
                f"min={fbg.min():.3f} max={fbg.max():.3f} n={len(fbg)}"
            )
    if not timeline.empty and "dg_first_ns" in timeline.columns:
        for col, label in [
            ("dg_issue_ns", "dispatch issue only  "),
            ("dg_first_ns", "first worker sees GO "),
            ("dg_median_ns", "median worker sees GO"),
            ("dg_last_ns", "last worker sees GO  "),
        ]:
            v = pd.to_numeric(timeline[col], errors="coerce").dropna()
            if v.empty:
                continue
            print(
                f"  done_to_go [{label}]: "
                f"median={v.median():.0f} ns  min={v.min():.0f}  max={v.max():.0f}  n={len(v)}"
            )
    if not timeline.empty and "gap_us" in timeline.columns:
        gap = pd.to_numeric(timeline["gap_us"], errors="coerce").dropna()
        if not gap.empty:
            print(
                f"  op2op gap_us (pack finish→unpack tile0): median={gap.median():.3f} us "
                f"min={gap.min():.3f} max={gap.max():.3f} n={len(gap)}"
            )


def print_dispatch_summary(rt_programs: pd.DataFrame, rt_gaps: pd.DataFrame) -> None:
    if rt_programs.empty and rt_gaps.empty():
        print("No real-time profiler data (run test with --use-realtime-profiler).", file=sys.stderr)
        return
    print(
        "Chip dispatch (RT profiler): go/done at chip enqueue; device BRISC_GO/DONE and NCRISC_GO/DONE "
        "are per-core dataflow kernel entry/exit in the CSV"
    )
    if not rt_programs.empty:
        for _, row in rt_programs.iterrows():
            gap_str = ""
            if int(row.get("gap_to_next_go_cycles", 0)) > 0:
                gap_str = f", gap_to_next_go={int(row.gap_to_next_go_cycles)} cycles ({row.gap_to_next_go_ns:.1f} ns)"
            print(
                f"  chip {int(row.chip)} prog {int(row.program_id)}: "
                f"go_cycles={int(row.go_cycles)} done_cycles={int(row.done_cycles)}, "
                f"duration={row.program_duration_ns:.1f} ns ({row.program_duration_us:.3f} us){gap_str}"
            )
    if not rt_gaps.empty:
        for _, row in rt_gaps.iterrows():
            gap_ns = (
                float(row["dispatch_gap_ns"]) if "dispatch_gap_ns" in row else float(row["dispatch_gap_us"]) * 1000.0
            )
            print(
                f"  transition {int(row.from_prog_id)}→{int(row.to_prog_id)}: "
                f"done→go gap={int(row.dispatch_gap_cycles)} cycles ({gap_ns:.1f} ns)"
            )


def aggregate_multi_run_summaries(runs_dir: Path, output_dir: Path) -> None:
    """Read per-run summary CSVs under runs_dir/run_* and write combined stats."""
    run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_"))
    if not run_dirs:
        print(f"No run_* directories under {runs_dir}", file=sys.stderr)
        return

    device_rows: list[dict] = []
    dispatch_rows: list[dict] = []
    for run_dir in run_dirs:
        run_name = run_dir.name
        chip_path = next(run_dir.glob("*_op_to_op_summary_chip.csv"), None)
        dispatch_path = next(run_dir.glob("*_op_to_op_dispatch_gaps.csv"), None)
        if chip_path and chip_path.is_file():
            chip = pd.read_csv(chip_path).iloc[0]
            device_rows.append({"run": run_name, **chip.to_dict()})
        if dispatch_path and dispatch_path.is_file():
            disp = pd.read_csv(dispatch_path)
            for _, row in disp.iterrows():
                dispatch_rows.append({"run": run_name, **row.to_dict()})

    if device_rows:
        device_df = pd.DataFrame(device_rows)
        out_path = output_dir / "multi_run_device_op_to_op_gap.csv"
        device_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(device_df)} runs)")
        for col in ("mean_gap_us", "median_gap_us"):
            if col in device_df.columns:
                s = device_df[col]
                print(
                    f"  device {col} across runs: min={s.min():.3f} max={s.max():.3f} "
                    f"mean={s.mean():.3f} std={s.std():.3f}"
                )

    if dispatch_rows:
        dispatch_df = pd.DataFrame(dispatch_rows)
        out_path = output_dir / "multi_run_dispatch_done_to_go.csv"
        dispatch_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(dispatch_df)} rows)")
        if "dispatch_gap_us" in dispatch_df.columns:
            s = dispatch_df["dispatch_gap_us"]
            print(
                f"  dispatch done→go us across runs: min={s.min():.3f} max={s.max():.3f} "
                f"mean={s.mean():.3f} std={s.std():.3f}"
            )


def print_gap_summary(gaps: pd.DataFrame, freq_mhz: float) -> None:
    if gaps.empty:
        print("No op-to-op gaps (need TRISC_2 FINISH_LAST_PUSH and TRISC_0 TILE_IDX tile 0).", file=sys.stderr)
        return

    print(f"Chip frequency: {freq_mhz:.3f} MHz")
    print(f"Gap = TRISC_0 TILE_IDX (tile 0) of program k+1 − TRISC_2 FINISH_LAST_PUSH of program k")
    print(f"Cores with gaps: {gaps.groupby(['core_x', 'core_y']).ngroups}")
    print(f"Total gap rows: {len(gaps)}")

    us = gaps["gap_cycles"] / freq_mhz
    print(f"  gap_us min={us.min():.2f} max={us.max():.2f} mean={us.mean():.2f} median={us.median():.2f}")

    first = gaps.sort_values(["core_x", "core_y", "from_prog_id"]).iloc[0]
    print(
        f"Example core ({int(first.core_x)},{int(first.core_y)}): "
        f"prog {int(first.from_prog_id)}→{int(first.to_prog_id)} "
        f"gap={int(first.gap_cycles)} cycles ({first.gap_cycles / freq_mhz:.2f} us)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export op-to-op latency CSVs from device profiler log.")
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument(
        "--rt-input-file", type=Path, default=None, help="profile_log_device_rt.csv from --use-realtime-profiler"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--min-prog-id",
        type=int,
        default=1,
        help="Exclude program ids below this (default 1 skips pre-compile PROG_ID=0)",
    )
    parser.add_argument(
        "--aggregate-runs-dir",
        type=Path,
        default=None,
        help="Parent directory containing run_1, run_2, ... subdirs; write multi-run summary CSVs and exit",
    )
    parser.add_argument(
        "--input-cb-depth-tiles",
        type=int,
        default=4,
        help="Input circular buffer depth in tiles (benchmark default: 4 = 2x --reader-push-tiles)",
    )
    parser.add_argument(
        "--tiles-per-core",
        type=int,
        default=4,
        help="Tiles processed per core per program (benchmark --num-pages-per-core default)",
    )
    parser.add_argument(
        "--reader-push-tiles",
        type=int,
        default=2,
        help="Reader reserve/read/push chunk size (benchmark --reader-push-tiles)",
    )
    parser.add_argument(
        "--reader-batch-push",
        action="store_true",
        help="Set if benchmark was run with --reader-batch-push",
    )
    args = parser.parse_args()

    import os

    if args.aggregate_runs_dir is not None:
        runs_dir = args.aggregate_runs_dir.resolve()
        out_dir = args.output_dir or runs_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        aggregate_multi_run_summaries(runs_dir, out_dir)
        return 0

    log_path = args.input_file or Path(os.environ.get("TT_METAL_HOME", ".")) / DEFAULT_LOG
    log_path = log_path.resolve()
    if not log_path.is_file():
        print(f"Input not found: {log_path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or log_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = log_path.stem

    freq_mhz = parse_chip_freq_mhz(log_path)
    df = load_profiler_csv(log_path)

    pack_finish, unpack_tile0_start, tiles = walk_core_markers(df)
    barriers = walk_barrier_markers(df)
    gaps = compute_gaps(pack_finish, unpack_tile0_start, args.min_prog_id)
    gaps_out = add_gap_us(gaps, freq_mhz)
    timeline = build_timeline(gaps_out, barriers, freq_mhz)
    progs = build_prog_table(df)

    rt_path = args.rt_input_file or (log_path.parent / "profile_log_device_rt.csv")
    rt_df = load_rt_profiler_csv(rt_path.resolve())
    rt_gaps, rt_skipped = build_rt_dispatch_gaps(rt_df, freq_mhz, args.min_prog_id)
    rt_programs = build_rt_program_table(rt_df, args.min_prog_id, freq_mhz)
    timeline = merge_dispatch_into_timeline(timeline, rt_gaps, rt_df, freq_mhz)
    dg_table = walk_done_to_go(df, freq_mhz, args.min_prog_id)
    if not dg_table.empty and not timeline.empty:
        # Map per (chip, to_prog_id) -> done/go for that transition's GO.
        # We surface four columns:
        #  - dg_issue_ns   : dispatch-only MCAST issue cost (~30 ns BH)
        #  - dg_first_ns   : nearest worker observes GO
        #  - dg_median_ns  : median worker observes GO
        #  - dg_last_ns    : farthest worker observes GO
        lookup = {(int(r.chip), int(r.program_id)): r for _, r in dg_table.iterrows()}
        for col in ["dg_issue_ns", "dg_first_ns", "dg_median_ns", "dg_last_ns"]:
            timeline[col] = timeline.apply(
                lambda r: lookup.get((int(r["chip"]), int(r["to_prog_id"])), pd.Series()).get(col, pd.NA),
                axis=1,
            )
    timeline = add_buffering_context_columns(
        timeline,
        freq_mhz,
        input_cb_depth_tiles=args.input_cb_depth_tiles,
        tiles_per_core=args.tiles_per_core,
        reader_push_tile_count=args.reader_push_tiles,
        reader_batch_push=args.reader_batch_push,
    )
    if rt_skipped > 0:
        print(
            f"RT dispatch: skipped {rt_skipped} duplicate program_id transition(s) "
            "(trace replay may emit repeated host runtime ids; use rows where from_prog_id < to_prog_id)",
            file=sys.stderr,
        )

    tiles_path = output_dir / f"{stem}_op_to_op_tiles.csv"
    gaps_path = output_dir / f"{stem}_op_to_op_gaps.csv"
    timeline_path = output_dir / f"{stem}_op_to_op_timeline.csv"
    progs_path = output_dir / f"{stem}_op_to_op_prog_ids.csv"
    summary_chip_path = output_dir / f"{stem}_op_to_op_summary_chip.csv"
    summary_per_core_path = output_dir / f"{stem}_op_to_op_summary_per_core.csv"
    rt_gaps_path = output_dir / f"{stem}_op_to_op_dispatch_gaps.csv"
    rt_programs_path = output_dir / f"{stem}_op_to_op_rt_programs.csv"
    complete_path = output_dir / f"{stem}_op_to_op_complete.csv"

    tiles.to_csv(tiles_path, index=False)
    gaps_out.to_csv(gaps_path, index=False)
    timeline.to_csv(timeline_path, index=False)
    timeline.to_csv(complete_path, index=False)
    progs.to_csv(progs_path, index=False)
    build_chip_summary(gaps_out).to_csv(summary_chip_path, index=False)
    build_per_core_summary(gaps_out).to_csv(summary_per_core_path, index=False)
    if not rt_gaps.empty:
        rt_gaps.to_csv(rt_gaps_path, index=False)
    if not rt_programs.empty:
        rt_programs.to_csv(rt_programs_path, index=False)

    print(f"Wrote {summary_chip_path}")
    print(f"Wrote {summary_per_core_path}")
    print(f"Wrote {gaps_path} ({len(gaps_out)} rows)")
    print(f"Wrote {timeline_path} ({len(timeline)} rows, per-core + barriers + chip dispatch go/done)")
    print(f"Wrote {complete_path} (same columns as timeline — full picture for HW review)")
    print(f"Wrote {tiles_path} ({len(tiles)} rows)")
    print(f"Wrote {progs_path} ({len(progs)} rows)")
    if not rt_gaps.empty:
        print(f"Wrote {rt_gaps_path} ({len(rt_gaps)} rows, chip-level done→go)")
    elif args.rt_input_file or rt_path.is_file():
        print(f"No RT dispatch gaps (check {rt_path})", file=sys.stderr)
    else:
        print("No RT CSV (re-run test with --use-realtime-profiler)", file=sys.stderr)
    if not rt_programs.empty:
        print(f"Wrote {rt_programs_path} ({len(rt_programs)} programs, go/done per enqueue)")
    print()
    print_gap_summary(gaps_out, freq_mhz)
    if not gaps_out.empty:
        chip = build_chip_summary(gaps_out)
        print(f"  All cores mean_gap_us={chip.iloc[0]['mean_gap_us']:.3f} median={chip.iloc[0]['median_gap_us']:.3f}")
    print()
    print_buffering_summary(timeline)
    print()
    print_done_go_validation_summary(timeline, rt_gaps, args.min_prog_id, freq_mhz)
    print()
    print_dispatch_summary(rt_programs, rt_gaps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
