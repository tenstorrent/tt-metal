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
UNPACK_RISC = "TRISC_0"
PACK_RISC = "TRISC_2"


def parse_chip_freq_mhz(log_path: Path) -> float:
    with log_path.open() as f:
        header = f.readline()
    match = re.search(r"CHIP_FREQ\[MHz\]:\s*([0-9.]+)", header)
    if not match:
        raise ValueError(f"Could not parse CHIP_FREQ from first line of {log_path}")
    return float(match.group(1))


def load_profiler_csv(log_path: Path) -> pd.DataFrame:
    df = pd.read_csv(log_path, skiprows=1)
    df.columns = df.columns.str.strip()
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
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--min-prog-id",
        type=int,
        default=1,
        help="Exclude program ids below this (default 1 skips pre-compile PROG_ID=0)",
    )
    args = parser.parse_args()

    import os

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
    gaps = compute_gaps(pack_finish, unpack_tile0_start, args.min_prog_id)
    gaps_out = add_gap_us(gaps, freq_mhz)
    progs = build_prog_table(df)

    tiles_path = output_dir / f"{stem}_op_to_op_tiles.csv"
    gaps_path = output_dir / f"{stem}_op_to_op_gaps.csv"
    progs_path = output_dir / f"{stem}_op_to_op_prog_ids.csv"
    summary_chip_path = output_dir / f"{stem}_op_to_op_summary_chip.csv"
    summary_per_core_path = output_dir / f"{stem}_op_to_op_summary_per_core.csv"

    tiles.to_csv(tiles_path, index=False)
    gaps_out.to_csv(gaps_path, index=False)
    progs.to_csv(progs_path, index=False)
    build_chip_summary(gaps_out).to_csv(summary_chip_path, index=False)
    build_per_core_summary(gaps_out).to_csv(summary_per_core_path, index=False)

    print(f"Wrote {summary_chip_path}")
    print(f"Wrote {summary_per_core_path}")
    print(f"Wrote {gaps_path} ({len(gaps_out)} rows)")
    print(f"Wrote {tiles_path} ({len(tiles)} rows)")
    print(f"Wrote {progs_path} ({len(progs)} rows)")
    print()
    print_gap_summary(gaps_out, freq_mhz)
    if not gaps_out.empty:
        chip = build_chip_summary(gaps_out)
        print(f"  All cores mean_gap_us={chip.iloc[0]['mean_gap_us']:.3f} median={chip.iloc[0]['median_gap_us']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
