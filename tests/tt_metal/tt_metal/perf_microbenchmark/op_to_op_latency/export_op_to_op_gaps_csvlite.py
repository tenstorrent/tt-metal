#!/usr/bin/env python3
"""Pandas-free export of op-to-op gap + done/go metrics from profile_log_device.csv."""

from __future__ import annotations

import argparse
import csv
import re
import statistics as st
import sys
from pathlib import Path

UNPACK_RISC = "TRISC_0"
PACK_RISC = "TRISC_2"


def parse_chip_freq_mhz(log_path: Path) -> float:
    with log_path.open() as f:
        header = f.readline()
    match = re.search(r"CHIP_FREQ\[MHz\]:\s*([0-9.]+)", header)
    if not match:
        raise ValueError(f"Could not parse CHIP_FREQ from {log_path}")
    return float(match.group(1))


def iter_markers(log_path: Path):
    with log_path.open(newline="") as f:
        f.readline()
        for row in csv.DictReader(f):
            yield row


def walk_core_markers(log_path: Path):
    pack_finish: dict[tuple, int] = {}
    unpack_tile0_start: dict[tuple, int] = {}
    cur_prog: dict[tuple, int | None] = {}

    for row in iter_markers(log_path):
        zone = row["zone name"].strip()
        risc = row["RISC processor type"].strip()
        if zone not in ("PROG_ID", "TILE_IDX", "FINISH_LAST_PUSH"):
            continue
        if risc not in (UNPACK_RISC, PACK_RISC):
            continue

        chip = int(row["PCIe slot"])
        core_x = int(row["core_x"])
        core_y = int(row["core_y"])
        t = int(row["time[cycles since reset]"])
        data = int(float(row["data"])) if row["data"] else 0
        key = (chip, core_x, core_y, risc)

        if zone == "PROG_ID" and risc == UNPACK_RISC:
            cur_prog[(chip, core_x, core_y)] = data
        elif zone == "TILE_IDX" and risc == UNPACK_RISC and data == 0:
            p = cur_prog.get((chip, core_x, core_y))
            if p is not None:
                unpack_tile0_start[(chip, core_x, core_y, p)] = t
        elif zone == "FINISH_LAST_PUSH" and risc == PACK_RISC:
            p = cur_prog.get((chip, core_x, core_y))
            if p is None:
                p = data
            pack_finish[(chip, core_x, core_y, p)] = t

    return pack_finish, unpack_tile0_start


def compute_gaps(
    pack_finish: dict[tuple, int],
    unpack_tile0_start: dict[tuple, int],
    min_prog_id: int,
) -> list[dict]:
    cores = {(k[0], k[1], k[2]) for k in pack_finish} | {(k[0], k[1], k[2]) for k in unpack_tile0_start}
    rows: list[dict] = []

    for chip, core_x, core_y in sorted(cores):
        prog_ids = sorted(
            {p for c, x, y, p in pack_finish if (c, x, y) == (chip, core_x, core_y) and p >= min_prog_id}
            | {p for c, x, y, p in unpack_tile0_start if (c, x, y) == (chip, core_x, core_y) and p >= min_prog_id}
        )
        for from_prog, to_prog in zip(prog_ids, prog_ids[1:]):
            end_key = (chip, core_x, core_y, from_prog)
            start_key = (chip, core_x, core_y, to_prog)
            if end_key not in pack_finish or start_key not in unpack_tile0_start:
                continue
            end_cycles = pack_finish[end_key]
            start_cycles = unpack_tile0_start[start_key]
            rows.append(
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
    return rows


def walk_done_to_go(log_path: Path, freq_mhz: float, min_prog_id: int) -> dict[tuple[int, int], dict]:
    disp_done: list[tuple[int, int, int]] = []
    disp_go: list[tuple[int, int]] = []
    worker_go: list[tuple[int, int, int, int]] = []
    dispatch_cores: set[tuple[int, int, int]] = set()

    for row in iter_markers(log_path):
        zone = row["zone name"].strip()
        if zone not in ("DISP_DONE_OBSERVED", "DISP_GO_ISSUED", "WORKER_GO_OBSERVED"):
            continue
        chip = int(row["PCIe slot"])
        core_x = int(row["core_x"])
        core_y = int(row["core_y"])
        t = int(row["time[cycles since reset]"])
        data = int(float(row["data"])) if row["data"] else 0
        if zone == "DISP_DONE_OBSERVED":
            disp_done.append((chip, t, data))
            dispatch_cores.add((chip, core_x, core_y))
        elif zone == "DISP_GO_ISSUED":
            disp_go.append((chip, t))
        elif zone == "WORKER_GO_OBSERVED":
            if (chip, core_x, core_y) in dispatch_cores:
                continue
            worker_go.append((chip, core_x, core_y, t))

    window_cycles = int(3000.0 / 1000.0 * freq_mhz)
    out: dict[tuple[int, int], dict] = {}

    disp_done_by_chip: dict[int, list[tuple[int, int]]] = {}
    for chip, t, pid in disp_done:
        disp_done_by_chip.setdefault(chip, []).append((t, pid))

    disp_go_by_chip: dict[int, list[int]] = {}
    for chip, t in disp_go:
        disp_go_by_chip.setdefault(chip, []).append(t)

    worker_by_chip: dict[int, list[int]] = {}
    for chip, _x, _y, t in worker_go:
        worker_by_chip.setdefault(chip, []).append(t)

    for chip, events in disp_done_by_chip.items():
        wg = sorted(worker_by_chip.get(chip, []))
        dg = sorted(disp_go_by_chip.get(chip, []))
        for d_t, d_pid in sorted(events):
            if d_pid < min_prog_id:
                continue
            lo, hi = d_t, d_t + window_cycles
            deltas = [w - d_t for w in wg if lo < w < hi]
            if not deltas:
                continue
            dg_after = [g for g in dg if g > d_t]
            issue_cycles = (dg_after[0] - d_t) if dg_after else None
            out[(chip, d_pid)] = {
                "dg_issue_ns": (issue_cycles / freq_mhz * 1000.0) if issue_cycles is not None else "",
                "dg_first_ns": min(deltas) / freq_mhz * 1000.0,
                "dg_median_ns": st.median(deltas) / freq_mhz * 1000.0,
                "dg_last_ns": max(deltas) / freq_mhz * 1000.0,
            }
    return out


def export_log(log_path: Path, output_path: Path, min_prog_id: int) -> int:
    freq_mhz = parse_chip_freq_mhz(log_path)
    pack_finish, unpack_tile0 = walk_core_markers(log_path)
    gaps = compute_gaps(pack_finish, unpack_tile0, min_prog_id)
    dg_lookup = walk_done_to_go(log_path, freq_mhz, min_prog_id)

    fieldnames = [
        "chip",
        "core_x",
        "core_y",
        "from_prog_id",
        "to_prog_id",
        "gap_cycles",
        "gap_us",
        "dg_issue_ns",
        "dg_first_ns",
        "dg_median_ns",
        "dg_last_ns",
        "_dg_issue_ns",
        "_dg_first_ns",
        "_dg_median_ns",
        "_dg_last_ns",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in gaps:
            gap_us = row["gap_cycles"] / freq_mhz
            dg = dg_lookup.get((row["chip"], row["to_prog_id"]), {})
            w.writerow(
                {
                    "chip": row["chip"],
                    "core_x": row["core_x"],
                    "core_y": row["core_y"],
                    "from_prog_id": row["from_prog_id"],
                    "to_prog_id": row["to_prog_id"],
                    "gap_cycles": row["gap_cycles"],
                    "gap_us": gap_us,
                    "dg_issue_ns": dg.get("dg_issue_ns", ""),
                    "dg_first_ns": dg.get("dg_first_ns", ""),
                    "dg_median_ns": dg.get("dg_median_ns", ""),
                    "dg_last_ns": dg.get("dg_last_ns", ""),
                    "_dg_issue_ns": dg.get("dg_issue_ns", ""),
                    "_dg_first_ns": dg.get("dg_first_ns", ""),
                    "_dg_median_ns": dg.get("dg_median_ns", ""),
                    "_dg_last_ns": dg.get("dg_last_ns", ""),
                }
            )
    return len(gaps)


def median_of(values: list[float]) -> float:
    vals = [v for v in values if v == v]
    return st.median(vals) if vals else float("nan")


def aggregate_runs(run_dirs: list[Path], min_prog_id: int) -> dict[str, float]:
    op2op, dg_first, dg_median, dg_last, dg_issue = [], [], [], [], []
    for run_dir in run_dirs:
        complete = run_dir / "profile_log_device_op_to_op_complete.csv"
        log = run_dir / "profile_log_device.csv"
        if not complete.is_file() and log.is_file():
            export_log(log, complete, min_prog_id)
        if not complete.is_file():
            continue
        with complete.open(newline="") as f:
            for row in csv.DictReader(f):
                if int(float(row["from_prog_id"])) < min_prog_id:
                    continue
                op2op.append(float(row["gap_us"]))
                for src, dst in (
                    ("dg_first_ns", dg_first),
                    ("dg_median_ns", dg_median),
                    ("dg_last_ns", dg_last),
                    ("dg_issue_ns", dg_issue),
                    ("_dg_median_ns", dg_median),
                ):
                    if src in row and row[src] not in ("", "nan"):
                        try:
                            dst.append(float(row[src]))
                        except ValueError:
                            pass
    return {
        "op2op_us_median": median_of(op2op),
        "dg_first_ns": median_of(dg_first),
        "dg_median_ns": median_of(dg_median),
        "dg_last_ns": median_of(dg_last),
        "dg_issue_ns": median_of(dg_issue),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--min-prog-id", type=int, default=3)
    parser.add_argument("--aggregate-runs-dir", type=Path)
    args = parser.parse_args()

    if args.aggregate_runs_dir is not None:
        runs = sorted(p for p in args.aggregate_runs_dir.glob("run_*") if p.is_dir())
        stats = aggregate_runs(runs, args.min_prog_id)
        print(f"op2op={stats['op2op_us_median']:.3f}us " f"dg_median={stats['dg_median_ns']:.0f}ns n_runs={len(runs)}")
        return 0

    out = args.output_file or args.input_file.parent / "profile_log_device_op_to_op_complete.csv"
    n = export_log(args.input_file, out, args.min_prog_id)
    print(f"Wrote {n} gap rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
