#!/usr/bin/env python3
"""Aggregate batch-3 compute_nops sweep into batch3_compute_sweep.csv."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TT_METAL_HOME = SCRIPT_DIR.parents[4]
OUT_DIR = TT_METAL_HOME / "generated/profiler/op_to_op_runs/batch3"
CSV_OUT = OUT_DIR / "batch3_compute_sweep.csv"

PYTHON = TT_METAL_HOME / "python_env/bin/python3"
EXPORT_PY = SCRIPT_DIR / "export_op_to_op_profiler_csv.py"

CORE_COUNTS = [1, 10, 40, 80, 110]
COMPUTE_NOPS_LIST = [0, 100, 500, 2000, 5000]
MIN_PROG_ID = 3
TRID_IN_FLIGHT = 2

CB = {
    1: (12, 2),
    10: (12, 8),
    40: (64, 64),
    80: (64, 64),
    110: (12, 64),
}

# batch 1 reference (2000 nops) for comparison column
BATCH1_REF = {
    1: (2.770, 1198),
    10: (2.731, 639),
    40: (5.981, 1070),
    80: (6.699, 1025),
    110: (5.644, 624),
}


def median(vals: list[float]) -> float:
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan")
    vals.sort()
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def reexport_run(run_dir: Path, in_cb: int) -> None:
    py = PYTHON if PYTHON.is_file() else Path(sys.executable)
    log = run_dir / "profile_log_device.csv"
    if not log.is_file():
        return
    rt = run_dir / "profile_log_device_rt.csv"
    cmd = [
        str(py),
        str(EXPORT_PY),
        "--input-file",
        str(log),
        "--min-prog-id",
        str(MIN_PROG_ID),
        "--tiles-per-core",
        "16",
        "--input-cb-depth-tiles",
        str(in_cb),
        "--reader-push-tiles",
        "2",
        "--output-dir",
        str(run_dir),
    ]
    if rt.is_file():
        cmd.extend(["--rt-input-file", str(rt)])
    subprocess.run(cmd, check=False, capture_output=True)


def aggregate_label(label: str, in_cb: int) -> dict[str, float]:
    base = TT_METAL_HOME / "generated/profiler/op_to_op_runs" / label
    run_dirs = sorted(base.glob("run_*"))[:3]
    op2op, dg_med, dg_first, dg_last = [], [], [], []
    for run_dir in run_dirs:
        reexport_run(run_dir, in_cb)
        complete = run_dir / "profile_log_device_op_to_op_complete.csv"
        if not complete.is_file():
            continue
        with complete.open(newline="") as f:
            for row in csv.DictReader(f):
                if int(float(row["from_prog_id"])) < MIN_PROG_ID:
                    continue
                op2op.append(float(row["gap_us"]))
                for col, dst in (
                    ("dg_median_ns", dg_med),
                    ("_dg_median_ns", dg_med),
                    ("dg_first_ns", dg_first),
                    ("dg_last_ns", dg_last),
                ):
                    if col in row and row[col] not in ("", "nan"):
                        try:
                            dst.append(float(row[col]))
                        except ValueError:
                            pass
    return {
        "op2op_us_median": median(op2op),
        "dg_median_ns": median(dg_med),
        "dg_first_ns": median(dg_first),
        "dg_last_ns": median(dg_last),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "num_cores",
        "compute_nops",
        "input_cb",
        "output_cb",
        "op2op_us_median",
        "dg_median_ns",
        "dg_first_ns",
        "dg_last_ns",
        "batch1_op2op_us",
        "batch1_dg_median_ns",
        "cb_label",
    ]
    rows: list[dict] = []

    for n in CORE_COUNTS:
        in_cb, out_cb = CB[n]
        b1_op2, b1_dg = BATCH1_REF.get(n, (float("nan"), float("nan")))
        for nops in COMPUTE_NOPS_LIST:
            label = f"batch3_cores{n}_nops{nops}_n2"
            stats = aggregate_label(label, in_cb)
            row = {
                "num_cores": n,
                "compute_nops": nops,
                "input_cb": in_cb,
                "output_cb": out_cb,
                "op2op_us_median": f"{stats['op2op_us_median']:.3f}",
                "dg_median_ns": f"{stats['dg_median_ns']:.1f}",
                "dg_first_ns": f"{stats['dg_first_ns']:.1f}",
                "dg_last_ns": f"{stats['dg_last_ns']:.1f}",
                "batch1_op2op_us": f"{b1_op2:.3f}" if nops == 2000 else "",
                "batch1_dg_median_ns": f"{b1_dg:.1f}" if nops == 2000 else "",
                "cb_label": f"in={in_cb}/out={out_cb}",
            }
            rows.append(row)
            print(
                f"cores={n:3d} nops={nops:5d}  op2op={stats['op2op_us_median']:.3f}us  "
                f"dg_median={stats['dg_median_ns']:.0f}ns  {row['cb_label']}"
            )

    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {CSV_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
