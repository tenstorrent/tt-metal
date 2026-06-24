#!/usr/bin/env python3
"""Rebuild  batch-2 CSVs from existing sweep logs (no pandas, no re-benchmark)."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

from export_op_to_op_gaps_csvlite import aggregate_runs

SCRIPT_DIR = Path(__file__).resolve().parent
TT_METAL_HOME = SCRIPT_DIR.parents[4]
PYTHON = TT_METAL_HOME / "python_env" / "bin" / "python3"
EXPORT_PY = SCRIPT_DIR / "export_op_to_op_profiler_csv.py"
OUT_DIR = TT_METAL_HOME / "generated/profiler/op_to_op_runs/batch2"
RUNS_ROOT = TT_METAL_HOME / "generated/profiler/op_to_op_runs"
SWEEP_LOG = OUT_DIR / "zero_compute_sweep.log"
READONLY_CSV = OUT_DIR / "read_only_bw.csv"
ZERO_CHART = OUT_DIR / "zero_compute_chart_data.csv"
SUMMARY = OUT_DIR / "batch2_summary.csv"

ZERO_CORES = [1, 2, 4, 8, 10, 16, 20, 32, 40, 56, 64, 80, 96, 110]
MIN_PROG_ID = 3
NUM_RUNS = 3
TRID_IN_FLIGHT = 2


def parse_zero_compute_bw(log_text: str) -> dict[int, dict]:
    """Parse 'picked: in_cb=.. out_cb=.. peak_in_gbps=..' blocks per core count."""
    rows: dict[int, dict] = {}
    current: int | None = None
    for line in log_text.splitlines():
        m = re.match(r"^=+\s*cores=(\d+)\s*=+", line)
        if m:
            current = int(m.group(1))
            continue
        if current is None:
            continue
        m = re.search(
            r"picked: in_cb=(\d+) out_cb=(\d+) peak_in_gbps=([0-9.]+) peak_out_gbps=([0-9.]+)",
            line,
        )
        if m:
            rows[current] = {
                "in_cb": int(m.group(1)),
                "out_cb": int(m.group(2)),
                "peak_in_gbps": float(m.group(3)),
                "peak_out_gbps": float(m.group(4)),
            }
    return rows


def reexport_runs(run_dirs: list[Path], in_cb: int, out_cb: int) -> None:
    py = PYTHON if PYTHON.is_file() else Path(sys.executable)
    for run_dir in run_dirs:
        log = run_dir / "profile_log_device.csv"
        if not log.is_file():
            continue
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
        subprocess.run(cmd, check=True, capture_output=True)


def load_readonly_summary() -> list[dict]:
    rows: list[dict] = []
    with READONLY_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main() -> int:
    if not SWEEP_LOG.is_file():
        print(f"Missing {SWEEP_LOG}", file=sys.stderr)
        return 1

    bw_by_core = parse_zero_compute_bw(SWEEP_LOG.read_text())
    zero_rows: list[dict] = []

    for n in ZERO_CORES:
        bw = bw_by_core.get(n)
        if bw is None:
            print(f"WARNING: no BW pick for cores={n}", file=sys.stderr)
            continue

        base = RUNS_ROOT / f"chart_cores{n}_n{TRID_IN_FLIGHT}"
        run_dirs = sorted(base.glob("run_*"))[:NUM_RUNS]
        if not run_dirs:
            print(f"WARNING: no runs under {base}", file=sys.stderr)
            continue

        reexport_runs(run_dirs, bw["in_cb"], bw["out_cb"])
        stats = aggregate_runs(run_dirs, MIN_PROG_ID)
        cb_label = f"in={bw['in_cb']}/out={bw['out_cb']}"
        zero_rows.append(
            {
                "num_cores": n,
                "peak_input_gbps": f"{bw['peak_in_gbps']:.4f}",
                "smallest_input_cb": bw["in_cb"],
                "peak_output_gbps": f"{bw['peak_out_gbps']:.4f}",
                "smallest_output_cb": bw["out_cb"],
                "op2op_us_median": f"{stats['op2op_us_median']:.3f}",
                "dg_first_ns": f"{stats['dg_first_ns']:.1f}",
                "dg_median_ns": f"{stats['dg_median_ns']:.1f}",
                "dg_last_ns": f"{stats['dg_last_ns']:.1f}",
                "dg_issue_ns": f"{stats['dg_issue_ns']:.1f}",
                "cb_label": cb_label,
                "trid_in_flight": TRID_IN_FLIGHT,
            }
        )
        print(
            f"cores={n:3d}  peak={bw['peak_in_gbps']:.2f} GB/s  "
            f"op2op={stats['op2op_us_median']:.3f}us  dg_median={stats['dg_median_ns']:.0f}ns  "
            f"{cb_label}  runs={len(run_dirs)}"
        )

    zero_fields = [
        "num_cores",
        "peak_input_gbps",
        "smallest_input_cb",
        "peak_output_gbps",
        "smallest_output_cb",
        "op2op_us_median",
        "dg_first_ns",
        "dg_median_ns",
        "dg_last_ns",
        "dg_issue_ns",
        "cb_label",
        "trid_in_flight",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ZERO_CHART.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=zero_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(zero_rows)

    summary_fields = [
        "section",
        "cores",
        "peak_bw_gbs",
        "peak_bw_gbps",
        "per_core_gbps",
        "smallest_in_cb",
        "smallest_out_cb",
        "pack_to_unpack_us",
        "dg_median_ns",
        "dg_last_ns",
        "cb_label",
        "notes",
    ]
    with SUMMARY.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for row in load_readonly_summary():
            nc = int(row["num_cores"])
            peak = float(row["peak_bw_gbs"])
            w.writerow(
                {
                    "section": "read_only_bw",
                    "cores": nc,
                    "peak_bw_gbs": f"{peak:.4f}",
                    "peak_bw_gbps": row["peak_bw_gbps"],
                    "per_core_gbps": row["per_core_gbps"],
                    "smallest_in_cb": row["smallest_in_cb"],
                    "smallest_out_cb": row["smallest_out_cb"],
                    "cb_label": f"in={row['smallest_in_cb']}/out={row['smallest_out_cb']}",
                    "notes": "read-only pipeline; BW=GB/s in logs x8=Gbps",
                }
            )
        for row in zero_rows:
            nc = int(row["num_cores"])
            peak = float(row["peak_input_gbps"])
            w.writerow(
                {
                    "section": "zero_compute",
                    "cores": nc,
                    "peak_bw_gbs": f"{peak:.4f}",
                    "peak_bw_gbps": f"{peak * 8:.4f}",
                    "per_core_gbps": f"{peak * 8 / nc:.4f}",
                    "smallest_in_cb": row["smallest_input_cb"],
                    "smallest_out_cb": row["smallest_output_cb"],
                    "pack_to_unpack_us": row["op2op_us_median"],
                    "dg_median_ns": row["dg_median_ns"],
                    "dg_last_ns": row["dg_last_ns"],
                    "cb_label": row["cb_label"],
                    "notes": "compute_nops=0; gap=pack->unpack not math-to-math",
                }
            )

    print(f"\nWrote {ZERO_CHART}")
    print(f"Wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
