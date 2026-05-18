#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Sweep dispatch + combine replay over (layer × galaxy_col) on an 8x1 LB.

For each combo this script:
  1. Runs `pytest test_dispatch_combine_replay.py -k "L<L> and col<C> and <links_id>"`
     under `python -m tracy` so per-op CSVs with OP CODE are produced.
  2. Post-processes the device log into `ops_perf_results_*_<name>.csv`.
  3. Copies that CSV out to a clean output dir.
  4. Deletes tracy artifacts (.tracy host trace, intermediate CSVs, timestamped
     `reports/<ts>/` dir).

After all combos, prints + saves a per-layer summary computing for BOTH
DispatchDeviceOperation and CombineDeviceOperation:
    per-chip median across timed iters
    max-across-8-chips per column
    max-across-4-cols → the layer's no-contention LB estimate

Example (single line for copy-paste):
    python run_dispatch_combine_replay_sweep.py --layers 5,20,40,55 --cols 0,1,2,3 --num-links-id linear-8-2link --capture-dir /localdev/nmilicevic/dispatch_captures --out-dir /localdev/nmilicevic/lb_dispatch_combine_results --warmup 2 --timed 5
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

TT_METAL_HOME = Path(os.environ.get("TT_METAL_HOME", "/localdev/nmilicevic/tt-metal"))
REPORTS_ROOT = TT_METAL_HOME / "generated" / "profiler" / "reports"
LOGS_DIR = TT_METAL_HOME / "generated" / "profiler" / ".logs"
TRACY_BIG_FILES = [
    "tracy_profile_log_host.tracy",
    "tracy_ops_times.csv",
    "tracy_ops_data.csv",
]

OPS_OF_INTEREST = ["DispatchDeviceOperation", "CombineDeviceOperation"]


def run_one(
    layer: int,
    col: int,
    links_id: str,
    capture_dir: str,
    out_dir: Path,
    warmup: int,
    timed: int,
    keep_reports_dir: bool,
) -> Path | None:
    """Run pytest + post-process + cleanup for one (layer, col) combo."""
    name = f"L{layer:02d}_col{col}_{links_id}"
    out_csv = out_dir / f"{name}.csv"
    print(f"\n=== {name} ===", flush=True)

    env = {
        **os.environ,
        "TT_METAL_DEVICE_PROFILER": "1",
        "TT_DS_PROFILE_OPS": "1",
        "TT_DS_DISPATCH_CAPTURE_DIR": capture_dir,
        "TT_DS_REPLAY_WARMUP": str(warmup),
        "TT_DS_REPLAY_TIMED": str(timed),
    }

    cmd_pytest = [
        "python",
        "-m",
        "tracy",
        "-p",
        "-v",
        "--disable-device-data-push-to-tracy",
        "-m",
        "pytest",
        "-v",
        "models/demos/deepseek_v3_d_p/tests/perf/test_dispatch_combine_replay.py",
        "-k",
        f"L{layer:02d} and col{col} and {links_id}",
    ]
    r = subprocess.run(cmd_pytest, env=env)
    if r.returncode != 0:
        print(f"  ! pytest exit {r.returncode}; continuing", file=sys.stderr)

    cmd_post = [
        "python",
        str(TT_METAL_HOME / "tools" / "tracy" / "process_ops_logs.py"),
        "-n",
        name,
    ]
    rr = subprocess.run(cmd_post, env=env)
    if rr.returncode != 0:
        print(f"  ! process_ops_logs exit {rr.returncode}", file=sys.stderr)

    # process_ops_logs.py -n <name> writes to:
    #   reports/<name>/ops_perf_results_<name>.csv  (current convention, no timestamp prefix)
    # Older versions used:
    #   reports/<timestamp>/ops_perf_results_<timestamp>_<name>.csv
    # Match either.
    candidates = [
        REPORTS_ROOT / name / f"ops_perf_results_{name}.csv",
    ]
    matches = [p for p in candidates if p.exists()]
    if not matches:
        # fall back to old timestamped pattern
        legacy_pattern = str(REPORTS_ROOT / "*" / f"ops_perf_results_*_{name}.csv")
        legacy_matches = sorted(glob.glob(legacy_pattern), key=os.path.getmtime, reverse=True)
        if legacy_matches:
            matches = [Path(legacy_matches[0])]
    if not matches:
        print(
            f"  ! no CSV at {REPORTS_ROOT / name / f'ops_perf_results_{name}.csv'} " f"or matching legacy pattern",
            file=sys.stderr,
        )
        return None
    produced = matches[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, out_csv)
    print(f"  saved → {out_csv}  ({out_csv.stat().st_size / 1024:.1f} KB)")

    ts_dir = produced.parent
    if not keep_reports_dir:
        shutil.rmtree(ts_dir, ignore_errors=True)
        print(f"  removed reports dir: {ts_dir}")
    for fname in TRACY_BIG_FILES:
        p = LOGS_DIR / fname
        if p.exists():
            try:
                size_mb = p.stat().st_size / (1024 * 1024)
                p.unlink()
                print(f"  removed {p.name} ({size_mb:.1f} MB)")
            except OSError as e:
                print(f"  ! failed to unlink {p}: {e}", file=sys.stderr)

    return out_csv


def aggregate_one_csv(csv: Path, op_code: str, n_warmup: int) -> dict | None:
    """For one LB run CSV, return per-chip stats for one op (Dispatch or Combine)."""
    try:
        df = pd.read_csv(csv, low_memory=False)
    except Exception as e:
        print(f"  ! failed to read {csv}: {e}", file=sys.stderr)
        return None
    if "OP CODE" not in df.columns:
        return None
    op_df = df[df["OP CODE"] == op_code].copy()
    if op_df.empty:
        return None
    op_df["dur_ns"] = pd.to_numeric(op_df["DEVICE KERNEL DURATION [ns]"], errors="coerce")
    op_df = op_df.dropna(subset=["dur_ns"]).sort_values(["DEVICE ID", "GLOBAL CALL COUNT"]).reset_index(drop=True)
    op_df["iter"] = op_df.groupby("DEVICE ID").cumcount()
    timed = op_df[op_df["iter"] >= n_warmup]
    if timed.empty:
        return None
    per_chip = timed.groupby("DEVICE ID")["dur_ns"].median()
    return {
        "n_chips": len(per_chip),
        "max_chip_med_ns": int(per_chip.max()),
        "median_chip_med_ns": int(per_chip.median()),
        "min_chip_med_ns": int(per_chip.min()),
    }


def summarize(out_dir: Path, layers: list[int], cols: list[int], links_id: str, warmup: int) -> pd.DataFrame:
    """Build per-layer summary: max-of-4-cols of (max-of-8-chips of median-of-iters) for each op."""
    rows = []
    headers = ["Layer"]
    for op in OPS_OF_INTEREST:
        for c in cols:
            headers.append(f"{op.replace('DeviceOperation', '')[:8]}_col{c}")
        headers.append(f"{op.replace('DeviceOperation', '')[:8]}_max")

    print("\n" + "=" * 130)
    print(f"{'Layer':<7}  " + " ".join(f"{h:>13}" for h in headers[1:]))
    print("-" * 130)

    for L in layers:
        row = {"layer_idx": L}
        line_cells = []
        for op in OPS_OF_INTEREST:
            per_col = {}
            for c in cols:
                name = f"L{L:02d}_col{c}_{links_id}"
                csv = out_dir / f"{name}.csv"
                if not csv.exists():
                    per_col[c] = None
                    continue
                s = aggregate_one_csv(csv, op, warmup)
                per_col[c] = s["max_chip_med_ns"] if s else None
            layer_max = max((v for v in per_col.values() if v is not None), default=None)
            short = op.replace("DeviceOperation", "")[:8].lower()
            for c in cols:
                col_key = f"{short}_col{c}_max_chip_med_ns"
                row[col_key] = per_col.get(c)
                line_cells.append(f"{per_col.get(c):>13,}" if per_col.get(c) is not None else f"{'--':>13}")
            row[f"{short}_layer_max_ns"] = layer_max
            line_cells.append(f"{layer_max:>13,}" if layer_max is not None else f"{'--':>13}")

        print(f"L{L:02d}     " + " ".join(line_cells))
        rows.append(row)
    print("=" * 130)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", required=True, help="Comma-separated MoE layer indices, e.g., 5,20,40,55")
    ap.add_argument("--cols", default="0,1,2,3", help="Comma-separated dispatch group columns (default: 0,1,2,3)")
    ap.add_argument("--num-links-id", default="linear-8-2link", help="Mesh config id (default: linear-8-2link)")
    ap.add_argument("--capture-dir", required=True, help="Path with L<NN>/indices.pt files")
    ap.add_argument("--out-dir", required=True, help="Where to save per-combo CSVs")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timed", type=int, default=5)
    ap.add_argument(
        "--skip-existing", action="store_true", help="Skip combos where the per-combo CSV already exists in out_dir."
    )
    ap.add_argument(
        "--summary-only", action="store_true", help="Skip running; just aggregate existing CSVs in out_dir."
    )
    ap.add_argument(
        "--keep-reports-dir",
        action="store_true",
        help="Keep the original generated/profiler/reports/<ts>/ dirs (default: delete).",
    )
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    cols = [int(x) for x in args.cols.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        print(f"Sweep plan: {len(layers)} layers × {len(cols)} cols × 1 links_id = {len(layers)*len(cols)} runs")
        print(f"  capture_dir: {args.capture_dir}")
        print(f"  out_dir:     {out_dir}")
        print(f"  links_id:    {args.num_links_id}")
        print(f"  warmup:      {args.warmup}")
        print(f"  timed:       {args.timed}")
        for L in layers:
            for c in cols:
                name = f"L{L:02d}_col{c}_{args.num_links_id}"
                existing = out_dir / f"{name}.csv"
                if args.skip_existing and existing.exists():
                    print(f"\n=== {name} (already exists, skipping) ===")
                    continue
                run_one(
                    L,
                    c,
                    args.num_links_id,
                    args.capture_dir,
                    out_dir,
                    args.warmup,
                    args.timed,
                    args.keep_reports_dir,
                )

    summary = summarize(out_dir, layers, cols, args.num_links_id, args.warmup)
    summary_path = out_dir / f"summary_{args.num_links_id}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
