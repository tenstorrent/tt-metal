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
import re
import shlex
import shutil
import subprocess
import sys
import time
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
    col: int | None,
    links_id: str,
    capture_dir: str,
    out_dir: Path,
    warmup: int,
    timed: int,
    keep_reports_dir: bool,
) -> Path | None:
    """Run pytest + post-process + cleanup for one combo.

    col=None → 8x4 mode: single run per layer, no col filter (parametrize variant skips non-zero cols).
    col=int  → 8x1 mode: one run per (layer, col).
    """
    if col is None:
        name = f"L{layer:02d}_{links_id}"
        kfilter = f"L{layer:02d} and {links_id}"
    else:
        name = f"L{layer:02d}_col{col}_{links_id}"
        kfilter = f"L{layer:02d} and col{col} and {links_id}"
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

    # tracy/__main__.py reconstructs the test command via `" ".join(originalArgs[1:])` and
    # runs it through `subprocess.Popen([cmd], shell=True)`. Without quoting, the multi-token
    # `-k "L05 and col0 and ..."` filter gets split into `-k L05 and col0 ...`, and pytest's
    # `and` becomes a positional file arg (zero items collected). Pre-quote with shlex.quote()
    # so the filter reassembles as a single shell-parsed token.
    cmd_pytest = [
        "python",
        "-m",
        "tracy",
        "-r",  # tells tracy to run generate_report() → writes reports/<timestamp>/ops_perf_results_<timestamp>.csv WITH OP CODE
        "-p",
        "-v",
        "-m",
        "pytest",
        "-v",
        "models/demos/deepseek_v3_d_p/tests/perf/test_dispatch_combine_replay.py",
        "-k",
        shlex.quote(kfilter),
    ]
    # Record state before pytest so we can identify NEW report dirs after.
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    pre_pytest_mtime = time.time()
    pre_pytest_dirs = {d.name for d in REPORTS_ROOT.iterdir() if d.is_dir()} if REPORTS_ROOT.exists() else set()

    r = subprocess.run(cmd_pytest, env=env)
    if r.returncode != 0:
        print(f"  ! pytest exit {r.returncode}; continuing", file=sys.stderr)

    # --- Inventory reports/ immediately after pytest, find the tracy auto-output
    #     timestamped dir (where generate_report wrote ops_perf_results_*.csv with OP CODE).
    post_pytest_dirs = list(REPORTS_ROOT.iterdir()) if REPORTS_ROOT.exists() else []
    new_dirs = [d for d in post_pytest_dirs if d.is_dir() and d.name not in pre_pytest_dirs]

    TS_PATTERN = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")
    new_ts_dirs = sorted(
        [d for d in new_dirs if TS_PATTERN.match(d.name)],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    produced = None
    if new_ts_dirs:
        ts_dir = new_ts_dirs[0]
        candidates = sorted(ts_dir.glob("ops_perf_results_*.csv"), key=os.path.getmtime, reverse=True)
        if candidates:
            produced = candidates[0]
            print(f"  found tracy auto-output: {produced.name}  ({produced.stat().st_size / 1024:.1f} KB)")

    # --- Path 2 fallback: re-run process_ops_logs.py -n <name> (device-only path, no OP CODE).
    if produced is None:
        print(f"  no timestamped dir from tracy auto-process; falling back to process_ops_logs.py -n {name}")
        cmd_post = [
            "python",
            str(TT_METAL_HOME / "tools" / "tracy" / "process_ops_logs.py"),
            "-n",
            name,
        ]
        rr = subprocess.run(cmd_post, env=env)
        if rr.returncode != 0:
            print(f"  ! process_ops_logs exit {rr.returncode}", file=sys.stderr)
        candidates = [REPORTS_ROOT / name / f"ops_perf_results_{name}.csv"]
        matches = [p for p in candidates if p.exists()]
        if not matches:
            legacy_pattern = str(REPORTS_ROOT / "*" / f"ops_perf_results_*_{name}.csv")
            legacy_matches = sorted(glob.glob(legacy_pattern), key=os.path.getmtime, reverse=True)
            if legacy_matches:
                matches = [Path(legacy_matches[0])]
        if not matches:
            print(
                f"  ! no CSV found via either path (timestamped tracy auto-output or {REPORTS_ROOT / name})",
                file=sys.stderr,
            )
            return None
        produced = matches[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, out_csv)
    print(f"  saved → {out_csv}  ({out_csv.stat().st_size / 1024:.1f} KB)")
    augment_csv_with_op_names(out_csv)

    # Cleanup: remove the entire produced.parent dir (timestamped OR named) — we've already
    # copied just the ops CSV. The big profile_log_device.csv (~5–30 MB) and any other
    # artifacts in that dir are not needed downstream.
    src_report_dir = produced.parent
    if not keep_reports_dir:
        shutil.rmtree(src_report_dir, ignore_errors=True)
        print(f"  removed source reports dir: {src_report_dir}")
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


# Position of each op within one iter of test_dispatch_combine_replay's _run_one_iter:
#   0 = dispatch, 1 = to_layout (skip in summary), 2 = combine
# Plus 1 startup op at op_idx==1 before the iter loop begins.
_OPS_PER_ITER = 3
_OP_POSITION = {
    "DispatchDeviceOperation": 0,
    "CombineDeviceOperation": 2,
}
_POSITION_TO_OP_NAME = {
    0: "DispatchDeviceOperation",
    1: "ToLayout",
    2: "CombineDeviceOperation",
}


def augment_csv_with_op_names(csv_path: Path) -> None:
    """If the CSV is device-only (no OP CODE column), synthesize OP CODE / CHIP_ID / OP_IDX /
    DEVICE ID / ITER_IN_DEV columns from GLOBAL CALL COUNT so each row is self-describing.

    GCC encoding (confirmed in our data):
      chip_id = GCC % 1024
      op_idx  = GCC // 1024   (1-indexed; op_idx=1 is a one-time startup op)
    For op_idx >= 2:
      iter_in_dev    = (op_idx - 2) // 3
      pos_in_iter    = (op_idx - 2) % 3
      OP CODE map    = {0: DispatchDeviceOperation, 1: ToLayout, 2: CombineDeviceOperation}
    """
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"  ! could not read {csv_path} to augment: {e}", file=sys.stderr)
        return
    if "OP CODE" in df.columns:
        return  # already has op names — leave alone
    if "GLOBAL CALL COUNT" not in df.columns:
        print(f"  ! {csv_path} has no GLOBAL CALL COUNT column; can't synthesize OP CODE", file=sys.stderr)
        return

    chip_id = df["GLOBAL CALL COUNT"] % 1024
    op_idx = df["GLOBAL CALL COUNT"] // 1024
    is_startup = op_idx == 1
    pos_in_iter = (op_idx - 2) % _OPS_PER_ITER
    iter_in_dev = (op_idx - 2) // _OPS_PER_ITER

    op_code = pos_in_iter.map(_POSITION_TO_OP_NAME).where(~is_startup, other="Startup")
    iter_in_dev_col = iter_in_dev.where(~is_startup, other=-1)
    pos_in_iter_col = pos_in_iter.where(~is_startup, other=-1)

    df.insert(0, "OP CODE", op_code)
    df.insert(1, "DEVICE ID", chip_id)
    df.insert(2, "CHIP_ID", chip_id)
    df.insert(3, "OP_IDX", op_idx)
    df.insert(4, "ITER_IN_DEV", iter_in_dev_col)
    df.insert(5, "POS_IN_ITER", pos_in_iter_col)
    df.to_csv(csv_path, index=False)
    print(f"  [augment] added OP CODE, DEVICE ID, CHIP_ID, OP_IDX, ITER_IN_DEV, POS_IN_ITER → {csv_path}")


def aggregate_one_csv(csv: Path, op_code: str, n_warmup: int) -> dict | None:
    """For one LB run CSV, return per-chip stats for one op (Dispatch or Combine).

    Supports two CSV variants:
      - Full tracy merge (has OP CODE + DEVICE ID columns): filter by OP CODE.
      - Device-only OPs csv (no OP CODE / DEVICE ID): attribute by GLOBAL CALL COUNT.
        chip_id = GCC % 1024, op_idx = GCC // 1024. Test sequence per iter is
        dispatch → to_layout → combine, with one startup op at op_idx=1.
    """
    try:
        df = pd.read_csv(csv, low_memory=False)
    except Exception as e:
        print(f"  ! failed to read {csv}: {e}", file=sys.stderr)
        return None
    if df.empty:
        return None
    df = df.copy()
    df["dur_ns"] = pd.to_numeric(df["DEVICE KERNEL DURATION [ns]"], errors="coerce")
    df = df.dropna(subset=["dur_ns"])
    if df.empty:
        return None

    if "OP CODE" in df.columns and "DEVICE ID" in df.columns:
        op_df = df[df["OP CODE"] == op_code].copy()
        if op_df.empty:
            return None
        op_df = op_df.sort_values(["DEVICE ID", "GLOBAL CALL COUNT"]).reset_index(drop=True)
        op_df["iter"] = op_df.groupby("DEVICE ID").cumcount()
        timed = op_df[op_df["iter"] >= n_warmup]
        if timed.empty:
            return None
        per_chip = timed.groupby("DEVICE ID")["dur_ns"].median()
    else:
        # Device-only fallback: attribute by GCC encoding.
        pos = _OP_POSITION.get(op_code)
        if pos is None:
            return None
        df["chip_id"] = df["GLOBAL CALL COUNT"] % 1024
        df["op_idx"] = df["GLOBAL CALL COUNT"] // 1024
        df = df[df["op_idx"] >= 2]  # drop op_idx=1 startup op
        df["op_idx_in_iter"] = (df["op_idx"] - 2) % _OPS_PER_ITER
        df["iter_in_dev"] = (df["op_idx"] - 2) // _OPS_PER_ITER
        op_df = df[df["op_idx_in_iter"] == pos]
        timed = op_df[op_df["iter_in_dev"] >= n_warmup]
        if timed.empty:
            return None
        per_chip = timed.groupby("chip_id")["dur_ns"].median()

    return {
        "n_chips": len(per_chip),
        "max_chip_med_ns": int(per_chip.max()),
        "median_chip_med_ns": int(per_chip.median()),
        "min_chip_med_ns": int(per_chip.min()),
    }


def aggregate_one_csv_8x4(csv: Path, op_code: str, n_warmup: int, num_cols: int = 4) -> dict | None:
    """For one Galaxy 8x4 run CSV (32 chips, one CSV per layer), return per-col stats.

    Uses position-based attribution (no OP CODE column) and `col = chip_id % num_cols`
    (same convention as analyze_galaxy_per_col.py). Returns dict mapping col_idx → stats.
    """
    try:
        df = pd.read_csv(csv, low_memory=False)
    except Exception as e:
        print(f"  ! failed to read {csv}: {e}", file=sys.stderr)
        return None
    if df.empty:
        return None
    df = df.copy()
    df["dur_ns"] = pd.to_numeric(df["DEVICE KERNEL DURATION [ns]"], errors="coerce")
    df = df.dropna(subset=["dur_ns"])
    if df.empty:
        return None

    pos = _OP_POSITION.get(op_code)
    if pos is None:
        return None

    if "OP CODE" in df.columns and "DEVICE ID" in df.columns:
        op_df = df[df["OP CODE"] == op_code].copy()
        if op_df.empty:
            return None
        op_df = op_df.sort_values(["DEVICE ID", "GLOBAL CALL COUNT"]).reset_index(drop=True)
        op_df["iter"] = op_df.groupby("DEVICE ID").cumcount()
        timed = op_df[op_df["iter"] >= n_warmup].copy()
        timed["col_idx"] = timed["DEVICE ID"] % num_cols
        chip_col = "DEVICE ID"
    else:
        df["chip_id"] = df["GLOBAL CALL COUNT"] % 1024
        df["op_idx"] = df["GLOBAL CALL COUNT"] // 1024
        df = df[df["op_idx"] >= 2]
        df["op_idx_in_iter"] = (df["op_idx"] - 2) % _OPS_PER_ITER
        df["iter_in_dev"] = (df["op_idx"] - 2) // _OPS_PER_ITER
        timed = df[(df["op_idx_in_iter"] == pos) & (df["iter_in_dev"] >= n_warmup)].copy()
        timed["col_idx"] = timed["chip_id"] % num_cols
        chip_col = "chip_id"

    if timed.empty:
        return None

    per_col_stats = {}
    for c in range(num_cols):
        col_df = timed[timed["col_idx"] == c]
        if col_df.empty:
            per_col_stats[c] = None
            continue
        per_chip_median = col_df.groupby(chip_col)["dur_ns"].median()
        per_col_stats[c] = {
            "n_chips": len(per_chip_median),
            "max_chip_med_ns": int(per_chip_median.max()),
            "median_chip_med_ns": int(per_chip_median.median()),
            "min_chip_med_ns": int(per_chip_median.min()),
        }
    return per_col_stats


def summarize_8x1(out_dir: Path, layers: list[int], cols: list[int], links_id: str, warmup: int) -> pd.DataFrame:
    """8x1 sweep summary: per-layer per-col table, 4 CSVs per layer (one per col)."""
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


def summarize_8x4(out_dir: Path, layers: list[int], links_id: str, warmup: int) -> pd.DataFrame:
    """8x4 sweep summary: per-layer per-col table extracted from a single 32-chip CSV per layer."""
    rows = []
    cols = [0, 1, 2, 3]
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
        name = f"L{L:02d}_{links_id}"
        csv = out_dir / f"{name}.csv"
        for op in OPS_OF_INTEREST:
            per_col_stats = aggregate_one_csv_8x4(csv, op, warmup) if csv.exists() else None
            per_col = {
                c: (per_col_stats[c]["max_chip_med_ns"] if per_col_stats and per_col_stats.get(c) else None)
                for c in cols
            }
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
    ap.add_argument(
        "--cols",
        default="0,1,2,3",
        help="Comma-separated dispatch group columns (default: 0,1,2,3). Ignored for --mesh 8x4.",
    )
    ap.add_argument(
        "--mesh",
        choices=["8x1", "8x4"],
        default="8x1",
        help="Mesh shape. 8x1 = LB/no-contention, runs N cols per layer. 8x4 = Galaxy with contention, single run per layer captures all 4 cols.",
    )
    ap.add_argument(
        "--num-links-id",
        default=None,
        help="Mesh config id. Defaults to linear-8-2link for 8x1 or mesh-8x4-2link for 8x4.",
    )
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

    if args.num_links_id is None:
        args.num_links_id = "mesh-8x4-2link" if args.mesh == "8x4" else "linear-8-2link"

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    cols = [int(x) for x in args.cols.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        if args.mesh == "8x4":
            n_runs = len(layers)
            print(f"Sweep plan (8x4): {n_runs} layers × 1 run each (no col iteration; single run captures all 4 cols)")
        else:
            n_runs = len(layers) * len(cols)
            print(f"Sweep plan (8x1): {len(layers)} layers × {len(cols)} cols = {n_runs} runs")
        print(f"  capture_dir: {args.capture_dir}")
        print(f"  out_dir:     {out_dir}")
        print(f"  links_id:    {args.num_links_id}")
        print(f"  warmup:      {args.warmup}")
        print(f"  timed:       {args.timed}")
        for L in layers:
            if args.mesh == "8x4":
                name = f"L{L:02d}_{args.num_links_id}"
                existing = out_dir / f"{name}.csv"
                if args.skip_existing and existing.exists():
                    print(f"\n=== {name} (already exists, skipping) ===")
                    continue
                run_one(
                    L,
                    None,
                    args.num_links_id,
                    args.capture_dir,
                    out_dir,
                    args.warmup,
                    args.timed,
                    args.keep_reports_dir,
                )
            else:
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

    if args.mesh == "8x4":
        summary = summarize_8x4(out_dir, layers, args.num_links_id, args.warmup)
    else:
        summary = summarize_8x1(out_dir, layers, cols, args.num_links_id, args.warmup)
    summary_path = out_dir / f"summary_{args.num_links_id}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
