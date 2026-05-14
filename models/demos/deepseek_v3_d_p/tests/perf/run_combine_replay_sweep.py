#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Sweep replay runs over (layer × dispatch_group_column) combos on an 8x1 LB.

For each combo this script:
  1. Runs `pytest test_combine_replay.py -k "L<L>_col<C> and <links_id>"` under
     `python -m tracy` so the device-side per-op CSV is produced with OP CODE.
  2. Post-processes the device log into `ops_perf_results_*_<name>.csv`.
  3. Copies that CSV out to a clean output dir (named `L<NN>_col<K>_<links>.csv`).
  4. Deletes the rest of the tracy artifacts (the .tracy host trace, the
     intermediate tracy_ops_*.csv files, and the timestamped `reports/<ts>/`
     directory) so disk usage stays bounded across many runs.

After all combos run, prints and saves a per-layer summary computing:
    per-chip median across timed iters
    max-across-8-chips per column
    max-across-4-cols → the layer's no-contention LB combine time estimate

Example:
    python run_combine_replay_sweep.py \\
      --layers 5,20,40,55 \\
      --cols 0,1,2,3 \\
      --num-links-id linear-8-2link \\
      --capture-dir /localdev/nmilicevic/combine_captures_1k \\
      --out-dir /localdev/nmilicevic/lb_replay_results \\
      --warmup 2 --timed 5
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
        "TT_DS_COMBINE_CAPTURE_DIR": capture_dir,
        "TT_DS_REPLAY_WARMUP": str(warmup),
        "TT_DS_REPLAY_TIMED": str(timed),
    }

    # 1. Run pytest under tracy (no `-r` to avoid shell-quoting bug; we'll
    #    invoke process_ops_logs.py manually below).
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
        "models/demos/deepseek_v3_d_p/tests/perf/test_combine_replay.py",
        "-k",
        f"L{layer:02d}_col{col} and {links_id}",
    ]
    r = subprocess.run(cmd_pytest, env=env)
    if r.returncode != 0:
        print(f"  ! pytest exit {r.returncode}; continuing", file=sys.stderr)
        # still try to post-process — sometimes pytest exits nonzero but the CSV is there

    # 2. Post-process device log into ops_perf_results_*_<name>.csv
    cmd_post = [
        "python",
        str(TT_METAL_HOME / "tools" / "tracy" / "process_ops_logs.py"),
        "-n",
        name,
    ]
    rr = subprocess.run(cmd_post, env=env)
    if rr.returncode != 0:
        print(f"  ! process_ops_logs exit {rr.returncode}", file=sys.stderr)

    # 3. Find the newest reports/<ts>/ dir that contains the matching CSV
    pattern = str(REPORTS_ROOT / "*" / f"ops_perf_results_*_{name}.csv")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not matches:
        print(f"  ! no CSV matched {pattern}", file=sys.stderr)
        return None
    produced = Path(matches[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, out_csv)
    print(f"  saved → {out_csv}  ({out_csv.stat().st_size / 1024:.1f} KB)")

    # 4. Cleanup
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


def aggregate_one_csv(csv: Path, n_warmup: int) -> dict | None:
    """Return per-chip stats for one LB run CSV."""
    try:
        df = pd.read_csv(csv, low_memory=False)
    except Exception as e:
        print(f"  ! failed to read {csv}: {e}", file=sys.stderr)
        return None
    if "OP CODE" not in df.columns:
        print(f"  ! {csv} has no OP CODE column (device-only CSV, no host trace?)", file=sys.stderr)
        return None
    combine = df[df["OP CODE"] == "CombineDeviceOperation"].copy()
    if combine.empty:
        print(f"  ! {csv} has no CombineDeviceOperation rows", file=sys.stderr)
        return None
    combine["dur_ns"] = pd.to_numeric(combine["DEVICE KERNEL DURATION [ns]"], errors="coerce")
    combine = combine.dropna(subset=["dur_ns"]).sort_values(["DEVICE ID", "GLOBAL CALL COUNT"]).reset_index(drop=True)
    combine["iter"] = combine.groupby("DEVICE ID").cumcount()
    timed = combine[combine["iter"] >= n_warmup]
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
    rows = []
    print("\n" + "=" * 92)
    print(f"{'Layer':<7} {'col0':>14} {'col1':>14} {'col2':>14} {'col3':>14}    {'LAYER max':>14}")
    print("-" * 92)
    for L in layers:
        per_col = {}
        for c in cols:
            name = f"L{L:02d}_col{c}_{links_id}"
            csv = out_dir / f"{name}.csv"
            if not csv.exists():
                per_col[c] = None
                continue
            s = aggregate_one_csv(csv, warmup)
            per_col[c] = s["max_chip_med_ns"] if s else None
        layer_max = max((v for v in per_col.values() if v is not None), default=None)
        cells = [(f"{per_col.get(c):>14,}" if per_col.get(c) is not None else f"{'--':>14}") for c in range(4)]
        layer_str = f"{layer_max:>14,}" if layer_max is not None else f"{'--':>14}"
        print(f"L{L:02d}     {cells[0]} {cells[1]} {cells[2]} {cells[3]}    {layer_str}")
        rows.append(
            {
                "layer_idx": L,
                "col0_max_chip_med_ns": per_col.get(0),
                "col1_max_chip_med_ns": per_col.get(1),
                "col2_max_chip_med_ns": per_col.get(2),
                "col3_max_chip_med_ns": per_col.get(3),
                "lb_layer_max_ns": layer_max,
            }
        )
    print("=" * 92)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", required=True, help="Comma-separated MoE layer indices, e.g., 5,20,40,55")
    ap.add_argument("--cols", default="0,1,2,3", help="Comma-separated dispatch group columns (default: 0,1,2,3)")
    ap.add_argument("--num-links-id", default="linear-8-2link", help="Mesh config id (default: linear-8-2link)")
    ap.add_argument("--capture-dir", required=True, help="Path with L<NN>/col<K>.pt files")
    ap.add_argument("--out-dir", required=True, help="Where to save per-combo CSVs")
    ap.add_argument("--warmup", type=int, default=3)
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
