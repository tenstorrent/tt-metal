#!/usr/bin/env python3
"""Generate plots for a generic_lut_activation_embedded experiment directory."""

import argparse
from pathlib import Path
import sys

try:
    from .frontier_scatter import main as frontier_main
except ImportError:
    from frontier_scatter import main as frontier_main


def existing(path):
    return Path(path).expanduser().resolve()


def tier_args(native_vs_embedded):
    if not native_vs_embedded:
        return []
    merged = existing(native_vs_embedded) / "merged"
    tiers = []
    for label, name in [
        ("best", "best_native_vs_embedded.csv"),
        ("best99", "best99_native_vs_embedded.csv"),
        ("best95", "best95_native_vs_embedded.csv"),
    ]:
        path = merged / name
        if path.exists():
            tiers += ["--tier", f"{label}={path}"]
    return tiers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run embedded experiment plot generators from explicit result directories."
    )
    parser.add_argument("--frontier", type=existing, help="frontier_4chip_* result directory")
    parser.add_argument("--ttnn", type=existing, help="TTNN reference CSV, usually merged/ttnn_ref.csv")
    parser.add_argument("--native-vs-embedded", type=existing, help="native_vs_embedded_4chip_* result directory")
    parser.add_argument("--all", action="store_true", help="Generate every plot type supported by available summary artifacts")
    parser.add_argument("--tiers", action="store_true", help="Also generate best/best99/best95 tier comparison plots")
    parser.add_argument("--outdir", type=existing, help="Output plot directory. Defaults to <frontier>/plots.")
    parser.add_argument(
        "--frontier-subdir",
        default="frontier_scatter",
        help="Optional subdirectory under --outdir for frontier scatters.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.frontier:
        raise SystemExit("plot_experiment: --frontier is required for frontier scatter plots")

    shards = sorted((args.frontier / "shards").glob("frontier*chip*.csv"))
    if not shards:
        raise SystemExit(f"plot_experiment: no frontier shard CSVs under {args.frontier / 'shards'}")

    outdir = args.outdir or (args.frontier / "plots")
    argv = [str(p) for p in shards] + ["--outdir", str(outdir)]
    if args.ttnn:
        argv += ["--ttnn", str(args.ttnn)]
    if args.frontier_subdir:
        argv += ["--frontier-subdir", args.frontier_subdir]
    if args.all or args.tiers:
        argv += tier_args(args.native_vs_embedded)

    old_argv = sys.argv
    try:
        sys.argv = ["frontier_scatter.py", *argv]
        frontier_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
