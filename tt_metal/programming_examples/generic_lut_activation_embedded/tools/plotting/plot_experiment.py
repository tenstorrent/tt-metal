#!/usr/bin/env python3
"""Generate plots for a generic_lut_activation_embedded experiment directory."""

import argparse
from pathlib import Path
import sys

try:
    from .frontier_scatter import main as frontier_main
    from .plot_pareto_io import main as pareto_io_main
    from .select_pareto_winners import main as select_pareto_main
except ImportError:
    from frontier_scatter import main as frontier_main
    from plot_pareto_io import main as pareto_io_main
    from select_pareto_winners import main as select_pareto_main


def existing(path):
    return Path(path).expanduser().resolve()


def repo_results_dir():
    return Path(__file__).resolve().parents[2] / "results"


def tier_args(native_vs_embedded):
    if not native_vs_embedded:
        return []
    root = existing(native_vs_embedded)
    csv_dir = root / "data" / "csv"
    if not csv_dir.exists():
        csv_dir = root / "merged"
    tiers = []
    for label, name in [
        ("best", "best_native_vs_embedded.csv"),
        ("best99", "best99_native_vs_embedded.csv"),
        ("best95", "best95_native_vs_embedded.csv"),
    ]:
        path = csv_dir / name
        if path.exists():
            tiers += ["--tier", f"{label}={path}"]
    return tiers


def frontier_csvs(frontier):
    csv_dir = frontier / "data" / "csv"
    patterns = [
        csv_dir / "frontier*chip*.csv",
        frontier / "shards" / "frontier*chip*.csv",
    ]
    shards = []
    for pattern in patterns:
        shards = sorted(pattern.parent.glob(pattern.name))
        if shards:
            return shards
    return []


def default_ttnn(frontier, native_vs_embedded):
    candidates = [
        frontier / "data" / "csv" / "ttnn_ref.csv",
    ]
    if frontier.name == "bf16":
        candidates.append(frontier.parent.parent / "native_vs_embedded" / "bf16" / "data" / "csv" / "ttnn_ref.csv")
    if native_vs_embedded:
        native_root = existing(native_vs_embedded)
        candidates += [
            native_root / "data" / "csv" / "ttnn_ref.csv",
            native_root / "merged" / "ttnn_ref.csv",
        ]
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_args():
    default_frontier = repo_results_dir() / "frontier" / "bf16"
    parser = argparse.ArgumentParser(
        description="Run embedded experiment plot generators from explicit result directories."
    )
    parser.add_argument(
        "--frontier",
        type=existing,
        default=default_frontier,
        help=f"Frontier result directory (default: {default_frontier})",
    )
    parser.add_argument("--ttnn", type=existing, help="TTNN reference CSV, usually merged/ttnn_ref.csv")
    parser.add_argument("--native-vs-embedded", type=existing, help="Native-vs-embedded result directory")
    parser.add_argument("--all", action="store_true", help="Generate every plot type supported by available summary artifacts")
    parser.add_argument("--tiers", action="store_true", help="Also generate best/best99/best95 tier comparison plots")
    parser.add_argument(
        "--select-pareto-winners",
        action="store_true",
        help="Write data/csv/pareto_winners.csv for later raw IO dumps.",
    )
    parser.add_argument(
        "--include-ttnn-dumps",
        action="store_true",
        help="Include TTNN rows in the Pareto winner manifest.",
    )
    parser.add_argument(
        "--ulp-by-input",
        action="store_true",
        help="Generate ULP-by-input plots from an existing Pareto dump manifest.",
    )
    parser.add_argument("--pareto-manifest", type=existing, help="Pareto winner dump manifest CSV")
    parser.add_argument(
        "--strict-ulp-dumps",
        action="store_true",
        help="Fail if any manifest dump is missing instead of skipping incomplete activations.",
    )
    parser.add_argument("--outdir", type=existing, help="Output plot directory. Defaults to <frontier>/plots.")
    parser.add_argument(
        "--frontier-subdir",
        default="scatter",
        help="Optional subdirectory under --outdir for frontier scatters.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    shards = frontier_csvs(args.frontier)
    if not shards:
        raise SystemExit(f"plot_experiment: no frontier shard CSVs under {args.frontier / 'data' / 'csv'}")

    outdir = args.outdir or (args.frontier / "plots")
    manifest = args.pareto_manifest or (args.frontier / "data" / "csv" / "pareto_winners.csv")
    ttnn = args.ttnn or default_ttnn(args.frontier, args.native_vs_embedded)

    if args.select_pareto_winners:
        select_argv = [str(p) for p in shards] + [
            "--frontier",
            str(args.frontier),
            "--out",
            str(manifest),
            "--dump-root",
            str(args.frontier / "data" / "dumps"),
            "--plot-root",
            str(outdir),
        ]
        if ttnn:
            select_argv += ["--ttnn", str(ttnn)]
        if args.include_ttnn_dumps:
            select_argv += ["--include-ttnn"]
        old_argv = sys.argv
        try:
            sys.argv = ["select_pareto_winners.py", *select_argv]
            select_pareto_main()
        finally:
            sys.argv = old_argv

    argv = [str(p) for p in shards] + ["--outdir", str(outdir)]
    if ttnn:
        argv += ["--ttnn", str(ttnn)]
    if manifest.exists():
        argv += ["--picked", str(manifest)]
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

    if args.ulp_by_input:
        io_argv = ["--manifest", str(manifest)]
        if args.strict_ulp_dumps:
            io_argv += ["--strict"]
        old_argv = sys.argv
        try:
            sys.argv = ["plot_pareto_io.py", *io_argv]
            pareto_io_main()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    main()
