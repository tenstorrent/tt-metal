#!/usr/bin/env python3
"""Generate ULP-by-input plots from a Pareto winner dump manifest."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

try:
    from .ulp_by_input import main as ulp_main
except ImportError:
    from ulp_by_input import main as ulp_main


def existing(path):
    return Path(path).expanduser().resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot raw IO dumps selected by select_pareto_winners.py.")
    parser.add_argument("--manifest", required=True, type=existing)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any selected dump is missing. Default: skip incomplete activations.",
    )
    parser.add_argument("--max-points", type=int, default=50000)
    return parser.parse_args()


def short_label(row):
    role = row.get("role") or "config"
    if role == "ttnn":
        return "TTNN"
    method = row.get("method") or "config"
    degree = row.get("degree") or ""
    segments = row.get("segments") or ""
    bits = [role.split("|")[0], method]
    if degree:
        bits.append(f"d{degree}")
    if segments:
        bits.append(f"s{segments}")
    return " ".join(bits)


def main():
    args = parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"plot_pareto_io: manifest not found: {args.manifest}")

    groups = defaultdict(list)
    with args.manifest.open() as f:
        for row in csv.DictReader(f):
            act = row.get("activation")
            dtype = row.get("dtype") or "bf16"
            dump = Path(row.get("dump_csv") or "")
            plot = Path(row.get("plot_png") or "")
            if not act or not dump or not plot:
                continue
            groups[(act, dtype, plot)].append(row)

    made = skipped = missing_count = 0
    for (act, dtype, plot), rows in sorted(groups.items()):
        missing = [row for row in rows if not Path(row["dump_csv"]).exists()]
        if missing:
            missing_count += len(missing)
            msg = f"plot_pareto_io: {act} {dtype} missing {len(missing)}/{len(rows)} dumps"
            if args.strict:
                first = missing[0]["dump_csv"]
                raise SystemExit(f"{msg}; first missing: {first}")
            print(f"# skip {msg}", file=sys.stderr)
            skipped += 1
            continue

        argv = [
            "ulp_by_input.py",
            "--activation",
            act,
            "--precision",
            dtype,
            "--out",
            str(plot),
            "--max-points",
            str(args.max_points),
        ]
        for row in rows:
            argv += ["--series", f"{short_label(row)}={row['dump_csv']}"]

        old_argv = sys.argv
        try:
            sys.argv = argv
            ulp_main()
        finally:
            sys.argv = old_argv
        made += 1

    print(f"# pareto IO plots made={made} skipped={skipped} missing_dumps={missing_count}")


if __name__ == "__main__":
    main()
