#!/usr/bin/env python3
"""Generate ULP-by-input plots from a Pareto winner dump manifest."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import subprocess
import sys

try:
    from .ulp_by_input import main as ulp_main
except ImportError:
    from ulp_by_input import main as ulp_main


def existing(path):
    return Path(path).expanduser().resolve()


LEGACY_TT_METAL_ROOT = None


def repo_root():
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())


def resolve_repo_path(path, repo):
    p = Path(path or "")
    if not p:
        return p
    global LEGACY_TT_METAL_ROOT
    if LEGACY_TT_METAL_ROOT:
        try:
            return repo / p.relative_to(LEGACY_TT_METAL_ROOT)
        except ValueError:
            pass
    return p


def plot_path(row, manifest, repo):
    p = resolve_repo_path(row.get("plot_png") or "", repo)
    if p and str(p).startswith(str(repo)):
        dtype = row.get("dtype") or "bf16"
        if p.parent.name == dtype and p.parent.parent.name == "ulp_by_input":
            return p.parent.parent / p.name
        return p
    dtype = row.get("dtype") or "bf16"
    act = row.get("activation")
    run_dir = manifest.parent.parent.parent
    return run_dir / "plots" / "ulp_by_input" / f"{act}.png"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot raw IO dumps selected by select_pareto_winners.py.")
    parser.add_argument("--manifest", required=True, type=existing)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any selected dump is missing. Default: skip incomplete activations.",
    )
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument(
        "--legacy-tt-metal-root",
        type=Path,
        help="Optional old checkout root used to remap legacy absolute manifest paths.",
    )
    return parser.parse_args()


def fnum(value):
    try:
        v = float(value)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def choose_ours(rows):
    frontier = [row for row in rows if row.get("role") != "ttnn"]
    if not frontier:
        return None, "no_frontier"

    ttnn_ulp = None
    for row in rows:
        if row.get("role") == "ttnn":
            ttnn_ulp = fnum(row.get("ttnn_maxulp") or row.get("max_ulp"))
            break
    if ttnn_ulp is None:
        ttnn_ulp = fnum(frontier[0].get("ttnn_maxulp"))

    def sort_key(row):
        runtime = fnum(row.get("runtime_us"))
        ulp = fnum(row.get("max_ulp"))
        return (
            float("inf") if runtime is None else runtime,
            float("inf") if ulp is None else ulp,
            row.get("csv") or row.get("dump_csv") or "",
        )

    if ttnn_ulp is not None:
        matches = [
            row for row in frontier if (fnum(row.get("max_ulp")) is not None and fnum(row.get("max_ulp")) <= ttnn_ulp)
        ]
        if matches:
            return min(matches, key=sort_key), "fastest_ttnn_ulp_match"

    ulps = [fnum(row.get("max_ulp")) for row in frontier]
    ulps = [ulp for ulp in ulps if ulp is not None]
    if not ulps:
        return min(frontier, key=sort_key), "fastest_no_numeric_ulp"
    best_ulp = min(ulps)
    best = [row for row in frontier if fnum(row.get("max_ulp")) == best_ulp]
    return min(best, key=sort_key), "fastest_min_ulp"


def choose_rows(rows):
    selected, reason = choose_ours(rows)
    if selected is None:
        return [], reason
    selected = dict(selected)
    selected["_label"] = "ours"
    chosen = [selected]
    ttnn = next((dict(row) for row in rows if row.get("role") == "ttnn"), None)
    if ttnn:
        ttnn["_label"] = "TTNN"
        chosen.append(ttnn)
    return chosen, reason


def main():
    args = parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"plot_pareto_io: manifest not found: {args.manifest}")

    repo = repo_root()
    global LEGACY_TT_METAL_ROOT
    LEGACY_TT_METAL_ROOT = args.legacy_tt_metal_root.expanduser().resolve() if args.legacy_tt_metal_root else None
    groups = defaultdict(list)
    with args.manifest.open() as f:
        for row in csv.DictReader(f):
            act = row.get("activation")
            dtype = row.get("dtype") or "bf16"
            dump = resolve_repo_path(row.get("dump_csv") or "", repo)
            plot = plot_path(row, args.manifest, repo)
            if not act or not dump or not plot:
                continue
            row = dict(row)
            row["dump_csv"] = str(dump)
            row["plot_png"] = str(plot)
            groups[(act, dtype, plot)].append(row)

    made = skipped = missing_count = fallback_count = 0
    for (act, dtype, plot), rows in sorted(groups.items()):
        rows, reason = choose_rows(rows)
        if reason != "fastest_ttnn_ulp_match":
            fallback_count += 1
            print(f"# {act} {dtype}: {reason}", file=sys.stderr)
        if not rows:
            skipped += 1
            continue
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
            "--split-series",
        ]
        for row in rows:
            argv += ["--series", f"{row['_label']}={row['dump_csv']}"]

        old_argv = sys.argv
        try:
            sys.argv = argv
            ulp_main()
        finally:
            sys.argv = old_argv
        made += 1

    print(
        f"# pareto IO plots made={made} skipped={skipped} " f"missing_dumps={missing_count} fallbacks={fallback_count}"
    )


if __name__ == "__main__":
    main()
