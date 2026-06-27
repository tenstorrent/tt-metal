#!/usr/bin/env python3
"""Select a bounded set of Pareto frontier configs for IO dump plots.

This does not run hardware. It converts frontier summary CSVs into a manifest
that a dump runner can consume, and predicts the canonical dump/plot locations.
"""

import argparse
import csv
import glob
import os
from pathlib import Path
import sys

try:
    from .frontier_scatter import (
        dtype_label,
        fnum,
        load,
        load_ttnn,
        pareto,
        slug,
        ttnn_for,
    )
except ImportError:
    from frontier_scatter import (
        dtype_label,
        fnum,
        load,
        load_ttnn,
        pareto,
        slug,
        ttnn_for,
    )


FIELDNAMES = [
    "activation",
    "dtype",
    "role",
    "csv",
    "coeff_csv",
    "method",
    "degree",
    "segments",
    "max_ulp",
    "runtime_us",
    "ttnn_maxulp",
    "ttnn_us",
    "dump_csv",
    "plot_png",
    "status",
]


def existing(path):
    return Path(path).expanduser().resolve()


def repo_results_dir():
    return Path(__file__).resolve().parents[2] / "results"


def frontier_csvs(frontier_dir):
    csv_dir = frontier_dir / "data" / "csv"
    patterns = [
        csv_dir / "frontier*chip*.csv",
        frontier_dir / "shards" / "frontier*chip*.csv",
    ]
    for pattern in patterns:
        shards = sorted(pattern.parent.glob(pattern.name))
        if shards:
            return shards
    return []


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        paths.extend(matches or [Path(pattern)])
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"select_pareto_winners: input CSV not found: {missing[0]}")
    return [str(p) for p in paths]


def default_ttnn(frontier_dir):
    candidates = [
        frontier_dir / "data" / "csv" / "ttnn_ref.csv",
        repo_results_dir() / "native_vs_embedded" / "bf16" / "data" / "csv" / "ttnn_ref.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def default_coeff_dir():
    fit = os.environ.get("TT_POLY_FIT_DIR")
    candidates = []
    if fit:
        candidates.append(Path(fit) / "data" / "coefficients")
    candidates += [
        Path.home() / "tt-polynomial-fitter" / "data" / "coefficients",
        Path.home() / "workspace" / "tt-polynomial-fitter" / "data" / "coefficients",
        Path("/localdev") / os.environ.get("USER", "") / "tt-polynomial-fitter" / "data" / "coefficients",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def group_rows(rows, explicit_dtype):
    grouped = {}
    for row in rows:
        key = (row["activation"], row["_dtype"]) if explicit_dtype else (row["activation"], None)
        grouped.setdefault(key, []).append(row)
    return grouped


def dedupe_configs(rows):
    best = {}
    for row in rows:
        precision = (row.get("precision") or row.get("dtype") or "").strip()
        key = ((row.get("csv") or "").strip(), precision)
        old = best.get(key)
        if old is None:
            best[key] = row
            continue
        old_key = (old["_ulp"], old["_us"], old.get("csv") or "")
        new_key = (row["_ulp"], row["_us"], row.get("csv") or "")
        if new_key < old_key:
            best[key] = row
    return list(best.values())


def select_for_group(front, tus, tulp, max_configs):
    if not front:
        return []

    def source_priority(row):
        metric = (row.get("metric") or row.get("source_metric") or "").strip()
        if metric == "ulp":
            return 0
        if metric == "max":
            return 1
        if metric == "mae":
            return 2
        return 0

    def by_runtime_then_ulp(point):
        us, ulp, row = point
        return (us, ulp, source_priority(row), row.get("csv") or "")

    if tulp is not None:
        ttnn_ulp_matches = [(us, ulp, row) for us, ulp, row in front if ulp <= tulp]
        if ttnn_ulp_matches:
            us, ulp, row = min(ttnn_ulp_matches, key=by_runtime_then_ulp)
            return [("fastest_ttnn_ulp_match", "ttnn_ulp_match", us, ulp, row)]

    best_ulp = min(ulp for _, ulp, _ in front)
    best_ulp_points = [(us, ulp, row) for us, ulp, row in front if ulp == best_ulp]
    us, ulp, row = min(best_ulp_points, key=by_runtime_then_ulp)
    return [("fastest_min_ulp", "fallback_min_ulp", us, ulp, row)]


def coeff_path(row, coeff_dir):
    name = (row.get("csv") or "").strip()
    if not name:
        return ""
    path = Path(name)
    if path.exists():
        return str(path.resolve())
    if coeff_dir:
        candidate = coeff_dir / name
        if candidate.exists():
            return str(candidate.resolve())
    return name


def manifest_row(act, dtype, role, status, us, ulp, row, tus, tulp, coeff_dir, dump_root, plot_root):
    dtype_name = dtype or (row.get("precision") or row.get("dtype") or "bf16")
    cfg_slug = slug(Path(row.get("csv") or "config").stem)
    role_slug = slug(role.replace("|", "_"))
    dump = dump_root / "frontier" / slug(dtype_name) / slug(act) / f"{role_slug}_{cfg_slug}.npz"
    plot = plot_root / "ulp_by_input" / f"{slug(act)}.png"
    return {
        "activation": act,
        "dtype": dtype_name,
        "role": role,
        "csv": row.get("csv") or "",
        "coeff_csv": coeff_path(row, coeff_dir),
        "method": row.get("method") or "",
        "degree": row.get("degree") or "",
        "segments": row.get("segments") or "",
        "max_ulp": ulp,
        "runtime_us": us,
        "ttnn_maxulp": "" if tulp is None else tulp,
        "ttnn_us": "" if tus is None else tus,
        "dump_csv": str(dump),
        "plot_png": str(plot),
        "status": status,
    }


def ttnn_manifest_row(act, dtype, tus, tulp, dump_root, plot_root):
    dtype_name = dtype or "bf16"
    dump = dump_root / "ttnn" / slug(dtype_name) / slug(act) / "ttnn.npz"
    plot = plot_root / "ulp_by_input" / f"{slug(act)}.png"
    return {
        "activation": act,
        "dtype": dtype_name,
        "role": "ttnn",
        "csv": "",
        "coeff_csv": "",
        "method": "ttnn",
        "degree": "",
        "segments": "",
        "max_ulp": tulp,
        "runtime_us": tus,
        "ttnn_maxulp": tulp,
        "ttnn_us": tus,
        "dump_csv": str(dump),
        "plot_png": str(plot),
        "status": "ttnn_ref",
    }


def parse_args():
    default_frontier = repo_results_dir() / "frontier" / "bf16"
    parser = argparse.ArgumentParser(description="Select frontier Pareto winners for raw IO dump plots.")
    parser.add_argument("csvs", nargs="*", help="Frontier shard CSV paths or globs")
    parser.add_argument("--frontier", type=existing, default=default_frontier)
    parser.add_argument("--ttnn", type=existing, help="TTNN reference CSV")
    parser.add_argument("--out", type=existing, help="Manifest CSV path")
    parser.add_argument("--dump-root", type=existing, help="Canonical raw dump root")
    parser.add_argument("--plot-root", type=existing, help="Canonical plot root")
    parser.add_argument("--coeff-dir", type=existing, help="Coefficient CSV directory")
    parser.add_argument("--max-configs-per-activation", type=int, default=3)
    parser.add_argument(
        "--include-ttnn", action="store_true", help="Add one TTNN manifest row per activation with a TTNN ref"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = expand_paths(args.csvs) if args.csvs else [str(p) for p in frontier_csvs(args.frontier)]
    if not paths:
        raise SystemExit(f"select_pareto_winners: no frontier shard CSVs under {args.frontier / 'data' / 'csv'}")

    out = args.out or (args.frontier / "data" / "csv" / "pareto_winners.csv")
    dump_root = args.dump_root or (args.frontier / "data" / "dumps")
    plot_root = args.plot_root or (args.frontier / "plots")
    coeff_dir = args.coeff_dir or default_coeff_dir()

    rows, explicit_dtype = load(paths)
    ttnn_path = args.ttnn or default_ttnn(args.frontier)
    ttnn, has_typed_ttnn = load_ttnn(ttnn_path) if ttnn_path else ({}, False)

    selected = []
    groups = group_rows(rows, explicit_dtype)
    for (act, dtype), group in sorted(groups.items()):
        compiled = [
            r
            for r in group
            if r["_ok"]
            and r["_us"] is not None
            and r["_ulp"] is not None
            and r["_us"] > 0
            and r["_ulp"] >= 0
            and (r.get("metric") in (None, "", "ulp"))
        ]
        compiled = dedupe_configs(compiled)
        front = pareto([(r["_us"], r["_ulp"], r) for r in compiled])
        tus, tulp = ttnn_for(ttnn, act, dtype, has_typed_ttnn)
        for role, status, us, ulp, row in select_for_group(front, tus, tulp, args.max_configs_per_activation):
            selected.append(
                manifest_row(act, dtype, role, status, us, ulp, row, tus, tulp, coeff_dir, dump_root, plot_root)
            )
        if args.include_ttnn and tus is not None and tulp is not None:
            selected.append(ttnn_manifest_row(act, dtype, tus, tulp, dump_root, plot_root))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)

    by_dtype = {}
    for row in selected:
        by_dtype[row["dtype"]] = by_dtype.get(row["dtype"], 0) + 1
    counts = ", ".join(f"{dtype_label(k)}={v}" for k, v in sorted(by_dtype.items())) or "0"
    print(f"# selected {len(selected)} manifest rows across {len(groups)} activation groups ({counts})")
    print(f"# manifest -> {out}")
    print(f"# raw dump root -> {dump_root}")
    print(f"# IO plots -> {plot_root / 'ulp_by_input'}")


if __name__ == "__main__":
    main()
