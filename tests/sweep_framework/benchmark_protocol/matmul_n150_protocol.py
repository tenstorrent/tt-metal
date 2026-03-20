# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Matmul (model-traced) N150 benchmark protocol — Milestone 1 CLI.

Subcommands:
  partition   — build smoke/train/holdout manifest from vectors_export
  write-json  — write subset JSON files for --vector-source file runs
  report      — summarize results_export JSON + manifest
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TESTS_ROOT = _REPO_ROOT / "tests"
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from sweep_framework.benchmark_protocol.matmul_n150_partition import (
    DEFAULT_MODULE_STEM,
    DEFAULT_SUITE_NAME,
    PartitionParams,
    build_manifest,
    filter_suite_vectors_by_traced_source,
    load_model_traced_suite,
    partition_hashes,
    write_suite_subset_json,
)
from sweep_framework.benchmark_protocol.matmul_n150_report import build_report


def _repo_root() -> Path:
    return _REPO_ROOT


def _default_vectors_export() -> Path:
    return Path(__file__).resolve().parent.parent / "vectors_export"


def _default_protocol_dir() -> Path:
    return Path(__file__).resolve().parent / "generated"


def cmd_partition(args: argparse.Namespace) -> int:
    export_dir = Path(args.vectors_export)
    params = PartitionParams(
        smoke_max=args.smoke_max,
        train_fraction_of_remainder=args.train_fraction,
    )
    suite = load_model_traced_suite(export_dir, module_stem=args.module_stem, suite_name=args.suite_name)
    unfiltered_count = len(suite)
    suite = filter_suite_vectors_by_traced_source(
        suite,
        include_pattern=args.source_include,
        exclude_pattern=args.source_exclude,
    )
    if not suite:
        print("No vectors left after traced_source filtering.", file=sys.stderr)
        return 1
    smoke, train, holdout, stats = partition_hashes(suite, params)
    manifest = build_manifest(
        smoke,
        train,
        holdout,
        stats,
        module_name=args.module_name,
        suite_name=args.suite_name,
        params=params,
        vectors_export_dir=str(export_dir),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"Wrote manifest ({stats}) -> {out} "
        f"[vectors: {len(suite)}/{unfiltered_count}, "
        f"include={args.source_include!r}, exclude={args.source_exclude!r}]"
    )
    return 0


def cmd_analyze_sources(args: argparse.Namespace) -> int:
    def _coerce_shape(v: object) -> tuple[int, ...] | None:
        if isinstance(v, (list, tuple)):
            try:
                return tuple(int(x) for x in v)
            except (TypeError, ValueError):
                return None
        if isinstance(v, str):
            try:
                parsed = ast.literal_eval(v)
            except (SyntaxError, ValueError):
                return None
            if isinstance(parsed, (list, tuple)):
                try:
                    return tuple(int(x) for x in parsed)
                except (TypeError, ValueError):
                    return None
        return None

    export_dir = Path(args.vectors_export)
    suite = load_model_traced_suite(export_dir, module_stem=args.module_stem, suite_name=args.suite_name)
    suite = filter_suite_vectors_by_traced_source(
        suite,
        include_pattern=args.source_include,
        exclude_pattern=args.source_exclude,
    )
    if not suite:
        print("No vectors to analyze after traced_source filtering.", file=sys.stderr)
        return 1

    source_counts: Counter[str] = Counter()
    weighted_shape_counts: Counter[str] = Counter()
    weighted_shape_work: defaultdict[str, float] = defaultdict(float)

    for vec in suite.values():
        src = str(vec.get("traced_source", "unknown"))
        source_counts[src] += 1

        sa = _coerce_shape(vec.get("input_a_shape"))
        sb = _coerce_shape(vec.get("input_b_shape"))
        if sa is None or sb is None:
            continue
        if len(sa) < 2 or len(sb) < 2:
            continue
        try:
            m, k, n = int(sa[-2]), int(sa[-1]), int(sb[-1])
            batch = 1
            for d in sa[:-2]:
                batch *= int(d)
        except (TypeError, ValueError):
            continue

        shape_key = f"A={tuple(sa)} B={tuple(sb)} src={src}"
        weighted_shape_counts[shape_key] += 1
        weighted_shape_work[shape_key] += float(batch * m * k * n)

    print("\n=== source counts ===")
    for src, count in source_counts.most_common(args.top_n):
        print(f"  {count:4d}  {src}")

    print("\n=== top weighted shape/source patterns ===")
    top_weighted = sorted(weighted_shape_work.items(), key=lambda kv: kv[1], reverse=True)[: args.top_n]
    for shape_key, work in top_weighted:
        print(f"  work={work:.3e}  count={weighted_shape_counts[shape_key]}  {shape_key}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vectors_total": len(suite),
            "source_include": args.source_include,
            "source_exclude": args.source_exclude,
            "source_counts": source_counts,
            "top_weighted_shape_patterns": [
                {
                    "shape_key": key,
                    "count": weighted_shape_counts[key],
                    "work_batch_mkn": work,
                }
                for key, work in top_weighted
            ],
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote source analysis JSON -> {out}")
    return 0


def cmd_write_json(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    export_dir = Path(args.vectors_export)
    suite = load_model_traced_suite(export_dir, module_stem=args.module_stem, suite_name=args.suite_name)
    out_dir = Path(args.output_dir)
    for part in ("smoke", "train", "holdout"):
        hashes = manifest.get(part, [])
        if not hashes:
            continue
        out_path = out_dir / f"matmul_n150_{part}.json"
        write_suite_subset_json(suite, hashes, out_path, suite_name=args.suite_name)
        print(f"Wrote {len(hashes)} vectors -> {out_path}")
    all_hashes = manifest.get("smoke", []) + manifest.get("train", []) + manifest.get("holdout", [])
    combined = out_dir / "matmul_n150_protocol_all.json"
    write_suite_subset_json(suite, all_hashes, combined, suite_name=args.suite_name)
    print(f"Wrote {len(all_hashes)} vectors -> {combined}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    paths = [Path(p) for p in (args.results or [])]
    if args.results_glob:
        from glob import glob

        paths.extend(Path(p) for p in sorted(glob(args.results_glob)))
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("No result files found.", file=sys.stderr)
        return 1
    report = build_report(manifest, paths)
    if args.json_out:
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote JSON report -> {outp}")
    print("\n=== report rows ===")
    print(f"  total_result_rows_raw: {report.get('total_result_rows_raw')}")
    print(f"  total_result_rows_deduped: {report.get('total_result_rows_deduped')}")
    print(f"  duplicate_result_rows_dropped: {report.get('duplicate_result_rows_dropped')}")

    def fmt_part(title: str, d: dict) -> None:
        print(f"\n=== {title} ===")
        for k in (
            "vectors_expected",
            "results_matched",
            "hashes_missing_in_results",
            "pass_count",
            "pass_rate",
            "timeout_or_hang_count",
            "e2e_ms_p50",
            "e2e_ms_p95",
            "e2e_samples",
            "memory_p50_peak_l1_per_core_bytes",
            "memory_p50_peak_l1_aggregate_bytes",
        ):
            if k in d:
                print(f"  {k}: {d[k]}")

    fmt_part("smoke", report["smoke"])
    fmt_part("train", report["train"])
    fmt_part("holdout", report["holdout"])
    print("\n=== train vs holdout ===")
    for k, v in report["train_vs_holdout"].items():
        print(f"  {k}: {v}")
    return 0


def main() -> int:
    root = _repo_root()
    default_ve = _default_vectors_export()
    default_gen = _default_protocol_dir()

    p = argparse.ArgumentParser(description="Matmul N150 benchmark protocol (local milestone 1)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("partition", help="Create manifest JSON from vectors_export")
    pp.add_argument(
        "--vectors-export",
        default=str(default_ve),
        help=f"vectors_export directory (default: {default_ve})",
    )
    pp.add_argument("--module-stem", default=DEFAULT_MODULE_STEM)
    pp.add_argument("--module-name", default=DEFAULT_MODULE_STEM, help="Recorded in manifest (sweep module name)")
    pp.add_argument("--suite-name", default=DEFAULT_SUITE_NAME)
    pp.add_argument("--smoke-max", type=int, default=16)
    pp.add_argument("--train-fraction", type=float, default=0.58, dest="train_fraction")
    pp.add_argument(
        "--source-include",
        default=None,
        help="Regex: keep only vectors whose traced_source matches",
    )
    pp.add_argument(
        "--source-exclude",
        default=None,
        help="Regex: drop vectors whose traced_source matches",
    )
    pp.add_argument(
        "--output",
        "-o",
        default=str(default_gen / "matmul_n150_protocol_manifest.json"),
        help="Manifest output path",
    )
    pp.set_defaults(func=cmd_partition)

    pw = sub.add_parser("write-json", help="Write slice JSON files for sweeps_runner --vector-source file")
    pw.add_argument("--manifest", "-m", default=str(default_gen / "matmul_n150_protocol_manifest.json"))
    pw.add_argument("--vectors-export", default=str(default_ve))
    pw.add_argument("--module-stem", default=DEFAULT_MODULE_STEM)
    pw.add_argument("--suite-name", default=DEFAULT_SUITE_NAME)
    pw.add_argument("--output-dir", "-o", default=str(default_gen))
    pw.set_defaults(func=cmd_write_json)

    pr = sub.add_parser("report", help="Summarize results_export JSON files using manifest")
    pr.add_argument("--manifest", "-m", default=str(default_gen / "matmul_n150_protocol_manifest.json"))
    pr.add_argument("--results", nargs="*", help="Explicit results_export .json files (OpTest list)")
    pr.add_argument("--results-glob", help="Glob for additional result files (e.g. results_export/model_traced_*.json)")
    pr.add_argument("--json-out", help="Write full report JSON to this path")
    pr.set_defaults(func=cmd_report)

    pa = sub.add_parser("analyze-sources", help="Show traced_source counts and top weighted shape patterns")
    pa.add_argument("--vectors-export", default=str(default_ve))
    pa.add_argument("--module-stem", default=DEFAULT_MODULE_STEM)
    pa.add_argument("--suite-name", default=DEFAULT_SUITE_NAME)
    pa.add_argument("--source-include", default=None, help="Regex source include filter")
    pa.add_argument("--source-exclude", default=None, help="Regex source exclude filter")
    pa.add_argument("--top-n", type=int, default=15, help="Rows to print in each section")
    pa.add_argument("--json-out", help="Optional JSON output for analysis")
    pa.set_defaults(func=cmd_analyze_sources)

    args = p.parse_args()
    if args.cmd == "report" and not args.results and not args.results_glob:
        # Default: all json in results_export that look like sweep exports
        rd = root / "tests" / "sweep_framework" / "results_export"
        if rd.is_dir():
            args.results = [str(x) for x in sorted(rd.glob("model_traced_*.json"))]

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
