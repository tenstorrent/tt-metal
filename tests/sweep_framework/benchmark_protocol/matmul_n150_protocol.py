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
import json
import sys
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
    print(f"Wrote manifest ({stats}) -> {out}")
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

    args = p.parse_args()
    if args.cmd == "report" and not args.results and not args.results_glob:
        # Default: all json in results_export that look like sweep exports
        rd = root / "tests" / "sweep_framework" / "results_export"
        if rd.is_dir():
            args.results = [str(x) for x in sorted(rd.glob("model_traced_*.json"))]

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
