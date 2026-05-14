#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Analyze Tracy ops_perf_results CSV for ``test_chunked_gdn_kernel_correctness``.

Uses DEVICE KERNEL DURATION [ns] as the timing metric. Ops between
``chunked_gdn_start`` and ``chunked_gdn_stop`` signposts are grouped per
parametrized test (num_heads=8/32/64 by default).

Usage:
    python analyze_chunked_gdn_profiler.py path/to/ops_perf_results_*.csv
    python analyze_chunked_gdn_profiler.py --summary path/to/ops_perf_results_*.csv
    python analyze_chunked_gdn_profiler.py --csv path/to/ops_perf_results_*.csv --json-out report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

DURATION_COL = "DEVICE KERNEL DURATION [ns]"
START_SIGNPOST = "chunked_gdn_op_start"
STOP_SIGNPOST = "chunked_gdn_op_stop"
INPUT_MEMORY_COL_RE = re.compile(r"^INPUT_\d+_MEMORY$")


def parse_test_label(attributes: str) -> str:
    """Turn signpost ATTRIBUTES into a short test label."""
    if not attributes or (isinstance(attributes, float) and pd.isna(attributes)):
        return "unknown"
    attrs = str(attributes).strip()
    m = re.search(r"num_heads=(\d+)", attrs)
    if m:
        return f"num_heads={m.group(1)}"
    return attrs.replace("; ", ", ")


def buffer_type_from_memory(memory: str) -> str | None:
    """
    Parse buffer type from Tracy MEMORY field, e.g. ``DEV_1_DRAM_INTERLEAVED`` -> ``DRAM``.
    """
    parts = memory.split("_")
    if len(parts) >= 3 and parts[0] == "DEV":
        return parts[2].upper()
    if "DRAM" in memory.upper():
        return "DRAM"
    if "L1" in memory.upper():
        return "L1"
    return None


def collect_input_buffer_types(row: pd.Series, memory_cols: list[str]) -> set[str]:
    types: set[str] = set()
    for col in memory_cols:
        val = row.get(col)
        if pd.isna(val) or str(val).strip() == "":
            continue
        buf = buffer_type_from_memory(str(val))
        if buf:
            types.add(buf)
    return types


def classify_memory_bucket(buffer_types: set[str]) -> str:
    """Exclusive bucket for an op based on its input buffer types."""
    has_dram = "DRAM" in buffer_types
    has_l1 = "L1" in buffer_types
    if has_dram and has_l1:
        return "mixed_dram_and_l1"
    if has_dram:
        return "dram_only"
    if has_l1:
        return "l1_only"
    return "unknown"


def iter_chunked_gdn_regions(df: pd.DataFrame):
    """Yield (test_label, attributes, op_rows) for each signpost-delimited region."""
    current_label: str | None = None
    current_attrs: str | None = None
    current_rows: list[pd.Series] = []

    for _, row in df.iterrows():
        if row.get("OP TYPE") == "signpost":
            op_code = row.get("OP CODE")
            if op_code == START_SIGNPOST:
                if current_rows and current_label is not None:
                    yield current_label, current_attrs, current_rows
                current_attrs = row.get("ATTRIBUTES")
                current_label = parse_test_label(current_attrs)
                current_rows = []
            elif op_code == STOP_SIGNPOST:
                if current_rows and current_label is not None:
                    yield current_label, current_attrs, current_rows
                current_label = None
                current_attrs = None
                current_rows = []
            continue

        if current_label is None:
            continue

        duration = row.get(DURATION_COL)
        if pd.isna(duration):
            continue
        current_rows.append(row)

    if current_rows and current_label is not None:
        yield current_label, current_attrs, current_rows


def analyze_region(
    rows: list[pd.Series],
    memory_cols: list[str],
) -> dict:
    by_op_code: dict[str, float] = defaultdict(float)
    op_code_counts: dict[str, int] = defaultdict(int)

    total_ns = 0.0
    any_dram_ns = 0.0
    any_l1_ns = 0.0
    exclusive_ns: dict[str, float] = defaultdict(float)
    exclusive_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        duration = float(row[DURATION_COL])
        total_ns += duration

        op_code = str(row.get("OP CODE", "unknown"))
        by_op_code[op_code] += duration
        op_code_counts[op_code] += 1

        buffer_types = collect_input_buffer_types(row, memory_cols)
        if "DRAM" in buffer_types:
            any_dram_ns += duration
        if "L1" in buffer_types:
            any_l1_ns += duration

        bucket = classify_memory_bucket(buffer_types)
        exclusive_ns[bucket] += duration
        exclusive_counts[bucket] += 1

    op_breakdown = []
    for op_code, ns in sorted(by_op_code.items(), key=lambda x: -x[1]):
        pct = (ns / total_ns * 100.0) if total_ns else 0.0
        op_breakdown.append(
            {
                "op_code": op_code,
                "duration_ns": ns,
                "duration_ms": ns / 1e6,
                "pct_of_total": pct,
                "call_count": op_code_counts[op_code],
            }
        )

    return {
        "total_duration_ns": total_ns,
        "total_duration_ms": total_ns / 1e6,
        "op_count": len(rows),
        "by_op_code": op_breakdown,
        "input_memory": {
            "any_dram_input": {
                "duration_ns": any_dram_ns,
                "duration_ms": any_dram_ns / 1e6,
                "pct_of_total": (any_dram_ns / total_ns * 100.0) if total_ns else 0.0,
            },
            "any_l1_input": {
                "duration_ns": any_l1_ns,
                "duration_ms": any_l1_ns / 1e6,
                "pct_of_total": (any_l1_ns / total_ns * 100.0) if total_ns else 0.0,
            },
            "exclusive": {
                bucket: {
                    "duration_ns": exclusive_ns[bucket],
                    "duration_ms": exclusive_ns[bucket] / 1e6,
                    "pct_of_total": (exclusive_ns[bucket] / total_ns * 100.0) if total_ns else 0.0,
                    "op_count": exclusive_counts[bucket],
                }
                for bucket in sorted(exclusive_ns.keys())
            },
        },
    }


def format_report(results: list[dict], csv_path: Path, *, summary_only: bool = False) -> str:
    lines = [
        "Chunked GDN profiler analysis",
        f"CSV: {csv_path}",
        f"Metric: {DURATION_COL}",
        "",
    ]

    if summary_only:
        lines.append(f"{'Test':<60} {'Total (ms)':>12} {'Ops':>6}")
        lines.append("-" * 80)
        for entry in results:
            s = entry["summary"]
            mem = s["input_memory"]
            lines.append(f"{entry['attributes']:<60} " f"{s['total_duration_ms']:>12.3f} " f"{s['op_count']:>6} ")
        return "\n".join(lines)

    for entry in results:
        lines.append("=" * 72)
        lines.append(f"Test: {entry['test_label']}")
        if entry.get("attributes"):
            lines.append(f"  {entry['attributes']}")
        summary = entry["summary"]
        lines.append(f"  Total: {summary['total_duration_ms']:.3f} ms ({summary['op_count']} ops)")
        lines.append("")
        lines.append("  Time by operation (OP CODE):")
        for op in summary["by_op_code"]:
            lines.append(
                f"    {op['op_code']:<32} {op['duration_ms']:>10.3f} ms  "
                f"{op['pct_of_total']:>6.2f}%  ({op['call_count']} calls)"
            )

        mem = summary["input_memory"]
        lines.append("")
        lines.append("  Time by input memory (ops may be counted in multiple buckets):")
        lines.append(
            f"    any DRAM input: {mem['any_dram_input']['duration_ms']:>10.3f} ms  "
            f"({mem['any_dram_input']['pct_of_total']:.2f}%)"
        )
        lines.append(
            f"    any L1 input:   {mem['any_l1_input']['duration_ms']:>10.3f} ms  "
            f"({mem['any_l1_input']['pct_of_total']:.2f}%)"
        )
        lines.append("")
        lines.append("  Exclusive input-memory buckets:")
        for bucket, stats in mem["exclusive"].items():
            lines.append(
                f"    {bucket:<22} {stats['duration_ms']:>10.3f} ms  "
                f"{stats['pct_of_total']:>6.2f}%  ({stats['op_count']} ops)"
            )
        lines.append("")

    return "\n".join(lines)


def analyze_csv(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path, low_memory=False)
    if DURATION_COL not in df.columns:
        raise ValueError(f"CSV missing required column: {DURATION_COL}")

    df[DURATION_COL] = pd.to_numeric(df[DURATION_COL], errors="coerce")
    memory_cols = [c for c in df.columns if INPUT_MEMORY_COL_RE.match(c)]

    results = []
    for test_label, attributes, rows in iter_chunked_gdn_regions(df):
        results.append(
            {
                "test_label": test_label,
                "attributes": attributes,
                "summary": analyze_region(rows, memory_cols),
            }
        )

    if not results:
        raise ValueError(
            f"No regions found between {START_SIGNPOST!r} and {STOP_SIGNPOST!r}. "
            "Was the test run with Tracy signposts enabled?"
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze chunked GDN Tracy profiler CSV (test_chunked_gdn_kernel_correctness)."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        help="Path to ops_perf_results_*.csv",
    )
    parser.add_argument(
        "--csv",
        dest="csv_flag",
        type=Path,
        help="Path to ops_perf_results_*.csv (alternative to positional)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print only per-profile totals and input-memory split (no per-op breakdown)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to write machine-readable JSON summary",
    )
    args = parser.parse_args()

    csv_path = args.csv_flag or args.csv
    if csv_path is None:
        parser.error("Provide a CSV path as a positional argument or via --csv")
    if not csv_path.is_file():
        parser.error(f"CSV not found: {csv_path}")

    results = analyze_csv(csv_path)
    report = format_report(results, csv_path, summary_only=args.summary)
    print(report)

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"Wrote JSON summary to {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
