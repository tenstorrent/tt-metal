#!/usr/bin/env python3
"""Parse LogOp lines from all_gather_minimal_matmul_async_program_factory.cpp in pytest logs.

Usage:
  python parse_agmm_factory_log.py agmm-factory.log
  python parse_agmm_factory_log.py agmm-factory.log --format csv -o agmm-factory-table.csv
  python parse_agmm_factory_log.py agmm-factory.log --format markdown

Fused loudbox tests log from all_gather_minimal_matmul_async_program_factory.cpp (once per
mesh chip, deduplicated). Separate (non-fused) tests use minimal_matmul_program_factory.cpp;
use --include-minimal-matmul to include those in a second table.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

AGMM_FACTORY = "all_gather_minimal_matmul_async_program_factory.cpp"
MINIMAL_MATMUL_FACTORY = "minimal_matmul_program_factory.cpp"

TEST_START_RE = re.compile(
    r"^models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async_loudbox\.py::"
    r"test_linear_loudbox\[([^\]]+)\]"
)

# Preferred column order (keys not present are omitted).
FIELD_ORDER = [
    "M_tiles_per_core",
    "N_tiles_per_core",
    "M_blocks_per_core",
    "N_blocks_per_core",
    "in0_cb_id",
    "in1_cb_id",
    "out_cb_id",
    "intermediate_cb_id",
    "M_tiles",
    "padded_M_tiles",
    "K_tiles",
    "padded_K_tiles",
    "N_tiles",
    "padded_N_tiles",
    "M_block_tiles",
    "K_block_tiles",
    "N_block_tiles",
    "subblock_h",
    "subblock_w",
    "in0_tile_size",
    "in1_tile_size",
    "out_tile_size",
    "in2_tile_size",
    "intermediate_tile_size",
    "intermediate_data_format",
    "in0_cb_num_tiles",
    "in1_cb_num_tiles",
    "out_cb_num_tiles",
    "interm_cb_num_tiles",
    "ternary_a_cb_id",
    "ternary_c_cb_id",
]

BLOCK_END_KEYS = frozenset({"interm_cb_num_tiles", "ternary_c_cb_id"})


def _kv_pattern(factory_file: str) -> re.Pattern[str]:
    escaped = re.escape(factory_file)
    return re.compile(rf"\|\s+Op\s+\|\s+([^:]+):\s+(.+?)\s+\({escaped}:\d+\)")


def _short_test_name(full_param: str) -> str:
    # blackhole-check-fused-m1024_k6144_n768-bh1x8partialgrid -> fused-m1024
    parts = full_param.split("-")
    if len(parts) >= 3 and parts[1] == "check":
        return "-".join(parts[2:])
    return full_param


def parse_log(
    log_text: str,
    factory_file: str = AGMM_FACTORY,
) -> dict[str, dict[str, str]]:
    """Return {test_param: field_dict} with one deduplicated block per test."""
    kv_re = _kv_pattern(factory_file)
    result: dict[str, dict[str, str]] = {}
    current_test: str | None = None
    current_block: dict[str, str] | None = None

    def flush_block() -> None:
        nonlocal current_block
        if current_test is None or not current_block:
            current_block = None
            return
        if current_test not in result:
            result[current_test] = dict(current_block)
        current_block = None

    for line in log_text.splitlines():
        test_match = TEST_START_RE.match(line)
        if test_match:
            flush_block()
            current_test = test_match.group(1)
            continue

        if current_test is None or factory_file not in line:
            continue

        kv_match = kv_re.search(line)
        if not kv_match:
            continue

        key, value = kv_match.group(1), kv_match.group(2)
        if key == "No config provided, using default block sizes and core grid":
            continue

        if key == "M_tiles_per_core":
            flush_block()
            current_block = {key: value}
        elif current_block is not None:
            current_block[key] = value
            if key in BLOCK_END_KEYS:
                flush_block()

    flush_block()
    return result


def ordered_fields(all_rows: dict[str, dict[str, str]]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for name in FIELD_ORDER:
        if any(name in row for row in all_rows.values()):
            keys.append(name)
            seen.add(name)
    for row in all_rows.values():
        for name in sorted(row):
            if name not in seen:
                keys.append(name)
                seen.add(name)
    return keys


def format_markdown(rows: dict[str, dict[str, str]], title: str) -> str:
    if not rows:
        return f"### {title}\n\n_(no LogOp lines from this factory in log)_\n"

    fields = ordered_fields(rows)
    tests = sorted(rows, key=_short_test_name)

    header = ["test case", *fields]
    lines = [
        f"### {title}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for test in tests:
        row = rows[test]
        cells = [_short_test_name(test), *[row.get(f, "—") for f in fields]]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"_Parsed {len(tests)} test case(s); values deduplicated across per-chip program builds._")
    return "\n".join(lines)


def write_csv(rows: dict[str, dict[str, str]], path: Path) -> None:
    fields = ordered_fields(rows)
    tests = sorted(rows, key=_short_test_name)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["test_case", *fields])
        writer.writeheader()
        for test in tests:
            out = {"test_case": _short_test_name(test)}
            out.update(rows[test])
            writer.writerow(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path, help="pytest log (e.g. agmm-factory.log)")
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="output format (default: markdown)",
    )
    parser.add_argument("-o", "--output", type=Path, help="write to file instead of stdout")
    parser.add_argument(
        "--include-minimal-matmul",
        action="store_true",
        help="also parse minimal_matmul_program_factory.cpp (separate / non-fused path)",
    )
    args = parser.parse_args()

    log_text = args.log_file.read_text()
    agmm_rows = parse_log(log_text, AGMM_FACTORY)

    if args.format == "csv":
        if not agmm_rows:
            print("No rows for all_gather_minimal_matmul_async_program_factory.cpp", file=sys.stderr)
            return 1
        out_path = args.output or Path("agmm-factory-table.csv")
        write_csv(agmm_rows, out_path)
        print(f"Wrote {out_path}", file=sys.stderr)
        if args.include_minimal_matmul:
            mm_rows = parse_log(log_text, MINIMAL_MATMUL_FACTORY)
            if mm_rows:
                mm_path = out_path.with_name(out_path.stem + "-minimal-matmul.csv")
                write_csv(mm_rows, mm_path)
                print(f"Wrote {mm_path}", file=sys.stderr)
        return 0

    sections = [
        format_markdown(
            agmm_rows,
            "all_gather_minimal_matmul_async_program_factory.cpp (fused AGMM)",
        )
    ]
    if args.include_minimal_matmul:
        mm_rows = parse_log(log_text, MINIMAL_MATMUL_FACTORY)
        sections.append(format_markdown(mm_rows, "minimal_matmul_program_factory.cpp (separate path)"))

    output = "\n".join(sections)
    if args.output:
        args.output.write_text(output + "\n")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
