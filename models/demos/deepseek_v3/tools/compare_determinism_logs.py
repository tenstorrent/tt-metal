#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare DeepSeek determinism JSONL logs from repeated runs.

This utility is intended for debugging non-deterministic DeepSeek behavior by
loading two per-step JSONL logs, optionally stripping volatile context fields,
and reporting the first mismatched records with compact summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two determinism JSONL logs and print a concise mismatch report."
    )
    parser.add_argument(
        "--ignore-context",
        action="store_true",
        help="Drop all fields named 'context' before comparing records.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N processed records (default: 1000). Use 0 to disable.",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=10,
        help="Maximum mismatches to print before stopping compare (default: 10).",
    )
    parser.add_argument("left_log", type=Path, help="First JSONL log to compare.")
    parser.add_argument("right_log", type=Path, help="Second JSONL log to compare.")
    return parser.parse_args()


def strip_context_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: strip_context_fields(v) for k, v in value.items() if k != "context"}
    if isinstance(value, list):
        return [strip_context_fields(v) for v in value]
    return value


def normalize_record(record: Any, *, ignore_context: bool) -> Any:
    if ignore_context:
        return strip_context_fields(record)
    return record


def read_jsonl(path: Path, *, ignore_context: bool, progress_every: int) -> list[Any]:
    print(f"[compare] Loading {path} ...", flush=True)
    if not path.exists():
        raise SystemExit(f"Input file does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"Input path is not a file: {path}")

    records: list[Any] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to parse JSON at {path}:{line_no}: {exc}") from exc
            records.append(normalize_record(record, ignore_context=ignore_context))
            if progress_every > 0 and len(records) % progress_every == 0:
                print(f"[compare] {path.name}: loaded {len(records)} records", flush=True)

    print(f"[compare] Loaded {len(records)} records from {path}", flush=True)
    return records


def compact_json(value: Any, *, max_chars: int = 220) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=True)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def compare_records(
    left_records: list[Any],
    right_records: list[Any],
    *,
    progress_every: int,
    max_mismatches: int,
) -> list[tuple[int, Any, Any]]:
    mismatches: list[tuple[int, Any, Any]] = []
    limit = min(len(left_records), len(right_records))

    for index in range(limit):
        line_no = index + 1
        if left_records[index] != right_records[index]:
            mismatches.append((line_no, left_records[index], right_records[index]))
            if len(mismatches) >= max_mismatches:
                return mismatches
        if progress_every > 0 and line_no % progress_every == 0:
            print(f"[compare] Compared {line_no} records ...", flush=True)

    return mismatches


def main() -> int:
    args = parse_args()
    left_records = read_jsonl(
        args.left_log,
        ignore_context=args.ignore_context,
        progress_every=args.progress_every,
    )
    right_records = read_jsonl(
        args.right_log,
        ignore_context=args.ignore_context,
        progress_every=args.progress_every,
    )

    print("[compare] Starting comparison ...", flush=True)
    mismatches = compare_records(
        left_records,
        right_records,
        progress_every=args.progress_every,
        max_mismatches=max(1, args.max_mismatches),
    )
    same_length = len(left_records) == len(right_records)

    if same_length and not mismatches:
        print("[compare] Success: logs are identical.", flush=True)
        return 0

    print("[compare] Determinism mismatch detected.", flush=True)
    if not same_length:
        print(
            f"[compare] Record count differs: left={len(left_records)} right={len(right_records)}",
            flush=True,
        )
    for line_no, left_value, right_value in mismatches:
        print(f"[compare] line {line_no} differs:", flush=True)
        print(f"  left : {compact_json(left_value)}", flush=True)
        print(f"  right: {compact_json(right_value)}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
