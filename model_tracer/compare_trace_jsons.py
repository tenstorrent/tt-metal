#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compare large TTNN model trace JSON files.

This tool compares 2+ trace JSON files with pairwise reports:
- metadata / operation counts
- missing and extra operations
- per-operation configuration count deltas
- exact normalized configuration multiset mismatches

By default, volatile fields are ignored to focus on logical configuration
equivalence:
  - config_id
  - config_hash
  - executions
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IGNORED_CONFIG_KEYS_DEFAULT = {"config_id", "config_hash", "executions"}


@dataclass
class TraceSummary:
    path: Path
    operations: dict[str, Any]
    metadata: dict[str, Any]
    total_operations: int
    total_configurations: int


def load_trace(path: Path) -> TraceSummary:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    operations = data.get("operations", {})
    metadata = data.get("metadata", {})
    total_configurations = sum(len(v.get("configurations", [])) for v in operations.values())
    return TraceSummary(
        path=path,
        operations=operations,
        metadata=metadata,
        total_operations=len(operations),
        total_configurations=total_configurations,
    )


def normalize_config(config: dict[str, Any], ignored_keys: set[str]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in ignored_keys}


def stable_serialize(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_counter_for_op(op_data: dict[str, Any], ignored_keys: set[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for cfg in op_data.get("configurations", []):
        normalized = normalize_config(cfg, ignored_keys)
        counter[stable_serialize(normalized)] += 1
    return counter


def config_locations_for_op(op_data: dict[str, Any], ignored_keys: set[str]) -> dict[str, list[dict[str, Any]]]:
    """Map serialized normalized config -> concrete locations in source list."""
    locations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, cfg in enumerate(op_data.get("configurations", [])):
        normalized = normalize_config(cfg, ignored_keys)
        key = stable_serialize(normalized)
        locations[key].append(
            {
                "index": idx,
                "config_id": cfg.get("config_id"),
                "config_hash": cfg.get("config_hash"),
            }
        )
    return locations


def format_num(n: int) -> str:
    return f"{n:,}"


def fingerprint(serialized_config: str) -> str:
    return hashlib.sha1(serialized_config.encode("utf-8")).hexdigest()[:12]


def trim(text: str, max_len: int = 240) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def canonical_configs(op_data: dict[str, Any], ignored_keys: set[str]) -> list[str]:
    """Return sorted canonical strings for each configuration (with duplicates)."""
    serialized: list[str] = []
    for cfg in op_data.get("configurations", []):
        normalized = normalize_config(cfg, ignored_keys)
        serialized.append(json.dumps(normalized, sort_keys=True, indent=2, ensure_ascii=True))
    serialized.sort()
    return serialized


def write_op_diff_file(
    out_dir: Path,
    pair_name: str,
    op_name: str,
    left: TraceSummary,
    right: TraceSummary,
    ignored_keys: set[str],
) -> Path:
    pair_dir = out_dir / safe_name(pair_name)
    pair_dir.mkdir(parents=True, exist_ok=True)
    out_path = pair_dir / f"{safe_name(op_name)}.diff"

    left_lines = canonical_configs(left.operations[op_name], ignored_keys)
    right_lines = canonical_configs(right.operations[op_name], ignored_keys)

    left_text = "[\n" + ",\n".join(left_lines) + "\n]\n"
    right_text = "[\n" + ",\n".join(right_lines) + "\n]\n"

    diff_lines = difflib.unified_diff(
        left_text.splitlines(keepends=True),
        right_text.splitlines(keepends=True),
        fromfile=f"{left.path.name}:{op_name}",
        tofile=f"{right.path.name}:{op_name}",
        lineterm="",
    )
    out_path.write_text("".join(diff_lines), encoding="utf-8")
    return out_path


def render_global_summary(traces: list[TraceSummary]) -> list[str]:
    lines = ["# Trace JSON Comparison Report", "", "## Input Files", ""]
    for tr in traces:
        lines.append(f"- `{tr.path}`")
    lines += ["", "## Top-Level Summary", ""]
    for tr in traces:
        lines.append(
            f"- `{tr.path.name}`: {format_num(tr.total_operations)} ops, "
            f"{format_num(tr.total_configurations)} configs"
        )
    return lines


def render_pairwise(
    left: TraceSummary,
    right: TraceSummary,
    ignored_keys: set[str],
    top_n_ops: int,
    sample_mismatches: int,
    op_diff_dir: Path | None,
    op_diff_max_ops: int,
) -> list[str]:
    lines: list[str] = []
    lines += [
        "",
        f"## Pair: `{left.path.name}` vs `{right.path.name}`",
        "",
    ]

    left_ops = set(left.operations.keys())
    right_ops = set(right.operations.keys())
    missing_in_right = sorted(left_ops - right_ops)
    extra_in_right = sorted(right_ops - left_ops)
    common_ops = sorted(left_ops & right_ops)

    lines += [
        f"- Left ops: {format_num(len(left_ops))}",
        f"- Right ops: {format_num(len(right_ops))}",
        f"- Common ops: {format_num(len(common_ops))}",
        f"- Missing in right: {format_num(len(missing_in_right))}",
        f"- Extra in right: {format_num(len(extra_in_right))}",
    ]

    if missing_in_right:
        lines += ["", "### Missing In Right (sample)", ""]
        for op in missing_in_right[:top_n_ops]:
            lines.append(f"- `{op}`")
        if len(missing_in_right) > top_n_ops:
            lines.append(f"- ... {len(missing_in_right) - top_n_ops} more")

    if extra_in_right:
        lines += ["", "### Extra In Right (sample)", ""]
        for op in extra_in_right[:top_n_ops]:
            lines.append(f"- `{op}`")
        if len(extra_in_right) > top_n_ops:
            lines.append(f"- ... {len(extra_in_right) - top_n_ops} more")

    count_deltas: list[tuple[str, int, int, int]] = []
    exact_mismatch_ops: list[str] = []

    for op in common_ops:
        l_cfgs = left.operations[op].get("configurations", [])
        r_cfgs = right.operations[op].get("configurations", [])
        l_count = len(l_cfgs)
        r_count = len(r_cfgs)
        if l_count != r_count:
            count_deltas.append((op, l_count, r_count, r_count - l_count))

        l_counter = config_counter_for_op(left.operations[op], ignored_keys)
        r_counter = config_counter_for_op(right.operations[op], ignored_keys)
        if l_counter != r_counter:
            exact_mismatch_ops.append(op)

    lines += [
        "",
        f"- Ops with config count delta: {format_num(len(count_deltas))}",
        f"- Ops with exact normalized config mismatch: {format_num(len(exact_mismatch_ops))}",
    ]

    if count_deltas:
        count_deltas.sort(key=lambda x: abs(x[3]), reverse=True)
        lines += ["", "### Largest Config Count Deltas", ""]
        for op, l_count, r_count, delta in count_deltas[:top_n_ops]:
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"- `{op}`: left={format_num(l_count)}, right={format_num(r_count)}, delta={sign}{format_num(delta)}"
            )

    if exact_mismatch_ops:
        lines += ["", "### Exact Normalized Mismatch Samples", ""]
        pair_name = f"{left.path.stem}__vs__{right.path.stem}"
        written_diffs = 0
        for op in exact_mismatch_ops[:top_n_ops]:
            l_counter = config_counter_for_op(left.operations[op], ignored_keys)
            r_counter = config_counter_for_op(right.operations[op], ignored_keys)
            l_locations = config_locations_for_op(left.operations[op], ignored_keys)
            r_locations = config_locations_for_op(right.operations[op], ignored_keys)
            left_only = l_counter - r_counter
            right_only = r_counter - l_counter
            left_only_total = sum(left_only.values())
            right_only_total = sum(right_only.values())
            lines.append(
                f"- `{op}`: left_only={format_num(left_only_total)}, right_only={format_num(right_only_total)}"
            )

            if sample_mismatches > 0:
                left_items = left_only.most_common(sample_mismatches)
                right_items = right_only.most_common(sample_mismatches)
                for serialized, count in left_items:
                    loc = l_locations[serialized][0] if l_locations[serialized] else {}
                    idx = loc.get("index")
                    cid = loc.get("config_id")
                    lines.append(
                        f"  - left-only cfg `{fingerprint(serialized)}` x{count} "
                        f"(op=`{op}`, idx={idx}, config_id={cid}): `{trim(serialized)}`"
                    )
                for serialized, count in right_items:
                    loc = r_locations[serialized][0] if r_locations[serialized] else {}
                    idx = loc.get("index")
                    cid = loc.get("config_id")
                    lines.append(
                        f"  - right-only cfg `{fingerprint(serialized)}` x{count} "
                        f"(op=`{op}`, idx={idx}, config_id={cid}): `{trim(serialized)}`"
                    )

            if op_diff_dir is not None and written_diffs < op_diff_max_ops:
                diff_path = write_op_diff_file(
                    out_dir=op_diff_dir,
                    pair_name=pair_name,
                    op_name=op,
                    left=left,
                    right=right,
                    ignored_keys=ignored_keys,
                )
                lines.append(f"  - op diff file: `{diff_path}`")
                written_diffs += 1

        if len(exact_mismatch_ops) > top_n_ops:
            lines.append(f"- ... {len(exact_mismatch_ops) - top_n_ops} more mismatching operations")
        if op_diff_dir is not None:
            lines += [
                "",
                f"- Wrote op-level diffs: {format_num(min(len(exact_mismatch_ops), op_diff_max_ops))}",
                f"- Diff directory: `{(op_diff_dir / safe_name(pair_name)).resolve()}`",
            ]

    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare 2+ TTNN trace JSON files.")
    parser.add_argument("json_files", nargs="+", help="Input JSON files (2 or more)")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Write markdown report to file (default: print to stdout only)",
    )
    parser.add_argument(
        "--top-n-ops",
        type=int,
        default=20,
        help="How many operation names to show in samples/lists (default: 20)",
    )
    parser.add_argument(
        "--sample-mismatches",
        type=int,
        default=2,
        help="How many left-only/right-only normalized configs to sample per op (default: 2)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Do strict compare (do not ignore config_id/config_hash/executions)",
    )
    parser.add_argument(
        "--op-diff-dir",
        default=None,
        help="Optional directory to write per-op unified diff files for mismatching ops",
    )
    parser.add_argument(
        "--op-diff-max-ops",
        type=int,
        default=50,
        help="Maximum mismatching ops per pair to emit diff files for (default: 50)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(p).resolve() for p in args.json_files]
    if len(paths) < 2:
        raise SystemExit("Need at least two JSON files")

    missing = [p for p in paths if not p.is_file()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise SystemExit(f"Missing files: {missing_str}")

    ignored_keys = set() if args.strict else set(IGNORED_CONFIG_KEYS_DEFAULT)
    op_diff_dir = Path(args.op_diff_dir).resolve() if args.op_diff_dir else None
    if op_diff_dir is not None:
        op_diff_dir.mkdir(parents=True, exist_ok=True)

    traces = [load_trace(p) for p in paths]
    report_lines = render_global_summary(traces)
    report_lines += ["", f"- Compare mode: {'strict' if args.strict else 'normalized'}"]
    if not args.strict:
        report_lines.append(f"- Ignored config keys: `{', '.join(sorted(ignored_keys))}`")

    for i in range(len(traces)):
        for j in range(i + 1, len(traces)):
            report_lines += render_pairwise(
                traces[i],
                traces[j],
                ignored_keys=ignored_keys,
                top_n_ops=args.top_n_ops,
                sample_mismatches=args.sample_mismatches,
                op_diff_dir=op_diff_dir,
                op_diff_max_ops=args.op_diff_max_ops,
            )

    output = "\n".join(report_lines) + "\n"

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Wrote report to {out_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
