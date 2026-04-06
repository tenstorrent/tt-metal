#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Validate a sweep trace against a master trace using direct config_hash matching.

The sweep trace carries ``sweep_source_hash`` on each configuration, which maps
directly to ``config_hash`` in the master trace.  This enables O(1) lookup
instead of fuzzy matching, and produces a structured diff report suitable for
CI (GitHub Actions step summary) and local debugging.

Usage:
    python tests/sweep_framework/validate_sweep_trace.py \\
        --master-trace model_tracer/traced_operations/ttnn_operations_master.json \\
        --sweep-trace  model_tracer/traced_operations/sweep_trace.json \\
        --output-report validation_summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Normalization — strip fields that are expected to differ between traces
# ---------------------------------------------------------------------------

IGNORED_KEYS = frozenset(
    {
        "config_hash",
        "config_id",
        "executions",
        "sweep_source_hash",
        "device_ids",
        "mesh_device",
        "hash",
    }
)


def normalize(obj: Any, *, _parent_key: str = "") -> Any:
    """Recursively normalize a config dict for comparison.

    Strips keys that are expected to vary between a master trace (from the
    database) and a sweep trace (from live execution) without representing a
    meaningful configuration difference.
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in sorted(obj.items()):
            if k in IGNORED_KEYS:
                continue
            # memory_config.hash is a device pointer, always different
            if k == "hash" and isinstance(v, (int, float)):
                continue
            # sub_core_grids: None is noise
            if k == "sub_core_grids" and v is None:
                continue
            result[k] = normalize(v, _parent_key=k)
        return result
    if isinstance(obj, list):
        return [normalize(item, _parent_key=_parent_key) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Diff engine
# ---------------------------------------------------------------------------

DIFF_CATEGORIES = {
    "tensor_placement": "tensor placement (mesh shape, distribution, shard vs replicate)",
    "memory_config": "memory config (buffer type, memory layout, shard spec)",
    "shard_spec": "shard spec (grid coordinates, shape, orientation)",
    "extra_key": "extra or missing key",
    "value": "argument value difference",
}


@dataclass
class Diff:
    path: str
    master_value: Any
    sweep_value: Any
    category: str

    @property
    def category_label(self) -> str:
        return DIFF_CATEGORIES.get(self.category, self.category)


def _classify(path: str) -> str:
    if "tensor_placement" in path or "placement" in path or "distribution_shape" in path:
        return "tensor_placement"
    if "shard_spec" in path:
        return "shard_spec"
    if "memory_config" in path or "memory_layout" in path or "buffer_type" in path:
        return "memory_config"
    return "value"


def deep_diff(master: Any, sweep: Any, prefix: str = "") -> list[Diff]:
    """Produce a list of leaf-level diffs between two normalized argument trees."""
    diffs: list[Diff] = []

    if isinstance(master, dict) and isinstance(sweep, dict):
        all_keys = sorted(set(master.keys()) | set(sweep.keys()))
        for k in all_keys:
            child_path = f"{prefix}.{k}" if prefix else k
            if k not in master:
                diffs.append(Diff(child_path, "<missing>", sweep[k], "extra_key"))
            elif k not in sweep:
                diffs.append(Diff(child_path, master[k], "<missing>", "extra_key"))
            else:
                diffs.extend(deep_diff(master[k], sweep[k], child_path))
    elif isinstance(master, list) and isinstance(sweep, list):
        for i in range(max(len(master), len(sweep))):
            child_path = f"{prefix}[{i}]"
            if i >= len(master):
                diffs.append(Diff(child_path, "<missing>", sweep[i], "extra_key"))
            elif i >= len(sweep):
                diffs.append(Diff(child_path, master[i], "<missing>", "extra_key"))
            else:
                diffs.extend(deep_diff(master[i], sweep[i], child_path))
    elif master != sweep:
        diffs.append(Diff(prefix, master, sweep, _classify(prefix)))

    return diffs


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class ConfigResult:
    config_hash: str
    op_name: str
    master_config_id: int | None
    sweep_config_id: int | None
    status: str  # "match", "diff", "missing_sweep", "missing_master", "incidental"
    diffs: list[Diff] = field(default_factory=list)


@dataclass
class ValidationReport:
    results: list[ConfigResult] = field(default_factory=list)

    @property
    def matched(self) -> list[ConfigResult]:
        return [r for r in self.results if r.status == "match"]

    @property
    def diffed(self) -> list[ConfigResult]:
        return [r for r in self.results if r.status == "diff"]

    @property
    def missing_sweep(self) -> list[ConfigResult]:
        return [r for r in self.results if r.status == "missing_sweep"]

    @property
    def incidental(self) -> list[ConfigResult]:
        return [r for r in self.results if r.status == "incidental"]

    @property
    def total_master(self) -> int:
        return len([r for r in self.results if r.status != "incidental"])

    @property
    def coverage(self) -> float:
        targeted = [r for r in self.results if r.status != "incidental"]
        if not targeted:
            return 0.0
        exercised = [r for r in targeted if r.status in ("match", "diff")]
        return len(exercised) / len(targeted)


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------


def validate(master_data: dict, sweep_data: dict) -> ValidationReport:
    """Join master and sweep traces by config_hash / sweep_source_hash."""
    report = ValidationReport()

    # Build master index: config_hash → (op_name, config_id, arguments)
    master_index: dict[str, tuple[str, int, dict]] = {}
    for op_name, op_info in master_data.get("operations", {}).items():
        for cfg in op_info.get("configurations", []):
            ch = cfg.get("config_hash")
            if ch:
                master_index[ch] = (op_name, cfg.get("config_id", 0), cfg.get("arguments", {}))

    # Track which master configs have been matched
    matched_hashes: set[str] = set()

    # Walk sweep configs
    for op_name, op_info in sweep_data.get("operations", {}).items():
        for cfg in op_info.get("configurations", []):
            source_hash = cfg.get("sweep_source_hash")
            sweep_cid = cfg.get("config_id", 0)

            if not source_hash:
                # No source hash — config wasn't driven by a sweep vector
                continue

            if source_hash not in master_index:
                # Sweep produced a config whose source hash isn't in the master
                report.results.append(
                    ConfigResult(
                        config_hash=source_hash,
                        op_name=op_name,
                        master_config_id=None,
                        sweep_config_id=sweep_cid,
                        status="incidental",
                    )
                )
                continue

            master_op, master_cid, master_args = master_index[source_hash]

            if master_op != op_name:
                # Incidental operation: sweep vector for ttnn.add triggered
                # a ttnn.typecast call — both carry the same sweep_source_hash
                report.results.append(
                    ConfigResult(
                        config_hash=source_hash,
                        op_name=op_name,
                        master_config_id=master_cid,
                        sweep_config_id=sweep_cid,
                        status="incidental",
                    )
                )
                continue

            # Direct match by hash — now compare arguments
            matched_hashes.add(source_hash)
            sweep_args = cfg.get("arguments", {})

            norm_master = normalize(master_args)
            norm_sweep = normalize(sweep_args)

            if norm_master == norm_sweep:
                report.results.append(
                    ConfigResult(
                        config_hash=source_hash,
                        op_name=op_name,
                        master_config_id=master_cid,
                        sweep_config_id=sweep_cid,
                        status="match",
                    )
                )
            else:
                diffs = deep_diff(norm_master, norm_sweep)
                report.results.append(
                    ConfigResult(
                        config_hash=source_hash,
                        op_name=op_name,
                        master_config_id=master_cid,
                        sweep_config_id=sweep_cid,
                        status="diff",
                        diffs=diffs,
                    )
                )

    # Report master configs with no sweep execution
    for ch, (op_name, cid, _args) in master_index.items():
        if ch not in matched_hashes:
            report.results.append(
                ConfigResult(
                    config_hash=ch,
                    op_name=op_name,
                    master_config_id=cid,
                    sweep_config_id=None,
                    status="missing_sweep",
                )
            )

    # Sort results for stable output
    report.results.sort(key=lambda r: (r.op_name, r.status, r.config_hash))
    return report


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _trunc(val: Any, max_len: int = 80) -> str:
    s = str(val)
    return s[: max_len - 3] + "..." if len(s) > max_len else s


def render_report(report: ValidationReport) -> str:
    lines: list[str] = []

    lines.append("# Sweep Trace Validation Report")
    lines.append("")
    lines.append(f"**Total master configs:** {report.total_master}")
    lines.append(f"**Exact matches:** {len(report.matched)}")
    lines.append(f"**With diffs:** {len(report.diffed)}")
    lines.append(f"**Not exercised by sweep:** {len(report.missing_sweep)}")
    lines.append(f"**Incidental (non-target ops):** {len(report.incidental)}")
    lines.append(f"**Coverage:** {report.coverage:.1%}")
    lines.append("")

    # Per-operation summary table
    op_stats: dict[str, dict[str, int]] = {}
    for r in report.results:
        if r.status == "incidental":
            continue
        stats = op_stats.setdefault(r.op_name, {"match": 0, "diff": 0, "missing_sweep": 0})
        stats[r.status] = stats.get(r.status, 0) + 1

    if op_stats:
        lines.append("## Per-operation summary")
        lines.append("")
        lines.append("| Operation | Match | Diff | Missing | Total |")
        lines.append("|-----------|------:|-----:|--------:|------:|")
        for op in sorted(op_stats):
            s = op_stats[op]
            total = sum(s.values())
            lines.append(f"| `{op}` | {s['match']} | {s['diff']} | {s['missing_sweep']} | {total} |")
        lines.append("")

    # Missing sweep configs (collapsed for brevity)
    if report.missing_sweep:
        lines.append("## Master configs not exercised by sweep")
        lines.append("")
        lines.append(f"{len(report.missing_sweep)} configs had no corresponding sweep execution.")
        lines.append("")
        lines.append("<details><summary>Show all</summary>")
        lines.append("")
        lines.append("| Operation | Config ID | Config Hash |")
        lines.append("|-----------|----------:|-------------|")
        for r in report.missing_sweep:
            lines.append(f"| `{r.op_name}` | {r.master_config_id} | `{r.config_hash[:16]}...` |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # Diff category summary
    if report.diffed:
        cat_counts: dict[str, int] = {}
        for r in report.diffed:
            for d in r.diffs:
                cat_counts[d.category] = cat_counts.get(d.category, 0) + 1

        lines.append("## Diff categories")
        lines.append("")
        lines.append("| Category | Count | Description |")
        lines.append("|----------|------:|-------------|")
        for cat in sorted(cat_counts, key=lambda c: -cat_counts[c]):
            lines.append(f"| `{cat}` | {cat_counts[cat]} | {DIFF_CATEGORIES.get(cat, cat)} |")
        lines.append("")

    # Detailed diffs
    if report.diffed:
        lines.append("## Detailed diffs")
        lines.append("")
        for r in report.diffed:
            lines.append(f"### `{r.op_name}` config_hash `{r.config_hash[:16]}...`")
            lines.append(f"master config_id={r.master_config_id}, sweep config_id={r.sweep_config_id}")
            lines.append("")
            lines.append("| Path | Category | Master | Sweep |")
            lines.append("|------|----------|--------|-------|")
            for d in r.diffs:
                lines.append(
                    f"| `{d.path}` | `{d.category}` | {_trunc(d.master_value, 40)} | {_trunc(d.sweep_value, 40)} |"
                )
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate sweep trace against master trace via config_hash join.",
    )
    parser.add_argument(
        "--master-trace",
        default="model_tracer/traced_operations/ttnn_operations_master.json",
        help="Path to master trace JSON (default: model_tracer/traced_operations/ttnn_operations_master.json)",
    )
    parser.add_argument(
        "--sweep-trace",
        nargs="+",
        default=["model_tracer/traced_operations/sweep_trace.json"],
        help="Path(s) to sweep trace JSON file(s). Multiple files are merged before validation.",
    )
    parser.add_argument(
        "--output-report",
        default="model_tracer/traced_operations/validation_summary.md",
        help="Write markdown report to file (default: validation_summary.md)",
    )
    parser.add_argument(
        "--ignore-categories",
        nargs="*",
        default=[],
        help="Diff categories to ignore for pass/fail (e.g. tensor_placement shard_spec)",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=None,
        help="Coverage threshold (0.0-1.0) below which the job fails. Default: no threshold.",
    )
    args = parser.parse_args()

    master_path = Path(args.master_trace)
    sweep_paths = [Path(p) for p in args.sweep_trace]

    if not master_path.is_file():
        print(f"ERROR: master trace not found: {master_path}", file=sys.stderr)
        return 1
    for sp in sweep_paths:
        if not sp.is_file():
            print(f"ERROR: sweep trace not found: {sp}", file=sys.stderr)
            return 1

    with open(master_path) as f:
        master_data = json.load(f)

    # Merge multiple sweep traces into a single operations dict
    sweep_data: dict = {"operations": {}}
    for sp in sweep_paths:
        with open(sp) as f:
            data = json.load(f)
        for op_name, op_info in data.get("operations", {}).items():
            existing = sweep_data["operations"].setdefault(op_name, {"configurations": []})
            existing["configurations"].extend(op_info.get("configurations", []))

    print(f"Loaded {len(sweep_paths)} sweep trace(s), {len(sweep_data['operations'])} operations")

    report = validate(master_data, sweep_data)
    rendered = render_report(report)

    if args.output_report:
        with open(args.output_report, "w") as f:
            f.write(rendered)
        print(f"Report written to: {args.output_report}")
    else:
        print(rendered)

    # Determine exit code
    ignore = set(args.ignore_categories)
    failing_diffs = [r for r in report.diffed if any(d.category not in ignore for d in r.diffs)]

    if failing_diffs:
        print(
            f"FAIL: {len(failing_diffs)} config(s) have diffs " f"(ignoring categories: {ignore or 'none'})",
            file=sys.stderr,
        )
        return 1

    if args.pass_threshold is not None and report.coverage < args.pass_threshold:
        print(
            f"FAIL: coverage {report.coverage:.1%} below threshold {args.pass_threshold:.1%}",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: {len(report.matched)} exact matches, coverage {report.coverage:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
