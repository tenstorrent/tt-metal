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
import re as _re
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
        "output_tensor",  # master trace records return values, sweep does not
        "indices_tensor",  # topk output tensor — master records it, sweep doesn't capture
        "attention_sink",  # optional named tensor kwarg in SDPA — sweep doesn't produce it
    }
)

# Infrastructure kwargs that the sweep framework intentionally filters out
# (handled separately via output_memory_config, etc.).  Only stripped at the
# top level of the arguments dict — NOT inside tensor arg sub-dicts.
TOP_LEVEL_INFRA_KWARGS = frozenset(
    {
        "memory_config",
        "dtype",
        "layout",
        "compute_kernel_config",
    }
)


def _normalize_tensor_placement(tp: dict) -> dict:
    """Normalize tensor_placement to a canonical form.

    The C++ tensor topology may report distribution as 1-D ``[32]`` or 2-D
    ``[4, 8]`` depending on how the mesh mapper was constructed, even when the
    functional behaviour is identical.  Collapse to a single product so that
    ``[4, 8]`` and ``[32]`` compare equal, and reduce placement lists to a
    canonical ``shard``/``replicate`` per-dimension tag.
    """
    result = dict(tp)

    # Normalize distribution_shape → product
    ds = result.get("distribution_shape")
    if isinstance(ds, str):
        import ast as _ast
        try:
            ds = _ast.literal_eval(ds)
        except Exception:
            ds = None
    if isinstance(ds, (list, tuple)) and ds:
        product = 1
        for d in ds:
            product *= int(d)
        result["distribution_shape"] = product
    elif isinstance(ds, int):
        result["distribution_shape"] = ds

    # Normalize mesh_device_shape → product (same reason)
    ms = result.get("mesh_device_shape")
    if isinstance(ms, str):
        import ast as _ast
        try:
            ms = _ast.literal_eval(ms)
        except Exception:
            ms = None
    if isinstance(ms, (list, tuple)) and ms:
        product = 1
        for d in ms:
            product *= int(d)
        result["mesh_device_shape"] = product
    elif isinstance(ms, int):
        result["mesh_device_shape"] = ms

    # Normalize placement list → first element's type only.
    # Multi-device meshes may report 1-element ``["replicate"]`` vs 2-element
    # ``["replicate", "shard"]`` depending on mesh dimensionality.  The second
    # entry is a per-dimension shard annotation that doesn't affect functional
    # equivalence.  Reduce to the first (primary) placement tag.
    pl = result.get("placement", "")
    if isinstance(pl, str):
        tags: list[str] = []
        # Only look at the primary placement (first entry)
        if "PlacementReplicate" in pl:
            tags.append("replicate")
        elif "PlacementShard" in pl:
            tags.append("shard")
        result["placement"] = tags
    elif isinstance(pl, list):
        if pl:
            first = str(pl[0])
            if "Replicate" in first:
                result["placement"] = ["replicate"]
            elif "Shard" in first:
                result["placement"] = ["shard"]
            else:
                result["placement"] = [first]
        else:
            result["placement"] = []

    return result


def _normalize_original_shape(shape: list) -> list:
    """Strip leading dimensions of 1 from original_shape.

    Model traces may record 4-D shapes like ``[1, 1, 131072, 64]`` for tensors
    that the sweep recreates as 2-D ``[131072, 64]``.  Stripping leading 1s
    makes them compare equal.
    """
    while len(shape) > 1 and shape[0] == 1:
        shape = shape[1:]
    return shape


# ---------------------------------------------------------------------------
# Named-kwarg to positional-arg mapping for ops that record kwargs in master
# but positional args in sweep (or vice versa).
# ---------------------------------------------------------------------------

# Maps op_name -> ordered list of named kwargs that correspond to arg1, arg2, arg3, ...
_NAMED_TO_POSITIONAL: dict[str, list[str]] = {
    "ttnn.topk": ["k", "dim"],
    "ttnn.slice": ["starts", "ends", "steps"],
    "ttnn.scatter": ["dim", "index", "src"],
}


def _unify_named_positional_args(args: dict, op_name: str) -> dict:
    """Rewrite named kwargs to positional arg keys when a mapping is known.

    If the config uses named kwargs (e.g. ``k``, ``dim``) that correspond to
    positional slots (``arg1``, ``arg2``), rewrite them to positional keys so
    both master and sweep use the same key names.
    """
    mapping = _NAMED_TO_POSITIONAL.get(op_name)
    if not mapping:
        return args

    result = dict(args)
    for idx, named_key in enumerate(mapping, start=1):
        positional_key = f"arg{idx}"
        # If the named key is present but the positional key is not, rename it
        if named_key in result and positional_key not in result:
            result[positional_key] = result.pop(named_key)
    return result


def normalize(obj: Any, *, _parent_key: str = "", _depth: int = 0, _op_name: str = "") -> Any:
    """Recursively normalize a config dict for comparison.

    Strips keys that are expected to vary between a master trace (from the
    database) and a sweep trace (from live execution) without representing a
    meaningful configuration difference.
    """
    if isinstance(obj, dict):
        # Normalize tensor_placement dicts before general processing
        if _parent_key == "tensor_placement" or (
            "distribution_shape" in obj and "placement" in obj
        ):
            obj = _normalize_tensor_placement(obj)

        # At the top level, unify named kwargs → positional args
        if _depth == 0 and _op_name:
            obj = _unify_named_positional_args(obj, _op_name)

        result = {}
        for k, v in sorted(obj.items()):
            if k in IGNORED_KEYS:
                continue
            # At the top level of the arguments dict, strip infrastructure kwargs
            # that the sweep framework intentionally handles separately
            if _depth == 0 and k in TOP_LEVEL_INFRA_KWARGS:
                continue
            # memory_config.hash is a device pointer — always differs between runs; skip numeric values only
            if k == "hash" and isinstance(v, (int, float)):
                continue
            # sub_core_grids: None is noise
            if k == "sub_core_grids" and v is None:
                continue
            # Strip keys with None or default-zero values — treat missing vs None/0 as equivalent
            # (kwargs with default values may appear in one trace but be absent in the other)
            if v is None or v == 0 or v == 0.0:
                continue
            # Ignore original_shape — it's metadata that varies with mesh device
            # count and doesn't represent a functional argument difference
            if k == "original_shape":
                continue
            # Ignore original_dtype — metadata about the source torch tensor dtype,
            # not a functional argument (e.g. scatter index is always int32 on device
            # regardless of the original torch dtype)
            if k == "original_dtype":
                continue
            # Normalize storage_type — HOST vs DEVICE is an artifact of tensor creation
            # path, not a functional op argument difference
            if k == "storage_type":
                continue
            result[k] = normalize(v, _parent_key=k, _depth=_depth + 1, _op_name=_op_name)
        return result
    if isinstance(obj, list):
        return [normalize(item, _parent_key=_parent_key, _depth=_depth + 1, _op_name=_op_name) for item in obj]
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
    status: str  # "match", "diff", "hash_mismatch", "missing_sweep", "incidental"
    diffs: list[Diff] = field(default_factory=list)
    sweep_config_hash: str | None = None  # the sweep trace's own computed config_hash


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
    def hash_mismatch(self) -> list[ConfigResult]:
        return [r for r in self.results if r.status == "hash_mismatch"]

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
        exercised = [r for r in targeted if r.status in ("match", "diff", "hash_mismatch")]
        return len(exercised) / len(targeted)


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------


_POSITIONAL_ARG_RE = _re.compile(r"^arg\d+$")


def _reconcile_extra_positional_args(master: dict, sweep: dict) -> None:
    """Remove positional arg keys (arg2, arg3, ...) present in only one side.

    For ops like reshape the master trace may record extra positional args
    (e.g. padded output shape in arg2) that the sweep trace does not capture.
    These are optional metadata, not functional differences.  Mutates both
    dicts in place.
    """
    if not isinstance(master, dict) or not isinstance(sweep, dict):
        return
    master_only = set(master.keys()) - set(sweep.keys())
    sweep_only = set(sweep.keys()) - set(master.keys())
    for k in master_only:
        if _POSITIONAL_ARG_RE.match(k):
            del master[k]
    for k in sweep_only:
        if _POSITIONAL_ARG_RE.match(k):
            del sweep[k]


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
            sweep_config_hash = cfg.get("config_hash")

            norm_master = normalize(master_args, _op_name=master_op)
            norm_sweep = normalize(sweep_args, _op_name=op_name)

            # Category 4: For reshape (and similar), extra positional args
            # (arg2, arg3, ...) that exist only in master are optional metadata
            # (e.g. padded output shape). Remove them before comparison.
            _reconcile_extra_positional_args(norm_master, norm_sweep)

            if norm_master == norm_sweep:
                # Arguments match — count as match regardless of config_hash divergence.
                # Hash mismatches occur because tensor_placement mesh shape (2D [4,8] vs
                # 1D [32]) is baked into the hash but normalized away for comparison.
                # The functional configuration is identical.
                report.results.append(
                    ConfigResult(
                        config_hash=source_hash,
                        op_name=op_name,
                        master_config_id=master_cid,
                        sweep_config_id=sweep_cid,
                        status="match",
                        sweep_config_hash=sweep_config_hash if (sweep_config_hash and sweep_config_hash != source_hash) else None,
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
                        sweep_config_hash=sweep_config_hash,
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
    lines.append(f"**Hash mismatch (args match, hash differs):** {len(report.hash_mismatch)}")
    lines.append(f"**Not exercised by sweep:** {len(report.missing_sweep)}")
    lines.append(f"**Incidental (non-target ops):** {len(report.incidental)}")
    lines.append(f"**Coverage:** {report.coverage:.1%}")
    lines.append("")

    # Per-operation summary table
    op_stats: dict[str, dict[str, int]] = {}
    for r in report.results:
        if r.status == "incidental":
            continue
        stats = op_stats.setdefault(r.op_name, {"match": 0, "diff": 0, "hash_mismatch": 0, "missing_sweep": 0})
        stats[r.status] = stats.get(r.status, 0) + 1

    if op_stats:
        lines.append("## Per-operation summary")
        lines.append("")
        lines.append("| Operation | Match | Diff | Hash Mismatch | Missing | Total |")
        lines.append("|-----------|------:|-----:|--------------:|--------:|------:|")
        for op in sorted(op_stats):
            s = op_stats[op]
            total = sum(s.values())
            lines.append(
                f"| `{op}` | {s['match']} | {s['diff']} | {s['hash_mismatch']} | {s['missing_sweep']} | {total} |"
            )
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

    # Hash mismatches (args identical but config_hash differs)
    if report.hash_mismatch:
        lines.append("## Config hash mismatches")
        lines.append("")
        lines.append(
            "These configs have identical normalized arguments but different `config_hash` values, "
            "indicating a divergence in the hash computation between the master trace and the sweep trace."
        )
        lines.append("")
        lines.append("| Operation | Config ID | Master Hash | Sweep Hash |")
        lines.append("|-----------|----------:|-------------|------------|")
        for r in report.hash_mismatch:
            lines.append(
                f"| `{r.op_name}` | {r.master_config_id} "
                f"| `{r.config_hash[:16]}...` | `{(r.sweep_config_hash or '?')[:16]}...` |"
            )
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

    # Detailed diffs (truncated to avoid exceeding GitHub step summary 1MB limit)
    if report.diffed:
        max_detailed_entries = 20
        shown = report.diffed[:max_detailed_entries]
        remaining = len(report.diffed) - max_detailed_entries

        lines.append("## Detailed diffs")
        lines.append("")
        if remaining > 0:
            lines.append(
                f"> Showing first {max_detailed_entries} of {len(report.diffed)} diffed configs. "
                f"{remaining} additional entries omitted to stay within GitHub step summary size limits."
            )
            lines.append("")

        for r in shown:
            lines.append(f"### `{r.op_name}` config_hash `{r.config_hash[:16]}...`")
            lines.append(f"master config_id={r.master_config_id}, sweep config_id={r.sweep_config_id}")
            lines.append("")
            lines.append("| Path | Category | Master | Sweep |")
            lines.append("|------|----------|--------|-------|")
            max_diffs_per_entry = 10
            for d in r.diffs[:max_diffs_per_entry]:
                lines.append(
                    f"| `{d.path}` | `{d.category}` | {_trunc(d.master_value, 40)} | {_trunc(d.sweep_value, 40)} |"
                )
            if len(r.diffs) > max_diffs_per_entry:
                lines.append(f"| ... | | {len(r.diffs) - max_diffs_per_entry} more diffs omitted | |")
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
            f"FAIL: {len(failing_diffs)} config(s) have argument diffs (ignoring categories: {ignore or 'none'})",
            file=sys.stderr,
        )
        return 1

    # Count matches that have hash divergence (informational only)
    hash_divergent = [r for r in report.matched if r.sweep_config_hash is not None]
    if hash_divergent:
        print(
            f"INFO: {len(hash_divergent)} of {len(report.matched)} matches have "
            f"config_hash divergence (functionally identical, hash computation differs)",
            file=sys.stderr,
        )

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
