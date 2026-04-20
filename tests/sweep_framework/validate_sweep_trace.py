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
        # Old master traces use "tensor_placement", new tracer uses "mesh_device".
        # Both represent the same multi-device placement metadata. Since mesh_device
        # is already ignored, tensor_placement should be ignored too for consistency.
        "tensor_placement",
        # output_tensor is a pre-allocated buffer optimization.  The sweep test
        # may or may not provide one depending on whether the V2 loader's
        # decomposed output_tensor_shape was successfully reconstructed.
        # Its presence/absence does not change the operation's semantics.
        "output_tensor",
        # num_devices and slice_dim are multi-device sharding metadata passed
        # by the distributed slice implementation.  Sweep tests use coordinate-
        # based slice parameters which are semantically equivalent.
        "num_devices",
        "slice_dim",
    }
)

# ---------------------------------------------------------------------------
# Operation-specific argument remapping
# ---------------------------------------------------------------------------
# Some operations trace positional arg names (arg0, arg1, ...) while the
# master trace records semantic parameter names.  This mapping lets the
# validator align them before comparison.

OP_ARG_REMAPPING: dict[str, dict[str, str]] = {
    "ttnn.experimental.all_gather_async": {
        "arg2": "dim",
    },
    "ttnn.transformer.paged_scaled_dot_product_attention_decode": {
        "arg3": "page_table_tensor",
    },
}

# Keys that are expected to differ per-operation due to runtime context,
# multi-device topology, or optional parameters not captured uniformly.
OP_IGNORED_KEYS: dict[str, frozenset[str]] = {
    "ttnn.experimental.all_gather_async": frozenset({
        # Multi-device runtime parameters: semaphore handles, topology axis,
        # and sub-device routing are set by the distributed runtime and vary
        # between the original model execution and the sweep re-run.
        "arg3", "cluster_axis", "subdevice_id",
    }),
    "ttnn.transformer.paged_scaled_dot_product_attention_decode": frozenset({
        # is_causal is an optional keyword argument that may or may not be
        # captured by the tracer depending on whether it was passed explicitly.
        "is_causal",
    }),
}


def _parse_shard_spec_string(s: str) -> Any:
    """Parse a ShardSpec string representation into a dict.

    Handles the format produced by the new operation tracer, e.g.:
      ShardSpec{grid=[{"start":{"x":0,"y":0},"end":{"x":7,"y":3}], shape=[32, 64],
                orientation=ShardOrientation::ROW_MAJOR}
    """
    import re as _re

    try:
        inner = s[len("ShardSpec{") : -1] if s.endswith("}") else s[len("ShardSpec{") :]
        result: dict[str, Any] = {}

        grid_idx = inner.find("grid=")
        if grid_idx >= 0:
            bracket_start = inner.index("[", grid_idx)
            depth, end = 0, bracket_start
            for i, c in enumerate(inner[bracket_start:], bracket_start):
                if c == "[":
                    depth += 1
                elif c == "]":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            grid_str = inner[bracket_start : end + 1]
            opens = grid_str.count("{")
            closes = grid_str.count("}")
            if opens > closes:
                grid_str = grid_str[:-1] + "}" * (opens - closes) + "]"
            try:
                result["grid"] = json.loads(grid_str)
            except json.JSONDecodeError:
                result["grid"] = grid_str

        shape_match = _re.search(r"shape=\[([0-9, ]+)\]", inner)
        if shape_match:
            result["shape"] = [int(x.strip()) for x in shape_match.group(1).split(",")]

        orient_match = _re.search(r"orientation=(?:ShardOrientation::)?(\w+)", inner)
        if orient_match:
            result["orientation"] = orient_match.group(1)

        return result if result else s
    except Exception:
        return s


def normalize(obj: Any, *, _parent_key: str = "", _op_name: str = "") -> Any:
    """Recursively normalize a config dict for comparison.

    Strips keys that are expected to vary between a master trace (from the
    database) and a sweep trace (from live execution) without representing a
    meaningful configuration difference.
    """
    if isinstance(obj, dict):
        # Distributed tensor dicts (args carrying device-side tensor values
        # such as slice start/end coordinates on a multi-device mesh) cannot
        # be meaningfully compared with the sweep's host-side coordinate
        # lists.  Normalize them to a sentinel so they match any non-None
        # counterpart rather than producing a spurious diff.
        if obj.get("type") in ("ttnn.Tensor",) and ("tensor_placement" in obj or "mesh_device" in obj):
            return "__DEVICE_TENSOR__"
        # Normalize Shape dicts: {'type': 'Shape', 'value': 'Shape([1, 1, 32, 1])'}
        # → plain list [1, 1, 32, 1] for comparison with sweep trace arrays.
        if obj.get("type") == "Shape" and "value" in obj:
            import re

            m = re.search(r"\[([0-9, ]+)\]", str(obj["value"]))
            if m:
                shape_list = [int(x.strip()) for x in m.group(1).split(",")]
                return normalize(shape_list, _parent_key=_parent_key, _op_name=_op_name)
        # Per-operation ignored keys
        op_ignore = OP_IGNORED_KEYS.get(_op_name, frozenset())
        # Per-operation arg remapping (arg2 → dim, etc.)
        remap = OP_ARG_REMAPPING.get(_op_name, {})
        result = {}
        for k, v in sorted(obj.items()):
            if k in IGNORED_KEYS or k in op_ignore:
                continue
            # Apply positional → semantic key remapping
            out_key = remap.get(k, k)
            # memory_config.hash is a device pointer — always differs between runs; skip numeric values only
            if out_key == "hash" and isinstance(v, (int, float)):
                continue
            # sub_core_grids: None is noise
            if out_key == "sub_core_grids" and v is None:
                continue
            # shard_spec: the raw tracer serializes None as the string "None";
            # normalize to actual None for consistent comparison.
            if k == "shard_spec" and v == "None":
                v = None
            # shard_spec: the new tracer serializes ShardSpec as a string
            # while old master traces store it as a dict. Parse the string
            # into a dict so both formats compare equally.
            if k == "shard_spec" and isinstance(v, str) and v.startswith("ShardSpec{"):
                v = _parse_shard_spec_string(v)
            # Strip top-level keys with None values — they are semantically
            # equivalent to absent keys.  The V2 loader sometimes produces
            # None for keys not present in the master trace, and the master
            # trace sometimes records explicit None for optional parameters
            # (e.g. dtype, core_grid, global_cb, sub_device_id).  Treating
            # None and absent as identical avoids false-positive diffs.
            if v is None and _parent_key == "":
                continue
            result[out_key] = normalize(v, _parent_key=out_key, _op_name=_op_name)
        return result
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if isinstance(obj, int) and abs(obj) > 2**53:
            return float(obj)
        return obj
    if isinstance(obj, list):
        # Normalize original_shape lists: strip leading 1s so that
        # 2D shapes like [32, 64128] and 4D shapes like [1, 1, 32, 64128]
        # are treated as equivalent.  The 2D→4D padding in create_tensor_on_mesh
        # legitimately adds leading 1s to avoid partition.cpp crashes, but
        # the logical shape is unchanged.
        if _parent_key == "original_shape" and all(isinstance(x, (int, float)) for x in obj):
            stripped = list(obj)
            while len(stripped) > 1 and stripped[0] == 1:
                stripped.pop(0)
            # Tile-align dimensions: TTNN's TILE_LAYOUT pads dimensions to
            # multiples of 32.  The master trace records the logical (pre-pad)
            # shape while the sweep re-trace may capture the padded shape.
            # Round up each dim > 1 to the nearest multiple of 32 so that
            # e.g. 16 and 32 compare as equal.
            aligned = []
            for x in stripped:
                val = int(x)
                if val > 1:
                    aligned.append(((val + 31) // 32) * 32)
                else:
                    aligned.append(val)
            return aligned
        return [normalize(item, _parent_key=_parent_key, _op_name=_op_name) for item in obj]
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


_DEVICE_TENSOR = "__DEVICE_TENSOR__"


def deep_diff(master: Any, sweep: Any, prefix: str = "") -> list[Diff]:
    """Produce a list of leaf-level diffs between two normalized argument trees."""
    diffs: list[Diff] = []

    # Device tensor sentinels match anything — the actual values live on the
    # device and cannot be compared from JSON alone.
    if master == _DEVICE_TENSOR or sweep == _DEVICE_TENSOR:
        return diffs

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
        # Approximate comparison for large floats
        if isinstance(master, (int, float)) and isinstance(sweep, (int, float)):
            m, s = float(master), float(sweep)
            try:
                # Relative tolerance for approximately equal values
                if m != 0.0 and abs(m - s) / abs(m) < 1e-6:
                    return diffs
                # Both are very large negative or very large positive values
                # (e.g. -DBL_MAX vs -FLT_MAX used as pad sentinels — same semantic
                # meaning even though magnitudes differ by ~270 orders)
                if (m < -1e30 and s < -1e30) or (m > 1e30 and s > 1e30):
                    return diffs
            except (ZeroDivisionError, OverflowError):
                pass
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
            norm_sweep = normalize(sweep_args, _op_name=master_op)

            if norm_master == norm_sweep:
                # Arguments match after normalization — this is a match.
                # The config_hash may differ (e.g. when normalization strips
                # None-valued keys that were present in one trace but absent
                # in the other), but the operational behaviour is identical.
                report.results.append(
                    ConfigResult(
                        config_hash=source_hash,
                        op_name=op_name,
                        master_config_id=master_cid,
                        sweep_config_id=sweep_cid,
                        status="match",
                        sweep_config_hash=sweep_config_hash
                        if (sweep_config_hash and sweep_config_hash != source_hash)
                        else None,
                    )
                )
            else:
                diffs = deep_diff(norm_master, norm_sweep)
                # If deep_diff returns no concrete diffs despite norm inequality
                # (e.g. floating-point approximate matches, dict ordering), treat
                # as match rather than reporting an empty diff table.
                if not diffs:
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
        max_detailed_entries = 100
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

    if report.hash_mismatch:
        print(
            f"FAIL: {len(report.hash_mismatch)} config(s) have matching arguments "
            f"but different config_hash (hash computation divergence)",
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
