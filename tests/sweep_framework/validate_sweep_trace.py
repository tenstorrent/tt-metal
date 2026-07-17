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
import re
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
    }
)


# Per-op kwarg → positional-index mapping. The model trace records kwargs
# under their semantic names (e.g. ``shape`` for ``ttnn.reshape``) while the
# sweep test calls the op with positional args, which the tracer captures as
# ``arg1``. Canonicalize both sides to the positional form before comparing.
KWARG_ALIASES = {
    "ttnn.reshape": {"shape": 1},
    "ttnn.embedding": {"weight": 1},
    "ttnn.repeat": {"repeat_dims": 1},
    "ttnn.unsqueeze": {"dim": 1},
    "ttnn.scatter": {"input": 0, "index": 1},
    "ttnn.linear": {"input_tensor_b": 1},
    "ttnn.experimental.paged_fill_cache": {"page_table": 2},
    "ttnn.experimental.paged_update_cache": {"input_tensor": 0, "input_tensor_b": 1},
    # Model traces sometimes record the gather input as the kwarg ``input_tensor``
    # and the gather dim as the kwarg ``dim``; the sweep calls all_gather_async
    # positionally so the tracer captures them as ``arg0``/``arg1``. Canonicalize
    # both to the positional form before comparing.
    "ttnn.experimental.all_gather_async": {"input_tensor": 0, "dim": 1},
}


def canonicalize_op_args(op_name: str, args: dict) -> dict:
    """Rename op-specific semantic kwargs to their positional ``argN`` form."""
    aliases = KWARG_ALIASES.get(op_name, {})
    if not aliases or not isinstance(args, dict):
        return args
    out = dict(args)
    for kwarg, idx in aliases.items():
        argN = f"arg{idx}"
        if kwarg in out and argN not in out:
            out[argN] = out.pop(kwarg)
    return out


import ast as _ast
import re as _re

_SHAPE_VALUE_RE = _re.compile(r"^Shape\(\[(.*)\]\)$")


_SET_REPR_RE = _re.compile(r"^\s*\{(.*)\}\s*$", _re.S)


def _coerce_set_repr(value):
    """Parse a python-set repr like '{X, Y, Z}' to a sorted tuple for comparison.

    Master and sweep traces both serialize sets via str(), which preserves
    insertion order — but the underlying set is unordered. Sorting makes the
    comparison order-independent.
    """
    if not isinstance(value, str):
        return value
    m = _SET_REPR_RE.match(value)
    if not m:
        return value
    inner = m.group(1).strip()
    if not inner:
        return ()
    # Split on top-level commas. Items here are all "MeshCoordinate([..])" so
    # we need to track bracket depth.
    parts = []
    depth = 0
    cur = []
    for ch in inner:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return tuple(sorted(parts))


def _coerce_shape_value(obj):
    """Convert {'type': 'Shape', 'value': 'Shape([1,2,3])'} → [1, 2, 3]."""
    if isinstance(obj, dict) and obj.get("type") == "Shape":
        m = _SHAPE_VALUE_RE.match(str(obj.get("value", "")).strip())
        if m:
            try:
                inner = m.group(1).strip()
                if not inner:
                    return []
                return [int(x.strip()) for x in inner.split(",") if x.strip()]
            except ValueError:
                return obj
    return obj


def _normalize_tensor_placement(tp):
    """Collapse fully-replicated placements across different mesh shapes.

    `[4, 8]` distribution + ['PlacementReplicate', 'PlacementReplicate']
    is semantically identical to `[32]` distribution + ['PlacementReplicate'].
    Same for any rectangular full-replication case. Reduce both to the 1-D
    fully-replicated form for comparison.
    """
    if not isinstance(tp, dict):
        return tp
    plac_raw = tp.get("placement")
    dist_raw = tp.get("distribution_shape")
    mesh_raw = tp.get("mesh_device_shape")
    try:
        plac = _ast.literal_eval(plac_raw) if isinstance(plac_raw, str) else plac_raw
        dist = _ast.literal_eval(dist_raw) if isinstance(dist_raw, str) else dist_raw
    except Exception:
        return tp
    if not isinstance(plac, (list, tuple)) or not isinstance(dist, (list, tuple)):
        return tp
    if not plac or any("Replicate" not in str(p) for p in plac):
        return tp
    # Fully replicated: collapse to 1-D
    total = 1
    for d in dist:
        try:
            total *= int(d)
        except (TypeError, ValueError):
            return tp
    out = dict(tp)
    out["placement"] = "['PlacementReplicate']"
    out["distribution_shape"] = f"[{total}]"
    if mesh_raw is not None:
        out["mesh_device_shape"] = f"[1, {total}]"
    return out


_PC_GRID_DASH = re.compile(r"compute_with_storage_grid_size=(\d+)-(\d+)")
_PC_GRID_PARENS_SPACED = re.compile(r"compute_with_storage_grid_size=\(x=(\d+),\s*y=(\d+)\)")


def _canonicalize_program_config_repr(s: str) -> str:
    """Canonicalize program_config repr drift across ttnn versions.

    Older tracer recorded `compute_with_storage_grid_size=(x=N,y=M)` with no
    space; newer ttnn produces `compute_with_storage_grid_size=N-M`. Both
    encode the same grid; we canonicalize both to `(x=N,y=M)` (no space) so
    the validator's string compare doesn't flag the format drift.
    """
    if "compute_with_storage_grid_size=" not in s:
        return s
    s = _PC_GRID_DASH.sub(r"compute_with_storage_grid_size=(x=\1,y=\2)", s)
    s = _PC_GRID_PARENS_SPACED.sub(r"compute_with_storage_grid_size=(x=\1,y=\2)", s)
    return s


def normalize(obj: Any, *, _parent_key: str = "") -> Any:
    """Recursively normalize a config dict for comparison.

    Strips keys that are expected to vary between a master trace (from the
    database) and a sweep trace (from live execution) without representing a
    meaningful configuration difference.
    """
    if isinstance(obj, dict):
        # Coerce wrapper Shape objects → list before per-key processing
        shape_form = _coerce_shape_value(obj)
        if shape_form is not obj:
            return shape_form
        # Normalize fully-replicated tensor_placement across mesh shapes
        if _parent_key == "tensor_placement":
            obj = _normalize_tensor_placement(obj)
        result = {}
        for k, v in sorted(obj.items()):
            if k in IGNORED_KEYS:
                continue
            # memory_config.hash is a device pointer — always differs between runs; skip numeric values only
            if k == "hash" and isinstance(v, (int, float)):
                continue
            # sub_core_grids: None is noise
            if k == "sub_core_grids" and v is None:
                continue
            result[k] = normalize(v, _parent_key=k)
        # shard_spec.grid is logically a set of CoreRanges — sort the entries so
        # master/sweep traces with the same grids in different orders compare equal.
        if "grid" in result and isinstance(result["grid"], list):
            try:
                result["grid"] = sorted(
                    result["grid"],
                    key=lambda g: (
                        (g.get("start", {}).get("x", 0), g.get("start", {}).get("y", 0))
                        if isinstance(g, dict)
                        else (0, 0)
                    ),
                )
            except (TypeError, AttributeError):
                pass
        return result
    if isinstance(obj, list):
        return [normalize(item, _parent_key=_parent_key) for item in obj]
    # Some traces serialize None as the string 'None' (e.g. shard_spec='None');
    # canonicalize so the diff doesn't flag it.
    if obj == "None":
        return None
    # Set repr "{a, b, c}" should compare as an unordered set.
    if isinstance(obj, str) and obj.startswith("{") and obj.endswith("}"):
        coerced = _coerce_set_repr(obj)
        if coerced is not obj:
            return coerced
    # Canonicalize program_config repr drift (`(x=N,y=M)` vs `N-M`).
    if isinstance(obj, str) and "compute_with_storage_grid_size=" in obj:
        return _canonicalize_program_config_repr(obj)
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
                # Skip extra keys with None value — the tracer captures function
                # defaults (dtype=None, memory_config=None) that the master trace
                # never had. These are not real diffs.
                # Also skip known sweep-framework output kwargs that the model
                # trace never captures (e.g. the output memory_config passed by
                # the sweep module to control placement).
                if sweep[k] is None or k in ("memory_config", "core_grid", "dtype"):
                    continue
                diffs.append(Diff(child_path, "<missing>", sweep[k], "extra_key"))
            elif k not in sweep:
                if master[k] is not None:
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

            # Skip if this master config was already matched by a previous
            # sweep trace (e.g. same config exercised by both model_traced
            # and lead_models scopes).
            if source_hash in matched_hashes:
                continue

            # Direct match by hash — now compare arguments
            matched_hashes.add(source_hash)
            sweep_args = cfg.get("arguments", {})
            sweep_config_hash = cfg.get("config_hash")

            norm_master = normalize(canonicalize_op_args(op_name, master_args))
            norm_sweep = normalize(canonicalize_op_args(op_name, sweep_args))

            if norm_master == norm_sweep:
                # Arguments match (after normalization) — that's a match.
                # The on-disk config_hash may differ if the trace was recorded
                # with an older repr format and the sweep used the newer one
                # (see _canonicalize_program_config_repr). The hash is a
                # pre-normalization fingerprint; with normalized args equal,
                # the hash difference is expected drift, not a real divergence.
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
                        status="diff" if diffs else "match",
                        diffs=diffs,
                        sweep_config_hash=sweep_config_hash,
                    )
                )

    # Build a lookup of matched configs' normalized arguments by op_name
    # to detect argument-level duplicates that the tracer collapsed.
    matched_norm_args: dict[str, set[str]] = {}  # op_name -> set of normalized arg JSON
    for ch in matched_hashes:
        if ch in master_index:
            m_op, _, m_args = master_index[ch]
            norm = json.dumps(normalize(canonicalize_op_args(m_op, m_args)), sort_keys=True)
            matched_norm_args.setdefault(m_op, set()).add(norm)

    # Also collect normalized args from ALL sweep traces (including incidental).
    # When a model makes N calls to the same op with identical args, the master
    # has N configs (each with a unique config_hash from the trace context) but
    # the sweep produces only 1 unique trace.  If the sweep trace's
    # sweep_source_hash doesn't match any master config_hash, the hash-level
    # match above won't fire and the dedup above only catches masters whose
    # normalized args match an already-hash-matched master.  By also comparing
    # against sweep trace args directly, we recover these "exercised but
    # hash-mismatched" configs.
    sweep_norm_args: dict[str, set[str]] = {}  # op_name -> set of normalized arg JSON
    for op_name, op_info in sweep_data.get("operations", {}).items():
        for cfg in op_info.get("configurations", []):
            sweep_args = cfg.get("arguments", {})
            norm = json.dumps(normalize(canonicalize_op_args(op_name, sweep_args)), sort_keys=True)
            sweep_norm_args.setdefault(op_name, set()).add(norm)

    # Report master configs with no sweep execution
    for ch, (op_name, cid, _args) in master_index.items():
        if ch not in matched_hashes:
            norm_args = json.dumps(normalize(canonicalize_op_args(op_name, _args)), sort_keys=True)
            # Check if an identical config (same normalized args) was already matched
            # via hash, or was exercised by any sweep trace with matching args.
            if norm_args in matched_norm_args.get(op_name, set()):
                matched_hashes.add(ch)
                report.results.append(
                    ConfigResult(
                        config_hash=ch,
                        op_name=op_name,
                        master_config_id=cid,
                        sweep_config_id=None,
                        status="match",
                    )
                )
                continue
            if norm_args in sweep_norm_args.get(op_name, set()):
                matched_hashes.add(ch)
                report.results.append(
                    ConfigResult(
                        config_hash=ch,
                        op_name=op_name,
                        master_config_id=cid,
                        sweep_config_id=None,
                        status="match",
                    )
                )
                continue
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
