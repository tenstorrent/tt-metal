#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Automated sweep trace validation against model trace.

Implements the logic from validate-sweep-trace.mdc and diagnose-config-hash-mismatch.mdc
as a single script that processes all operations in one pass, or a single operation.

Usage (all operations):
    python tests/sweep_framework/validate_all_traces.py \
        --model-trace-split model_tracer/traced_operations/ttnn_operations_master_v2_reconstructed_split \
        --sweep-traces-dir model_tracer/traced_operations \
        --output-report validation_report.txt

Usage (single operation):
    python tests/sweep_framework/validate_all_traces.py \
        --model-trace-split model_tracer/traced_operations/ttnn_operations_master_v2_reconstructed_split \
        --sweep-traces-dir model_tracer/traced_operations \
        --operation ttnn_add

    python tests/sweep_framework/validate_all_traces.py \
        --model-trace-split model_tracer/traced_operations/ttnn_operations_master_v2_reconstructed_split \
        --sweep-trace model_tracer/traced_operations/sweep_trace_add_model_traced_split \
        --operation ttnn.add
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Normalization helpers (validate-sweep-trace rule)
# ---------------------------------------------------------------------------

EXCLUDED_ARG_KEYS = {
    "multi_device_global_semaphore",
    "barrier_semaphore",
}


def _is_global_semaphore(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("type") == "global_semaphore"


def _is_set_with_mesh_coordinate(obj: Any) -> bool:
    if not (isinstance(obj, dict) and obj.get("type") == "set"):
        return False
    value = obj.get("value")
    if isinstance(value, list):
        return any(isinstance(v, dict) and "MeshCoordinate" in str(v) for v in value)
    return "MeshCoordinate" in str(value) if value else False


def normalize_for_comparison(obj: Any, *, _parent_key: str = "") -> Any:
    """Strip ignored keys for argument comparison per validate-sweep-trace rule."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in ("config_hash", "config_id", "executions"):
                continue
            if k in EXCLUDED_ARG_KEYS:
                continue
            if k == "device_ids":
                continue
            if k == "hash" and isinstance(v, (int, float)):
                continue
            if k == "value" and _is_global_semaphore(obj):
                continue
            if k == "value" and _is_set_with_mesh_coordinate(obj):
                continue
            normalized_v = normalize_for_comparison(v, _parent_key=k)
            if k == "shard_spec" and normalized_v == "None":
                normalized_v = None
            result[k] = normalized_v
        return result
    elif isinstance(obj, list):
        return [normalize_for_comparison(item, _parent_key=_parent_key) for item in obj]
    return obj


def serialize_normalized(args: dict) -> str:
    return json.dumps(normalize_for_comparison(args), sort_keys=True)


# ---------------------------------------------------------------------------
# Similarity scoring (for close-match fallback)
# ---------------------------------------------------------------------------

TENSOR_SUBFIELDS = [
    "original_shape",
    "original_dtype",
    "layout",
    ("memory_config", "memory_layout"),
    ("memory_config", "buffer_type"),
    "tensor_placement",
]


def _get_nested(obj: Any, path: str | tuple) -> Any:
    if not isinstance(obj, dict):
        return None
    if isinstance(path, str):
        return obj.get(path)
    cur = obj
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _is_tensor_arg(val: Any) -> bool:
    if isinstance(val, dict):
        return val.get("type") == "ttnn.Tensor"
    if isinstance(val, list) and val:
        return isinstance(val[0], dict) and val[0].get("type") == "ttnn.Tensor"
    return False


def compute_similarity(sweep_args: dict, model_args: dict) -> float:
    """Compute similarity between two normalized argument dicts."""
    sweep_n = normalize_for_comparison(sweep_args)
    model_n = normalize_for_comparison(model_args)

    all_keys = set(sweep_n.keys()) | set(model_n.keys())
    if not all_keys:
        return 1.0

    total_points = 0.0
    earned_points = 0.0

    for key in all_keys:
        sv = sweep_n.get(key)
        mv = model_n.get(key)

        if _is_tensor_arg(sv) or _is_tensor_arg(mv):
            s_tensors = sv if isinstance(sv, list) else ([sv] if sv else [])
            m_tensors = mv if isinstance(mv, list) else ([mv] if mv else [])

            max_len = max(len(s_tensors), len(m_tensors))
            for i in range(max_len):
                st = s_tensors[i] if i < len(s_tensors) else {}
                mt = m_tensors[i] if i < len(m_tensors) else {}
                for sf in TENSOR_SUBFIELDS:
                    total_points += 1
                    if _get_nested(st, sf) == _get_nested(mt, sf):
                        earned_points += 1
        else:
            total_points += 1
            if json.dumps(sv, sort_keys=True) == json.dumps(mv, sort_keys=True):
                earned_points += 1

    return earned_points / total_points if total_points > 0 else 1.0


# ---------------------------------------------------------------------------
# Config summary string (for unmatched reporting)
# ---------------------------------------------------------------------------


def _config_summary(args: dict) -> str:
    """Short human-readable summary of a config's key arguments."""
    parts = []
    arg0 = args.get("arg0")
    if isinstance(arg0, list) and arg0:
        shapes = []
        for t in arg0:
            if not isinstance(t, dict):
                continue
            s = t.get("original_shape") or t.get("shape")
            if s:
                shapes.append(str(s))
        if shapes:
            parts.append(" + ".join(shapes))
    elif isinstance(arg0, dict):
        s = arg0.get("original_shape") or arg0.get("shape")
        if s:
            parts.append(str(s))

    for k in ("dim", "num_links", "cluster_axis", "topology"):
        if k in args:
            parts.append(f"{k}={args[k]}")

    return ", ".join(parts) if parts else "(no summary)"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    sweep_config_id: int
    model_config_id: int
    match_type: str  # "exact" or "close"
    similarity: float = 1.0


@dataclass
class HashDiff:
    path: str
    model_value: Any
    sweep_value: Any
    category: int
    category_desc: str


@dataclass
class HashDiagnostic:
    sweep_config_id: int
    model_config_id: int
    model_hash: str
    sweep_hash: str
    hashes_match: bool
    diffs: list[HashDiff] = field(default_factory=list)
    categories: set[int] = field(default_factory=set)


@dataclass
class OpResult:
    op_name: str
    model_config_count: int
    sweep_config_count: int
    exact_matches: list[MatchResult] = field(default_factory=list)
    close_matches: list[MatchResult] = field(default_factory=list)
    unmatched_sweep: list[tuple[int, str, int | None, float]] = field(default_factory=list)
    unmatched_model: list[tuple[int, str]] = field(default_factory=list)
    hash_diagnostics: list[HashDiagnostic] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Matching procedure (validate-sweep-trace rule)
# ---------------------------------------------------------------------------


def match_configs(
    op_name: str,
    model_configs: list[dict],
    sweep_configs: list[dict],
) -> OpResult:
    result = OpResult(
        op_name=op_name,
        model_config_count=len(model_configs),
        sweep_config_count=len(sweep_configs),
    )

    model_ids = list(range(len(model_configs)))
    model_serialized = {i: serialize_normalized(model_configs[i].get("arguments", {})) for i in model_ids}
    consumed_model = set()

    sweep_with_ids = []
    for i, sc in enumerate(sweep_configs):
        sid = sc.get("config_id", i + 1)
        sweep_with_ids.append((sid, sc))

    unmatched_sweep_ids: list[tuple[int, dict]] = []

    for sid, sc in sweep_with_ids:
        sweep_ser = serialize_normalized(sc.get("arguments", {}))
        matched = False
        for mi in model_ids:
            if mi in consumed_model:
                continue
            if sweep_ser == model_serialized[mi]:
                mid = model_configs[mi].get("config_id", mi + 1)
                result.exact_matches.append(MatchResult(sid, mid, "exact"))
                consumed_model.add(mi)
                matched = True
                break
        if not matched:
            unmatched_sweep_ids.append((sid, sc))

    for sid, sc in unmatched_sweep_ids:
        best_mi = None
        best_sim = 0.0
        best_mid = None
        for mi in model_ids:
            if mi in consumed_model:
                continue
            sim = compute_similarity(
                sc.get("arguments", {}),
                model_configs[mi].get("arguments", {}),
            )
            if sim > best_sim:
                best_sim = sim
                best_mi = mi
                best_mid = model_configs[mi].get("config_id", mi + 1)

        if best_mi is not None and best_sim >= 0.80:
            result.close_matches.append(MatchResult(sid, best_mid, "close", best_sim))
            consumed_model.add(best_mi)
        else:
            summary = _config_summary(sc.get("arguments", {}))
            result.unmatched_sweep.append((sid, summary, best_mid, best_sim))

    for mi in model_ids:
        if mi not in consumed_model:
            mid = model_configs[mi].get("config_id", mi + 1)
            summary = _config_summary(model_configs[mi].get("arguments", {}))
            result.unmatched_model.append((mid, summary))

    return result


# ---------------------------------------------------------------------------
# Hash diagnostic (diagnose-config-hash-mismatch rule)
# ---------------------------------------------------------------------------

CATEGORY_LABELS = {
    1: "memory_config.hash -- device-specific, NOT fixable in sweep module",
    2: "shard_spec serialization -- fix in tracer normalization",
    3: "tensor_placement -- fixable in sweep module",
    4: "hardware tuple -- environment-dependent",
    5: "mesh config -- environment-dependent",
    6: "extra/missing keys -- fixable in sweep module",
    7: "argument value difference -- fixable in sweep module",
    8: "shard_spec.grid coordinates -- device/run-specific, NOT fixable in sweep module",
}


def _extract_hardware(config: dict) -> tuple | None:
    execs = config.get("executions", [])
    if not execs:
        return None
    mi = execs[0].get("machine_info", {})
    board = mi.get("board_type", "")
    series = mi.get("device_series", "")
    if isinstance(series, list):
        series = series[0] if series else ""
    count = mi.get("card_count", 1)
    return (board, series, count)


def _extract_mesh(config: dict) -> dict | None:
    execs = config.get("executions", [])
    if not execs:
        return None
    mi = execs[0].get("machine_info", {})
    tp = mi.get("tensor_placements")
    if not tp:
        return None
    return tp[0] if isinstance(tp, list) and tp else None


def _normalize_for_hash(obj: Any) -> Any:
    """Mirror the tracer's _normalize_for_hash: strip excluded arg keys,
    memory_config.hash, and canonicalize shard_spec None → "None" so diffs
    only show TRUE hash-affecting differences."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in EXCLUDED_ARG_KEYS:
                continue
            if k == "hash" and isinstance(v, (int, float)):
                continue
            nv = _normalize_for_hash(v)
            if k == "shard_spec" and nv is None:
                nv = "None"
            result[k] = nv
        return result
    elif isinstance(obj, list):
        return [_normalize_for_hash(item) for item in obj]
    return obj


def _diff_arguments_raw(
    model_args: dict,
    sweep_args: dict,
    prefix: str = "arguments",
) -> list[HashDiff]:
    """Deep-diff argument dicts after applying _normalize_for_hash to both sides,
    so only TRUE hash-affecting differences are reported."""
    diffs = []

    all_keys = set(model_args.keys()) | set(sweep_args.keys())
    for key in sorted(all_keys):
        mv = model_args.get(key)
        sv = sweep_args.get(key)

        if key not in model_args:
            diffs.append(
                HashDiff(
                    f"{prefix}.{key}",
                    "<missing>",
                    sv,
                    6,
                    CATEGORY_LABELS[6],
                )
            )
            continue
        if key not in sweep_args:
            diffs.append(
                HashDiff(
                    f"{prefix}.{key}",
                    mv,
                    "<missing>",
                    6,
                    CATEGORY_LABELS[6],
                )
            )
            continue

        if isinstance(mv, dict) and isinstance(sv, dict):
            diffs.extend(_diff_arguments_raw(mv, sv, f"{prefix}.{key}"))
        elif isinstance(mv, list) and isinstance(sv, list):
            max_len = max(len(mv), len(sv))
            for i in range(max_len):
                if i >= len(mv):
                    diffs.append(
                        HashDiff(
                            f"{prefix}.{key}[{i}]",
                            "<missing>",
                            sv[i],
                            6,
                            CATEGORY_LABELS[6],
                        )
                    )
                elif i >= len(sv):
                    diffs.append(
                        HashDiff(
                            f"{prefix}.{key}[{i}]",
                            mv[i],
                            "<missing>",
                            6,
                            CATEGORY_LABELS[6],
                        )
                    )
                elif isinstance(mv[i], dict) and isinstance(sv[i], dict):
                    diffs.extend(_diff_arguments_raw(mv[i], sv[i], f"{prefix}.{key}[{i}]"))
                elif mv[i] != sv[i]:
                    cat, desc = _classify_leaf_diff(f"{prefix}.{key}[{i}]", mv[i], sv[i])
                    diffs.append(HashDiff(f"{prefix}.{key}[{i}]", mv[i], sv[i], cat, desc))
        elif mv != sv:
            cat, desc = _classify_leaf_diff(f"{prefix}.{key}", mv, sv)
            diffs.append(HashDiff(f"{prefix}.{key}", mv, sv, cat, desc))

    return diffs


def _classify_leaf_diff(path: str, model_val: Any, sweep_val: Any) -> tuple[int, str]:
    if ".memory_config.hash" in path or path.endswith(".hash"):
        if isinstance(model_val, (int, float)) and isinstance(sweep_val, (int, float)):
            return 1, CATEGORY_LABELS[1]

    if "shard_spec" in path:
        m_none = model_val is None or model_val == "None"
        s_none = sweep_val is None or sweep_val == "None"
        if m_none and s_none:
            return 2, CATEGORY_LABELS[2]

    # shard_spec.grid coordinate values (start.x, start.y, end.x, end.y)
    # are device/run-specific CoreRange coordinates
    if "shard_spec.grid" in path and any(path.endswith(suffix) for suffix in (".x", ".y")):
        return 8, CATEGORY_LABELS[8]

    if "tensor_placement" in path or "placement" in path:
        return 3, CATEGORY_LABELS[3]

    return 7, CATEGORY_LABELS[7]


def diagnose_hash_pair(
    op_name: str,
    model_config: dict,
    sweep_config: dict,
    sweep_cid: int,
    model_cid: int,
) -> HashDiagnostic:
    m_hash = model_config.get("config_hash", "")
    s_hash = sweep_config.get("config_hash", "")

    diag = HashDiagnostic(
        sweep_config_id=sweep_cid,
        model_config_id=model_cid,
        model_hash=m_hash,
        sweep_hash=s_hash,
        hashes_match=(m_hash == s_hash),
    )

    if diag.hashes_match:
        return diag

    m_hw = _extract_hardware(model_config)
    s_hw = _extract_hardware(sweep_config)
    if m_hw != s_hw:
        diag.diffs.append(HashDiff("hardware", m_hw, s_hw, 4, CATEGORY_LABELS[4]))
        diag.categories.add(4)

    m_mesh = _extract_mesh(model_config)
    s_mesh = _extract_mesh(sweep_config)
    if m_mesh != s_mesh:
        diag.diffs.append(HashDiff("mesh", m_mesh, s_mesh, 5, CATEGORY_LABELS[5]))
        diag.categories.add(5)

    m_args = _normalize_for_hash(model_config.get("arguments", {}))
    s_args = _normalize_for_hash(sweep_config.get("arguments", {}))
    arg_diffs = _diff_arguments_raw(m_args, s_args)
    for d in arg_diffs:
        diag.diffs.append(d)
        diag.categories.add(d.category)

    return diag


# ---------------------------------------------------------------------------
# Discovery: find sweep trace split dirs for each op
# ---------------------------------------------------------------------------


def _normalize_op_name(name: str) -> str:
    """Normalize an operation name to the directory form (underscores).

    Accepts either dotted (``ttnn.add``) or underscored (``ttnn_add``) forms.
    """
    return name.replace(".", "_")


def discover_sweep_traces(
    model_split_dir: Path,
    sweep_traces_dir: Path,
    *,
    operation: str | None = None,
    sweep_trace_dir: Path | None = None,
) -> dict[str, tuple[Path, Path]]:
    """
    Returns {op_name: (model_json, sweep_json)} for ops that have both
    a model trace split and a sweep trace split.

    Parameters
    ----------
    operation : str, optional
        Restrict to a single operation.  Accepts either the directory form
        (``ttnn_add``) or the dotted form (``ttnn.add``).
    sweep_trace_dir : Path, optional
        A specific sweep trace split directory to use instead of
        scanning *sweep_traces_dir* for all ``sweep_trace_*_split/`` dirs.
    """
    pairs: dict[str, tuple[Path, Path]] = {}

    op_filter = _normalize_op_name(operation) if operation else None

    model_op_dirs: dict[str, Path] = {}
    if model_split_dir.is_dir():
        if op_filter:
            candidate = model_split_dir / op_filter
            model_json = candidate / "ttnn_operations_master.json"
            if candidate.is_dir() and model_json.is_file():
                model_op_dirs[op_filter] = model_json
        else:
            for d in model_split_dir.iterdir():
                if d.is_dir():
                    model_json = d / "ttnn_operations_master.json"
                    if model_json.is_file():
                        model_op_dirs[d.name] = model_json

    if sweep_trace_dir is not None:
        sweep_split_dirs = [sweep_trace_dir] if sweep_trace_dir.is_dir() else []
    else:
        sweep_split_dirs = sorted(
            d
            for d in sweep_traces_dir.iterdir()
            if d.is_dir() and "sweep_trace" in d.name and d.name.endswith("_split")
        )

    for ssd in sweep_split_dirs:
        if op_filter:
            op_dir = ssd / op_filter
            if op_dir.is_dir():
                sweep_json = op_dir / "ttnn_operations_master.json"
                if sweep_json.is_file() and op_filter in model_op_dirs and op_filter not in pairs:
                    pairs[op_filter] = (model_op_dirs[op_filter], sweep_json)
        else:
            for op_dir in ssd.iterdir():
                if not op_dir.is_dir():
                    continue
                sweep_json = op_dir / "ttnn_operations_master.json"
                if not sweep_json.is_file():
                    continue
                op_dir_name = op_dir.name
                if op_dir_name in model_op_dirs and op_dir_name not in pairs:
                    pairs[op_dir_name] = (model_op_dirs[op_dir_name], sweep_json)

    return pairs


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    model_split_dir: Path,
    sweep_traces_dir: Path,
    op_results: list[OpResult],
) -> str:
    lines: list[str] = []

    total_model = sum(r.model_config_count for r in op_results)
    total_sweep = sum(r.sweep_config_count for r in op_results)

    lines.append("=== Sweep Trace Validation Report ===")
    lines.append(f"Model trace split: {model_split_dir}")
    lines.append(f"Sweep traces dir:  {sweep_traces_dir}")
    lines.append(f"Operations validated: {len(op_results)}")
    lines.append(f"Total model configs: {total_model}  |  Total sweep configs: {total_sweep}")
    lines.append("")

    for r in op_results:
        exact_count = len(r.exact_matches)
        close_count = len(r.close_matches)
        total_matched = exact_count + close_count

        lines.append(f"--- Operation: {r.op_name} ---")
        lines.append(f"  Model configs: {r.model_config_count} | Sweep configs: {r.sweep_config_count}")
        lines.append(f"  Exact matches: {exact_count}/{r.sweep_config_count}")
        if close_count:
            lines.append(f"  Close matches: {close_count}/{r.sweep_config_count} (>=80% similarity)")
        lines.append(f"  Unmatched sweep configs: {len(r.unmatched_sweep)}")
        lines.append(f"  Unmatched model configs: {len(r.unmatched_model)}")
        lines.append("")

        lines.append("  Config ID mapping (sweep -> model):")
        if r.exact_matches:
            lines.append("    Exact matches:")
            for m in sorted(r.exact_matches, key=lambda x: x.sweep_config_id):
                lines.append(f"      sweep config_id {m.sweep_config_id:<4} -> model config_id {m.model_config_id}")
        if r.close_matches:
            lines.append("    Close matches:")
            for m in sorted(r.close_matches, key=lambda x: x.sweep_config_id):
                lines.append(
                    f"      sweep config_id {m.sweep_config_id:<4} -> model config_id {m.model_config_id} "
                    f"({m.similarity * 100:.1f}% similarity)"
                )
        if r.unmatched_sweep:
            lines.append("    Unmatched sweep configs:")
            for sid, summary, best_mid, best_sim in r.unmatched_sweep:
                cand = ""
                if best_mid is not None:
                    cand = f" (best candidate: model config_id {best_mid} at {best_sim * 100:.1f}%)"
                lines.append(f"      sweep config_id {sid}: no model match{cand} -- {summary}")
        if r.unmatched_model:
            lines.append("    Unmatched model configs:")
            for mid, summary in r.unmatched_model:
                lines.append(f"      model config_id {mid}: no sweep match -- {summary}")
        lines.append("")

        hash_mismatches = [d for d in r.hash_diagnostics if not d.hashes_match]
        hash_matches = [d for d in r.hash_diagnostics if d.hashes_match]

        if hash_mismatches:
            lines.append("  config_hash diagnostics:")
            lines.append(f"    Pairs with matching hash: {len(hash_matches)}")
            lines.append(f"    Pairs with different hash: {len(hash_mismatches)}")
            for diag in hash_mismatches:
                lines.append(
                    f"    Pair: sweep config_id {diag.sweep_config_id} -> model config_id {diag.model_config_id}"
                )
                cats = sorted(diag.categories)
                lines.append(f"      Root causes: {', '.join(f'Category {c}' for c in cats)}")
                for d in diag.diffs[:10]:
                    lines.append(f"        {d.path}: model={_trunc(d.model_value)} sweep={_trunc(d.sweep_value)}")
                    lines.append(f"          Category {d.category}: {d.category_desc}")
                if len(diag.diffs) > 10:
                    lines.append(f"        ... and {len(diag.diffs) - 10} more diffs")
            lines.append("")
        else:
            lines.append(f"  config_hash: all {len(r.hash_diagnostics)} matched pairs have identical hashes")
            lines.append("")

    grand_exact = sum(len(r.exact_matches) for r in op_results)
    grand_close = sum(len(r.close_matches) for r in op_results)
    grand_unmatched_s = sum(len(r.unmatched_sweep) for r in op_results)
    grand_unmatched_m = sum(len(r.unmatched_model) for r in op_results)
    grand_matched = grand_exact + grand_close
    coverage = (grand_matched / total_sweep * 100) if total_sweep > 0 else 0.0

    lines.append("=== Summary ===")
    lines.append(
        f"Total sweep configs: {total_sweep} | "
        f"Matched: {grand_exact} exact + {grand_close} close | "
        f"Unmatched sweep: {grand_unmatched_s} | "
        f"Unmatched model: {grand_unmatched_m}"
    )
    lines.append(f"Coverage: {coverage:.1f}%  (of sweep configs matched)")
    lines.append("")

    cat_counts: dict[int, int] = {}
    total_hash_pairs = 0
    total_hash_match = 0
    for r in op_results:
        for diag in r.hash_diagnostics:
            total_hash_pairs += 1
            if diag.hashes_match:
                total_hash_match += 1
            else:
                for c in diag.categories:
                    cat_counts[c] = cat_counts.get(c, 0) + 1

    lines.append("=== config_hash Summary ===")
    lines.append(f"Total matched pairs analyzed: {total_hash_pairs}")
    lines.append(f"Pairs with identical config_hash: {total_hash_match}")
    lines.append(f"Pairs with different config_hash: {total_hash_pairs - total_hash_match}")
    if cat_counts:
        lines.append("Root cause breakdown:")
        for cat in sorted(cat_counts):
            lines.append(f"  Category {cat} ({CATEGORY_LABELS.get(cat, '?')}): {cat_counts[cat]} pairs")
        sweep_fixable = sum(cat_counts.get(c, 0) for c in (3, 6, 7))
        tracer_fix = sum(cat_counts.get(c, 0) for c in (1, 2))
        env_dep = sum(cat_counts.get(c, 0) for c in (4, 5, 8))
        lines.append(f"Actionable in sweep module: {sweep_fixable} pair-categories (Categories 3, 6, 7)")
        lines.append(f"Requires tracer fix: {tracer_fix} pair-categories (Categories 1, 2)")
        lines.append(f"Environment/device-dependent: {env_dep} pair-categories (Categories 4, 5, 8)")
    lines.append("")

    lines.append("=== Full Config ID Match List ===")
    lines.append(
        f"{'Operation':<50} {'Sweep ID':>10} {'Model ID':>10} {'Match':>8} {'Similarity':>12} {'Hash Match':>12}"
    )
    lines.append("-" * 104)
    for r in op_results:
        all_matches = sorted(
            [(m, "exact") for m in r.exact_matches] + [(m, "close") for m in r.close_matches],
            key=lambda x: x[0].sweep_config_id,
        )
        for m, mt in all_matches:
            hash_status = "?"
            for diag in r.hash_diagnostics:
                if diag.sweep_config_id == m.sweep_config_id:
                    hash_status = "YES" if diag.hashes_match else "NO"
                    break
            sim_str = "100.0%" if mt == "exact" else f"{m.similarity * 100:.1f}%"
            lines.append(
                f"{r.op_name:<50} {m.sweep_config_id:>10} {m.model_config_id:>10} {mt:>8} {sim_str:>12} {hash_status:>12}"
            )

    return "\n".join(lines)


def _trunc(val: Any, max_len: int = 60) -> str:
    s = str(val)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _find_config_by_id(
    configs: list[dict],
    config_id: int,
) -> dict | None:
    for i, c in enumerate(configs):
        cid = c.get("config_id", i + 1)
        if cid == config_id:
            return c
    return None


def run_validation(
    model_split_dir: Path,
    sweep_traces_dir: Path,
    *,
    operation: str | None = None,
    sweep_trace_dir: Path | None = None,
) -> list[OpResult]:
    pairs = discover_sweep_traces(
        model_split_dir,
        sweep_traces_dir,
        operation=operation,
        sweep_trace_dir=sweep_trace_dir,
    )

    if not pairs:
        print("WARNING: No matching operation pairs found.", file=sys.stderr)
        return []

    op_results: list[OpResult] = []

    for op_dir_name in sorted(pairs.keys()):
        model_json, sweep_json = pairs[op_dir_name]

        with open(model_json) as f:
            model_data = json.load(f)
        with open(sweep_json) as f:
            sweep_data = json.load(f)

        for op_name, op_info in model_data.get("operations", {}).items():
            model_configs = op_info.get("configurations", [])

            sweep_op_info = sweep_data.get("operations", {}).get(op_name)
            if sweep_op_info is None:
                op_results.append(
                    OpResult(
                        op_name=op_name,
                        model_config_count=len(model_configs),
                        sweep_config_count=0,
                    )
                )
                continue

            sweep_configs = sweep_op_info.get("configurations", [])

            result = match_configs(op_name, model_configs, sweep_configs)

            all_matched = result.exact_matches + result.close_matches
            for m in all_matched:
                model_cfg = _find_config_by_id(model_configs, m.model_config_id)
                sweep_cfg = _find_config_by_id(sweep_configs, m.sweep_config_id)
                if model_cfg and sweep_cfg:
                    diag = diagnose_hash_pair(
                        op_name,
                        model_cfg,
                        sweep_cfg,
                        m.sweep_config_id,
                        m.model_config_id,
                    )
                    result.hash_diagnostics.append(diag)

            op_results.append(result)

    return op_results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate sweep traces against model traces for all operations, or a single operation.",
    )
    parser.add_argument(
        "--model-trace-split",
        required=True,
        type=str,
        help="Path to the model trace split directory",
    )
    parser.add_argument(
        "--sweep-traces-dir",
        type=str,
        default=None,
        help="Path to the directory containing sweep_trace_*_split/ directories (required unless --sweep-trace is given)",
    )
    parser.add_argument(
        "--sweep-trace",
        type=str,
        default=None,
        help="Path to a single sweep trace split directory (e.g. sweep_trace_add_model_traced_split). "
        "Use instead of --sweep-traces-dir to target one sweep.",
    )
    parser.add_argument(
        "-o",
        "--operation",
        type=str,
        default=None,
        help="Validate a single operation by name. Accepts either the directory form "
        "(ttnn_add) or dotted form (ttnn.add).",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Path to write the validation report (default: stdout)",
    )

    args = parser.parse_args()

    if args.sweep_traces_dir is None and args.sweep_trace is None:
        parser.error("one of --sweep-traces-dir or --sweep-trace is required")

    model_split_dir = Path(args.model_trace_split).resolve()
    sweep_trace_dir: Path | None = Path(args.sweep_trace).resolve() if args.sweep_trace else None
    sweep_traces_dir: Path | None = Path(args.sweep_traces_dir).resolve() if args.sweep_traces_dir else None

    if not model_split_dir.is_dir():
        print(f"ERROR: Model trace split directory not found: {model_split_dir}", file=sys.stderr)
        return 1
    if sweep_traces_dir is not None and not sweep_traces_dir.is_dir():
        print(f"ERROR: Sweep traces directory not found: {sweep_traces_dir}", file=sys.stderr)
        return 1
    if sweep_trace_dir is not None and not sweep_trace_dir.is_dir():
        print(f"ERROR: Sweep trace directory not found: {sweep_trace_dir}", file=sys.stderr)
        return 1

    print(f"Model trace split: {model_split_dir}", file=sys.stderr)
    if sweep_trace_dir:
        print(f"Sweep trace dir:   {sweep_trace_dir}", file=sys.stderr)
    else:
        print(f"Sweep traces dir:  {sweep_traces_dir}", file=sys.stderr)
    if args.operation:
        print(f"Operation filter:  {args.operation}", file=sys.stderr)

    # When --sweep-trace is given without --sweep-traces-dir, use its parent
    # as the sweep_traces_dir so the rest of the pipeline has a valid Path.
    effective_sweep_traces_dir = sweep_traces_dir or sweep_trace_dir.parent

    op_results = run_validation(
        model_split_dir,
        effective_sweep_traces_dir,
        operation=args.operation,
        sweep_trace_dir=sweep_trace_dir,
    )

    report = generate_report(model_split_dir, effective_sweep_traces_dir, op_results)

    if args.output_report:
        output_path = Path(args.output_report)
        output_path.write_text(report)
        print(f"\nReport written to: {output_path}", file=sys.stderr)
    else:
        print(report)

    any_unmatched = any(r.unmatched_sweep for r in op_results)
    return 1 if any_unmatched else 0


if __name__ == "__main__":
    sys.exit(main())
