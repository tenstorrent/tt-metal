#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Homogenize multiple master JSON files into a single consolidated trace.

Given N master JSON files produced by independent invocations of
``generic_ops_tracer.py`` (typically the per-leg outputs of a matrix trace
job), this script merges them into one canonical master JSON representing
a single logical trace_run:

- Configurations are deduplicated by argument signature (same md5 the tracer
  uses: ``md5(json.dumps(args, sort_keys=True, default=str))``).
- Matching configurations have their ``executions`` lists unioned; duplicate
  (source, machine_info) pairs collapse and take ``max(count)`` — the same
  rule ``update_master_file()`` applies.
- Every ``executions[].trace_uid`` is rewritten to a single fresh UUID so the
  merged file represents one artifact for the DB loader.
- Hardware profile (board_type, device_series, card_count) must be uniform
  across inputs; otherwise we refuse to merge.
- Top-level ``metadata`` is recomputed (``trace_uid``, ``models``,
  ``unique_operations``, ``total_configurations``, ``operations_summary``,
  ``last_updated``).

Usage:
    python model_tracer/homogenize_master_json.py \\
        --input path/to/dir \\
        --output merged.json

    python model_tracer/homogenize_master_json.py \\
        --input leg1.json --input leg2.json --input leg3.json \\
        --output merged.json \\
        --trace-uid <explicit-uuid>

Exit codes:
    0  Merged master JSON written to --output
    1  Input discovery, hardware-mismatch, or IO failure
"""

import argparse
import copy
import glob
import hashlib
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path


def _args_signature(args):
    """Match the md5 signature used by update_master_file() for dedup."""
    return hashlib.md5(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()


def _machine_info_signature(machine_info):
    if machine_info is None:
        return None
    return json.dumps(machine_info, sort_keys=True, default=str)


def _hardware_key(machine_info):
    """Fields that must agree across a homogenized batch."""
    if not isinstance(machine_info, dict):
        return None
    return (
        machine_info.get("board_type"),
        machine_info.get("device_series"),
        machine_info.get("card_count"),
    )


def collect_input_files(inputs):
    """Expand --input values (files, directories, globs) into a sorted file list."""
    files = []
    seen = set()
    for spec in inputs:
        path = Path(spec)
        if path.is_dir():
            candidates = sorted(path.glob("*.json"))
        elif any(ch in spec for ch in "*?["):
            candidates = sorted(Path(p) for p in glob.glob(spec))
        else:
            candidates = [path]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(candidate)
    return files


def _collect_hardware_keys(master_data):
    """Return the set of distinct hardware keys seen in a single master JSON."""
    keys = set()
    for op_data in master_data.get("operations", {}).values():
        for cfg in op_data.get("configurations", []):
            for execution in cfg.get("executions", []):
                hw = _hardware_key(execution.get("machine_info"))
                if hw is not None:
                    keys.add(hw)
    return keys


def validate_hardware_uniformity(loaded_inputs):
    """Raise ValueError if (board_type, device_series, card_count) differs.

    ``loaded_inputs`` is a list of (path, data) pairs.
    """
    observed = {}
    for path, data in loaded_inputs:
        for hw in _collect_hardware_keys(data):
            observed.setdefault(hw, []).append(str(path))
    if len(observed) > 1:
        lines = ["Hardware profile mismatch across inputs — refusing to homogenize:"]
        for hw, sources in observed.items():
            lines.append(f"  {hw}: {', '.join(sources)}")
        raise ValueError("\n".join(lines))
    return next(iter(observed), None)


def _merge_executions(acc_executions, incoming_executions, trace_uid):
    """Union incoming executions into acc_executions.

    Matches the execution-merging rules used by update_master_file():
      - Same source + same machine_info ⇒ take max(count), overwrite trace_uid
      - Otherwise ⇒ append a new execution entry
    Every resulting execution has its trace_uid rewritten to the homogenized one.
    """
    for incoming in incoming_executions:
        incoming = copy.deepcopy(incoming)
        incoming["trace_uid"] = trace_uid

        matched = None
        incoming_sig = _machine_info_signature(incoming.get("machine_info"))
        for existing in acc_executions:
            if existing.get("source") != incoming.get("source"):
                continue
            if _machine_info_signature(existing.get("machine_info")) == incoming_sig:
                matched = existing
                break

        if matched is None:
            acc_executions.append(incoming)
        else:
            matched["count"] = max(matched.get("count", 1), incoming.get("count", 1))
            matched["trace_uid"] = trace_uid


def _merge_configuration(acc_configs, incoming_cfg, trace_uid):
    """Merge one configuration into an operation's configuration list."""
    incoming_args = incoming_cfg.get("arguments", {})
    incoming_sig = _args_signature(incoming_args)

    match = None
    for existing in acc_configs:
        if _args_signature(existing.get("arguments", {})) == incoming_sig:
            match = existing
            break

    if match is None:
        new_cfg = copy.deepcopy(incoming_cfg)
        new_cfg["executions"] = []
        _merge_executions(new_cfg["executions"], incoming_cfg.get("executions", []), trace_uid)
        acc_configs.append(new_cfg)
    else:
        _merge_executions(match.setdefault("executions", []), incoming_cfg.get("executions", []), trace_uid)
        # Preserve an existing sweep_source_hash, but inherit one if only the
        # incoming side has it (symmetric with update_master_file).
        if "sweep_source_hash" not in match and "sweep_source_hash" in incoming_cfg:
            match["sweep_source_hash"] = incoming_cfg["sweep_source_hash"]


def _reassign_config_ids(merged):
    """Renumber config_id sequentially across the merged output."""
    next_id = 1
    for op_name in sorted(merged["operations"].keys()):
        for cfg in merged["operations"][op_name]["configurations"]:
            cfg["config_id"] = next_id
            next_id += 1


def _compute_metadata(merged, trace_uid, input_models):
    """Recompute top-level metadata from the merged content."""
    operations = merged["operations"]
    operations_summary = {op_name: len(op_data["configurations"]) for op_name, op_data in sorted(operations.items())}
    total_configurations = sum(operations_summary.values())
    return {
        "trace_uid": trace_uid,
        "models": sorted(set(input_models)),
        "unique_operations": len(operations),
        "total_configurations": total_configurations,
        "operations_summary": operations_summary,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def merge_master_jsons(loaded_inputs, trace_uid):
    """Merge already-loaded master JSON dicts into a single canonical dict."""
    merged = {"operations": {}, "metadata": {}}
    all_models = []

    for _path, data in loaded_inputs:
        all_models.extend(data.get("metadata", {}).get("models", []))
        for op_name, op_data in data.get("operations", {}).items():
            acc_op = merged["operations"].setdefault(op_name, {"configurations": []})
            for cfg in op_data.get("configurations", []):
                _merge_configuration(acc_op["configurations"], cfg, trace_uid)

    _reassign_config_ids(merged)
    merged["metadata"] = _compute_metadata(merged, trace_uid, all_models)
    return merged


def load_master_jsons(files):
    loaded = []
    for path in files:
        with open(path, "r") as f:
            loaded.append((path, json.load(f)))
    return loaded


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Merge multiple master JSON files into one homogenized trace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        action="append",
        required=True,
        help="Input master JSON file, directory of *.json, or glob. May be passed multiple times.",
    )
    parser.add_argument("--output", "-o", required=True, help="Path to write the homogenized master JSON.")
    parser.add_argument(
        "--trace-uid",
        default=None,
        help="Explicit trace_uid to stamp on every execution (default: fresh UUID4).",
    )
    args = parser.parse_args(argv)

    files = collect_input_files(args.input)
    if not files:
        print("❌ No input master JSON files found.", file=sys.stderr)
        return 1

    print(f"📂 Homogenizing {len(files)} master JSON file(s):")
    for f in files:
        print(f"   - {f}")

    loaded = load_master_jsons(files)

    try:
        hw = validate_hardware_uniformity(loaded)
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1

    trace_uid = args.trace_uid or str(uuid.uuid4())
    print(f"🔖 Homogenized trace_uid: {trace_uid}")
    if hw is not None:
        print(f"🖥️  Hardware: board_type={hw[0]}, device_series={hw[1]}, card_count={hw[2]}")

    merged = merge_master_jsons(loaded, trace_uid)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2, sort_keys=True, default=str)

    meta = merged["metadata"]
    print(
        f"✅ Wrote {output_path}: {meta['unique_operations']} unique ops, "
        f"{meta['total_configurations']} configurations, {len(meta['models'])} model source(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
