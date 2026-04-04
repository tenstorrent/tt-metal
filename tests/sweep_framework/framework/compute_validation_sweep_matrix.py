#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Compute the GitHub Actions matrix for model trace sweep validation.

Validation keeps hardware-grouped vector generation for now, but routing must
still flow through the shared CI ownership helpers in ``matrix_runner_config``.
This script converts generated vector files into matrix rows that use real
logical ``test_group_name`` values while preserving validation-specific context
as metadata.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from compute_sweep_matrix import chunk_modules
from constants import format_hardware_suffix, parse_hardware_suffix, strip_grouping_suffix
from matrix_runner_config import GENERATION_MANIFEST_FILENAME, get_runner_config, get_test_group_name_for_hardware_group


def _load_generation_manifest(vectors_path: Path) -> dict:
    """Load generation manifest metadata if present."""
    manifest_path = vectors_path / GENERATION_MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}

    try:
        with open(manifest_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Failed to read generation manifest at {manifest_path}: {error}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, dict):
        print(f"Generation manifest at {manifest_path} is not a JSON object", file=sys.stderr)
        sys.exit(1)

    return data


def _modules_from_manifest_or_dir(vectors_path: Path, manifest: dict) -> list[str]:
    """Resolve vector file stems, preferring the manifest as the source of truth."""
    manifest_files = manifest.get("vector_files")
    if isinstance(manifest_files, list):
        files = [name for name in manifest_files if isinstance(name, str) and name.endswith(".json")]
        modules = sorted({Path(name).stem for name in files if Path(name).name != GENERATION_MANIFEST_FILENAME})
        if modules:
            return modules

    return sorted([path.stem for path in vectors_path.glob("*.json") if path.name != GENERATION_MANIFEST_FILENAME])


def _normalize_hardware_group(board_type, device_series, card_count):
    """Normalize manifest hardware metadata into the same tuple format as filename parsing."""
    synthetic_module_name = f"validation{format_hardware_suffix(board_type, device_series, card_count)}"
    return parse_hardware_suffix(synthetic_module_name)


def _get_hardware_display_label(hardware_group) -> str:
    """Return a stable display label for metadata and batch descriptions."""
    if hardware_group is None:
        return "default"
    board_type, device_series, card_count = hardware_group
    return f"{board_type}_{device_series}_{card_count}c"


def _get_trace_ids_by_hardware(trace_ids: list[int], registry: dict) -> dict:
    """Group trace IDs by the normalized hardware tuple declared in the workflow manifest."""
    trace_ids_by_hardware = defaultdict(list)

    for trace_id in trace_ids:
        if trace_id not in registry:
            raise KeyError(f"Trace {trace_id} not found in manifest registry")

        hardware = registry[trace_id].get("hardware") or {}
        hardware_group = _normalize_hardware_group(
            hardware.get("board_type"),
            hardware.get("device_series"),
            hardware.get("card_count"),
        )
        trace_ids_by_hardware[hardware_group].append(trace_id)

    return trace_ids_by_hardware


def compute_validation_matrix(
    manifest_path: Path,
    master_json_path: Path,
    vectors_dir: Path,
    validation_scope: str,
    batch_size: int = 10,
) -> dict:
    """Build matrix entries for validation sweeps using shared routing helpers."""
    with open(master_json_path, "r", encoding="utf-8") as file:
        master_json = json.load(file)

    trace_ids = master_json.get("metadata", {}).get("trace_run_ids", [])
    if not trace_ids:
        print("No trace_run_ids found in reconstructed master JSON", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = yaml.safe_load(file) or {}

    registry = {entry["trace_id"]: entry for entry in manifest.get("registry", []) if entry.get("trace_id") is not None}

    generation_manifest = _load_generation_manifest(vectors_dir)
    grouping_mode = generation_manifest.get("vector_grouping_mode")
    if grouping_mode and grouping_mode != "hw":
        print(
            f"Validation expects hardware-grouped vectors, but generation manifest declares '{grouping_mode}'",
            file=sys.stderr,
        )
        sys.exit(1)

    vector_modules = _modules_from_manifest_or_dir(vectors_dir, generation_manifest)
    if not vector_modules:
        print(f"No vector JSON files found in {vectors_dir}", file=sys.stderr)
        sys.exit(1)

    trace_ids_by_hardware = _get_trace_ids_by_hardware(trace_ids, registry)

    hardware_modules = defaultdict(list)
    unmatched_modules = []
    for module in vector_modules:
        hardware_group = parse_hardware_suffix(module)
        if hardware_group is None:
            unmatched_modules.append(module)
        else:
            hardware_modules[hardware_group].append(module)

    include = []
    grouped_items = sorted(hardware_modules.items(), key=lambda item: item[0])
    if unmatched_modules:
        grouped_items.append((None, unmatched_modules))

    for hardware_group, grouped_modules in grouped_items:
        base_modules = sorted({strip_grouping_suffix(module) for module in grouped_modules})
        if not base_modules:
            continue

        hardware_label = _get_hardware_display_label(hardware_group)
        test_group_name = get_test_group_name_for_hardware_group(hardware_group)
        runner_config = get_runner_config(test_group_name)
        runner_batches = chunk_modules(base_modules, batch_size)
        total_batches = len(runner_batches)
        trace_id_list = sorted(trace_ids_by_hardware.get(hardware_group, []))

        for index, batch in enumerate(runner_batches, start=1):
            include.append(
                {
                    **runner_config,
                    "batch_display": f"{validation_scope}:{hardware_label}:{index}/{total_batches}",
                    "module_selector": batch,
                    "suite_name": "model_traced",
                    "validation_scope": validation_scope,
                    "vectors_artifact_name": f"sweeps-vectors-{validation_scope}",
                    "trace_ids": trace_id_list,
                    "hardware_group": hardware_label,
                }
            )

    return {"include": include}


def compute_combined_validation_matrix(
    manifest_path: Path,
    master_json_path: Path,
    vectors_root: Path,
    scope_target: str,
    batch_size: int = 10,
) -> dict:
    """Build a combined matrix for one or both validation scopes."""
    if scope_target == "all":
        scopes = ["model_traced", "lead_models"]
    elif scope_target in {"model_traced", "lead_models"}:
        scopes = [scope_target]
    else:
        print(f"Unsupported validation scope target: {scope_target}", file=sys.stderr)
        sys.exit(1)

    include = []
    for scope in scopes:
        vectors_dir = vectors_root / scope
        if not vectors_dir.exists():
            print(f"Missing vectors directory for scope '{scope}': {vectors_dir}", file=sys.stderr)
            sys.exit(1)

        scope_matrix = compute_validation_matrix(
            manifest_path=manifest_path,
            master_json_path=master_json_path,
            vectors_dir=vectors_dir,
            validation_scope=scope,
            batch_size=batch_size,
        )
        include.extend(scope_matrix.get("include", []))

    return {"include": include}


def main():
    """Parse arguments and print the matrix JSON to stdout."""
    parser = argparse.ArgumentParser(description="Compute model trace validation sweep matrix.")
    parser.add_argument("--manifest-path", required=True, help="Path to model_tracer/sweep_manifest.yaml")
    parser.add_argument("--master-json-path", required=True, help="Path to reconstructed ttnn_operations_master.json")
    parser.add_argument("--vectors-dir", required=False, help="Directory containing generated vector JSON files")
    parser.add_argument(
        "--vectors-root",
        required=False,
        help="Root directory containing per-scope validation vectors directories",
    )
    parser.add_argument(
        "--validation-scope",
        required=False,
        choices=["model_traced", "lead_models"],
        help="Validation scope represented by the supplied vectors directory",
    )
    parser.add_argument(
        "--scope-target",
        required=False,
        choices=["model_traced", "lead_models", "all"],
        help="Top-level validation scope target used to combine one or both per-scope matrices",
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Maximum number of modules per matrix batch")
    args = parser.parse_args()

    if args.vectors_root and args.scope_target:
        matrix = compute_combined_validation_matrix(
            manifest_path=Path(args.manifest_path),
            master_json_path=Path(args.master_json_path),
            vectors_root=Path(args.vectors_root),
            scope_target=args.scope_target,
            batch_size=args.batch_size,
        )
    elif args.vectors_dir and args.validation_scope:
        matrix = compute_validation_matrix(
            manifest_path=Path(args.manifest_path),
            master_json_path=Path(args.master_json_path),
            vectors_dir=Path(args.vectors_dir),
            validation_scope=args.validation_scope,
            batch_size=args.batch_size,
        )
    else:
        parser.error("Provide either --vectors-dir with --validation-scope, or --vectors-root with --scope-target.")

    print(json.dumps(matrix, separators=(",", ":")))


if __name__ == "__main__":
    main()
