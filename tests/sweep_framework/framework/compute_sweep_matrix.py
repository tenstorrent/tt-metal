#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Compute the sweep test matrix for GitHub Actions CI.

This script analyzes generated sweep vector files and produces a matrix
configuration that maps test modules to appropriate hardware runners.

The matrix output includes:
- Module batching to parallelize execution
- Runner assignment based on mesh topology (for lead models)
- Suite name configuration per run type

Environment Variables (from GitHub Actions context):
- GITHUB_EVENT_SCHEDULE: Cron schedule expression
- GITHUB_EVENT_NAME: Event type (schedule, workflow_dispatch)
- SWEEP_NAME: Selected sweep type from workflow_dispatch
- MEASURE_DEVICE_PERF: Whether device performance measurement is enabled
- VECTORS_DIR: Directory containing vector JSON files (default: /tmp/vectors)

Output:
Prints JSON matrix to stdout in format required by GitHub Actions.
"""

import argparse
import os
import json
import sys
from collections import defaultdict
from pathlib import Path

# Import grouping utilities from shared constants module
from constants import format_hardware_suffix, get_mesh_shape_string, parse_hardware_suffix, strip_grouping_suffix


def chunk_module_lists(items, size):
    """Split modules into batches while preserving list structure."""
    return [items[i : i + size] for i in range(0, len(items), size)] if items else []


def chunk_modules(items, size):
    """Split modules into batches of specified size."""
    return [",".join(batch) for batch in chunk_module_lists(items, size)]


# Alias for backward compatibility and clearer naming in this context
def get_mesh_shape(module_name):
    """Extract mesh shape string from module name (wrapper for get_mesh_shape_string)."""
    return get_mesh_shape_string(module_name)


def get_hardware_group(module_name):
    """Extract normalized hardware group from module name suffix."""
    return parse_hardware_suffix(module_name)


def get_runner_config_for_hardware_group(hardware_group):
    """Map a normalized hardware tuple to CI runner metadata."""
    if hardware_group is None:
        return {
            "test_group_name": "wormhole-n150-sweeps",
            "arch": "wormhole_b0",
            "runs_on": "tt-ubuntu-2204-n150-stable",
            "runner_label": "N150",
            "tt_smi_cmd": "tt-smi -r",
        }

    board_type, device_series, card_count = hardware_group

    if board_type == "blackhole" or device_series == "p150b":
        return {
            "test_group_name": "blackhole-p150b-sweeps",
            "arch": "blackhole",
            "runs_on": "tt-ubuntu-2204-p150b-viommu-stable",
            "runner_label": "p150b",
            "tt_smi_cmd": "tt-smi -r",
        }

    if device_series == "tt_galaxy_wh":
        return {
            "test_group_name": "wormhole-galaxy-sweeps",
            "arch": "wormhole_b0",
            "runs_on": ["topology-6u", "in-service", "bare-metal"],
            "runner_label": "topology-6u",
            "tt_smi_cmd": "tt-smi -glx_reset_auto",
        }

    if device_series == "n300" and card_count == 4:
        return {
            "test_group_name": "wormhole-n300-llmbox-sweeps",
            "arch": "wormhole_b0",
            "runs_on": "tt-ubuntu-2204-n300-llmbox-stable",
            "runner_label": "n300-llmbox",
            "tt_smi_cmd": "tt-smi -r",
        }

    if device_series == "n300":
        return {
            "test_group_name": "wormhole-n300-sweeps",
            "arch": "wormhole_b0",
            "runs_on": "tt-ubuntu-2204-n300-stable",
            "runner_label": "N300",
            "tt_smi_cmd": "tt-smi -r",
        }

    return {
        "test_group_name": "wormhole-n150-sweeps",
        "arch": "wormhole_b0",
        "runs_on": "tt-ubuntu-2204-n150-stable",
        "runner_label": "N150",
        "tt_smi_cmd": "tt-smi -r",
    }


def _format_hardware_group_label(hardware_group):
    """Build a readable label for logs and batch display."""
    if hardware_group is None:
        return "default"

    board_type, device_series, card_count = hardware_group
    return f"{board_type}/{device_series}/{card_count}c"


def _format_hardware_group_slug(hardware_group):
    """Build a stable hardware token for matrix metadata and naming."""
    if hardware_group is None:
        return "default"

    return format_hardware_suffix(*hardware_group).replace(".hw_", "", 1)


def _normalize_trace_ids(value):
    """Normalize trace IDs to a sorted unique list."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]

    trace_ids = set()
    for item in items:
        if item is None:
            continue
        try:
            trace_ids.add(int(item))
        except (TypeError, ValueError):
            continue

    return sorted(trace_ids)


def _load_vector_metadata(vectors_path):
    """Load vector metadata needed for matrix enrichment."""
    modules = []
    metadata = {}

    for vector_file in sorted(vectors_path.glob("*.json")):
        module_name = vector_file.stem
        modules.append(module_name)
        trace_ids = set()

        try:
            with open(vector_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Error: Failed to read vector JSON {vector_file}: {exc}", file=sys.stderr)
            sys.exit(1)

        if isinstance(data, dict):
            for suite_data in data.values():
                if not isinstance(suite_data, dict):
                    continue
                for vector in suite_data.values():
                    if not isinstance(vector, dict):
                        continue
                    trace_ids.update(_normalize_trace_ids(vector.get("trace_ids")))

        metadata[module_name] = {"trace_ids": sorted(trace_ids)}

    return modules, metadata


def _build_source_modules_by_base(modules):
    """Map base module names to the grouped vector files that back them."""
    source_modules = defaultdict(list)
    for module in modules:
        source_modules[strip_grouping_suffix(module)].append(module)
    return source_modules


def _collect_source_modules(batch_modules, source_modules_by_base):
    """Collect grouped vector file names for a batch of base module names."""
    source_modules = []
    seen = set()
    for base_module in batch_modules:
        for module in source_modules_by_base.get(base_module, [base_module]):
            if module in seen:
                continue
            seen.add(module)
            source_modules.append(module)
    return source_modules


def _aggregate_trace_ids(source_modules, vector_metadata_by_module):
    """Aggregate trace IDs for the vector files included in a matrix row."""
    trace_ids = set()
    for module in source_modules:
        trace_ids.update(vector_metadata_by_module.get(module, {}).get("trace_ids", []))
    return sorted(trace_ids)


def _infer_common_hardware_group(source_modules):
    """Infer a single hardware group when all grouped source modules agree."""
    hardware_groups = set()
    for module in source_modules:
        hardware_group = get_hardware_group(module)
        if hardware_group is not None:
            hardware_groups.add(hardware_group)
    if len(hardware_groups) == 1:
        return next(iter(hardware_groups))
    return None


def _build_matrix_entry(
    runner_config,
    batch_modules,
    source_modules,
    suite_name,
    batch_display,
    vector_metadata_by_module,
    hardware_group=None,
    test_group_name=None,
):
    """Construct one enriched matrix row."""
    resolved_hardware_group = (
        hardware_group if hardware_group is not None else _infer_common_hardware_group(source_modules)
    )
    return {
        **runner_config,
        "test_group_name": test_group_name or runner_config["test_group_name"],
        "module_selector": ",".join(batch_modules),
        "batch_display": batch_display,
        "suite_name": suite_name,
        "hardware_group": _format_hardware_group_slug(resolved_hardware_group),
        "trace_ids": _aggregate_trace_ids(source_modules, vector_metadata_by_module),
    }


def get_lead_models_mesh_runner_config():
    """
    Configuration: Map mesh shapes to runner configurations.

    Each entry specifies which runner handles which mesh shapes.
    Multiple mesh shapes can be handled by the same runner.

    To add new mesh configurations or runners, add entries to this list.

    Note: 'runs_on' can be either:
      - A string: Single runner label (e.g., "tt-ubuntu-2204-n150-stable")
      - A list: Multiple runner labels for GitHub Actions matrix
                (e.g., ["topology-6u", "arch-wormhole_b0", "in-service", "pipeline-functional"])
    """
    return [
        {
            # Single-chip operations (1x1 mesh) - runs on N150
            "mesh_shapes": ["1x1"],
            "test_group_name": "lead-models-single-chip",
            "arch": "wormhole_b0",
            "runs_on": "tt-ubuntu-2204-n150-stable",
            "runner_label": "N150",
            "tt_smi_cmd": "tt-smi -r",
            "suite_name": "model_traced",
        },
        {
            # Multi-chip operations - pinned to g04glx03 Galaxy runner
            "mesh_shapes": ["1x2", "1x4", "1x8", "2x4", "4x8", "8x4", "2x16", "16x2"],
            "test_group_name": "lead-models-galaxy",
            "arch": "wormhole_b0",
            "runs_on": "g04glx03",
            "runner_label": "g04glx03",
            "tt_smi_cmd": "tt-smi -r",
            "suite_name": "model_traced",
        },
    ]


def compute_lead_models_matrix(modules, batch_size, vector_metadata_by_module):
    """
    Compute matrix for lead models run with mesh-aware runner assignment.

    Args:
        modules: List of module names (from vector JSON filenames)
        batch_size: Number of modules per batch

    Returns:
        Tuple of (include_entries, batches, ccl_batches)
    """
    config = get_lead_models_mesh_runner_config()

    # Group modules by mesh shape, with hardware-based fallback when mesh suffixes are absent.
    mesh_shape_modules = defaultdict(list)
    hardware_modules = defaultdict(list)
    unmatched_modules = []

    for module in modules:
        mesh_shape = get_mesh_shape(module)
        if mesh_shape:
            mesh_shape_modules[mesh_shape].append(module)
            continue

        hardware_group = get_hardware_group(module)
        if hardware_group:
            hardware_modules[hardware_group].append(module)
        else:
            unmatched_modules.append(module)

    # Build set of all configured mesh shapes for validation
    configured_shapes = set()
    for runner_config in config:
        configured_shapes.update(runner_config["mesh_shapes"])

    # Warn about mesh shapes without runner config
    for mesh_shape in mesh_shape_modules.keys():
        if mesh_shape not in configured_shapes:
            print(
                f"Warning: Mesh shape '{mesh_shape}' has no runner config, " f"modules will be skipped", file=sys.stderr
            )

    # Create matrix entries based on runner config
    include_entries = []
    batches = []

    for runner_config in config:
        # Collect all modules for this runner's mesh shapes
        runner_modules = []
        for mesh_shape in runner_config["mesh_shapes"]:
            runner_modules.extend(mesh_shape_modules.get(mesh_shape, []))

        # Hardware-grouped files are already routed to the correct runner family.
        for hardware_group, mods in hardware_modules.items():
            _, device_series, card_count = hardware_group
            wants_galaxy = device_series == "tt_galaxy_wh" or card_count > 1
            is_galaxy_runner = runner_config["test_group_name"] == "lead-models-galaxy"
            if wants_galaxy == is_galaxy_runner:
                runner_modules.extend(mods)

        # Route modules without a grouping suffix to the first (default) runner config,
        # which is conventionally the single-chip N150 runner.
        is_default_runner = runner_config == config[0]
        if is_default_runner:
            runner_modules.extend(unmatched_modules)

        if not runner_modules:
            continue

        # Strip grouping suffixes to get base module names that sweeps_runner can find.
        # The VectorExportSource will automatically load grouped JSON variants.
        base_modules = sorted(set(strip_grouping_suffix(m) for m in runner_modules))
        source_modules_by_base = _build_source_modules_by_base(runner_modules)

        # For Galaxy runners (multi-chip), split into 3 parallel jobs
        # For single-chip runners, use the standard batch size
        is_galaxy = runner_config["test_group_name"] == "lead-models-galaxy"
        if is_galaxy:
            galaxy_jobs = 3
            galaxy_batch_size = max(1, -(-len(base_modules) // galaxy_jobs))
            runner_batch_lists = chunk_module_lists(base_modules, galaxy_batch_size)
        else:
            # Standard batching for single-chip
            runner_batch_lists = chunk_module_lists(base_modules, batch_size)

        runner_batches = [",".join(batch) for batch in runner_batch_lists]
        batches.extend(runner_batches)

        # Create matrix entries
        mesh_label = "+".join(runner_config["mesh_shapes"])
        for batch_modules in runner_batch_lists:
            source_modules = _collect_source_modules(batch_modules, source_modules_by_base)
            include_entries.append(
                _build_matrix_entry(
                    runner_config,
                    batch_modules,
                    source_modules,
                    runner_config["suite_name"],
                    f"{mesh_label}:{','.join(batch_modules)}",
                    vector_metadata_by_module,
                )
            )

    # Log summary
    total_base_modules = len(set(strip_grouping_suffix(m) for m in modules))
    print(
        f"Lead models run: {len(modules)} vector files ({total_base_modules} unique modules), "
        f"{len(include_entries)} matrix entries",
        file=sys.stderr,
    )
    for mesh_shape, mods in sorted(mesh_shape_modules.items()):
        unique_base = len(set(strip_grouping_suffix(m) for m in mods))
        print(f"  mesh {mesh_shape}: {len(mods)} vectors ({unique_base} unique modules)", file=sys.stderr)
    for hardware_group, mods in sorted(hardware_modules.items()):
        unique_base = len(set(strip_grouping_suffix(m) for m in mods))
        board_type, device_series, card_count = hardware_group
        print(
            f"  hardware {board_type}/{device_series}/{card_count}c: {len(mods)} vectors ({unique_base} unique modules)",
            file=sys.stderr,
        )
    if unmatched_modules:
        unique_base = len(set(strip_grouping_suffix(m) for m in unmatched_modules))
        print(
            f"  no grouping suffix (default runner): {len(unmatched_modules)} vectors ({unique_base} unique modules)",
            file=sys.stderr,
        )

    return include_entries, batches, []  # No CCL batches for lead models


def compute_standard_matrix(modules, batch_size, suite_name, vector_metadata_by_module):
    """
    Compute matrix for standard runs (nightly, comprehensive, model_traced).

    Args:
        modules: List of module names (may include mesh suffixes from JSON filenames)
        batch_size: Number of modules per batch
        suite_name: Suite name override (None means no override)

    Returns:
        Tuple of (include_entries, batches, ccl_batches)
    """
    if suite_name == "model_traced":
        return compute_model_traced_hardware_matrix(modules, batch_size, suite_name, vector_metadata_by_module)

    # Strip grouping suffixes to get base module names that sweeps_runner can find.
    # The VectorExportSource will automatically load grouped JSON variants.
    base_modules = sorted(set(strip_grouping_suffix(m) for m in modules))

    ccl_modules = [m for m in base_modules if m.startswith("ccl.")]

    # Create batches for all modules using base names
    regular_batch_lists = chunk_module_lists(base_modules, batch_size)
    regular_batches = [",".join(batch) for batch in regular_batch_lists]
    batches = list(regular_batches)

    # Generate CCL-only batches for dedicated runners
    ccl_batch_lists = chunk_module_lists(ccl_modules, batch_size)
    ccl_batches = [",".join(batch) for batch in ccl_batch_lists]

    include_entries = []

    # Standard wormhole runner entries
    wormhole_template = {
        "test_group_name": "wormhole-n150-sweeps",
        "arch": "wormhole_b0",
        "runs_on": "tt-ubuntu-2204-n150-stable",
        "runner_label": "N150",
        "tt_smi_cmd": "tt-smi -r",
    }

    for batch_modules in regular_batch_lists:
        include_entries.append(
            _build_matrix_entry(
                wormhole_template,
                batch_modules,
                batch_modules,
                suite_name,
                ",".join(batch_modules),
                vector_metadata_by_module,
            )
        )

    # CCL-specific entries for N300 runners
    if ccl_batches:
        n300_template = {
            "test_group_name": "n300-llmbox-ccl",
            "arch": "wormhole_b0",
            "runs_on": "tt-ubuntu-2204-n300-llmbox-viommu-stable",
            "runner_label": "n300-llmbox",
            "tt_smi_cmd": "tt-smi -r",
        }
        for batch_modules in ccl_batch_lists:
            include_entries.append(
                _build_matrix_entry(
                    n300_template,
                    batch_modules,
                    batch_modules,
                    "generality_suite_fabric_1d",
                    f"ccl:{','.join(batch_modules)}",
                    vector_metadata_by_module,
                )
            )

    return include_entries, batches, ccl_batches


def compute_model_traced_hardware_matrix(modules, batch_size, suite_name, vector_metadata_by_module):
    """Compute matrix for model_traced runs using hardware-grouped vector files."""
    hardware_modules = defaultdict(list)
    unmatched_modules = []

    for module in modules:
        hardware_group = get_hardware_group(module)
        if hardware_group:
            hardware_modules[hardware_group].append(module)
        else:
            unmatched_modules.append(module)

    include_entries = []
    batches = []

    grouped_items = sorted(hardware_modules.items(), key=lambda item: item[0])
    if unmatched_modules:
        grouped_items.append((None, unmatched_modules))

    for hardware_group, grouped_modules in grouped_items:
        base_modules = sorted(set(strip_grouping_suffix(m) for m in grouped_modules))
        runner_config = get_runner_config_for_hardware_group(hardware_group)
        hardware_label = _format_hardware_group_label(hardware_group)
        source_modules_by_base = _build_source_modules_by_base(grouped_modules)
        runner_batch_lists = chunk_module_lists(base_modules, batch_size)
        runner_batches = [",".join(batch) for batch in runner_batch_lists]
        batches.extend(runner_batches)

        for batch_modules in runner_batch_lists:
            source_modules = _collect_source_modules(batch_modules, source_modules_by_base)
            include_entries.append(
                _build_matrix_entry(
                    runner_config,
                    batch_modules,
                    source_modules,
                    suite_name,
                    f"{hardware_label}:{','.join(batch_modules)}",
                    vector_metadata_by_module,
                    hardware_group=hardware_group,
                )
            )

    total_base_modules = len(set(strip_grouping_suffix(m) for m in modules))
    print(
        f"Model traced run: {len(modules)} vector files ({total_base_modules} unique modules), "
        f"{len(include_entries)} matrix entries",
        file=sys.stderr,
    )
    for hardware_group, grouped_modules in grouped_items:
        unique_base = len(set(strip_grouping_suffix(m) for m in grouped_modules))
        print(
            f"  hardware {_format_hardware_group_label(hardware_group)}: "
            f"{len(grouped_modules)} vectors ({unique_base} unique modules)",
            file=sys.stderr,
        )

    return include_entries, batches, []


def compute_validation_matrix(modules, suite_name, vector_metadata_by_module):
    """Compute validation matrix with one row per hardware group."""
    hardware_modules = defaultdict(list)
    unmatched_modules = []

    for module in modules:
        hardware_group = get_hardware_group(module)
        if hardware_group:
            hardware_modules[hardware_group].append(module)
        else:
            unmatched_modules.append(module)

    include_entries = []
    batches = []

    grouped_items = sorted(hardware_modules.items(), key=lambda item: item[0])
    if unmatched_modules:
        grouped_items.append((None, unmatched_modules))

    for hardware_group, grouped_modules in grouped_items:
        base_modules = sorted(set(strip_grouping_suffix(m) for m in grouped_modules))
        if not base_modules:
            continue

        batches.append(",".join(base_modules))
        runner_config = get_runner_config_for_hardware_group(hardware_group)
        hardware_label = _format_hardware_group_label(hardware_group)
        hardware_slug = _format_hardware_group_slug(hardware_group)
        include_entries.append(
            _build_matrix_entry(
                runner_config,
                base_modules,
                grouped_modules,
                suite_name,
                hardware_label,
                vector_metadata_by_module,
                hardware_group=hardware_group,
                test_group_name=f"validation-{hardware_slug}",
            )
        )

    total_base_modules = len(set(strip_grouping_suffix(m) for m in modules))
    print(
        f"Validation run: {len(modules)} vector files ({total_base_modules} unique modules), "
        f"{len(include_entries)} matrix entries",
        file=sys.stderr,
    )
    for hardware_group, grouped_modules in grouped_items:
        unique_base = len(set(strip_grouping_suffix(m) for m in grouped_modules))
        print(
            f"  hardware {_format_hardware_group_label(hardware_group)}: "
            f"{len(grouped_modules)} vectors ({unique_base} unique modules)",
            file=sys.stderr,
        )

    return include_entries, batches, []


def main():
    """Main entry point for matrix computation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["auto", "standard", "validation"],
        default=os.environ.get("MATRIX_PROFILE", "auto"),
        help="Matrix scheduling profile. Defaults to MATRIX_PROFILE or auto-detection.",
    )
    parser.add_argument(
        "--vectors-dir",
        default=os.environ.get("VECTORS_DIR", "/tmp/vectors"),
        help="Directory containing vector JSON files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Optional batch size override for standard scheduling modes.",
    )
    args = parser.parse_args()

    # Read environment variables from GitHub Actions context
    schedule_expr = os.environ.get("GITHUB_EVENT_SCHEDULE", "")
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    sweep_name = os.environ.get("SWEEP_NAME", "")
    measure_device_perf = os.environ.get("MEASURE_DEVICE_PERF", "false")
    vectors_dir = args.vectors_dir

    # Read vector files from directory
    vectors_path = Path(vectors_dir)
    if not vectors_path.exists():
        print(f"Error: Vectors directory not found: {vectors_dir}", file=sys.stderr)
        sys.exit(1)

    modules, vector_metadata_by_module = _load_vector_metadata(vectors_path)

    if not modules:
        print(f"Error: No vector JSON files found in {vectors_dir}", file=sys.stderr)
        sys.exit(1)

    # Detect run type (mutually exclusive with explicit precedence)
    is_comprehensive = False
    is_model_traced = False
    is_lead_models = False

    # Prefer explicit sweep selection over schedule-based detection
    if sweep_name == "ALL SWEEPS (Lead Models)":
        is_lead_models = True
    elif sweep_name == "ALL SWEEPS (Model Traced)":
        is_model_traced = True
    elif sweep_name == "ALL SWEEPS (Comprehensive)":
        is_comprehensive = True
    else:
        # Fallback to schedule-based detection when no explicit sweep is selected
        # Schedule expressions must match ttnn-run-sweeps.yaml:
        #   - "0 2 * * *"  -> lead models (2 AM daily)
        #   - "0 3 * * *"  -> model traced (3 AM daily)
        #   - "0 4 * * 3,6" -> comprehensive (4 AM Wed/Sat)
        #   - "30 4 * * 0,1,2,4,5" -> nightly (falls through to else)
        if schedule_expr == "0 2 * * *":
            is_lead_models = True
        elif schedule_expr == "0 3 * * *":
            is_model_traced = True
        elif schedule_expr == "0 4 * * 3,6":
            is_comprehensive = True

    # Determine batch size
    # Use smaller batch size for comprehensive runs or when device performance measurement is enabled
    # (must stay under 256 total batches due to GitHub Actions limit)
    device_perf_enabled = event_name == "schedule" or measure_device_perf == "true"
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif is_comprehensive or device_perf_enabled:
        batch_size = 3
    else:
        batch_size = 10

    # Compute matrix based on run type
    if args.profile == "validation":
        include_entries, batches, ccl_batches = compute_validation_matrix(
            modules,
            "model_traced",
            vector_metadata_by_module,
        )
    elif is_lead_models:
        include_entries, batches, ccl_batches = compute_lead_models_matrix(
            modules, batch_size, vector_metadata_by_module
        )
    else:
        # Determine suite name for standard runs
        if is_model_traced:
            suite_name = "model_traced"
        elif is_comprehensive:
            suite_name = None
        else:
            suite_name = "nightly"

        include_entries, batches, ccl_batches = compute_standard_matrix(
            modules,
            batch_size,
            suite_name,
            vector_metadata_by_module,
        )

    # Validation
    total_batches = len(batches)
    if total_batches > 256:
        print(
            f"Total batch count ({total_batches}) exceeds GitHub Actions matrix limit of 256. "
            f"Please adjust BATCH_SIZE or reduce the number of modules.",
            file=sys.stderr,
        )
        sys.exit(1)

    total_matrix_entries = len(include_entries)
    if total_matrix_entries > 256:
        print(
            f"Total matrix entry count ({total_matrix_entries}) exceeds GitHub Actions matrix "
            f"limit of 256. Please adjust batching logic or reduce modules.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Output matrix JSON
    result = {
        "module": modules,
        "batches": batches,
        "ccl_batches": ccl_batches,
        "include": include_entries,
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
