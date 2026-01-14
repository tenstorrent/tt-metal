#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

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

import os
import json
import sys
from collections import defaultdict
from pathlib import Path


def chunk_modules(items, size):
    """Split modules into batches of specified size."""
    return [",".join(items[i : i + size]) for i in range(0, len(items), size)] if items else []


def get_mesh_shape(module_name):
    """Extract mesh shape from module name (e.g., 'op__mesh_2x4' -> '2x4')."""
    if "__mesh_" in module_name:
        return module_name.split("__mesh_")[1]
    return None


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
            # Single-chip operations (1x1 mesh)
            "mesh_shapes": ["1x1"],
            "test_group_name": "lead-models-single-chip",
            "arch": "wormhole_b0",
            "runs_on": "tt-ubuntu-2204-n150-stable",
            "tt_smi_cmd": "tt-smi -r",
            "suite_name": "model_traced",
        },
        {
            # Small multi-chip operations (N300 configurations: 2-8 chips)
            # N300 llmbox supports up to 8 chips (1x4, 1x8, 2x4)
            "mesh_shapes": [],
            "test_group_name": "lead-models-n300",
            "arch": "wormhole_b0",
            "runs_on": "tt-ubuntu-2204-n300-llmbox-viommu-stable",
            "tt_smi_cmd": "tt-smi -r",
            "suite_name": "model_traced",
        },
        {
            # Large multi-chip operations (Galaxy TG: 32 chips)
            # Galaxy supports 4x8 and similar large mesh configurations
            "mesh_shapes": ["1x2", "1x4", "1x8", "2x4", "4x8", "8x4", "2x16", "16x2"],
            "test_group_name": "lead-models-galaxy",
            "arch": "wormhole_b0",
            "runs_on": [
                "topology-6u",  # 32-chip galaxy topology
                "arch-wormhole_b0",  # Wormhole B0 architecture
                "in-service",  # Available for use
                "pipeline-functional",  # Functional pipeline (use pipeline-model for model-specific)
            ],
            "tt_smi_cmd": "tt-smi -r",
            "suite_name": "model_traced",
        },
    ]


def compute_lead_models_matrix(modules, batch_size):
    """
    Compute matrix for lead models run with mesh-aware runner assignment.

    Args:
        modules: List of module names (from vector JSON filenames)
        batch_size: Number of modules per batch

    Returns:
        Tuple of (include_entries, batches, ccl_batches)
    """
    config = get_lead_models_mesh_runner_config()

    # Group modules by mesh shape
    mesh_shape_modules = defaultdict(list)
    unmatched_modules = []

    for module in modules:
        mesh_shape = get_mesh_shape(module)
        if mesh_shape:
            mesh_shape_modules[mesh_shape].append(module)
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

        if not runner_modules:
            continue

        # Create batches for this runner
        runner_batches = chunk_modules(sorted(runner_modules), batch_size)
        batches.extend(runner_batches)

        # Create matrix entries
        mesh_label = "+".join(runner_config["mesh_shapes"])
        for batch in runner_batches:
            include_entries.append(
                {
                    "test_group_name": runner_config["test_group_name"],
                    "arch": runner_config["arch"],
                    "runs_on": runner_config["runs_on"],
                    "tt_smi_cmd": runner_config["tt_smi_cmd"],
                    "module_selector": batch,
                    "batch_display": f"{mesh_label}:{batch}",
                    "suite_name": runner_config["suite_name"],
                }
            )

    # Log summary
    print(f"Lead models run: {len(modules)} modules, {len(include_entries)} matrix entries", file=sys.stderr)
    for mesh_shape, mods in sorted(mesh_shape_modules.items()):
        print(f"  mesh {mesh_shape}: {len(mods)} modules", file=sys.stderr)

    return include_entries, batches, []  # No CCL batches for lead models


def compute_standard_matrix(modules, batch_size, suite_name):
    """
    Compute matrix for standard runs (nightly, comprehensive, model_traced).

    Args:
        modules: List of module names
        batch_size: Number of modules per batch
        suite_name: Suite name override (None means no override)

    Returns:
        Tuple of (include_entries, batches, ccl_batches)
    """
    ccl_modules = [m for m in modules if m.startswith("ccl.")]

    # Create batches for all modules
    regular_batches = chunk_modules(modules, batch_size)
    batches = list(regular_batches)

    # Generate CCL-only batches for dedicated runners
    ccl_batches = chunk_modules(ccl_modules, batch_size)

    include_entries = []

    # Standard wormhole runner entries
    wormhole_template = {
        "test_group_name": "wormhole-n150-sweeps",
        "arch": "wormhole_b0",
        "runs_on": "tt-ubuntu-2204-n150-stable",
        "tt_smi_cmd": "tt-smi -r",
    }

    for batch in regular_batches:
        include_entries.append(
            {
                **wormhole_template,
                "module_selector": batch,
                "batch_display": batch,
                "suite_name": suite_name,
            }
        )

    # CCL-specific entries for N300 runners
    if ccl_batches:
        n300_template = {
            "test_group_name": "n300-llmbox-ccl",
            "arch": "wormhole_b0",
            "runs_on": "tt-ubuntu-2204-n300-llmbox-viommu-stable",
            "tt_smi_cmd": "tt-smi -r",
        }
        for batch in ccl_batches:
            include_entries.append(
                {
                    **n300_template,
                    "module_selector": batch,
                    "batch_display": f"ccl:{batch}",
                    "suite_name": "generality_suite_fabric_1d",
                }
            )

    return include_entries, batches, ccl_batches


def main():
    """Main entry point for matrix computation."""
    # Read environment variables from GitHub Actions context
    schedule_expr = os.environ.get("GITHUB_EVENT_SCHEDULE", "")
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    sweep_name = os.environ.get("SWEEP_NAME", "")
    measure_device_perf = os.environ.get("MEASURE_DEVICE_PERF", "false")
    vectors_dir = os.environ.get("VECTORS_DIR", "/tmp/vectors")

    # Read vector files from directory
    vectors_path = Path(vectors_dir)
    if not vectors_path.exists():
        print(f"Error: Vectors directory not found: {vectors_dir}", file=sys.stderr)
        sys.exit(1)

    modules = sorted([f.stem for f in vectors_path.glob("*.json")])

    if not modules:
        print(f"Error: No vector JSON files found in {vectors_dir}", file=sys.stderr)
        sys.exit(1)

    # Detect run type
    is_comprehensive = schedule_expr == "0 4 * * 3, 6" or sweep_name == "ALL SWEEPS (Comprehensive)"
    is_model_traced = schedule_expr == "0 4 * * *" or sweep_name == "ALL SWEEPS (Model Traced)"
    is_lead_models = schedule_expr == "0 3 * * *" or sweep_name == "ALL SWEEPS (Lead Models)"

    # Determine batch size
    # Use smaller batch size for comprehensive runs or when device performance measurement is enabled
    # (must stay under 256 total batches due to GitHub Actions limit)
    device_perf_enabled = event_name == "schedule" or measure_device_perf == "true"
    if is_comprehensive or device_perf_enabled:
        batch_size = 3
    else:
        batch_size = 10

    # Compute matrix based on run type
    if is_lead_models:
        include_entries, batches, ccl_batches = compute_lead_models_matrix(modules, batch_size)
    else:
        # Determine suite name for standard runs
        if is_model_traced:
            suite_name = "model_traced"
        elif is_comprehensive:
            suite_name = None
        else:
            suite_name = "nightly"

        include_entries, batches, ccl_batches = compute_standard_matrix(modules, batch_size, suite_name)

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
