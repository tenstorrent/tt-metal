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

import os
import json
import sys
from collections import defaultdict
from pathlib import Path

# Import mesh utilities from shared constants module
from constants import get_mesh_shape_string, strip_mesh_suffix


def chunk_modules(items, size):
    """Split modules into batches of specified size."""
    return [",".join(items[i : i + size]) for i in range(0, len(items), size)] if items else []


# Alias for backward compatibility and clearer naming in this context
def get_mesh_shape(module_name):
    """Extract mesh shape string from module name (wrapper for get_mesh_shape_string)."""
    return get_mesh_shape_string(module_name)


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
            # Multi-chip operations - runs on Galaxy (32-chip topology)
            # Handles all mesh shapes larger than 1x1
            "mesh_shapes": ["1x2", "1x4", "1x8", "2x4", "4x8", "8x4", "2x16", "16x2"],
            "test_group_name": "lead-models-galaxy",
            "arch": "wormhole_b0",
            "runs_on": [
                "topology-6u",  # 32-chip galaxy topology
                "arch-wormhole_b0",  # Wormhole B0 architecture
                "in-service",  # Available for use
                "pipeline-functional",  # Functional pipeline
            ],
            "runner_label": "topology-6u",
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

        # Strip mesh suffixes to get base module names that sweeps_runner can find
        # The VectorExportSource will automatically load mesh-variant JSONs
        base_modules = sorted(set(strip_mesh_suffix(m) for m in runner_modules))

        # For Galaxy runners (multi-chip), run all modules as a single batch
        # For single-chip runners, use the standard batch size
        is_galaxy = runner_config["test_group_name"] == "lead-models-galaxy"
        if is_galaxy:
            # Single batch with all modules
            runner_batches = [",".join(base_modules)] if base_modules else []
        else:
            # Standard batching for single-chip
            runner_batches = chunk_modules(base_modules, batch_size)

        batches.extend(runner_batches)

        # Create matrix entries
        mesh_label = "+".join(runner_config["mesh_shapes"])
        for batch in runner_batches:
            include_entries.append(
                {
                    "test_group_name": runner_config["test_group_name"],
                    "arch": runner_config["arch"],
                    "runs_on": runner_config["runs_on"],
                    "runner_label": runner_config["runner_label"],
                    "tt_smi_cmd": runner_config["tt_smi_cmd"],
                    "module_selector": batch,
                    "batch_display": f"{mesh_label}:{batch}",
                    "suite_name": runner_config["suite_name"],
                }
            )

    # Log summary
    total_base_modules = len(set(strip_mesh_suffix(m) for m in modules))
    print(
        f"Lead models run: {len(modules)} vector files ({total_base_modules} unique modules), "
        f"{len(include_entries)} matrix entries",
        file=sys.stderr,
    )
    for mesh_shape, mods in sorted(mesh_shape_modules.items()):
        unique_base = len(set(strip_mesh_suffix(m) for m in mods))
        print(f"  mesh {mesh_shape}: {len(mods)} vectors ({unique_base} unique modules)", file=sys.stderr)

    return include_entries, batches, []  # No CCL batches for lead models


def get_model_traced_mesh_runner_config(available_mesh_shapes=None):
    """
    Configuration: Map mesh shapes to runner configurations for model_traced runs.

    This configuration defines which hardware runners handle which mesh shapes
    based on device count (rows * cols).

    Dynamically assigns mesh shapes to runners based on their device requirements:
    - 1 device (1x1): P150B (Blackhole) or N150 (Wormhole)
    - 2 devices (1x2, 2x1): N300
    - 4 devices (1x4, 2x2, 4x1): T3000
    - 8-32 devices: Galaxy (32-chip topology)

    Args:
        available_mesh_shapes: Optional set of mesh shapes that actually exist in vector files.
                               Only these mesh shapes will be assigned to runners.

    Note: 'runs_on' can be either:
      - A string: Single runner label (e.g., "tt-ubuntu-2204-n150-stable")
      - A list: Multiple runner labels for GitHub Actions matrix
    """
    # Define runners with device count criteria instead of hardcoded mesh shapes
    runner_definitions = [
        {
            # P150B (Blackhole) - single chip
            "test_group_name": "P150B-1x1",
            "arch": "blackhole",
            "runs_on": [
                "arch-blackhole",
                "in-service",
                "pipeline-functional",
            ],
            "runner_label": "BH",
            "tt_smi_cmd": "tt-smi -r",
            "suite_name": "model_traced",
            "device_count": 1,  # Handles 1-device meshes
        },
        {
            # N300 - 2-chip configurations
            "test_group_name": "N300",
            "arch": "wormhole_b0",
            "runs_on": "tt-ubuntu-2204-n300-llmbox-viommu-stable",
            "runner_label": "N300",
            "tt_smi_cmd": "tt-smi -r",
            "suite_name": "model_traced",
            "device_count": 2,  # Handles 2-device meshes (1x2, 2x1)
        },
        {
            # T3000 - 4-chip configurations
            "test_group_name": "T3000",
            "arch": "wormhole_b0",
            "runs_on": [
                "arch-wormhole_b0",
                "config-t3000",
                "in-service",
                "pipeline-functional",
            ],
            "runner_label": "T3000",
            "tt_smi_cmd": "tt-smi -r",
            "suite_name": "model_traced",
            "device_count": 4,  # Handles 4-device meshes (1x4, 2x2, 4x1)
        },
        {
            # Galaxy - 8-32 chip configurations
            "test_group_name": "Galaxy",
            "arch": "wormhole_b0",
            "runs_on": [
                "topology-6u",  # 32-chip galaxy topology
                "arch-wormhole_b0",  # Wormhole B0 architecture
                "in-service",  # Available for use
                "pipeline-functional",  # Functional pipeline
            ],
            "runner_label": "topology-6u",
            "tt_smi_cmd": "tt-smi -r",
            "suite_name": "model_traced",
            "device_count_range": (8, 32),  # Handles 8-32 device meshes
        },
    ]

    # If no available mesh shapes provided, return empty config
    if available_mesh_shapes is None or not available_mesh_shapes:
        return []

    # Build dynamic config by assigning available mesh shapes to runners
    dynamic_config = []
    for runner_def in runner_definitions:
        assigned_shapes = []

        # Assign mesh shapes to this runner based on device count
        for mesh_shape in available_mesh_shapes:
            try:
                # Parse mesh shape (e.g., "1x2" -> rows=1, cols=2)
                rows, cols = map(int, mesh_shape.split("x"))
                device_count = rows * cols

                # Check if this mesh fits this runner's device count criteria
                if "device_count_range" in runner_def:
                    # Range-based (e.g., Galaxy: 8-32 devices)
                    min_devices, max_devices = runner_def["device_count_range"]
                    if min_devices <= device_count <= max_devices:
                        assigned_shapes.append(mesh_shape)
                elif "device_count" in runner_def:
                    # Exact match (e.g., N300: 2 devices)
                    if device_count == runner_def["device_count"]:
                        assigned_shapes.append(mesh_shape)
            except (ValueError, AttributeError):
                continue

        # Only add runner if it has mesh shapes assigned
        if assigned_shapes:
            runner_config = {
                "test_group_name": runner_def["test_group_name"],
                "arch": runner_def["arch"],
                "runs_on": runner_def["runs_on"],
                "runner_label": runner_def["runner_label"],
                "tt_smi_cmd": runner_def["tt_smi_cmd"],
                "suite_name": runner_def["suite_name"],
                "mesh_shapes": sorted(assigned_shapes),  # Dynamically assigned!
            }
            dynamic_config.append(runner_config)

    return dynamic_config


def compute_model_traced_matrix(modules, batch_size):
    """
    Compute matrix for model_traced runs with mesh-aware runner assignment.
    Uses dedicated mesh configuration for model_traced tests.

    Dynamically determines which mesh shapes have actual test vectors and only
    creates jobs for those mesh shapes.

    Args:
        modules: List of module names (from vector JSON filenames)
        batch_size: Number of modules per batch

    Returns:
        Tuple of (include_entries, batches, ccl_batches)
    """
    # Group modules by mesh shape
    mesh_shape_modules = defaultdict(list)
    unmatched_modules = []

    for module in modules:
        mesh_shape = get_mesh_shape(module)
        if mesh_shape:
            mesh_shape_modules[mesh_shape].append(module)
        else:
            unmatched_modules.append(module)

    # Discover which mesh shapes actually have test vectors
    available_mesh_shapes = set(mesh_shape_modules.keys())
    if available_mesh_shapes:
        print(f"Discovered mesh shapes with test vectors: {sorted(available_mesh_shapes)}", file=sys.stderr)

    # Get configuration filtered to only available mesh shapes
    config = get_model_traced_mesh_runner_config(available_mesh_shapes)

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
        # Process each mesh shape separately to create per-mesh batches
        for mesh_shape in runner_config["mesh_shapes"]:
            shape_modules = mesh_shape_modules.get(mesh_shape, [])

            if not shape_modules:
                continue

            # Strip mesh suffixes to get base module names that sweeps_runner can find
            # The VectorExportSource will automatically load mesh-variant JSONs
            base_modules = sorted(set(strip_mesh_suffix(m) for m in shape_modules))

            # Batch modules for this specific mesh shape
            mesh_batches = chunk_modules(base_modules, batch_size)
            batches.extend(mesh_batches)

            # Create matrix entries for each batch in this mesh shape
            for batch in mesh_batches:
                include_entries.append(
                    {
                        "test_group_name": runner_config["test_group_name"],
                        "arch": runner_config["arch"],
                        "runs_on": runner_config["runs_on"],
                        "runner_label": runner_config["runner_label"],
                        "tt_smi_cmd": runner_config["tt_smi_cmd"],
                        "module_selector": batch,
                        "batch_display": f"{mesh_shape}:{batch}",
                        "suite_name": "model_traced",
                        "mesh_device_shape": mesh_shape,  # Pass mesh shape to runner
                    }
                )

    # Log summary
    total_base_modules = len(set(strip_mesh_suffix(m) for m in modules))
    print(
        f"Model traced run: {len(modules)} vector files ({total_base_modules} unique modules), "
        f"{len(include_entries)} matrix entries",
        file=sys.stderr,
    )
    for mesh_shape, mods in sorted(mesh_shape_modules.items()):
        unique_base = len(set(strip_mesh_suffix(m) for m in mods))
        print(f"  mesh {mesh_shape}: {len(mods)} vectors ({unique_base} unique modules)", file=sys.stderr)

    return include_entries, batches, []  # No CCL batches for model_traced


def compute_standard_matrix(modules, batch_size, suite_name):
    """
    Compute matrix for standard runs (nightly, comprehensive, model_traced).

    Args:
        modules: List of module names (may include mesh suffixes from JSON filenames)
        batch_size: Number of modules per batch
        suite_name: Suite name override (None means no override)

    Returns:
        Tuple of (include_entries, batches, ccl_batches)
    """
    # For model_traced runs, use mesh-aware configuration
    if suite_name == "model_traced":
        return compute_model_traced_matrix(modules, batch_size)

    # For nightly/comprehensive, use original logic (non-mesh-aware)
    # Strip mesh suffixes to get base module names that sweeps_runner can find
    # The VectorExportSource will automatically load mesh-variant JSONs
    base_modules = sorted(set(strip_mesh_suffix(m) for m in modules))

    ccl_modules = [m for m in base_modules if m.startswith("ccl.")]

    # Create batches for all modules using base names
    regular_batches = chunk_modules(base_modules, batch_size)
    batches = list(regular_batches)

    # Generate CCL-only batches for dedicated runners
    ccl_batches = chunk_modules(ccl_modules, batch_size)

    include_entries = []

    # Standard wormhole runner entries
    wormhole_template = {
        "test_group_name": "wormhole-n150-sweeps",
        "arch": "wormhole_b0",
        "runs_on": "tt-ubuntu-2204-n150-stable",
        "runner_label": "N150",
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
            "runner_label": "n300-llmbox",
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
