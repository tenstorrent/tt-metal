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
from constants import get_mesh_shape_string, strip_mesh_suffix, parse_hardware_suffix, get_runner_config_for_hardware


def chunk_modules(items, size):
    """Split modules into batches of specified size."""
    return [",".join(items[i : i + size]) for i in range(0, len(items), size)] if items else []


# Alias for backward compatibility and clearer naming in this context
def get_mesh_shape(module_name):
    """Extract mesh shape string from module name (wrapper for get_mesh_shape_string)."""
    return get_mesh_shape_string(module_name)


def get_lead_models_mesh_runner_config():
    """Static runner config for lead models runs (unchanged from original).

    Uses two groups: single-chip (N150) and multi-chip (Galaxy).
    """
    return [
        {
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


def build_mesh_runner_config_from_modules(modules):
    """Dynamically build runner config from vector filenames.

    Reads hardware from the ``__hw_<name>`` filename suffix (written by
    the vector generator from traced_machine_info).  Groups all mesh
    shapes for the same hardware into ONE runner config, producing one
    CI job per hardware type (e.g., one N300 job, one Galaxy job).

    Args:
        modules: List of module filenames (stems) including mesh/hw suffixes.

    Returns:
        List of runner config dicts — one per unique hardware that has vectors.
    """
    # Group mesh shapes by hardware
    hw_to_meshes = defaultdict(set)
    for module in modules:
        mesh_str = get_mesh_shape_string(module)
        hw_name = parse_hardware_suffix(module)
        if mesh_str:
            hw_to_meshes[hw_name or ""].add(mesh_str)

    configs = []
    for hw_name in sorted(hw_to_meshes.keys()):
        mesh_shapes = sorted(hw_to_meshes[hw_name])
        if hw_name:
            runner = get_runner_config_for_hardware(hw_name)
            if runner is None:
                print(
                    f"Warning: Unknown hardware '{hw_name}', skipping {mesh_shapes}",
                    file=sys.stderr,
                )
                continue
            configs.append(
                {
                    "mesh_shapes": mesh_shapes,
                    "test_group_name": f"model-traced-{hw_name}",
                    "suite_name": "model_traced",
                    **runner,
                }
            )
        else:
            # Legacy suffix without __hw_ — fall back to default N150 runner
            configs.append(
                {
                    "mesh_shapes": mesh_shapes,
                    "test_group_name": "model-traced-default",
                    "suite_name": "model_traced",
                    "arch": "wormhole_b0",
                    "runs_on": "tt-ubuntu-2204-n150-stable",
                    "runner_label": "N150",
                    "tt_smi_cmd": "tt-smi -r",
                }
            )
    return configs


def compute_lead_models_matrix(modules, batch_size, dynamic_hw=False):
    """
    Compute matrix for mesh-aware runner assignment.

    Args:
        modules: List of module names (from vector JSON filenames)
        batch_size: Number of modules per batch
        dynamic_hw: If True, derive hardware from __hw_ filename suffix
                    (for model_traced runs). If False, use static config
                    (for lead models runs).

    Returns:
        Tuple of (include_entries, batches, ccl_batches)
    """
    # Group modules by mesh shape first to discover what shapes exist
    mesh_shape_modules = defaultdict(list)
    unmatched_modules = []

    for module in modules:
        mesh_shape = get_mesh_shape(module)
        if mesh_shape:
            mesh_shape_modules[mesh_shape].append(module)
        else:
            unmatched_modules.append(module)

    if dynamic_hw:
        # Model traced: read hardware from __hw_ suffix in filenames
        config = build_mesh_runner_config_from_modules(modules)
    else:
        # Lead models: use static config (N150 + Galaxy)
        config = get_lead_models_mesh_runner_config()

    # Create matrix entries based on runner config
    include_entries = []
    batches = []

    for runner_config in config:
        # Collect all modules for this runner's mesh shapes
        runner_modules = []
        for mesh_shape in runner_config["mesh_shapes"]:
            runner_modules.extend(mesh_shape_modules.get(mesh_shape, []))

        # Route modules without a mesh suffix to the first (default) runner config.
        # Only for lead models (static config); dynamic_hw skips legacy modules.
        is_default_runner = runner_config == config[0]
        if is_default_runner and not dynamic_hw:
            runner_modules.extend(unmatched_modules)

        if not runner_modules:
            continue

        if dynamic_hw:
            # --- Model traced: one sub-job per mesh shape ---
            # Each sub-job sets MESH_DEVICE_SHAPE so only matching vectors run.
            # UI shows: Run sweeps (model-traced-tt-galaxy-wh, 4x8: add,linear)
            for mesh_shape in runner_config["mesh_shapes"]:
                shape_modules = mesh_shape_modules.get(mesh_shape, [])
                if not shape_modules:
                    continue
                base_modules = sorted(set(strip_mesh_suffix(m) for m in shape_modules))
                # T3K runners are scarce — cap to 5 jobs per mesh shape
                if "t3k" in runner_config["test_group_name"]:
                    max_jobs = 5
                    effective_batch_size = max(1, -(-len(base_modules) // max_jobs))
                else:
                    effective_batch_size = 5
                shape_batches = chunk_modules(base_modules, effective_batch_size)
                batches.extend(shape_batches)
                for batch in shape_batches:
                    include_entries.append(
                        {
                            "test_group_name": runner_config["test_group_name"],
                            "arch": runner_config["arch"],
                            "runs_on": runner_config["runs_on"],
                            "runner_label": runner_config["runner_label"],
                            "tt_smi_cmd": runner_config["tt_smi_cmd"],
                            "module_selector": batch,
                            "batch_display": f"{mesh_shape}:{batch}",
                            "suite_name": runner_config["suite_name"],
                            "mesh_shapes_filter": mesh_shape,
                        }
                    )
            # Log unmatched modules (no __mesh_/__hw_ suffix) — these are
            # legacy vectors that cannot be routed to specific hardware.
            # They are skipped in hardware-routed model_traced runs.
            if is_default_runner and unmatched_modules:
                base_unmatched = sorted(set(strip_mesh_suffix(m) for m in unmatched_modules))
                print(
                    f"Warning: {len(base_unmatched)} legacy modules without mesh/hw suffix "
                    f"will be skipped (no hardware routing info): {', '.join(base_unmatched[:5])}{'...' if len(base_unmatched) > 5 else ''}",
                    file=sys.stderr,
                )
        else:
            # --- Lead models: original behavior (all mesh shapes grouped per runner) ---
            base_modules = sorted(set(strip_mesh_suffix(m) for m in runner_modules))
            is_galaxy = "galaxy" in runner_config["test_group_name"]
            if is_galaxy:
                galaxy_jobs = 3
                galaxy_batch_size = max(1, -(-len(base_modules) // galaxy_jobs))
                runner_batches = chunk_modules(base_modules, galaxy_batch_size)
            else:
                runner_batches = chunk_modules(base_modules, batch_size)
            batches.extend(runner_batches)
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
                        "mesh_shapes_filter": "",
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
    if unmatched_modules:
        unique_base = len(set(strip_mesh_suffix(m) for m in unmatched_modules))
        print(
            f"  no mesh suffix (default runner): {len(unmatched_modules)} vectors ({unique_base} unique modules)",
            file=sys.stderr,
        )

    return include_entries, batches, []  # No CCL batches for lead models


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
                "mesh_shapes_filter": "",
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
                    "mesh_shapes_filter": "",
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
        include_entries, batches, ccl_batches = compute_lead_models_matrix(modules, batch_size, dynamic_hw=False)
    elif is_model_traced:
        # Model traced runs use mesh-aware routing so vectors execute
        # on the exact hardware they were traced on.
        include_entries, batches, ccl_batches = compute_lead_models_matrix(modules, batch_size, dynamic_hw=True)
    else:
        # Determine suite name for standard runs
        if is_comprehensive:
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

    # Split include entries by test_group_name for per-hardware job definitions.
    # Each hardware group becomes a separate matrix output so the workflow
    # can create distinct parent jobs with sub-jobs underneath.
    hw_groups = defaultdict(list)
    for entry in include_entries:
        hw_groups[entry["test_group_name"]].append(entry)

    # Output matrix JSON
    result = {
        "module": modules,
        "batches": batches,
        "ccl_batches": ccl_batches,
        "include": include_entries,
        # Per-hardware sub-matrices — each is a list of matrix entries
        # that share the same runner. The workflow creates a separate
        # job definition per hardware group, each with its own matrix.
        "hw_groups": {name: entries for name, entries in sorted(hw_groups.items())},
        "hw_group_names": sorted(hw_groups.keys()),
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
