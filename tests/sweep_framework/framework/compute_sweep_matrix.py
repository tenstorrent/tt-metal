#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Compute the sweep test matrix for GitHub Actions CI.

This script analyzes generated sweep vector files and produces a matrix
configuration that maps test modules to appropriate hardware runners.

The matrix output includes:
- Module batching to parallelize execution
- Runner assignment based on mesh topology (for lead models and model traced)
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
from constants import (
    get_mesh_shape_string,
    strip_mesh_suffix,
    parse_hardware_suffix,
    get_runner_config_for_hardware,
)


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

    Note: 'runs_on' can be either:
      - A string: Single runner label (e.g., "tt-ubuntu-2204-n150-stable")
      - A list: Multiple runner labels for GitHub Actions matrix
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
            "mesh_shapes": ["1x2", "1x4", "1x8", "2x4", "4x8", "8x4", "2x16", "16x2"],
            "test_group_name": "lead-models-galaxy",
            "arch": "wormhole_b0",
            "runs_on": "g04glx03",
            "runner_label": "g04glx03",
            "tt_smi_cmd": "tt-smi -r",
            "suite_name": "model_traced",
        },
    ]


def compute_lead_models_matrix(modules, batch_size):
    """Compute matrix for lead models run with mesh-aware runner assignment."""
    config = get_lead_models_mesh_runner_config()

    mesh_shape_modules = defaultdict(list)
    unmatched_modules = []

    for module in modules:
        mesh_shape = get_mesh_shape(module)
        if mesh_shape:
            mesh_shape_modules[mesh_shape].append(module)
        else:
            unmatched_modules.append(module)

    configured_shapes = set()
    for runner_config in config:
        configured_shapes.update(runner_config["mesh_shapes"])

    for mesh_shape in mesh_shape_modules.keys():
        if mesh_shape not in configured_shapes:
            print(f"Warning: Mesh shape '{mesh_shape}' has no runner config, modules will be skipped", file=sys.stderr)

    include_entries = []
    batches = []

    for runner_config in config:
        runner_modules = []
        for mesh_shape in runner_config["mesh_shapes"]:
            runner_modules.extend(mesh_shape_modules.get(mesh_shape, []))

        is_default_runner = runner_config == config[0]
        if is_default_runner:
            runner_modules.extend(unmatched_modules)

        if not runner_modules:
            continue

        base_modules = sorted(set(strip_mesh_suffix(m) for m in runner_modules))

        is_galaxy = runner_config["test_group_name"] == "lead-models-galaxy"
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
                }
            )

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

    return include_entries, batches, []


def compute_model_traced_matrix(modules, batch_size):
    """
    Compute matrix for model-traced runs with hardware-aware runner assignment.

    Reads the __hw_<name> suffix embedded in vector filenames (written by
    sweeps_parameter_generator from traced_machine_info) and emits one CI job
    per (hardware, mesh_shape) combination routed to the appropriate runner.

    Each matrix entry sets mesh_shapes_filter so MESH_DEVICE_SHAPE is passed
    to the runner, ensuring VectorExportSource only loads vectors for the exact
    mesh shape traced on that hardware.
    """
    # Group modules by (hardware_id, mesh_shape_string)
    hw_mesh_modules = defaultdict(list)  # (hw_id, mesh_str) -> [module_stem]
    legacy_modules = []  # no __hw_ or __mesh_ suffix

    for module in modules:
        hw_id = parse_hardware_suffix(module)
        mesh_str = get_mesh_shape_string(module)
        if hw_id and mesh_str:
            hw_mesh_modules[(hw_id, mesh_str)].append(module)
        else:
            legacy_modules.append(module)

    include_entries = []
    batches = []

    for hw_id, mesh_str in sorted(hw_mesh_modules.keys()):
        runner = get_runner_config_for_hardware(hw_id)
        if runner is None:
            print(f"Warning: Unknown hardware '{hw_id}' — skipping {mesh_str} vectors", file=sys.stderr)
            continue

        shape_modules = hw_mesh_modules[(hw_id, mesh_str)]
        base_modules = sorted(set(strip_mesh_suffix(m) for m in shape_modules))

        # T3K runners are scarce — cap to 5 parallel jobs per group
        if "t3k" in hw_id:
            max_jobs = 5
            effective_batch = max(1, -(-len(base_modules) // max_jobs))
        else:
            effective_batch = batch_size

        shape_batches = chunk_modules(base_modules, effective_batch)
        batches.extend(shape_batches)

        for batch in shape_batches:
            include_entries.append(
                {
                    "test_group_name": f"model-traced-{hw_id}",
                    "arch": runner["arch"],
                    "runs_on": runner["runs_on"],
                    "runner_label": runner["runner_label"],
                    "tt_smi_cmd": runner["tt_smi_cmd"],
                    "module_selector": batch,
                    "batch_display": f"{mesh_str}:{batch}",
                    "suite_name": "model_traced",
                    # Passed as MESH_DEVICE_SHAPE env var: VectorExportSource filters
                    # to only the vectors for this specific mesh shape / hardware.
                    "mesh_shapes_filter": mesh_str,
                }
            )

    # Legacy vectors (no __hw_ suffix): fall back to N150, no mesh filtering.
    if legacy_modules:
        base_legacy = sorted(set(strip_mesh_suffix(m) for m in legacy_modules))
        print(
            f"Warning: {len(base_legacy)} legacy module(s) without __hw_ suffix — "
            f"routing to default N150 runner: {', '.join(base_legacy[:5])}"
            f"{'...' if len(base_legacy) > 5 else ''}",
            file=sys.stderr,
        )
        legacy_batches = chunk_modules(base_legacy, batch_size)
        batches.extend(legacy_batches)
        for batch in legacy_batches:
            include_entries.append(
                {
                    "test_group_name": "model-traced-default",
                    "arch": "wormhole_b0",
                    "runs_on": "tt-ubuntu-2204-n150-stable",
                    "runner_label": "N150",
                    "tt_smi_cmd": "tt-smi -r",
                    "module_selector": batch,
                    "batch_display": batch,
                    "suite_name": "model_traced",
                    "mesh_shapes_filter": "",
                }
            )

    total_base = len(set(strip_mesh_suffix(m) for m in modules))
    print(
        f"Model traced run: {len(modules)} vector files ({total_base} unique modules), "
        f"{len(include_entries)} matrix entries",
        file=sys.stderr,
    )
    for (hw_id, mesh_str), mods in sorted(hw_mesh_modules.items()):
        unique_base = len(set(strip_mesh_suffix(m) for m in mods))
        print(f"  hw={hw_id} mesh={mesh_str}: {len(mods)} vectors ({unique_base} unique modules)", file=sys.stderr)

    return include_entries, batches, []


def compute_standard_matrix(modules, batch_size, suite_name):
    """Compute matrix for standard runs (nightly, comprehensive)."""
    base_modules = sorted(set(strip_mesh_suffix(m) for m in modules))
    ccl_modules = [m for m in base_modules if m.startswith("ccl.")]

    regular_batches = chunk_modules(base_modules, batch_size)
    batches = list(regular_batches)
    ccl_batches = chunk_modules(ccl_modules, batch_size)

    include_entries = []

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
    schedule_expr = os.environ.get("GITHUB_EVENT_SCHEDULE", "")
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    sweep_name = os.environ.get("SWEEP_NAME", "")
    measure_device_perf = os.environ.get("MEASURE_DEVICE_PERF", "false")
    vectors_dir = os.environ.get("VECTORS_DIR", "/tmp/vectors")

    vectors_path = Path(vectors_dir)
    if not vectors_path.exists():
        print(f"Error: Vectors directory not found: {vectors_dir}", file=sys.stderr)
        sys.exit(1)

    modules = sorted([f.stem for f in vectors_path.glob("*.json")])

    if not modules:
        print(f"Error: No vector JSON files found in {vectors_dir}", file=sys.stderr)
        sys.exit(1)

    is_comprehensive = False
    is_model_traced = False
    is_lead_models = False

    if sweep_name == "ALL SWEEPS (Lead Models)":
        is_lead_models = True
    elif sweep_name == "ALL SWEEPS (Model Traced)":
        is_model_traced = True
    elif sweep_name == "ALL SWEEPS (Comprehensive)":
        is_comprehensive = True
    else:
        # Schedule-based detection:
        #   "0 2 * * *"       -> lead models
        #   "0 3 * * *"       -> model traced
        #   "0 4 * * 3,6"     -> comprehensive
        #   "30 4 * * 0,1,2,4,5" -> nightly
        if schedule_expr == "0 2 * * *":
            is_lead_models = True
        elif schedule_expr == "0 3 * * *":
            is_model_traced = True
        elif schedule_expr == "0 4 * * 3,6":
            is_comprehensive = True

    device_perf_enabled = event_name == "schedule" or measure_device_perf == "true"
    if is_comprehensive or device_perf_enabled:
        batch_size = 3
    else:
        batch_size = 10

    if is_lead_models:
        include_entries, batches, ccl_batches = compute_lead_models_matrix(modules, batch_size)
    elif is_model_traced:
        # Route each job to the exact hardware the vectors were traced on.
        # sweeps_parameter_generator embeds __mesh_<r>x<c>__hw_<id> in filenames;
        # compute_model_traced_matrix reads those suffixes and assigns the right runner.
        include_entries, batches, ccl_batches = compute_model_traced_matrix(modules, batch_size)
    else:
        suite_name = None if is_comprehensive else "nightly"
        include_entries, batches, ccl_batches = compute_standard_matrix(modules, batch_size, suite_name)

    if len(batches) > 256:
        print(f"Total batch count ({len(batches)}) exceeds GitHub Actions matrix limit of 256.", file=sys.stderr)
        sys.exit(1)

    if len(include_entries) > 256:
        print(
            f"Total matrix entry count ({len(include_entries)}) exceeds GitHub Actions matrix limit of 256.",
            file=sys.stderr,
        )
        sys.exit(1)

    result = {
        "module": modules,
        "batches": batches,
        "ccl_batches": ccl_batches,
        "include": include_entries,
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
