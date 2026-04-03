#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Compute the sweep test matrix for GitHub Actions CI.

This script analyzes generated sweep vector files and produces a matrix
configuration that maps test modules to appropriate hardware runners.

Environment Variables (from GitHub Actions context):
- GITHUB_EVENT_SCHEDULE: Cron schedule expression
- GITHUB_EVENT_NAME: Event type (schedule, workflow_dispatch)
- SWEEP_NAME: Selected sweep type from workflow_dispatch
- MEASURE_DEVICE_PERF: Whether device performance measurement is enabled
- VECTORS_DIR: Directory containing vector JSON files (default: /tmp/vectors)

Output:
Prints GitHub Actions output lines to stdout (matrix + per-hw matrices).
"""

import os
import json
import sys
from collections import defaultdict
from pathlib import Path

from constants import get_mesh_shape_string, parse_hardware_suffix, strip_grouping_suffix
from matrix_runner_config import (
    GENERATION_MANIFEST_FILENAME,
    HW_GROUP_MATRIX_KEYS,
    LEAD_MODELS_BATCH_POLICY,
    LEAD_MODELS_DEFAULT_TEST_GROUP,
    LEAD_MODELS_MESH_TEST_GROUPS,
    LEAD_MODELS_SUITE_NAME,
    MODEL_TRACED_MESH_TEST_GROUPS,
    SCHEDULE_TYPES,
    SWEEP_TYPES,
    get_lead_models_test_group_name_for_hardware_group,
    get_runner_config,
    get_test_group_name_for_hardware_group,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def chunk_modules(items, size):
    """Split modules into batches of specified size."""
    return [",".join(items[i : i + size]) for i in range(0, len(items), size)] if items else []


def _get_runner(name):
    """Look up a runner config by test_group_name."""
    return get_runner_config(name)


def _hw_label(hardware_group):
    """Readable label like 'wormhole/n300/4c' or 'default'."""
    if hardware_group is None:
        return "default"
    board_type, device_series, card_count = hardware_group
    return f"{board_type}/{device_series}/{card_count}c"


def _log_module_groups(header, modules, groups):
    """Print a summary of module grouping to stderr."""
    total_base = len(set(strip_grouping_suffix(m) for m in modules))
    print(
        f"{header}: {len(modules)} vector files ({total_base} unique modules), "
        f"{sum(1 for g in groups for _ in g[1])} matrix entries",
        file=sys.stderr,
    )
    for label, entries in groups:
        unique = len(set(strip_grouping_suffix(m) for m in entries))
        print(f"  {label}: {len(entries)} vectors ({unique} unique modules)", file=sys.stderr)


def _build_entries(runner_config, batches, batch_display_prefix, suite_name):
    """Create matrix include entries for a set of batches using a runner config."""
    return [
        {
            **runner_config,
            "module_selector": batch,
            "batch_display": f"{batch_display_prefix}:{batch}" if batch_display_prefix else batch,
            "suite_name": suite_name,
        }
        for batch in batches
    ]


def _load_generation_manifest(vectors_path):
    """Load generation manifest metadata if present."""
    manifest_path = vectors_path / GENERATION_MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}

    try:
        with open(manifest_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: Failed to read generation manifest at {manifest_path}: {e}", file=sys.stderr)
        return {}

    if not isinstance(data, dict):
        print(f"Warning: Generation manifest at {manifest_path} is not a JSON object", file=sys.stderr)
        return {}
    return data


def _modules_from_manifest_or_dir(vectors_path, manifest):
    """Resolve module stems from manifest vector_files or by scanning directory."""
    manifest_files = manifest.get("vector_files")
    if isinstance(manifest_files, list):
        files = [name for name in manifest_files if isinstance(name, str) and name.endswith(".json")]
        if files:
            modules = sorted({Path(name).stem for name in files if Path(name).name != GENERATION_MANIFEST_FILENAME})
            if modules:
                return modules

    return sorted([f.stem for f in vectors_path.glob("*.json") if f.name != GENERATION_MANIFEST_FILENAME])


def _group_modules_by_preference(modules, grouping_mode):
    """Group modules by mesh, hardware, or unmatched based on routing preference."""
    mesh_modules = defaultdict(list)
    hw_modules = defaultdict(list)
    unmatched = []

    for module in modules:
        mesh = get_mesh_shape_string(module)
        hw = parse_hardware_suffix(module)

        if grouping_mode == "mesh":
            if mesh:
                mesh_modules[mesh].append(module)
            elif hw:
                hw_modules[hw].append(module)
            else:
                unmatched.append(module)
        else:
            if hw:
                hw_modules[hw].append(module)
            elif mesh:
                mesh_modules[mesh].append(module)
            else:
                unmatched.append(module)

    return mesh_modules, hw_modules, unmatched


def _get_test_group_for_mesh_shape(mesh_shape, mesh_test_groups, run_label, default_test_group=None):
    """Resolve a mesh shape to a logical test group with optional fallback."""
    test_group_name = mesh_test_groups.get(mesh_shape)
    if test_group_name is not None:
        return test_group_name

    if default_test_group is not None:
        print(
            f"Warning: Mesh shape '{mesh_shape}' has no {run_label} runner mapping; using default runner",
            file=sys.stderr,
        )
        return default_test_group

    print(
        f"Warning: Mesh shape '{mesh_shape}' has no {run_label} runner mapping; modules will be skipped",
        file=sys.stderr,
    )
    return None


def _batch_modules_for_test_group(base_modules, batch_size, batch_policy=None):
    """Batch base module names using fixed size or policy-defined parallel jobs."""
    parallel_jobs = (batch_policy or {}).get("parallel_jobs")
    if parallel_jobs:
        size = max(1, -(-len(base_modules) // parallel_jobs))
        return chunk_modules(base_modules, size)
    return chunk_modules(base_modules, batch_size)


def _append_routed_group(
    include_entries, batches, log_groups, label, modules, test_group_name, batch_size, suite_name, batch_policy=None
):
    """Convert routed modules for one logical group into matrix entries."""
    if not modules or test_group_name is None:
        return

    base = sorted(set(strip_grouping_suffix(m) for m in modules))
    runner_config = _get_runner(test_group_name)
    runner_batches = _batch_modules_for_test_group(base, batch_size, batch_policy)
    batches.extend(runner_batches)
    include_entries.extend(_build_entries(runner_config, runner_batches, label, suite_name))
    log_groups.append((label, modules))


# ── Matrix computation per run type ─────────────────────────────────────────


def compute_lead_models_matrix(modules, batch_size):
    """Compute matrix for lead models with mesh-aware runner assignment."""
    mesh_modules, hw_modules, unmatched = _group_modules_by_preference(modules, "mesh")

    include_entries = []
    batches = []
    log_groups = []
    routed_modules = defaultdict(list)

    for mesh_shape, mods in sorted(mesh_modules.items()):
        test_group_name = _get_test_group_for_mesh_shape(
            mesh_shape,
            LEAD_MODELS_MESH_TEST_GROUPS,
            "lead_models",
        )
        if test_group_name is not None:
            routed_modules[test_group_name].extend(mods)

    for hw_group, mods in sorted(hw_modules.items()):
        routed_modules[get_lead_models_test_group_name_for_hardware_group(hw_group)].extend(mods)

    if unmatched:
        routed_modules[LEAD_MODELS_DEFAULT_TEST_GROUP].extend(unmatched)

    lead_models_group_order = [LEAD_MODELS_DEFAULT_TEST_GROUP]
    for test_group_name in dict.fromkeys(LEAD_MODELS_MESH_TEST_GROUPS.values()):
        if test_group_name not in lead_models_group_order:
            lead_models_group_order.append(test_group_name)
    for test_group_name in routed_modules:
        if test_group_name not in lead_models_group_order:
            lead_models_group_order.append(test_group_name)

    for test_group_name in lead_models_group_order:
        label_meshes = [mesh for mesh, group in LEAD_MODELS_MESH_TEST_GROUPS.items() if group == test_group_name]
        batch_label = "+".join(label_meshes) if label_meshes else test_group_name
        _append_routed_group(
            include_entries,
            batches,
            log_groups,
            batch_label,
            routed_modules.get(test_group_name, []),
            test_group_name,
            batch_size,
            LEAD_MODELS_SUITE_NAME,
            LEAD_MODELS_BATCH_POLICY.get(test_group_name),
        )

    # Log
    for mesh, mods in sorted(mesh_modules.items()):
        log_groups.append((f"mesh {mesh}", mods))
    for hw, mods in sorted(hw_modules.items()):
        log_groups.append((f"hardware {_hw_label(hw)}", mods))
    if unmatched:
        log_groups.append(("no grouping suffix (default runner)", unmatched))
    _log_module_groups("Lead models run", modules, log_groups)

    return include_entries, batches


def compute_model_traced_matrix(modules, batch_size, suite_name, grouping_mode=None):
    """Compute matrix for model_traced runs using mesh/hardware grouped vector files."""
    mode = grouping_mode if grouping_mode in {"mesh", "hw"} else "hw"
    if grouping_mode not in {"mesh", "hw", None}:
        print(f"Warning: Unsupported grouping_mode '{grouping_mode}', defaulting to hardware routing", file=sys.stderr)

    mesh_modules, hw_modules, unmatched = _group_modules_by_preference(modules, mode)

    include_entries = []
    batches = []
    log_groups = []

    for mesh_shape, mods in sorted(mesh_modules.items(), key=lambda x: x[0]):
        test_group_name = _get_test_group_for_mesh_shape(
            mesh_shape,
            MODEL_TRACED_MESH_TEST_GROUPS,
            "model_traced",
            "wormhole-n150-sweeps",
        )
        _append_routed_group(
            include_entries,
            batches,
            log_groups,
            f"mesh {mesh_shape}",
            mods,
            test_group_name,
            batch_size,
            suite_name,
        )

    grouped = sorted(hw_modules.items(), key=lambda x: x[0])
    if unmatched:
        grouped.append((None, unmatched))

    for hw_group, mods in grouped:
        _append_routed_group(
            include_entries,
            batches,
            log_groups,
            f"hardware {_hw_label(hw_group)}",
            mods,
            get_test_group_name_for_hardware_group(hw_group),
            batch_size,
            suite_name,
        )

    _log_module_groups(f"Model traced run ({mode}-grouped)", modules, log_groups)
    return include_entries, batches


def compute_standard_matrix(modules, batch_size, suite_name):
    """Compute matrix for nightly/comprehensive runs."""
    base_modules = sorted(set(strip_grouping_suffix(m) for m in modules))
    ccl_modules = [m for m in base_modules if m.startswith("ccl.")]
    regular_modules = [m for m in base_modules if not m.startswith("ccl.")]

    # Keep CCL modules off the default runner path; they get their own N300 lane below.
    regular_batches = chunk_modules(regular_modules, batch_size)
    ccl_batches = chunk_modules(ccl_modules, batch_size)

    n150_config = _get_runner("wormhole-n150-sweeps")
    include_entries = _build_entries(n150_config, regular_batches, "", suite_name)

    if ccl_batches:
        ccl_config = _get_runner("n300-llmbox-ccl")
        include_entries.extend(_build_entries(ccl_config, ccl_batches, "ccl", "generality_suite_fabric_1d"))

    return include_entries, list(regular_batches), ccl_batches


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    """Main entry point."""
    schedule_expr = os.environ.get("GITHUB_EVENT_SCHEDULE", "")
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    sweep_name = os.environ.get("SWEEP_NAME", "")
    measure_device_perf = os.environ.get("MEASURE_DEVICE_PERF", "false")
    vectors_dir = os.environ.get("VECTORS_DIR", "/tmp/vectors")

    vectors_path = Path(vectors_dir)
    if not vectors_path.exists():
        print(f"Error: Vectors directory not found: {vectors_dir}", file=sys.stderr)
        sys.exit(1)

    manifest = _load_generation_manifest(vectors_path)
    modules = _modules_from_manifest_or_dir(vectors_path, manifest)
    if not modules:
        print(f"Error: No vector JSON files found in {vectors_dir}", file=sys.stderr)
        sys.exit(1)

    # Detect run type: explicit sweep name takes precedence, then schedule cron
    run_type = SWEEP_TYPES.get(sweep_name) or SCHEDULE_TYPES.get(schedule_expr, "nightly")

    # Batch size: smaller for comprehensive or device-perf runs (must stay under 256 batches)
    device_perf_enabled = event_name == "schedule" or measure_device_perf == "true"
    batch_size = 3 if (run_type == "comprehensive" or device_perf_enabled) else 10

    # Compute matrix
    ccl_batches = []
    if run_type == "lead_models":
        include_entries, batches = compute_lead_models_matrix(modules, batch_size)
    elif run_type == "model_traced":
        grouping_mode = manifest.get("vector_grouping_mode")
        include_entries, batches = compute_model_traced_matrix(modules, batch_size, "model_traced", grouping_mode)
    else:
        suite_name = None if run_type == "comprehensive" else run_type  # "nightly" or None
        include_entries, batches, ccl_batches = compute_standard_matrix(modules, batch_size, suite_name)

    # Validate GitHub Actions limits
    for label, count in [("batch", len(batches)), ("matrix entry", len(include_entries))]:
        if count > 256:
            print(f"Total {label} count ({count}) exceeds GitHub Actions limit of 256.", file=sys.stderr)
            sys.exit(1)

    # Output: combined matrix + per-hardware matrices (piped to $GITHUB_OUTPUT)
    compact = {"separators": (",", ":")}
    result = {
        "module": modules,
        "batches": batches,
        "ccl_batches": ccl_batches,
        "include": include_entries,
    }
    print("matrix=" + json.dumps(result, **compact))

    for hw_key, group_names in HW_GROUP_MATRIX_KEYS.items():
        hw_entries = [e for e in include_entries if e.get("test_group_name", "") in group_names]
        print(f"{hw_key}-matrix=" + json.dumps({"include": hw_entries}, **compact))


if __name__ == "__main__":
    main()
