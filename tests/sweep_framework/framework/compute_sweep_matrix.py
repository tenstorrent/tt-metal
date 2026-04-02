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

# ── Runner config registry ───────────────────────────────────────────────────
# All runner configurations live here.  Every function that needs a runner
# config looks it up by test_group_name rather than constructing dicts inline.

RUNNER_CONFIGS = {
    "wormhole-n150-sweeps": {
        "test_group_name": "wormhole-n150-sweeps",
        "arch": "wormhole_b0",
        "runs_on": "tt-ubuntu-2204-n150-stable",
        "runner_label": "N150",
        "tt_smi_cmd": "tt-smi -r",
    },
    "wormhole-n300-sweeps": {
        "test_group_name": "wormhole-n300-sweeps",
        "arch": "wormhole_b0",
        "runs_on": "tt-ubuntu-2204-n300-stable",
        "runner_label": "N300",
        "tt_smi_cmd": "tt-smi -r",
    },
    "n300-llmbox-ccl": {
        "test_group_name": "n300-llmbox-ccl",
        "arch": "wormhole_b0",
        "runs_on": "tt-ubuntu-2204-n300-llmbox-viommu-stable",
        "runner_label": "n300-llmbox",
        "tt_smi_cmd": "tt-smi -r",
    },
    "blackhole-p150b-sweeps": {
        "test_group_name": "blackhole-p150b-sweeps",
        "arch": "blackhole",
        "runs_on": "tt-ubuntu-2204-p150b-viommu-stable",
        "runner_label": "p150b",
        "tt_smi_cmd": "tt-smi -r",
    },
    "wormhole-t3k-sweeps": {
        "test_group_name": "wormhole-t3k-sweeps",
        "arch": "wormhole_b0",
        "runs_on": ["config-t3000", "arch-wormhole_b0", "in-service", "pipeline-functional"],
        "runner_label": "config-t3000",
        "tt_smi_cmd": "tt-smi -r",
    },
    "wormhole-galaxy-sweeps": {
        "test_group_name": "wormhole-galaxy-sweeps",
        "arch": "wormhole_b0",
        "runs_on": ["topology-6u", "in-service", "bare-metal"],
        "runner_label": "topology-6u",
        "tt_smi_cmd": "tt-smi -glx_reset_auto",
    },
    "lead-models-single-chip": {
        "test_group_name": "lead-models-single-chip",
        "arch": "wormhole_b0",
        "runs_on": "tt-ubuntu-2204-n150-stable",
        "runner_label": "N150",
        "tt_smi_cmd": "tt-smi -r",
    },
    "lead-models-galaxy": {
        "test_group_name": "lead-models-galaxy",
        "arch": "wormhole_b0",
        "runs_on": "g04glx03",
        "runner_label": "g04glx03",
        "tt_smi_cmd": "tt-smi -r",
    },
}

# Maps GitHub Actions output key → test_group_names for per-hw matrix splitting.
HW_GROUP_MATRIX_KEYS = {
    "n150": ["wormhole-n150-sweeps", "lead-models-single-chip"],
    "n300": ["wormhole-n300-sweeps", "n300-llmbox-ccl"],
    "p150b": ["blackhole-p150b-sweeps"],
    "t3k": ["wormhole-t3k-sweeps"],
    "galaxy": ["wormhole-galaxy-sweeps", "lead-models-galaxy"],
}

# Lead models mesh shape → runner config mapping.
LEAD_MODELS_RUNNERS = [
    {
        "mesh_shapes": ["1x1"],
        "config": "lead-models-single-chip",
        "suite_name": "model_traced",
        "is_default": True,  # receives unmatched modules
    },
    {
        "mesh_shapes": ["1x2", "1x4", "1x8", "2x4", "4x8", "8x4", "2x16", "16x2"],
        "config": "lead-models-galaxy",
        "suite_name": "model_traced",
        "galaxy_jobs": 3,  # split into N parallel jobs instead of fixed batch_size
    },
]

# Sweep name → (run_type, schedule_cron) for detection.
_SWEEP_TYPES = {
    "ALL SWEEPS (Lead Models)": "lead_models",
    "ALL SWEEPS (Model Traced)": "model_traced",
    "ALL SWEEPS (Comprehensive)": "comprehensive",
    "ALL SWEEPS (Nightly)": "nightly",
}
_SCHEDULE_TYPES = {
    "0 2 * * *": "lead_models",
    "0 3 * * *": "model_traced",
    "0 4 * * 3,6": "comprehensive",
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def chunk_modules(items, size):
    """Split modules into batches of specified size."""
    return [",".join(items[i : i + size]) for i in range(0, len(items), size)] if items else []


def _get_runner(name):
    """Look up a runner config by test_group_name."""
    return dict(RUNNER_CONFIGS[name])


def _runner_for_hardware_group(hardware_group):
    """Map a hardware tuple (from parse_hardware_suffix) to a runner config."""
    if hardware_group is None:
        return _get_runner("wormhole-n150-sweeps")

    board_type, device_series, card_count = hardware_group

    if board_type == "blackhole" or device_series == "p150b":
        return _get_runner("blackhole-p150b-sweeps")
    if device_series == "tt_galaxy_wh":
        return _get_runner("wormhole-galaxy-sweeps")
    if device_series == "n300" and card_count == 4:
        return _get_runner("wormhole-t3k-sweeps")
    if device_series == "n300":
        return _get_runner("wormhole-n300-sweeps")
    return _get_runner("wormhole-n150-sweeps")


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


# ── Matrix computation per run type ─────────────────────────────────────────


def compute_lead_models_matrix(modules, batch_size):
    """Compute matrix for lead models with mesh-aware runner assignment."""
    # Group modules by mesh shape, then by hardware suffix, then unmatched.
    mesh_modules = defaultdict(list)
    hw_modules = defaultdict(list)
    unmatched = []

    for module in modules:
        mesh = get_mesh_shape_string(module)
        if mesh:
            mesh_modules[mesh].append(module)
        else:
            hw = parse_hardware_suffix(module)
            (hw_modules[hw] if hw else unmatched).append(module)

    configured_shapes = {s for r in LEAD_MODELS_RUNNERS for s in r["mesh_shapes"]}
    for mesh in mesh_modules:
        if mesh not in configured_shapes:
            print(f"Warning: Mesh shape '{mesh}' has no runner config, modules will be skipped", file=sys.stderr)

    include_entries = []
    batches = []
    log_groups = []

    for runner_def in LEAD_MODELS_RUNNERS:
        runner_config = _get_runner(runner_def["config"])
        is_galaxy = "galaxy" in runner_def["config"]

        # Collect modules for this runner's mesh shapes
        runner_modules = []
        for mesh in runner_def["mesh_shapes"]:
            runner_modules.extend(mesh_modules.get(mesh, []))

        # Route hardware-grouped modules
        for hw_group, mods in hw_modules.items():
            _, device_series, card_count = hw_group
            wants_galaxy = device_series == "tt_galaxy_wh" or card_count > 1
            if wants_galaxy == is_galaxy:
                runner_modules.extend(mods)

        # Unmatched modules go to the default runner
        if runner_def.get("is_default"):
            runner_modules.extend(unmatched)

        if not runner_modules:
            continue

        base = sorted(set(strip_grouping_suffix(m) for m in runner_modules))

        # Galaxy: split into N parallel jobs; standard: use batch_size
        galaxy_jobs = runner_def.get("galaxy_jobs")
        if galaxy_jobs:
            size = max(1, -(-len(base) // galaxy_jobs))
            runner_batches = chunk_modules(base, size)
        else:
            runner_batches = chunk_modules(base, batch_size)

        batches.extend(runner_batches)
        mesh_label = "+".join(runner_def["mesh_shapes"])
        include_entries.extend(_build_entries(runner_config, runner_batches, mesh_label, runner_def["suite_name"]))
        log_groups.append((mesh_label, runner_modules))

    # Log
    for mesh, mods in sorted(mesh_modules.items()):
        log_groups.append((f"mesh {mesh}", mods))
    for hw, mods in sorted(hw_modules.items()):
        log_groups.append((f"hardware {_hw_label(hw)}", mods))
    if unmatched:
        log_groups.append(("no grouping suffix (default runner)", unmatched))
    _log_module_groups("Lead models run", modules, log_groups)

    return include_entries, batches


def compute_model_traced_matrix(modules, batch_size, suite_name):
    """Compute matrix for model_traced runs using hardware-grouped vector files."""
    hw_modules = defaultdict(list)
    unmatched = []

    for module in modules:
        hw = parse_hardware_suffix(module)
        (hw_modules[hw] if hw else unmatched).append(module)

    include_entries = []
    batches = []
    log_groups = []

    grouped = sorted(hw_modules.items(), key=lambda x: x[0])
    if unmatched:
        grouped.append((None, unmatched))

    for hw_group, mods in grouped:
        base = sorted(set(strip_grouping_suffix(m) for m in mods))
        runner_config = _runner_for_hardware_group(hw_group)
        label = _hw_label(hw_group)
        runner_batches = chunk_modules(base, batch_size)
        batches.extend(runner_batches)
        include_entries.extend(_build_entries(runner_config, runner_batches, label, suite_name))
        log_groups.append((f"hardware {label}", mods))

    _log_module_groups("Model traced run", modules, log_groups)
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

    modules = sorted([f.stem for f in vectors_path.glob("*.json")])
    if not modules:
        print(f"Error: No vector JSON files found in {vectors_dir}", file=sys.stderr)
        sys.exit(1)

    # Detect run type: explicit sweep name takes precedence, then schedule cron
    run_type = _SWEEP_TYPES.get(sweep_name) or _SCHEDULE_TYPES.get(schedule_expr, "nightly")

    # Batch size: smaller for comprehensive or device-perf runs (must stay under 256 batches)
    device_perf_enabled = event_name == "schedule" or measure_device_perf == "true"
    batch_size = 3 if (run_type == "comprehensive" or device_perf_enabled) else 10

    # Compute matrix
    ccl_batches = []
    if run_type == "lead_models":
        include_entries, batches = compute_lead_models_matrix(modules, batch_size)
    elif run_type == "model_traced":
        include_entries, batches = compute_model_traced_matrix(modules, batch_size, "model_traced")
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
