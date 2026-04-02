#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Compute the sweep test matrix for GitHub Actions CI.

This script reads the generated export manifest and produces a matrix
configuration that maps test modules to appropriate hardware runners.

Environment Variables (from GitHub Actions context):
- GITHUB_EVENT_SCHEDULE: Cron schedule expression
- GITHUB_EVENT_NAME: Event type (schedule, workflow_dispatch)
- SWEEP_NAME: Selected sweep type from workflow_dispatch
- MEASURE_DEVICE_PERF: Whether device performance measurement is enabled
- VECTORS_DIR: Directory containing exported vectors and export_manifest.json

Output:
Prints GitHub Actions output lines to stdout (matrix + per-hw matrices).
"""

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from tests.sweep_framework.framework.constants import normalize_hardware_group

HardwareGroup = tuple[str, str, int]
MeshShape = tuple[int, int]
EXPORT_MANIFEST_NAME = "export_manifest.json"


@dataclass(frozen=True)
class ManifestEntry:
    module_name: str
    base_module_name: str
    grouping_kind: str
    hardware_group: HardwareGroup | None
    mesh_shapes: tuple[MeshShape, ...]
    suite_names: tuple[str, ...]
    trace_ids: tuple[int, ...]


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

# Runner test group -> execution capability profile mapping.
CAPABILITY_PROFILE_BY_TEST_GROUP = {
    "wormhole-n150-sweeps": "wormhole_n150_host",
    "lead-models-single-chip": "wormhole_n150_host",
    "wormhole-n300-sweeps": "wormhole_n300_1c_host",
    "n300-llmbox-ccl": "wormhole_t3k_host",
    "blackhole-p150b-sweeps": "blackhole_p150b_host",
    "wormhole-t3k-sweeps": "wormhole_t3k_host",
    "wormhole-galaxy-sweeps": "wormhole_galaxy_host",
    "lead-models-galaxy": "wormhole_galaxy_host",
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


def _normalize_mesh_shape(value: Any) -> MeshShape | None:
    """Convert manifest mesh metadata to a canonical (rows, cols) tuple."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return None
    return None


def _normalize_trace_ids(values: Any) -> tuple[int, ...]:
    """Convert manifest trace IDs to a sorted tuple of ints."""
    if values is None:
        return ()
    if not isinstance(values, list):
        values = [values]

    trace_ids = set()
    for value in values:
        try:
            trace_ids.add(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(trace_ids))


def _manifest_entry_from_raw(raw_entry: dict[str, Any]) -> ManifestEntry:
    """Parse a manifest file entry into a typed planning record."""
    module_name = str(raw_entry.get("module_name") or "").strip()
    base_module_name = str(raw_entry.get("base_module_name") or "").strip()
    if not module_name or not base_module_name:
        raise RuntimeError(f"Manifest entry is missing module naming fields: {raw_entry}")

    raw_hardware = raw_entry.get("hardware_group")
    hardware_group = None
    if isinstance(raw_hardware, dict):
        hardware_group = normalize_hardware_group(
            raw_hardware.get("board_type"),
            raw_hardware.get("device_series"),
            raw_hardware.get("card_count"),
        )

    mesh_shapes = tuple(
        sorted(
            {
                mesh_shape
                for mesh_shape in (_normalize_mesh_shape(value) for value in raw_entry.get("mesh_shapes", []))
                if mesh_shape is not None
            }
        )
    )
    suite_names = tuple(sorted(str(name) for name in raw_entry.get("suite_names", []) if str(name).strip()))
    return ManifestEntry(
        module_name=module_name,
        base_module_name=base_module_name,
        grouping_kind=str(raw_entry.get("grouping_kind") or "ungrouped"),
        hardware_group=hardware_group,
        mesh_shapes=mesh_shapes,
        suite_names=suite_names,
        trace_ids=_normalize_trace_ids(raw_entry.get("trace_ids")),
    )


def load_manifest_entries(vectors_path: Path) -> list[ManifestEntry]:
    """Load planning entries from the generated export manifest."""
    manifest_path = vectors_path / EXPORT_MANIFEST_NAME
    if not manifest_path.exists():
        raise RuntimeError(f"Export manifest not found: {manifest_path}")

    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read export manifest {manifest_path}: {exc}") from exc

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise RuntimeError(f"Export manifest {manifest_path} is missing a valid 'files' list.")

    entries = [_manifest_entry_from_raw(raw_entry) for raw_entry in raw_files if isinstance(raw_entry, dict)]
    if not entries:
        raise RuntimeError(f"Export manifest {manifest_path} does not contain any planning entries.")
    return entries


def _get_runner(name):
    """Look up a runner config by test_group_name."""
    return dict(RUNNER_CONFIGS[name])


def _runner_for_hardware_group(hardware_group):
    """Map normalized manifest hardware metadata to a runner config."""
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


def _mesh_shape_string(mesh_shape: MeshShape) -> str:
    return f"{mesh_shape[0]}x{mesh_shape[1]}"


def _log_module_groups(header, entries, groups):
    """Print a summary of module grouping to stderr."""
    total_base = len({entry.base_module_name for entry in entries})
    print(
        f"{header}: {len(entries)} vector files ({total_base} unique modules), "
        f"{sum(len(group_entries) for _, group_entries in groups)} routed entries",
        file=sys.stderr,
    )
    for label, entries in groups:
        unique = len({entry.base_module_name for entry in entries})
        print(f"  {label}: {len(entries)} vectors ({unique} unique modules)", file=sys.stderr)


def _build_entries(runner_config, batches, batch_display_prefix, suite_name):
    """Create matrix include entries for a set of batches using a runner config."""
    test_group_name = runner_config.get("test_group_name")
    capability_profile = CAPABILITY_PROFILE_BY_TEST_GROUP.get(test_group_name)
    if capability_profile is None:
        raise RuntimeError(f"Missing capability profile mapping for test group '{test_group_name}'")

    return [
        {
            **runner_config,
            "module_selector": batch,
            "batch_display": f"{batch_display_prefix}:{batch}" if batch_display_prefix else batch,
            "suite_name": suite_name,
            "capability_profile": capability_profile,
        }
        for batch in batches
    ]


# ── Matrix computation per run type ─────────────────────────────────────────


def compute_lead_models_matrix(entries: list[ManifestEntry], batch_size: int):
    """Compute matrix for lead models with manifest-declared mesh routing."""
    configured_shapes = {shape for runner in LEAD_MODELS_RUNNERS for shape in runner["mesh_shapes"]}
    runner_modules: dict[str, list[str]] = defaultdict(list)
    log_groups: dict[str, list[ManifestEntry]] = defaultdict(list)

    for entry in entries:
        mesh_labels = sorted({_mesh_shape_string(mesh_shape) for mesh_shape in entry.mesh_shapes})
        matched_configs = []

        # Planner follows the same contract as runtime selection: grouping_kind
        # is authoritative, while other manifest metadata is descriptive only.
        if entry.grouping_kind == "mesh" and mesh_labels:
            unknown_shapes = [mesh for mesh in mesh_labels if mesh not in configured_shapes]
            for mesh in unknown_shapes:
                print(f"Warning: Mesh shape '{mesh}' has no runner config, modules will be skipped", file=sys.stderr)
            for runner_def in LEAD_MODELS_RUNNERS:
                if set(mesh_labels).intersection(runner_def["mesh_shapes"]):
                    matched_configs.append(runner_def["config"])
            log_label = f"mesh {'+'.join(mesh_labels)}"
        elif entry.hardware_group is not None:
            _, device_series, card_count = entry.hardware_group
            wants_galaxy = device_series == "tt_galaxy_wh" or card_count > 1
            matched_configs.append("lead-models-galaxy" if wants_galaxy else "lead-models-single-chip")
            log_label = f"hardware {_hw_label(entry.hardware_group)}"
        else:
            default_runner = next(runner_def for runner_def in LEAD_MODELS_RUNNERS if runner_def.get("is_default"))
            matched_configs.append(default_runner["config"])
            log_label = "no grouping metadata (default runner)"

        if not matched_configs:
            continue

        for config_name in sorted(set(matched_configs)):
            runner_modules[config_name].append(entry.base_module_name)
        log_groups[log_label].append(entry)

    include_entries = []
    batches = []

    for runner_def in LEAD_MODELS_RUNNERS:
        runner_config = _get_runner(runner_def["config"])
        runner_bases = sorted(set(runner_modules.get(runner_def["config"], [])))
        if not runner_bases:
            continue

        galaxy_jobs = runner_def.get("galaxy_jobs")
        if galaxy_jobs:
            size = max(1, -(-len(runner_bases) // galaxy_jobs))
            runner_batches = chunk_modules(runner_bases, size)
        else:
            runner_batches = chunk_modules(runner_bases, batch_size)

        batches.extend(runner_batches)
        mesh_label = "+".join(runner_def["mesh_shapes"])
        include_entries.extend(_build_entries(runner_config, runner_batches, mesh_label, runner_def["suite_name"]))
    _log_module_groups("Lead models run", entries, sorted(log_groups.items()))

    return include_entries, batches


def compute_model_traced_matrix(entries: list[ManifestEntry], batch_size: int, suite_name: str):
    """Compute matrix for model_traced runs using manifest hardware routing."""
    hw_modules: dict[HardwareGroup | None, list[ManifestEntry]] = defaultdict(list)
    for entry in entries:
        hw_modules[entry.hardware_group].append(entry)

    include_entries = []
    batches = []
    log_groups = []

    grouped = sorted(hw_modules.items(), key=lambda item: (item[0] is None, item[0]))

    for hw_group, grouped_entries in grouped:
        base = sorted({entry.base_module_name for entry in grouped_entries})
        runner_config = _runner_for_hardware_group(hw_group)
        label = _hw_label(hw_group)
        runner_batches = chunk_modules(base, batch_size)
        batches.extend(runner_batches)
        include_entries.extend(_build_entries(runner_config, runner_batches, label, suite_name))
        log_groups.append((f"hardware {label}", grouped_entries))

    _log_module_groups("Model traced run", entries, log_groups)
    return include_entries, batches


def compute_standard_matrix(entries: list[ManifestEntry], batch_size: int, suite_name: str | None):
    """Compute matrix for nightly/comprehensive runs."""
    base_modules = sorted({entry.base_module_name for entry in entries})
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

    try:
        manifest_entries = load_manifest_entries(vectors_path)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    modules = sorted(entry.module_name for entry in manifest_entries)

    # Detect run type: explicit sweep name takes precedence, then schedule cron
    run_type = _SWEEP_TYPES.get(sweep_name) or _SCHEDULE_TYPES.get(schedule_expr, "nightly")

    # Batch size: smaller for comprehensive or device-perf runs (must stay under 256 batches)
    device_perf_enabled = event_name == "schedule" or measure_device_perf == "true"
    batch_size = 3 if (run_type == "comprehensive" or device_perf_enabled) else 10

    # Compute matrix
    ccl_batches = []
    if run_type == "lead_models":
        include_entries, batches = compute_lead_models_matrix(manifest_entries, batch_size)
    elif run_type == "model_traced":
        include_entries, batches = compute_model_traced_matrix(manifest_entries, batch_size, "model_traced")
    else:
        suite_name = None if run_type == "comprehensive" else run_type  # "nightly" or None
        include_entries, batches, ccl_batches = compute_standard_matrix(manifest_entries, batch_size, suite_name)

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
