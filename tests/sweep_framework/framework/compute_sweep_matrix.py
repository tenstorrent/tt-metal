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
from pathlib import Path

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from tests.sweep_framework.framework.vector_routing import (
    CAPABILITY_PROFILE_BY_TEST_GROUP,
    EXPORT_MANIFEST_NAME,
    HW_GROUP_MATRIX_KEYS,
    ManifestRoutingEntry,
    get_runner_config,
    manifest_entry_from_raw,
    runner_for_hardware_group,
)

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


def load_manifest_entries(vectors_path: Path) -> list[ManifestRoutingEntry]:
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

    entries = [
        manifest_entry_from_raw(raw_entry, strict=True) for raw_entry in raw_files if isinstance(raw_entry, dict)
    ]
    if not entries:
        raise RuntimeError(f"Export manifest {manifest_path} does not contain any planning entries.")
    return [entry for entry in entries if entry is not None]


def _hw_label(hardware_group):
    """Readable label like 'wormhole/n300/4c' or 'default'."""
    if hardware_group is None:
        return "default"
    board_type, device_series, card_count = hardware_group
    return f"{board_type}/{device_series}/{card_count}c"


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


def compute_lead_models_matrix(entries: list[ManifestRoutingEntry], batch_size: int):
    """Compute matrix for lead models using model-traced routing + galaxy sharding."""
    return compute_model_traced_matrix(
        entries,
        batch_size,
        "model_traced",
        galaxy_shards=3,
        run_label="Lead models run",
        runner_overrides_by_group={
            "wormhole-galaxy-sweeps": {
                "runs_on": "g04glx03",
                "runner_label": "g04glx03",
                "tt_smi_cmd": "tt-smi -r",
            }
        },
    )


def compute_model_traced_matrix(
    entries: list[ManifestRoutingEntry],
    batch_size: int,
    suite_name: str,
    *,
    galaxy_shards: int | None = None,
    run_label: str = "Model traced run",
    runner_overrides_by_group: dict[str, dict[str, object]] | None = None,
):
    """Compute matrix for model_traced runs using manifest hardware routing."""
    hw_modules: dict[tuple[str, str, int] | None, list[ManifestRoutingEntry]] = defaultdict(list)
    for entry in entries:
        hw_modules[entry.hardware_group].append(entry)

    include_entries = []
    batches = []
    log_groups = []

    grouped = sorted(hw_modules.items(), key=lambda item: (item[0] is None, item[0]))

    for hw_group, grouped_entries in grouped:
        base = sorted({entry.base_module_name for entry in grouped_entries})
        runner_config = runner_for_hardware_group(hw_group)
        if runner_overrides_by_group:
            group_name = runner_config.get("test_group_name")
            override = runner_overrides_by_group.get(group_name) if isinstance(group_name, str) else None
            if override:
                runner_config.update(override)
        label = _hw_label(hw_group)
        shard_count = galaxy_shards if runner_config.get("test_group_name") == "wormhole-galaxy-sweeps" else None
        if shard_count and shard_count > 0:
            shard_size = max(1, -(-len(base) // shard_count))
            runner_batches = chunk_modules(base, shard_size)
        else:
            runner_batches = chunk_modules(base, batch_size)
        batches.extend(runner_batches)
        include_entries.extend(_build_entries(runner_config, runner_batches, label, suite_name))
        log_groups.append((f"hardware {label}", grouped_entries))

    _log_module_groups(run_label, entries, log_groups)
    return include_entries, batches


def compute_standard_matrix(entries: list[ManifestRoutingEntry], batch_size: int, suite_name: str | None):
    """Compute matrix for nightly/comprehensive runs."""
    base_modules = sorted({entry.base_module_name for entry in entries})
    ccl_modules = [m for m in base_modules if m.startswith("ccl.")]
    regular_modules = [m for m in base_modules if not m.startswith("ccl.")]

    # Keep CCL modules off the default runner path; they get their own N300 lane below.
    regular_batches = chunk_modules(regular_modules, batch_size)
    ccl_batches = chunk_modules(ccl_modules, batch_size)

    n150_config = get_runner_config("wormhole-n150-sweeps")
    include_entries = _build_entries(n150_config, regular_batches, "", suite_name)

    if ccl_batches:
        ccl_config = get_runner_config("n300-llmbox-ccl")
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
