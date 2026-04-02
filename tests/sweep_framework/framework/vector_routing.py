#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.sweep_framework.framework.constants import normalize_hardware_group
from tests.sweep_framework.framework.vector_metadata import normalize_mesh_shape

HardwareGroup = tuple[str, str, int]
MeshShape = tuple[int, int]

EXPORT_MANIFEST_NAME = "export_manifest.json"


@dataclass(frozen=True)
class ManifestRoutingEntry:
    module_name: str
    base_module_name: str
    grouping_kind: str
    hardware_group: HardwareGroup | None
    mesh_shapes: tuple[MeshShape, ...]
    suite_names: tuple[str, ...]
    trace_ids: tuple[int, ...]


@dataclass(frozen=True)
class ManifestFileEntry:
    module_name: str
    base_module_name: str
    file_path: Path
    grouping_kind: str
    hardware_group: HardwareGroup | None
    mesh_shapes: tuple[MeshShape, ...]
    suite_names: tuple[str, ...]
    trace_ids: tuple[int, ...]


# Runner config registry.
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
}

# Maps GitHub Actions output key to test_group_names for per-hw matrix splitting.
HW_GROUP_MATRIX_KEYS = {
    "n150": ["wormhole-n150-sweeps"],
    "n300": ["wormhole-n300-sweeps", "n300-llmbox-ccl"],
    "p150b": ["blackhole-p150b-sweeps"],
    "t3k": ["wormhole-t3k-sweeps"],
    "galaxy": ["wormhole-galaxy-sweeps"],
}

# Runner test group -> execution capability profile mapping.
CAPABILITY_PROFILE_BY_TEST_GROUP = {
    "wormhole-n150-sweeps": "wormhole_n150_host",
    "wormhole-n300-sweeps": "wormhole_n300_1c_host",
    "n300-llmbox-ccl": "wormhole_t3k_host",
    "blackhole-p150b-sweeps": "blackhole_p150b_host",
    "wormhole-t3k-sweeps": "wormhole_t3k_host",
    "wormhole-galaxy-sweeps": "wormhole_galaxy_host",
}

# Lead-model runs may pin specific runner groups differently from default routing.
RUNNER_OVERRIDES_BY_RUN_TYPE = {
    "lead_models": {
        "wormhole-galaxy-sweeps": {
            "runs_on": "g04glx03",
            "runner_label": "g04glx03",
            "tt_smi_cmd": "tt-smi -r",
        }
    }
}


def normalize_manifest_hardware_group(value: Any) -> HardwareGroup | None:
    if not isinstance(value, dict):
        return None
    return normalize_hardware_group(value.get("board_type"), value.get("device_series"), value.get("card_count"))


def normalize_manifest_mesh_shapes(value: Any) -> tuple[MeshShape, ...]:
    if not isinstance(value, list):
        return ()
    mesh_shapes = set()
    for mesh_shape in value:
        normalized = normalize_mesh_shape(mesh_shape)
        if normalized is not None:
            mesh_shapes.add(normalized)
    return tuple(sorted(mesh_shapes))


def normalize_manifest_trace_ids(values: Any) -> tuple[int, ...]:
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


def manifest_entry_from_raw(raw_entry: dict[str, Any], *, strict: bool = True) -> ManifestRoutingEntry | None:
    module_name = str(raw_entry.get("module_name") or "").strip()
    base_module_name = str(raw_entry.get("base_module_name") or "").strip()
    if not module_name or not base_module_name:
        if strict:
            raise RuntimeError(f"Manifest entry is missing module naming fields: {raw_entry}")
        return None

    suite_names = tuple(sorted(str(name) for name in raw_entry.get("suite_names", []) if str(name).strip()))
    return ManifestRoutingEntry(
        module_name=module_name,
        base_module_name=base_module_name,
        grouping_kind=str(raw_entry.get("grouping_kind") or "ungrouped"),
        hardware_group=normalize_manifest_hardware_group(raw_entry.get("hardware_group")),
        mesh_shapes=normalize_manifest_mesh_shapes(raw_entry.get("mesh_shapes")),
        suite_names=suite_names,
        trace_ids=normalize_manifest_trace_ids(raw_entry.get("trace_ids")),
    )


def manifest_file_entry_from_raw(
    raw_entry: dict[str, Any],
    *,
    repo_root: Path,
    strict: bool = False,
) -> ManifestFileEntry | None:
    routing_entry = manifest_entry_from_raw(raw_entry, strict=strict)
    if routing_entry is None:
        return None

    file_path_str = str(raw_entry.get("file_path") or "").strip()
    if not file_path_str:
        if strict:
            raise RuntimeError(f"Manifest entry is missing file_path: {raw_entry}")
        return None

    file_path = Path(file_path_str)
    if not file_path.is_absolute():
        file_path = repo_root / file_path

    return ManifestFileEntry(
        module_name=routing_entry.module_name,
        base_module_name=routing_entry.base_module_name,
        file_path=file_path,
        grouping_kind=routing_entry.grouping_kind,
        hardware_group=routing_entry.hardware_group,
        mesh_shapes=routing_entry.mesh_shapes,
        suite_names=routing_entry.suite_names,
        trace_ids=routing_entry.trace_ids,
    )


def load_manifest_raw_files(manifest_path: Path) -> list[dict[str, Any]]:
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
    return [raw_entry for raw_entry in raw_files if isinstance(raw_entry, dict)]


def load_manifest_routing_entries(manifest_path: Path, *, strict: bool = True) -> list[ManifestRoutingEntry]:
    entries = [
        manifest_entry_from_raw(raw_entry, strict=strict) for raw_entry in load_manifest_raw_files(manifest_path)
    ]
    filtered = [entry for entry in entries if entry is not None]
    if strict and not filtered:
        raise RuntimeError(f"Export manifest {manifest_path} does not contain any planning entries.")
    return filtered


def load_manifest_file_entries(
    manifest_path: Path,
    *,
    repo_root: Path,
    strict: bool = False,
) -> list[ManifestFileEntry]:
    entries = [
        manifest_file_entry_from_raw(raw_entry, repo_root=repo_root, strict=strict)
        for raw_entry in load_manifest_raw_files(manifest_path)
    ]
    filtered = [entry for entry in entries if entry is not None]
    if strict and not filtered:
        raise RuntimeError(f"Export manifest {manifest_path} does not contain any file entries.")
    return filtered


def get_runner_config(name: str) -> dict[str, Any]:
    """Look up a runner config by test_group_name."""
    return dict(RUNNER_CONFIGS[name])


def runner_for_hardware_group(hardware_group: HardwareGroup | None) -> dict[str, Any]:
    """Map normalized manifest hardware metadata to a runner config."""
    if hardware_group is None:
        return get_runner_config("wormhole-n150-sweeps")

    board_type, device_series, card_count = hardware_group
    if board_type == "blackhole" or device_series == "p150b":
        return get_runner_config("blackhole-p150b-sweeps")
    if device_series == "tt_galaxy_wh":
        return get_runner_config("wormhole-galaxy-sweeps")
    if device_series == "n300" and card_count == 4:
        return get_runner_config("wormhole-t3k-sweeps")
    if device_series == "n300":
        return get_runner_config("wormhole-n300-sweeps")
    return get_runner_config("wormhole-n150-sweeps")


def is_manifest_entry_eligible(
    *,
    grouping_kind: str,
    hardware_group: HardwareGroup | None,
    mesh_shapes: tuple[MeshShape, ...],
    profile: Any,
) -> bool:
    # grouping_kind is authoritative for runtime selection. Non-grouping
    # metadata is retained in the manifest for observability/debugging.
    if grouping_kind == "hw":
        if hardware_group and hardware_group not in profile.allowed_hardware_groups:
            return False
        return True
    if grouping_kind == "mesh":
        if mesh_shapes and not set(mesh_shapes).intersection(profile.allowed_mesh_shapes):
            return False
        return True

    # Fallback for legacy/ungrouped entries that may carry both hints.
    if hardware_group and hardware_group not in profile.allowed_hardware_groups:
        return False
    if mesh_shapes and not set(mesh_shapes).intersection(profile.allowed_mesh_shapes):
        return False
    return True
