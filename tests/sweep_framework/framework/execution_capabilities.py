#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import yaml

if __package__ in (None, ""):
    REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from tests.sweep_framework.framework.constants import normalize_hardware_group, parse_hardware_suffix, parse_mesh_suffix

HardwareGroup = tuple[str, str, int]
MeshShape = tuple[int, int]

DEFAULT_PROFILE_ENV_VAR = "TT_SWEEP_CAPABILITY_PROFILE"
DEFAULT_PROFILES_PATH = pathlib.Path(__file__).with_name("execution_capability_profiles.yaml")


@dataclass(frozen=True)
class ExecutionCapabilityProfile:
    name: str
    host_match: HardwareGroup
    can_run_hardware_groups: frozenset[HardwareGroup]
    can_run_mesh_shapes: frozenset[MeshShape]


@dataclass(frozen=True)
class VectorRequirement:
    hardware_groups: frozenset[HardwareGroup]
    mesh_shapes: frozenset[MeshShape]


@dataclass(frozen=True)
class VectorFileSummary:
    module_name: str
    file_path: pathlib.Path
    requirement: VectorRequirement
    trace_ids: frozenset[int]


def normalize_mesh_shape(mesh_shape: Any) -> MeshShape | None:
    if mesh_shape is None:
        return None
    if isinstance(mesh_shape, str):
        try:
            parsed = ast.literal_eval(mesh_shape)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            mesh_shape = parsed
        elif "x" in mesh_shape.lower():
            parts = mesh_shape.lower().split("x", 1)
            try:
                mesh_shape = [int(parts[0]), int(parts[1])]
            except (TypeError, ValueError):
                return None
    if isinstance(mesh_shape, (list, tuple)) and len(mesh_shape) == 2:
        try:
            return (int(mesh_shape[0]), int(mesh_shape[1]))
        except (TypeError, ValueError):
            return None
    return None


def normalize_trace_ids(value: Any) -> frozenset[int]:
    if value is None:
        return frozenset()
    if not isinstance(value, (list, tuple, set)):
        value = [value]

    trace_ids: set[int] = set()
    for item in value:
        try:
            trace_ids.add(int(item))
        except (TypeError, ValueError):
            continue
    return frozenset(trace_ids)


def load_execution_capability_profiles(
    profiles_path: pathlib.Path | None = None,
) -> dict[str, ExecutionCapabilityProfile]:
    path = profiles_path or DEFAULT_PROFILES_PATH
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    profiles: dict[str, ExecutionCapabilityProfile] = {}
    raw_profiles = data.get("profiles", {})
    for name, raw_profile in raw_profiles.items():
        raw_host_match = raw_profile.get("host_match", {})
        can_run = raw_profile.get("can_run", {})
        host_match = normalize_hardware_group(
            raw_host_match.get("board_type"),
            raw_host_match.get("device_series"),
            raw_host_match.get("card_count"),
        )
        hardware_groups = frozenset(
            normalize_hardware_group(
                group.get("board_type"),
                group.get("device_series"),
                group.get("card_count"),
            )
            for group in can_run.get("hardware_groups", [])
        )
        mesh_shapes = frozenset(
            normalized
            for normalized in (normalize_mesh_shape(mesh_shape) for mesh_shape in can_run.get("mesh_shapes", []))
            if normalized is not None
        )
        profiles[name] = ExecutionCapabilityProfile(
            name=name,
            host_match=host_match,
            can_run_hardware_groups=hardware_groups,
            can_run_mesh_shapes=mesh_shapes,
        )
    return profiles


def _import_get_machine_info():
    model_tracer_path = pathlib.Path(__file__).resolve().parents[3] / "model_tracer"
    model_tracer_path_str = str(model_tracer_path)
    if model_tracer_path_str not in sys.path:
        sys.path.insert(0, model_tracer_path_str)

    from generic_ops_tracer import get_machine_info

    return get_machine_info


def detect_host_hardware_group() -> HardwareGroup | None:
    try:
        get_machine_info = _import_get_machine_info()
        machine_info = get_machine_info()
        if not machine_info:
            return None
        return normalize_hardware_group(
            machine_info.get("board_type"),
            machine_info.get("device_series"),
            machine_info.get("card_count"),
        )
    except Exception:
        return None


def select_profile_for_host(
    profiles: dict[str, ExecutionCapabilityProfile],
    host_group: HardwareGroup | None = None,
) -> ExecutionCapabilityProfile:
    effective_host_group = host_group or detect_host_hardware_group()
    if effective_host_group is None:
        raise RuntimeError("Could not detect host hardware group for execution capability profile selection.")

    matching = [profile for profile in profiles.values() if profile.host_match == effective_host_group]
    if not matching:
        raise RuntimeError(f"No execution capability profile matches detected host group {effective_host_group}.")
    if len(matching) > 1:
        names = ", ".join(sorted(profile.name for profile in matching))
        raise RuntimeError(
            f"Multiple execution capability profiles match detected host group {effective_host_group}: {names}"
        )
    return matching[0]


def resolve_active_profile(
    profiles: dict[str, ExecutionCapabilityProfile] | None = None,
    profile_name: str | None = None,
    profile_env_var: str = DEFAULT_PROFILE_ENV_VAR,
) -> ExecutionCapabilityProfile:
    loaded_profiles = profiles or load_execution_capability_profiles()
    explicit_name = profile_name or os.environ.get(profile_env_var, "").strip()
    if explicit_name:
        if explicit_name not in loaded_profiles:
            available = ", ".join(sorted(loaded_profiles))
            raise RuntimeError(
                f"Unknown execution capability profile '{explicit_name}'. Available profiles: {available}"
            )
        return loaded_profiles[explicit_name]
    return select_profile_for_host(loaded_profiles)


def _extract_mesh_shape_from_entry(entry: dict[str, Any]) -> MeshShape | None:
    mesh_shape = normalize_mesh_shape(entry.get("mesh_device_shape"))
    if mesh_shape is not None:
        return mesh_shape

    placements = entry.get("tensor_placements")
    if isinstance(placements, list) and placements:
        return normalize_mesh_shape(placements[0].get("mesh_device_shape"))
    return None


def extract_traced_machine_entries(vector_data: dict[str, Any]) -> list[dict[str, Any]]:
    traced_machine_info = vector_data.get("traced_machine_info")
    if isinstance(traced_machine_info, dict):
        return [traced_machine_info]
    if isinstance(traced_machine_info, list):
        return [entry for entry in traced_machine_info if isinstance(entry, dict)]
    return []


def requirements_from_vector_data(vector_data: dict[str, Any], module_name: str | None = None) -> VectorRequirement:
    hardware_groups: set[HardwareGroup] = set()
    mesh_shapes: set[MeshShape] = set()

    if module_name:
        suffix_hardware = parse_hardware_suffix(module_name)
        suffix_mesh = parse_mesh_suffix(module_name)
        if suffix_hardware is not None:
            hardware_groups.add(suffix_hardware)
        if suffix_mesh is not None:
            mesh_shapes.add(suffix_mesh)

    for entry in extract_traced_machine_entries(vector_data):
        hardware_group = normalize_hardware_group(
            entry.get("board_type"),
            entry.get("device_series"),
            entry.get("card_count"),
        )
        if hardware_group[0] != "unknown" and hardware_group[1] != "unknown" and hardware_group[2] > 0:
            hardware_groups.add(hardware_group)
        mesh_shape = _extract_mesh_shape_from_entry(entry)
        if mesh_shape is not None:
            mesh_shapes.add(mesh_shape)

    return VectorRequirement(hardware_groups=frozenset(hardware_groups), mesh_shapes=frozenset(mesh_shapes))


def requirements_from_module_name(module_name: str) -> VectorRequirement:
    hardware_groups = set()
    mesh_shapes = set()

    hardware_group = parse_hardware_suffix(module_name)
    if hardware_group is not None:
        hardware_groups.add(hardware_group)

    mesh_shape = parse_mesh_suffix(module_name)
    if mesh_shape is not None:
        mesh_shapes.add(mesh_shape)

    return VectorRequirement(hardware_groups=frozenset(hardware_groups), mesh_shapes=frozenset(mesh_shapes))


def merge_requirements(*requirements: VectorRequirement) -> VectorRequirement:
    hardware_groups: set[HardwareGroup] = set()
    mesh_shapes: set[MeshShape] = set()
    for requirement in requirements:
        hardware_groups.update(requirement.hardware_groups)
        mesh_shapes.update(requirement.mesh_shapes)
    return VectorRequirement(hardware_groups=frozenset(hardware_groups), mesh_shapes=frozenset(mesh_shapes))


def is_requirement_eligible(requirement: VectorRequirement, profile: ExecutionCapabilityProfile) -> bool:
    if requirement.hardware_groups and not requirement.hardware_groups.intersection(profile.can_run_hardware_groups):
        return False
    if requirement.mesh_shapes and not requirement.mesh_shapes.intersection(profile.can_run_mesh_shapes):
        return False
    return True


def summarize_vector_file(file_path: pathlib.Path) -> VectorFileSummary:
    module_name = file_path.stem
    requirement = requirements_from_module_name(module_name)
    trace_ids: set[int] = set()

    with open(file_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        for suite_data in data.values():
            if not isinstance(suite_data, dict):
                continue
            for vector_data in suite_data.values():
                if not isinstance(vector_data, dict):
                    continue
                requirement = merge_requirements(requirement, requirements_from_vector_data(vector_data, module_name))
                trace_ids.update(normalize_trace_ids(vector_data.get("trace_ids")))

    return VectorFileSummary(
        module_name=module_name,
        file_path=file_path,
        requirement=requirement,
        trace_ids=frozenset(trace_ids),
    )


def load_vector_file_summaries(vectors_dir: pathlib.Path) -> list[VectorFileSummary]:
    return [summarize_vector_file(vector_file) for vector_file in sorted(vectors_dir.glob("*.json"))]


def select_eligible_vector_summaries(
    summaries: list[VectorFileSummary],
    profile: ExecutionCapabilityProfile,
) -> list[VectorFileSummary]:
    return [summary for summary in summaries if is_requirement_eligible(summary.requirement, profile)]
