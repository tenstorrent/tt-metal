#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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

from tests.sweep_framework.framework.constants import normalize_hardware_group
from tests.sweep_framework.framework.vector_metadata import normalize_mesh_shape

HardwareGroup = tuple[str, str, int]
MeshShape = tuple[int, int]

DEFAULT_PROFILE_ENV_VAR = "TT_SWEEP_CAPABILITY_PROFILE"
DEFAULT_PROFILES_PATH = pathlib.Path(__file__).with_name("execution_capability_profiles.yaml")


@dataclass(frozen=True)
class ExecutionCapabilityProfile:
    name: str
    host_group: HardwareGroup
    allowed_hardware_groups: frozenset[HardwareGroup]
    allowed_mesh_shapes: frozenset[MeshShape]


def _normalize_hardware_spec(value: Any) -> HardwareGroup | None:
    if isinstance(value, dict):
        return normalize_hardware_group(value.get("board_type"), value.get("device_series"), value.get("card_count"))
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return normalize_hardware_group(value[0], value[1], value[2])
    return None


def load_execution_capability_profiles(
    profiles_path: pathlib.Path | None = None,
) -> dict[str, ExecutionCapabilityProfile]:
    path = profiles_path or DEFAULT_PROFILES_PATH
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    profiles: dict[str, ExecutionCapabilityProfile] = {}
    for name, raw_profile in (data.get("profiles") or {}).items():
        host_group = normalize_hardware_group(
            raw_profile.get("board_type"),
            raw_profile.get("device_series"),
            raw_profile.get("card_count"),
        )
        allowed_hardware_groups = frozenset(
            hardware_group
            for hardware_group in (_normalize_hardware_spec(group) for group in raw_profile.get("hardware_groups", []))
            if hardware_group is not None
        )
        allowed_mesh_shapes = frozenset(
            mesh_shape
            for mesh_shape in (normalize_mesh_shape(shape) for shape in raw_profile.get("mesh_shapes", []))
            if mesh_shape is not None
        )
        profiles[name] = ExecutionCapabilityProfile(
            name=name,
            host_group=host_group,
            allowed_hardware_groups=allowed_hardware_groups,
            allowed_mesh_shapes=allowed_mesh_shapes,
        )
    return profiles


def _import_get_machine_info():
    model_tracer_path = pathlib.Path(__file__).resolve().parents[3] / "model_tracer"
    model_tracer_path_str = str(model_tracer_path)
    if model_tracer_path_str not in sys.path:
        sys.path.insert(0, model_tracer_path_str)

    from generic_ops_tracer import get_machine_info  # pyright: ignore[reportMissingImports]

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

    host_group = detect_host_hardware_group()
    if host_group is None:
        raise RuntimeError("Could not detect host hardware group for execution capability profile selection.")

    matching = [profile for profile in loaded_profiles.values() if profile.host_group == host_group]
    if not matching:
        raise RuntimeError(f"No execution capability profile matches detected host group {host_group}.")
    if len(matching) > 1:
        names = ", ".join(sorted(profile.name for profile in matching))
        raise RuntimeError(f"Multiple execution capability profiles match detected host group {host_group}: {names}")
    return matching[0]
