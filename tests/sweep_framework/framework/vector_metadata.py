#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from typing import Any

from model_tracer.mesh_metadata import infer_mesh_shape

from tests.sweep_framework.framework.constants import normalize_hardware_group

HardwareGroup = tuple[str, str, int]
MeshShape = tuple[int, int]


def normalize_mesh_shape(value: Any) -> MeshShape | None:
    """Normalize mixed mesh-shape metadata into a canonical tuple."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            value = parsed
        elif "x" in value.lower():
            parts = value.lower().split("x", 1)
            try:
                value = [int(parts[0]), int(parts[1])]
            except (TypeError, ValueError):
                return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return None
    return None


def normalize_mesh_shape_for_manifest(mesh_shape: Any) -> list[int] | None:
    """Normalize mesh metadata to a JSON-friendly [rows, cols] list."""
    normalized = normalize_mesh_shape(mesh_shape)
    return [normalized[0], normalized[1]] if normalized is not None else None


def extract_traced_machine_entries(vector_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize traced_machine_info to a list of dict entries."""
    traced_machine_info = vector_data.get("traced_machine_info")
    if isinstance(traced_machine_info, dict):
        return [traced_machine_info]
    if isinstance(traced_machine_info, list):
        return [entry for entry in traced_machine_info if isinstance(entry, dict)]
    return []


def get_hardware_from_vector(vector_data: dict[str, Any]) -> HardwareGroup | None:
    """Extract a normalized hardware tuple from traced_machine_info when present."""
    for entry in extract_traced_machine_entries(vector_data):
        board_type = entry.get("board_type")
        device_series = entry.get("device_series")
        card_count = entry.get("card_count")

        if isinstance(device_series, list):
            device_series = device_series[0] if device_series else None

        if not board_type and not device_series and card_count is None:
            continue

        return normalize_hardware_group(board_type, device_series, card_count)

    return None


def get_mesh_shape_from_vector(vector_data: dict[str, Any]) -> MeshShape | None:
    """Infer mesh shape from traced_machine_info when present."""
    machine_entries = extract_traced_machine_entries(vector_data)
    if not machine_entries:
        return None

    first_entry = machine_entries[0]
    direct_mesh_shape = normalize_mesh_shape(first_entry.get("mesh_device_shape"))
    if direct_mesh_shape is not None:
        return direct_mesh_shape

    for entry in machine_entries:
        tensor_placements = entry.get("tensor_placements")
        placement_mesh_shape = None
        distribution_shape = None
        if isinstance(tensor_placements, list) and tensor_placements:
            placement_mesh_shape = tensor_placements[0].get("mesh_device_shape")
            distribution_shape = tensor_placements[0].get("distribution_shape")

        mesh_shape = infer_mesh_shape(
            mesh_shape=entry.get("mesh_device_shape") or placement_mesh_shape,
            distribution_shape=distribution_shape,
            device_ids=entry.get("device_ids"),
            device_count=entry.get("device_count"),
            device_series=entry.get("device_series"),
            card_count=entry.get("card_count"),
        )
        normalized = normalize_mesh_shape(mesh_shape)
        if normalized is not None:
            return normalized

    return None
