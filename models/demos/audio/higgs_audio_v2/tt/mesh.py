# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import ttnn

HIGGS_AUDIO_MESH_SHAPE_ENV = "HIGGS_AUDIO_MESH_SHAPE"


def parse_mesh_shape(mesh_shape=None) -> ttnn.MeshShape:
    if mesh_shape is None:
        return ttnn.MeshShape(1, 1)
    if isinstance(mesh_shape, ttnn.MeshShape):
        return mesh_shape
    if isinstance(mesh_shape, str):
        stripped = mesh_shape.strip().lower()
        if not stripped:
            return ttnn.MeshShape(1, 1)
        parts = [part for part in stripped.replace(",", "x").replace(" ", "x").split("x") if part]
        if len(parts) != 2:
            raise ValueError(f"Expected mesh shape as '<rows>x<cols>', got {mesh_shape!r}")
        return ttnn.MeshShape(int(parts[0]), int(parts[1]))
    raise TypeError(f"Unsupported mesh shape type: {type(mesh_shape)!r}")


def format_mesh_shape(mesh_shape) -> str:
    shape = parse_mesh_shape(mesh_shape)
    rows, cols = list(shape)
    return f"{rows}x{cols}"


def resolve_mesh_shape(mesh_shape=None) -> ttnn.MeshShape:
    if mesh_shape is not None:
        return parse_mesh_shape(mesh_shape)
    value = os.environ.get(HIGGS_AUDIO_MESH_SHAPE_ENV)
    if value:
        return parse_mesh_shape(value)
    return ttnn.MeshShape(1, 1)


def required_device_count(mesh_shape) -> int:
    rows, cols = list(parse_mesh_shape(mesh_shape))
    return rows * cols


def visible_device_count() -> int:
    device_ids = ttnn.get_device_ids()
    return len(device_ids) if device_ids else 0


def has_enough_devices(mesh_shape) -> bool:
    return visible_device_count() >= required_device_count(mesh_shape)


def open_higgs_mesh_device(
    mesh_shape=None,
    **kwargs,
):
    resolved_shape = resolve_mesh_shape(mesh_shape)
    required_devices = required_device_count(resolved_shape)
    available_devices = visible_device_count()
    if available_devices < required_devices:
        raise RuntimeError(
            f"Higgs Audio requested mesh {format_mesh_shape(resolved_shape)} "
            f"({required_devices} devices), but only {available_devices} devices are visible."
        )
    if required_devices > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(mesh_shape=resolved_shape, **kwargs)


def close_higgs_mesh_device(mesh_device) -> None:
    if mesh_device is not None:
        reset_fabric = mesh_device.get_num_devices() > 1
        ttnn.close_mesh_device(mesh_device)
        if reset_fabric:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
