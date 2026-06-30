# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh setup and replicated tensor helpers for Seamless M4T v2."""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
import torch
import ttnn

MESH_SHAPE_SINGLE = (1, 1)
MESH_SHAPE = (1, 4)
_NUM_DEVICES = MESH_SHAPE[0] * MESH_SHAPE[1]

_MESH_FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def mesh_num_devices(device: ttnn.Device) -> int:
    if hasattr(device, "get_num_devices"):
        return max(1, int(device.get_num_devices()))
    return 1


def get_tp(device: ttnn.Device) -> int:
    """Tensor-parallelism degree for the current mesh."""
    return mesh_num_devices(device)


def mesh_cluster_axis(device: ttnn.Device) -> int:
    """CCL cluster axis: 1 for MeshShape(1, N), 0 for MeshShape(N, 1) or single-device."""
    if mesh_num_devices(device) <= 1:
        return 0
    if hasattr(device, "shape"):
        rows, cols = int(device.shape[0]), int(device.shape[1])
        if rows == 1 and cols > 1:
            return 1
        if cols == 1 and rows > 1:
            return 0
    return 0


def mesh_replicate_mapper(device: ttnn.Device):
    if mesh_num_devices(device) > 1:
        return ttnn.ReplicateTensorToMesh(device)
    return None


def mesh_mapper(device: ttnn.Device):
    """Replicate batch-1 host tensors on every device in a multi-device mesh."""
    return mesh_replicate_mapper(device)


@contextmanager
def mesh_default_device(mesh: ttnn.Device):
    """Set ttnn default device for mask builders and mesh readback."""
    try:
        original = ttnn.GetDefaultDevice()
    except Exception:
        original = None
    ttnn.SetDefaultDevice(mesh)
    try:
        yield
    finally:
        ttnn.SetDefaultDevice(original)


DEVICE_PARAMS_FULL = {"l1_small_size": 65536, **_MESH_FABRIC}

DEVICE_PARAMS_FULL_DECODE_TRACE = {
    "l1_small_size": 65536,
    "trace_region_size": 450_000_000,
    **_MESH_FABRIC,
}

DEVICE_PARAMS_TEXT = {"l1_small_size": 32768, "num_command_queues": 2, **_MESH_FABRIC}
DEVICE_PARAMS_TEXT_SINGLE = {"l1_small_size": 32768, "num_command_queues": 2}

DEVICE_PARAMS_E2E_2CQ_GENERATE = {
    "l1_small_size": 65536,
    "trace_region_size": 450_000_000,
    "num_command_queues": 2,
    **_MESH_FABRIC,
}

DEVICE_PARAMS_E2E_2CQ_GENERATE_SINGLE = {
    "l1_small_size": 65536,
    "trace_region_size": 450_000_000,
    "num_command_queues": 2,
}


def seamless_mesh_shape_from_env() -> tuple[int, int]:
    mesh_env = os.environ.get("MESH_DEVICE")
    if mesh_env in {"P150": MESH_SHAPE_SINGLE, "BH-QB": MESH_SHAPE}:
        return {"P150": MESH_SHAPE_SINGLE, "BH-QB": MESH_SHAPE}[mesh_env]
    if "TT_MESH_WIDTH" in os.environ:
        return (1, int(os.environ["TT_MESH_WIDTH"]))
    try:
        return MESH_SHAPE if ttnn.get_num_devices() >= _NUM_DEVICES else MESH_SHAPE_SINGLE
    except Exception:
        return MESH_SHAPE_SINGLE


def _requires_bh_qb() -> bool:
    try:
        return ttnn.get_num_devices() != _NUM_DEVICES
    except Exception:
        return True


def _requires_mesh_shape(mesh_shape: tuple[int, int]) -> bool:
    try:
        return ttnn.get_num_devices() < mesh_shape[0] * mesh_shape[1]
    except Exception:
        return True


def _mesh_device_param(device_params: dict):
    return pytest.param(
        MESH_SHAPE,
        device_params,
        id="1x4",
        marks=pytest.mark.skipif(
            _requires_bh_qb(),
            reason=f"requires exactly {_NUM_DEVICES} devices (MeshShape{MESH_SHAPE})",
        ),
    )


def _mesh_device_param_for_shape(mesh_shape: tuple[int, int], device_params: dict, *, id: str):
    num_devices = mesh_shape[0] * mesh_shape[1]
    return pytest.param(
        mesh_shape,
        device_params,
        id=id,
        marks=pytest.mark.skipif(
            _requires_mesh_shape(mesh_shape),
            reason=f"requires at least {num_devices} device(s) (MeshShape{mesh_shape})",
        ),
    )


MESH_DEVICE_PARAMETRIZE_FULL = (
    "mesh_device,device_params",
    [_mesh_device_param(DEVICE_PARAMS_FULL)],
)

MESH_DEVICE_PARAMETRIZE_TEXT = (
    "mesh_device,device_params",
    [_mesh_device_param(DEVICE_PARAMS_TEXT)],
)

MESH_DEVICE_PARAMETRIZE_TEXT_SINGLE_AND_1X4 = (
    "mesh_device,device_params",
    [
        _mesh_device_param_for_shape(MESH_SHAPE_SINGLE, DEVICE_PARAMS_TEXT_SINGLE, id="1x1"),
        _mesh_device_param_for_shape(MESH_SHAPE, DEVICE_PARAMS_TEXT, id="1x4"),
    ],
)

MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE = (
    "mesh_device,device_params",
    [_mesh_device_param(DEVICE_PARAMS_E2E_2CQ_GENERATE)],
)

MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE_SINGLE_AND_1X4 = (
    "mesh_device,device_params",
    [
        _mesh_device_param_for_shape(MESH_SHAPE_SINGLE, DEVICE_PARAMS_E2E_2CQ_GENERATE_SINGLE, id="1x1"),
        _mesh_device_param_for_shape(MESH_SHAPE, DEVICE_PARAMS_E2E_2CQ_GENERATE, id="1x4"),
    ],
)


def from_torch_uint32_rm(
    device: ttnn.Device,
    t: torch.Tensor,
    *,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.int32).cpu(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
        mesh_mapper=mesh_mapper(device),
    )


def from_torch_bfloat16_tile(
    device: ttnn.Device,
    t: torch.Tensor,
    *,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Upload bf16 host data already in TILE layout."""
    host = ttnn.from_torch(
        t.to(torch.bfloat16).cpu().contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mesh_mapper(device),
    )
    return ttnn.to_device(host, device, memory_config=memory_config)


def open_seamless_mesh_device(*, enable_decode_trace: bool = False, enable_2cq: bool = False):
    """Open the Seamless M4T v2 demo mesh, defaulting to 1x1 on P150 and 1x4 on BH-QB."""
    mesh_shape_tuple = seamless_mesh_shape_from_env()
    requested_devices = mesh_shape_tuple[0] * mesh_shape_tuple[1]
    num_devices = ttnn.get_num_devices()
    if num_devices < requested_devices:
        raise RuntimeError(
            f"Seamless M4T v2 requested {requested_devices} devices (MeshShape{mesh_shape_tuple}), found {num_devices}"
        )

    if enable_decode_trace and enable_2cq:
        device_params = dict(DEVICE_PARAMS_E2E_2CQ_GENERATE)
    elif enable_decode_trace:
        device_params = dict(DEVICE_PARAMS_FULL_DECODE_TRACE)
    else:
        device_params = dict(DEVICE_PARAMS_FULL)

    if mesh_shape_tuple == MESH_SHAPE_SINGLE:
        device_params.pop("fabric_config", None)
    fabric_config = device_params.pop("fabric_config", None)
    if fabric_config is not None:
        ttnn.set_fabric_config(fabric_config)

    mesh_shape = ttnn.MeshShape(*mesh_shape_tuple)
    mesh = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
    return mesh, (int(mesh_shape[0]), int(mesh_shape[1]))
