# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh setup for Seamless M4T v2: shapes, fabric, pytest params, replicated uploads, demo open.

P150 (1 device): ``MeshShape(1, 1)`` — no fabric.
BH QB (4 devices): ``MeshShape(1, 4)`` — ``FABRIC_1D`` for head-parallel CCL.

Tests::

    @pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_FULL, indirect=["mesh_device", "device_params"])
    def test_foo(mesh_device, device_params, ...):
        with mesh_default_device(mesh_device):
            ...
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import pytest
import torch
import ttnn


# ---------------------------------------------------------------------------
# Mesh runtime (replication, CCL axis)
# ---------------------------------------------------------------------------


def mesh_num_devices(device: ttnn.Device) -> int:
    if hasattr(device, "get_num_devices"):
        return max(1, int(device.get_num_devices()))
    return 1


def mesh_cluster_axis(device: ttnn.Device) -> int:
    """CCL cluster axis: ``1`` for ``MeshShape(1, N)``, ``0`` for ``MeshShape(N, 1)``."""
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
    """Set ``ttnn`` default device for mask builders / mesh readback (replaces tests/conftest)."""
    original = None
    try:
        original = ttnn.GetDefaultDevice()
    except Exception:
        original = None
    ttnn.SetDefaultDevice(mesh)
    try:
        yield
    finally:
        ttnn.SetDefaultDevice(original)


# ---------------------------------------------------------------------------
# Pytest mesh shape + device_params (P150 vs BH QB)
# ---------------------------------------------------------------------------

MESH_SHAPE_P150 = (1, 1)
MESH_SHAPE_BH_QB = (1, 4)

_MESH_FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}

DEVICE_PARAMS_P150_FULL = {"l1_small_size": 65536}
DEVICE_PARAMS_BH_QB_FULL = {"l1_small_size": 65536, **_MESH_FABRIC}

# Demo / ``generate(use_decode_trace=True)``: KV-decode Metal trace replay.
DEVICE_PARAMS_P150_FULL_DECODE_TRACE = {
    "l1_small_size": 65536,
    "trace_region_size": 450_000_000,
}
DEVICE_PARAMS_BH_QB_FULL_DECODE_TRACE = {
    "l1_small_size": 65536,
    "trace_region_size": 450_000_000,
    **_MESH_FABRIC,
}

DEVICE_PARAMS_P150_TEXT = {"l1_small_size": 32768}
DEVICE_PARAMS_BH_QB_TEXT = {"l1_small_size": 32768, "num_command_queues": 2, **_MESH_FABRIC}

DEVICE_PARAMS_P150_E2E_2CQ = {"l1_small_size": 32768, "num_command_queues": 2}
DEVICE_PARAMS_BH_QB_E2E_2CQ = {"l1_small_size": 32768, "num_command_queues": 2, **_MESH_FABRIC}

DEVICE_PARAMS_P150_E2E_2CQ_TRACE = {
    "l1_small_size": 32768,
    "trace_region_size": 450_000_000,
    "num_command_queues": 2,
}
DEVICE_PARAMS_BH_QB_E2E_2CQ_TRACE = {
    "l1_small_size": 32768,
    "trace_region_size": 450_000_000,
    "num_command_queues": 2,
    **_MESH_FABRIC,
}

DEVICE_PARAMS_P150_E2E_2CQ_GENERATE = {
    "l1_small_size": 65536,
    "trace_region_size": 450_000_000,
    "num_command_queues": 2,
}
DEVICE_PARAMS_BH_QB_E2E_2CQ_GENERATE = {
    "l1_small_size": 65536,
    "trace_region_size": 450_000_000,
    "num_command_queues": 2,
    **_MESH_FABRIC,
}


def _requires_num_devices(n: int) -> bool:
    try:
        return ttnn.get_num_devices() != n
    except Exception:
        return True


def _mesh_device_param(mesh_shape: tuple[int, int], device_params: dict, *, id_suffix: str):
    n = mesh_shape[0] * mesh_shape[1]
    return pytest.param(
        mesh_shape,
        device_params,
        id=id_suffix,
        marks=pytest.mark.skipif(
            _requires_num_devices(n),
            reason=f"requires exactly {n} device(s) (mesh {id_suffix})",
        ),
    )


MESH_DEVICE_PARAMETRIZE_FULL = (
    "mesh_device,device_params",
    [
        _mesh_device_param(MESH_SHAPE_P150, DEVICE_PARAMS_P150_FULL, id_suffix="1x1"),
        _mesh_device_param(MESH_SHAPE_BH_QB, DEVICE_PARAMS_BH_QB_FULL, id_suffix="1x4"),
    ],
)

MESH_DEVICE_PARAMETRIZE_TEXT = (
    "mesh_device,device_params",
    [
        _mesh_device_param(MESH_SHAPE_P150, DEVICE_PARAMS_P150_TEXT, id_suffix="1x1"),
        _mesh_device_param(MESH_SHAPE_BH_QB, DEVICE_PARAMS_BH_QB_TEXT, id_suffix="1x4"),
    ],
)

MESH_DEVICE_PARAMETRIZE_E2E_2CQ = (
    "mesh_device,device_params",
    [
        _mesh_device_param(MESH_SHAPE_P150, DEVICE_PARAMS_P150_E2E_2CQ, id_suffix="1x1"),
        _mesh_device_param(MESH_SHAPE_BH_QB, DEVICE_PARAMS_BH_QB_E2E_2CQ, id_suffix="1x4"),
    ],
)

MESH_DEVICE_PARAMETRIZE_E2E_2CQ_TRACE = (
    "mesh_device,device_params",
    [
        _mesh_device_param(MESH_SHAPE_P150, DEVICE_PARAMS_P150_E2E_2CQ_TRACE, id_suffix="1x1"),
        _mesh_device_param(MESH_SHAPE_BH_QB, DEVICE_PARAMS_BH_QB_E2E_2CQ_TRACE, id_suffix="1x4"),
    ],
)

MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE = (
    "mesh_device,device_params",
    [
        _mesh_device_param(MESH_SHAPE_P150, DEVICE_PARAMS_P150_E2E_2CQ_GENERATE, id_suffix="1x1"),
        _mesh_device_param(MESH_SHAPE_BH_QB, DEVICE_PARAMS_BH_QB_E2E_2CQ_GENERATE, id_suffix="1x4"),
    ],
)


# ---------------------------------------------------------------------------
# Replicated ``from_torch`` uploads (PCC / perf)
# ---------------------------------------------------------------------------


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
    """Upload bf16 host data already in TILE layout (avoids per-chip ``TilizeDeviceOperation`` on mesh)."""
    host = ttnn.from_torch(
        t.to(torch.bfloat16).cpu().contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mesh_mapper(device),
    )
    return ttnn.to_device(host, device, memory_config=memory_config)


def from_torch_bfloat16_rm(
    device: ttnn.Device,
    t: torch.Tensor,
    *,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    return ttnn.from_torch(
        t.to(torch.bfloat16).cpu().contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem,
        mesh_mapper=mesh_mapper(device),
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def open_seamless_mesh_device(*, enable_decode_trace: bool = False):
    """Open mesh for ``demo.py``: (1,1) on P150, (1,4) on BH QB.

    When ``enable_decode_trace=True``, reserve ``trace_region_size`` for
    ``TTSeamlessM4Tv2Model.generate(..., use_decode_trace=True)``.
    """
    num_devices = ttnn.get_num_devices()
    if num_devices >= 4:
        mesh_shape = ttnn.MeshShape(*MESH_SHAPE_BH_QB)
        device_params = dict(DEVICE_PARAMS_BH_QB_FULL_DECODE_TRACE if enable_decode_trace else DEVICE_PARAMS_BH_QB_FULL)
    else:
        mesh_shape = ttnn.MeshShape(*MESH_SHAPE_P150)
        device_params = dict(DEVICE_PARAMS_P150_FULL_DECODE_TRACE if enable_decode_trace else DEVICE_PARAMS_P150_FULL)

    fabric_config = device_params.pop("fabric_config", None)
    if fabric_config is not None:
        ttnn.set_fabric_config(fabric_config)

    mesh = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
    return mesh, (int(mesh_shape[0]), int(mesh_shape[1]))
