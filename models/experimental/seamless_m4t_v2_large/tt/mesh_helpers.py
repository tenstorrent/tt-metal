# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh setup for Seamless M4T v2: BH QB shape, fabric, pytest params, replicated uploads, demo open.

Blackhole QB (4 devices): ``MeshShape(1, 4)`` with ``FABRIC_1D`` for head-parallel CCL.

Tests::

    @pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_FULL, indirect=["mesh_device", "device_params"])
    def test_foo(mesh_device, device_params, ...):
        with mesh_default_device(mesh_device):
            ...
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
import ttnn

# ---------------------------------------------------------------------------
# Mesh runtime (replication, CCL axis)
# ---------------------------------------------------------------------------

MESH_SHAPE = (1, 4)
_NUM_DEVICES = MESH_SHAPE[0] * MESH_SHAPE[1]

_MESH_FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def mesh_num_devices(device: ttnn.Device) -> int:
    if hasattr(device, "get_num_devices"):
        return max(1, int(device.get_num_devices()))
    return 1


def get_tp(device: ttnn.Device) -> int:
    """Tensor-parallelism degree = number of devices on the 1×4 mesh (TP=4)."""
    return mesh_num_devices(device)


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
    """Replicate batch-1 host tensors on every device in the multi-device mesh."""
    return mesh_replicate_mapper(device)


@contextmanager
def mesh_default_device(mesh: ttnn.Device):
    """Set ``ttnn`` default device for mask builders and mesh readback."""
    original = None
    try:
        original = ttnn.GetDefaultDevice()
    except Exception:
        # Default device may be unset before tests open a mesh.
        original = None
    ttnn.SetDefaultDevice(mesh)
    try:
        yield
    finally:
        ttnn.SetDefaultDevice(original)


# ---------------------------------------------------------------------------
# Pytest mesh shape + device_params (BH QB only)
# ---------------------------------------------------------------------------

DEVICE_PARAMS_FULL = {"l1_small_size": 65536, **_MESH_FABRIC}

# Demo / ``generate(use_decode_trace=True)``: KV-decode Metal trace replay.
DEVICE_PARAMS_FULL_DECODE_TRACE = {
    "l1_small_size": 65536,
    "trace_region_size": 450_000_000,
    **_MESH_FABRIC,
}

DEVICE_PARAMS_TEXT = {"l1_small_size": 32768, "num_command_queues": 2, **_MESH_FABRIC}

DEVICE_PARAMS_E2E_2CQ_GENERATE = {
    "l1_small_size": 65536,
    "trace_region_size": 450_000_000,
    "num_command_queues": 2,
    **_MESH_FABRIC,
}


def _requires_bh_qb() -> bool:
    try:
        return ttnn.get_num_devices() != _NUM_DEVICES
    except Exception:
        # Treat probe failures as "wrong host" so tests skip instead of crashing at collection.
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


MESH_DEVICE_PARAMETRIZE_FULL = (
    "mesh_device,device_params",
    [_mesh_device_param(DEVICE_PARAMS_FULL)],
)

MESH_DEVICE_PARAMETRIZE_TEXT = (
    "mesh_device,device_params",
    [_mesh_device_param(DEVICE_PARAMS_TEXT)],
)

MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE = (
    "mesh_device,device_params",
    [_mesh_device_param(DEVICE_PARAMS_E2E_2CQ_GENERATE)],
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


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def open_seamless_mesh_device(*, enable_decode_trace: bool = False, enable_2cq: bool = False):
    """Open ``MeshShape(1, 4)`` for ``demo.py`` on a Blackhole QB host.

    When ``enable_decode_trace=True``, reserve ``trace_region_size`` for
    ``TTSeamlessM4Tv2Model.generate(..., use_decode_trace=True)``.
    When ``enable_2cq=True``, open with ``num_command_queues=2`` so CQ1 can
    stage H2D copies while CQ0 executes the decode trace.
    ``enable_2cq`` is only effective when combined with ``enable_decode_trace``.
    """
    num_devices = ttnn.get_num_devices()
    if num_devices < _NUM_DEVICES:
        raise RuntimeError(
            f"Seamless M4T v2 requires {_NUM_DEVICES} devices (MeshShape{MESH_SHAPE}), found {num_devices}"
        )

    if enable_decode_trace and enable_2cq:
        device_params = dict(DEVICE_PARAMS_E2E_2CQ_GENERATE)
    elif enable_decode_trace:
        device_params = dict(DEVICE_PARAMS_FULL_DECODE_TRACE)
    else:
        device_params = dict(DEVICE_PARAMS_FULL)

    fabric_config = device_params.pop("fabric_config", None)
    if fabric_config is not None:
        ttnn.set_fabric_config(fabric_config)

    mesh_shape = ttnn.MeshShape(*MESH_SHAPE)
    mesh = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
    return mesh, (int(mesh_shape[0]), int(mesh_shape[1]))
