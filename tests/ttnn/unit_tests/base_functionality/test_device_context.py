# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for TTNN DeviceContext and independent streams (CCL all-gather + matmul).

Covers sub-device context (set_current_sub_device, sub_device context manager), grid size APIs,
and dual sub-devices: CCL row + matmul row with stream 1 = matmul busy loop, stream 2 = CCL
all-gather (multi-device). Variable subdevice test runs workload on 2, 4, and 3 subdevices
(matmul + CCL when multi-device).
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.common.utility_functions import skip_for_slow_dispatch


# Small matmul shape for fast tests
TEST_M, TEST_K, TEST_N = 256, 256, 256
NUM_ITERATIONS = 10
NUM_ITERATIONS_VARIABLE = 3
CCL_CLUSTER_AXIS = 1
CORES_PER_SUBDEVICE = 2


def _linear(input_tensor, weight, *, core_grid=None, sub_device_id=None, queue_id=None):
    if core_grid is None:
        device = input_tensor.device()
        grid_size = device.compute_with_storage_grid_size()
        core_grid = ttnn.CoreGrid(x=grid_size.x, y=grid_size.y)
    kwargs = {
        "transpose_a": False,
        "transpose_b": True,
        "core_grid": core_grid,
        "sub_device_id": sub_device_id,
    }
    if queue_id is not None:
        kwargs["queue_id"] = queue_id
    return ttnn.linear(input_tensor, weight, **kwargs)


def setup_dual_sub_devices_small(device):
    """Two sub-devices with 2 cores each: CCL (row 0), matmul (row 1). Same as two_independent_streams.py."""
    n = CORES_PER_SUBDEVICE
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))})
    matmul_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(n - 1, 1))})
    ccl_sub_device = ttnn.SubDevice([ccl_crs])
    matmul_sub_device = ttnn.SubDevice([matmul_crs])
    ccl_sub_device_id = ttnn.SubDeviceId(0)
    matmul_sub_device_id = ttnn.SubDeviceId(1)
    sub_device_manager = device.create_sub_device_manager([ccl_sub_device, matmul_sub_device], 0)
    device.load_sub_device_manager(sub_device_manager)
    device.set_sub_device_stall_group([ccl_sub_device_id, matmul_sub_device_id])
    core_grid_2x1 = ttnn.CoreGrid(x=n, y=1)
    return (
        sub_device_manager,
        ccl_sub_device_id,
        matmul_sub_device_id,
        core_grid_2x1,
        core_grid_2x1,
    )


def setup_n_sub_devices(device, n_subdevices, cores_per_subdevice=CORES_PER_SUBDEVICE):
    """Create n_subdevices sub-devices, each with cores_per_subdevice cores in one row (row 0, 1, ... n-1)."""
    sub_devices = []
    for row in range(n_subdevices):
        crs = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, row),
                    ttnn.CoreCoord(cores_per_subdevice - 1, row),
                )
            }
        )
        sub_devices.append(ttnn.SubDevice([crs]))
    sub_device_ids = [ttnn.SubDeviceId(i) for i in range(n_subdevices)]
    sub_device_manager = device.create_sub_device_manager(sub_devices, 0)
    device.load_sub_device_manager(sub_device_manager)
    device.set_sub_device_stall_group(sub_device_ids)
    core_grid = ttnn.CoreGrid(x=cores_per_subdevice, y=1)
    return sub_device_manager, sub_device_ids, core_grid


def teardown_sub_devices(device, sub_device_manager):
    device.reset_sub_device_stall_group()
    device.clear_loaded_sub_device_manager()
    device.remove_sub_device_manager(sub_device_manager)


# -----------------------------------------------------------------------------
# Minimal API tests (execution context)
# -----------------------------------------------------------------------------


def test_set_current_sub_device_returns_guard(mesh_device):
    """set_current_sub_device returns a CurrentSubDeviceGuard; it restores the previous sub-device when destroyed."""
    guard = mesh_device.set_current_sub_device(ttnn.SubDeviceId(0))
    assert guard is not None
    # Guard goes out of scope here; C++ destructor restores the previous sub-device.


def test_sub_device_context_manager(mesh_device):
    """sub_device(device, id) context manager enters and exits without error."""
    with ttnn.sub_device(mesh_device, 0):
        pass


def trivial_workload(mesh_device, input_mapper, weight, core_grid):
    input_torch = torch.randn(TEST_M, TEST_K, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_torch, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=input_mapper)
    x = input_tensor
    for _ in range(NUM_ITERATIONS_VARIABLE):
        x = _linear(x, weight, core_grid=core_grid)
    ttnn.deallocate(x)


def test_set_current_sub_device_example(mesh_device):
    """Example: set_current_sub_device returns a guard; when the guard goes out of scope, previous context is restored."""
    guard = mesh_device.set_current_sub_device(ttnn.SubDeviceId(0))
    assert guard is not None
    # Run a small workload while the guard is active (current sub-device is 0).
    input_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    core_grid = mesh_device.core_grid
    weight_torch = torch.randn(TEST_K, TEST_N, dtype=torch.bfloat16)
    weight = ttnn.from_torch(weight_torch, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=weights_mapper)
    trivial_workload(mesh_device, input_mapper, weight, core_grid)
    del guard  # Guard destroyed here; previous current sub-device is restored.


@skip_for_slow_dispatch(reason_str="Uses nested sub_device context; requires fast dispatch")
def test_sub_device_nested_guards(mesh_device):
    """Three levels of nested sub_device context: enter 0, then 1, then 2; run workload in innermost; exit restores in reverse order."""
    sub_device_manager, sub_device_ids, core_grid = setup_n_sub_devices(mesh_device, 3)
    try:
        input_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        weights_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        weight_torch = torch.randn(TEST_K, TEST_N, dtype=torch.bfloat16)
        weight = ttnn.from_torch(weight_torch, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=weights_mapper)
        with ttnn.sub_device(mesh_device, sub_device_ids[0]):
            trivial_workload(mesh_device, input_mapper, weight, core_grid)
            with ttnn.sub_device(mesh_device, sub_device_ids[1]):
                with ttnn.sub_device(mesh_device, sub_device_ids[2]):
                    # Workload in innermost context (sub-device 2)
                    trivial_workload(mesh_device, input_mapper, weight, core_grid)

                # Back to sub-device 1
                trivial_workload(mesh_device, input_mapper, weight, core_grid)
            # Back to sub-device 0
        # Back to default
        ttnn.synchronize_device(mesh_device)
    finally:
        teardown_sub_devices(mesh_device, sub_device_manager)


# -----------------------------------------------------------------------------
# Two independent streams: CCL all-gather + matmul busy loop (multi-device)
# -----------------------------------------------------------------------------


@skip_for_slow_dispatch(reason_str="Uses 2 command queues for parallel matmul + all-gather; requires fast dispatch")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "num_command_queues": 2}],
    indirect=True,
)
def test_two_independent_streams_all_gather_and_matmul(mesh_device):
    """Stream 1: matmul on matmul subdevice (CQ 1). Stream 2: all-gather on CCL subdevice (CQ 0).

    Uses two command queues so matmul and all-gather can run in parallel on the device instead
    of serializing (single CQ would order them and the final sync would appear as matmul waiting
    for all-gather). With 2 CQs, both streams are independent and overlap on different cores.

    Core count: 2 devices × (2 matmul cores + 2 CCL cores) = 8 cores total (4 for matmul, 4 for
    all-gather). This is expected for a 2-chip mesh with dual subdevices per chip.
    """
    num_devices = mesh_device.get_num_devices()
    if num_devices < 2:
        pytest.skip("Need at least 2 devices for all-gather stream; mesh has %d" % num_devices)

    CCL_CQ = 0
    MATMUL_CQ = 1

    sub_device_manager, ccl_sub_device_id, matmul_sub_device_id, matmul_core_grid, _ = setup_dual_sub_devices_small(
        mesh_device
    )
    try:
        # Tensors for stream 1 (matmuls)
        input_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        input_torch = torch.randn(TEST_M, TEST_K, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(
            input_torch, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=input_mapper
        )
        weights_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        weight_torch = torch.randn(TEST_K, TEST_N, dtype=torch.bfloat16)
        weight = ttnn.from_torch(weight_torch, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=weights_mapper)

        # Sharded tensors for stream 2 (all-gathers)
        shard_rows = max(32, TEST_M // num_devices)
        ag_shape = (shard_rows * num_devices, TEST_N)
        ag_torch = torch.randn(ag_shape, dtype=torch.bfloat16)
        shard_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        sharded_tensors = [
            ttnn.from_torch(
                ag_torch.clone(),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=shard_mapper,
            )
            for _ in range(NUM_ITERATIONS)
        ]

        # Two CQs: matmul on CQ 1, all-gather on CQ 0 so they can run in parallel (no sync between them).
        # Use num_workers_per_link=1 so CCL fits on the small subdevice (2 cores).
        ag_kwargs = dict(
            dim=0,
            cluster_axis=CCL_CLUSTER_AXIS,
            topology=ttnn.Topology.Ring,
            subdevice_id=ccl_sub_device_id,
            num_workers_per_link=1,
            queue_id=CCL_CQ,
        )
        ag_kwargs_no_sd = {k: v for k, v in ag_kwargs.items() if k != "subdevice_id"}

        x = input_tensor
        for i in range(NUM_ITERATIONS):
            x = _linear(
                x,
                weight,
                core_grid=matmul_core_grid,
                sub_device_id=matmul_sub_device_id,
                queue_id=MATMUL_CQ,
            )
            with ttnn.sub_device(mesh_device, ccl_sub_device_id):
                gathered = ttnn.all_gather(sharded_tensors[i % len(sharded_tensors)], **ag_kwargs_no_sd)
            ttnn.deallocate(gathered)
        del x

        ttnn.synchronize_device(mesh_device)
    finally:
        teardown_sub_devices(mesh_device, sub_device_manager)


# -----------------------------------------------------------------------------
# Variable number of subdevices (2, 4, 3): run workload on each configuration
# -----------------------------------------------------------------------------


@skip_for_slow_dispatch(reason_str="Uses 2 command queues and CCL all-gather; requires fast dispatch")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "num_command_queues": 2}],
    indirect=True,
)
def test_execution_context_variable_subdevices_workload(mesh_device):
    """Create 2, then 4, then 3 subdevices; each run runs workload on all subdevices.

    Workload: matmul chain on every subdevice; when num_devices >= 2, subdevice 0 also runs
    all_gather (CCL). Exercises execution context and multiple subdevices with mixed matmul + CCL.
    """
    num_devices = mesh_device.get_num_devices()
    for num_subdevices in [2, 4, 3]:
        sub_device_manager, sub_device_ids, core_grid = setup_n_sub_devices(mesh_device, num_subdevices)
        try:
            input_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
            weights_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
            # Shared weight for matmul on all subdevices
            weight_torch = torch.randn(TEST_K, TEST_N, dtype=torch.bfloat16)
            weight = ttnn.from_torch(
                weight_torch, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=weights_mapper
            )

            # Per-subdevice matmul chain (each subdevice gets its own input and runs a short chain)
            for sub_device_id in sub_device_ids:
                input_torch = torch.randn(TEST_M, TEST_K, dtype=torch.bfloat16)
                input_tensor = ttnn.from_torch(
                    input_torch, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=input_mapper
                )
                with ttnn.sub_device(mesh_device, sub_device_id):
                    x = input_tensor
                    for _ in range(NUM_ITERATIONS_VARIABLE):
                        x = _linear(x, weight, core_grid=core_grid)
                    ttnn.deallocate(x)

            # When multi-device, run all_gather on subdevice 0 (CCL) so we exercise CCL + execution context
            if num_devices >= 2:
                shard_rows = max(32, TEST_M // num_devices)
                ag_shape = (shard_rows * num_devices, TEST_N)
                ag_torch = torch.randn(ag_shape, dtype=torch.bfloat16)
                shard_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
                sharded = ttnn.from_torch(
                    ag_torch, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard_mapper
                )
                ag_kwargs = dict(
                    dim=0,
                    cluster_axis=CCL_CLUSTER_AXIS,
                    topology=ttnn.Topology.Ring,
                    subdevice_id=sub_device_ids[0],
                    num_workers_per_link=1,
                )
                with ttnn.sub_device(mesh_device, sub_device_ids[0]):
                    gathered = ttnn.all_gather(sharded, **ag_kwargs)
                ttnn.deallocate(gathered)

            ttnn.synchronize_device(mesh_device)
        finally:
            teardown_sub_devices(mesh_device, sub_device_manager)
