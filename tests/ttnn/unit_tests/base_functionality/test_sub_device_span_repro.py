# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Minimal repro for the "Programs must be executed on a single sub-device" guard.

A single TTNN op compiles to one Program. At enqueue time,
ProgramImpl::determine_sub_device_ids (tt_metal/impl/program/program.cpp:1787)
intersects every kernel group's core_ranges against each loaded sub-device's
worker_cores. FDMeshCommandQueue::enqueue_mesh_workload then asserts the
result has size exactly 1 (tt_metal/distributed/fd_mesh_command_queue.cpp:273).

Here we load a 2-sub-device manager that partitions the compute grid and then
issue an op (`ttnn.multiply_`) whose `sub_core_grids` deliberately spans both
halves. The op's kernel groups intersect sd0 AND sd1, so the enqueue path
fires a TT_FATAL — which surfaces in Python as RuntimeError. The test passes
if (and only if) that RuntimeError is raised.
"""

import pytest
import torch

import ttnn


@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True, ids=["8x1"])
def test_program_spanning_two_subdevices_raises(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    assert grid.y >= 2, f"need at least 2 rows of compute cores, got {grid.y}"

    mid_y = grid.y // 2
    sd0_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, mid_y - 1))})
    sd1_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, mid_y), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    spanning_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})

    sd_manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([sd0_cores]), ttnn.SubDevice([sd1_cores])], 3200)
    mesh_device.load_sub_device_manager(sd_manager)

    try:
        mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)])

        a = ttnn.from_torch(
            torch.randn(1, 1, 32, 32, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        b = ttnn.from_torch(
            torch.randn(1, 1, 32, 32, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        with pytest.raises(RuntimeError):
            ttnn.multiply_(a, b, sub_core_grids=spanning_cores)
            ttnn.synchronize_device(mesh_device)
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(sd_manager)
