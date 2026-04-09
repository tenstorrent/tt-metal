# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimal test to reproduce all_reduce on N300 with Gemma4 shapes.

Tests with and without sub_device_manager setup to isolate the root cause
of the fabric deadlock between dispatch and CCL.
"""

import pytest
import torch

import ttnn


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("num_iters", [1, 5], ids=["single", "5_layers"])
@pytest.mark.parametrize("shape", [[1, 1, 1, 2880], [1, 1, 32, 2560]], ids=["1x1x1x2880", "1x1x32x2560"])
def test_allreduce_n300_with_subdevice(mesh_device, num_iters, shape):
    """Test all_reduce on N300 with sub_device_manager (matching official CCL tests)."""
    # shape = [1, 1, 32, 2560]  # E4B O_proj output

    # Set up sub-device manager — required to isolate CCL from dispatch on fabric
    compute_grid = mesh_device.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([worker_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    for i in range(num_iters):
        torch_input = torch.randn(shape, dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        tt_output = ttnn.all_reduce(
            tt_input,
            cluster_axis=1,
            num_links=1,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            subdevice_id=worker_sub_device_id,
        )
        tt_input.deallocate(True)

        assert tt_output.shape == ttnn.Shape(shape), f"Iter {i}: expected {shape}, got {tt_output.shape}"
        tt_output.deallocate(True)
        print(f"  iter {i} passed", flush=True)

    mesh_device.clear_loaded_sub_device_manager()
    mesh_device.remove_sub_device_manager(sub_device_manager)
    print(f"PASSED: {num_iters} all_reduce calls with shape {shape}")


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("shape", [[1, 1, 1, 2880], [1, 1, 32, 2560]], ids=["1x1x1x2880", "1x1x32x2560"])
def test_allreduce_n300_no_subdevice(mesh_device, shape):
    """Test all_reduce on N300 WITHOUT sub_device_manager (expected to hang?)."""
    # shape = [1, 1, 32, 2560]

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_output = ttnn.all_reduce(
        tt_input,
        cluster_axis=1,
        num_links=1,
        topology=ttnn.Topology.Linear,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_input.deallocate(True)

    assert tt_output.shape == ttnn.Shape(shape)
    tt_output.deallocate(True)
    print("PASSED: all_reduce without subdevice")
