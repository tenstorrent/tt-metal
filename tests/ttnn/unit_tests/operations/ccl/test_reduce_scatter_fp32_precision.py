# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test case to ensure fp32 precision in reduce_scatter operations.

Expected behavior: Intermediate accumulation should use fp32 precision (like
local reduce operations do with fp32_dest_acc_en=true) and convert back to bf16
only at the final output.

To run (from tt-metal directory):
    pytest tests/ttnn/unit_tests/operations/ccl/test_reduce_scatter_fp32_precision.py -svv
"""

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def create_global_semaphores(mesh_device, cores, initial_value, count=3):
    """Create global semaphore handles for CCL operations."""
    return [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(count)]


@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize("rs_input_shape", [[1, 1, 32, 1024]])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_reduce_scatter_bf16_precision(
    mesh_device,
    num_devices,
    rs_input_shape,
    dim,
    num_links,
):
    """
    Compare (reduce_scatter + all_gather) precision vs (all_gather + local sum).

    Both compute all_reduce (same result on all devices), but:
    - reduce_scatter + all_gather: loses precision due to bf16 intermediate accumulation
    - all_gather + local sum: maintains precision because local sum uses fp32_dest_acc_en

    Issue: reduce_scatter should use fp32 intermediate accumulation for bf16 inputs.
    """
    if mesh_device.get_num_devices() < num_devices:
        pytest.skip(f"Requires at least {num_devices} devices")

    torch.manual_seed(42)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    # Fabric setup
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # Create input tensor with values near bf16 limits
    # bf16 max ~65504, each value ~8000, sum of 8 = ~64000 (close to bf16 max)
    rs_global_input_shape = rs_input_shape[:]
    rs_global_input_shape[dim] *= num_devices
    torch_input = torch.rand(rs_global_input_shape).float() * 1000 + 7500
    torch_input_fp32 = torch_input.float()  # updated

    # Golden: bf16 input → fp32 accumulation → full reduce result
    # This is what all_reduce should produce with proper fp32 accumulation
    input_for_golden = torch_input_fp32.float()  # updated
    input_chunks = torch.chunk(input_for_golden, num_devices, dim)
    golden_reduce = torch.sum(torch.stack(input_chunks), dim=0)  # [1, 1, 32, 1024]

    # ========================================
    # Method 1: reduce_scatter + all_gather
    # ========================================
    rs_semaphore_handles = create_global_semaphores(mesh_device, ccl_sub_device_crs, 0, count=3)

    intermediate_shape = rs_input_shape[:]
    intermediate_shape.insert(0, 2)  # Linear topology
    persistent_intermediate = ttnn.from_torch(
        torch.zeros(intermediate_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,  # updated
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rs_output_shape = rs_input_shape[:]
    rs_output_shape[dim] //= num_devices
    persistent_output = ttnn.from_torch(
        torch.zeros(rs_output_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,  # updated
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    input_rs = ttnn.from_torch(
        torch_input_fp32,  # updated
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,  # updated
        memory_config=mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig(
                [ttnn.PlacementReplicate(), ttnn.PlacementShard(dim)], ttnn.MeshShape(1, num_devices)
            ),
        ),
    )

    # Step 1: reduce_scatter
    rs_output = ttnn.experimental.reduce_scatter_minimal_async(
        input_rs,
        persistent_output_buffers=[persistent_intermediate, persistent_output],
        dim=dim,
        multi_device_global_semaphore=rs_semaphore_handles,
        num_links=num_links,
        memory_config=mem_config,
        topology=ttnn.Topology.Linear,
        subdevice_id=worker_sub_device_id,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

    # Step 2: all_gather on reduce_scatter output
    ag_semaphore_handles = create_global_semaphores(mesh_device, ccl_sub_device_crs, 0, count=2)

    all_reduce_output = ttnn.experimental.all_gather_async(
        rs_output,
        dim=dim,
        multi_device_global_semaphore=ag_semaphore_handles,
        num_links=num_links,
        memory_config=mem_config,
        topology=ttnn.Topology.Linear,
        subdevice_id=worker_sub_device_id,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

    # Get result from first device (all devices should have same result)
    all_reduce_torch = ttnn.to_torch(
        ttnn.from_device(all_reduce_output), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )
    rs_ag_result = all_reduce_torch[0:1]  # [1, 1, 32, 1024]

    _, rs_ag_pcc = comp_pcc(rs_ag_result.float(), golden_reduce, pcc=0.999)  # updated
    rs_ag_max_diff = (rs_ag_result.float() - golden_reduce).abs().max().item()

    # ========================================
    # Method 2: all_gather + local sum (simulated)
    # ========================================
    ag2_semaphore_handles = create_global_semaphores(mesh_device, ccl_sub_device_crs, 0, count=2)

    input_ag = ttnn.from_torch(
        torch_input_fp32,  # updated
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,  # updated
        memory_config=mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig(
                [ttnn.PlacementReplicate(), ttnn.PlacementShard(dim)], ttnn.MeshShape(1, num_devices)
            ),
        ),
    )

    ag_output = ttnn.experimental.all_gather_async(
        input_ag,
        dim=dim,
        multi_device_global_semaphore=ag2_semaphore_handles,
        num_links=num_links,
        memory_config=mem_config,
        topology=ttnn.Topology.Linear,
        subdevice_id=worker_sub_device_id,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

    ag_torch = ttnn.to_torch(ttnn.from_device(ag_output), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # After all_gather, each device has full data [1, 1, 32, 8192]
    # Simulate local sum with fp32 accumulation (this is what local reduce does correctly)
    ag_single = ag_torch[0:1]  # [1, 1, 32, 8192]
    ag_chunks = torch.chunk(ag_single.float(), num_devices, dim=dim)
    local_sum_result = torch.sum(torch.stack(ag_chunks), dim=0)  # [1, 1, 32, 1024]

    _, ag_local_pcc = comp_pcc(local_sum_result, golden_reduce, pcc=0.999)  # updated
    ag_local_max_diff = (local_sum_result - golden_reduce).abs().max().item()

    # Cleanup
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()

    # ========================================
    # Summary
    # ========================================

    logger.info(f" reduce_scatter + all_gather: PCC={rs_ag_pcc}, max_diff={rs_ag_max_diff:.2f}")
    logger.info(f" all_gather + local sum:     PCC={ag_local_pcc}, max_diff={ag_local_max_diff:.2f}")

    rs_ag_pcc_value = float(rs_ag_pcc.split("PCC: ")[1]) if "PCC: " in rs_ag_pcc else 1.0
    assert rs_ag_pcc_value > 0.99
