# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.common.utility_functions import skip_for_blackhole, skip_for_wormhole_b0


def run_fused_broadcast_impl(
    mesh_device,
    root_coord,
    mesh_shape,
    output_shape,
    input_dtype,
    layout,
    topology,
    num_links=1,
    num_iters=1,
    mem_config=None,
):
    """Implementation for fused broadcast testing following the CCL test pattern."""

    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    if mem_config is None:
        mem_config = ttnn.DRAM_MEMORY_CONFIG

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

    logger.info(f"Output shape: {output_shape}")
    logger.info(f"Root coord: {root_coord}")
    logger.info(f"Mesh shape: {mesh_shape}")

    # Create input tensors for each iteration
    input_tensor_mesh_list = []
    input_tensor_golden_list = []

    for iter_idx in range(num_iters):
        # Create input tensor on root device only
        torch_input = torch.randn(output_shape, dtype=torch.bfloat16)
        input_tensor_golden_list.append(torch_input)

        # Create mesh tensor - input only exists on root device, zeros elsewhere
        device_tensors = []
        mesh_rows, mesh_cols = mesh_shape

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                if (row, col) == root_coord:
                    device_tensors.append(torch_input)
                else:
                    device_tensors.append(torch.zeros_like(torch_input))

        # Concatenate along last dimension for mesh mapper
        mesh_tensor = torch.cat(device_tensors, dim=-1)

        input_tensor_mesh = ttnn.from_torch(
            mesh_tensor,
            device=mesh_device,
            layout=layout,
            dtype=input_dtype,
            memory_config=mem_config,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(
                    [ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)], ttnn.MeshShape(mesh_rows, mesh_cols)
                ),
            ),
        )
        input_tensor_mesh_list.append(input_tensor_mesh)

    # Run fused broadcast operation
    tt_out_tensor_list = []
    for i in range(num_iters):
        tt_out_tensor = ttnn.fused_broadcast(
            input_tensor_mesh_list[i],
            root_coord=ttnn.MeshCoordinate(root_coord[0], root_coord[1]),
            mesh_shape=ttnn.MeshCoordinate(mesh_shape[0], mesh_shape[1]),
            topology=topology,
            num_links=num_links,
            memory_config=mem_config,
            subdevice_id=worker_sub_device_id,
        )
        tt_out_tensor_list.append(tt_out_tensor)

    logger.info("Waiting for op to complete")
    ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
    logger.info("Op completed")

    # Validate results - all devices should have the same data as the root
    passed = True
    for iter_idx in range(len(tt_out_tensor_list)):
        tt_out_tensor = tt_out_tensor_list[iter_idx]
        expected_tensor = input_tensor_golden_list[iter_idx]

        # Convert mesh tensor back to torch
        output_tensor_torch = ttnn.to_torch(
            tt_out_tensor,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1),
        )

        # Check each device's output
        slice_size = output_shape[-1]  # Last dimension was sharded across devices
        num_devices = mesh_rows * mesh_cols

        for device_idx in range(num_devices):
            start = device_idx * slice_size
            end = start + slice_size
            device_output = output_tensor_torch[..., start:end]

            # All devices should have the same data as the original input
            if input_dtype == ttnn.bfloat16:
                eq, output = comp_equal(device_output, expected_tensor)
            else:
                eq, output = comp_pcc(device_output, expected_tensor)

            if not eq:
                logger.error(f"Output mismatch for device {device_idx}")
                passed = False
                assert eq, f"Device {device_idx} FAILED: {output}"

    # Cleanup
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()

    if not passed:
        pytest.fail("Fused broadcast test failed - output mismatch detected")


@pytest.mark.parametrize(
    "root_coord, output_shape, layout, input_dtype, mem_config",
    [
        ((0, 1), [1, 1, 1, 1024], ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        # ((4, 2), (2, 0), [1, 1, 64, 64], ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
        # ((4, 2), (1, 1), [2, 1, 32, 64], ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
    ],
)
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize("num_iters", [1])
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["t3k"], indirect=True)
def test_fused_broadcast(
    mesh_shape,
    root_coord,
    output_shape,
    layout,
    input_dtype,
    mem_config,
    topology,
    num_iters,
    mesh_device,
    function_level_defaults,
):
    """Test fused broadcast operation ensures all devices receive the same data."""

    if not mesh_device:
        pytest.skip("Test requires mesh device")

    run_fused_broadcast_impl(
        mesh_device,
        root_coord,
        mesh_shape,
        output_shape,
        input_dtype,
        layout,
        topology,
        num_links=1,
        num_iters=num_iters,
        mem_config=mem_config,
    )


if __name__ == "__main__":
    # Run basic tests if executed directly
    pytest.main([__file__, "-v"])
