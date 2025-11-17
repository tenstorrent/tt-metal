# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.common.utility_functions import skip_for_blackhole, skip_for_wormhole_b0
import numpy as np


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

    print("output shape: ", output_shape)
    for iter_idx in range(num_iters):
        # Create input tensor on root device only
        torch_input = torch.randn(output_shape, dtype=torch.bfloat16)
        input_tensor_golden_list.append(torch_input)

        # Create mesh tensor - input only exists on root device, zeros elsewhere
        device_tensors = []
        mesh_rows, mesh_cols = mesh_shape
        num_devices = mesh_rows * mesh_cols
        cluster_axis = -1  # last dimension
        for device_idx in range(num_devices):
            row = device_idx // mesh_cols
            col = device_idx % mesh_cols
            if (row, col) == root_coord:
                device_tensors.append(torch_input)
            else:
                device_tensors.append(torch.zeros_like(torch_input))

        # Reshape device_tensors to match mesh shape (4, 2, 1, 1, 1, 32)
        output_shape_1 = (1, 32)
        mesh_tensor = torch.stack(device_tensors, dim=0).reshape(mesh_rows, mesh_cols, *output_shape_1)
        # Remove squeezing, just use the correct shape
        input_tensor_mesh = ttnn.from_torch(
            mesh_tensor,
            device=mesh_device,
            layout=layout,
            dtype=input_dtype,
            memory_config=mem_config,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(
                    [ttnn.PlacementShard(0), ttnn.PlacementShard(1)], ttnn.MeshShape(mesh_rows, mesh_cols)
                ),
            ),
        )
        input_tensor_mesh_list.append(input_tensor_mesh)

    # Run fused broadcast operation
    print("input tensor mesh list length: ", len(input_tensor_mesh_list))
    print("input tensor mesh shape: ", input_tensor_mesh_list[0])
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
        ((1, 0), [1, 1, 1, 32], ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        # ((4, 2), (2, 0), [1, 1, 64, 64], ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
        # ((4, 2), (1, 1), [2, 1, 32, 64], ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
    ],
)
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize("num_iters", [1])
@pytest.mark.parametrize("mesh_shape", [(4, 2)])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 2), id="4x2_grid")], indirect=True)
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


@pytest.mark.parametrize("r_star", [1])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 2), id="4x2_grid")], indirect=True)
def test_unfused_seed_ops_ttnn(mesh_device, r_star):
    """
    Unfused seed: P2P replicate input across TP ranks in row r*, MM for Q-proj, then broadcast Q down each column.
    """
    # Mesh shape: 4 rows x 2 cols
    mesh_shape = (4, 2)
    NUM_ROWS, NUM_COLS = mesh_shape
    HIDDEN_SIZE = 1536
    Q_DIM = 24576
    BATCH = 1
    input_dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    mem_config = ttnn.DRAM_MEMORY_CONFIG

    # Step 1: P2P replicate input across TP ranks in row r*
    input_tensor = torch.randn((BATCH, HIDDEN_SIZE), dtype=torch.bfloat16)
    # Place input on (r*, 0), zeros elsewhere in row r*
    tp_input_mesh = torch.zeros((NUM_ROWS, NUM_COLS, BATCH, HIDDEN_SIZE), dtype=torch.bfloat16)
    tp_input_mesh[r_star, 0] = input_tensor
    tt_tp_input = ttnn.from_torch(
        tp_input_mesh,
        device=mesh_device,
        layout=layout,
        dtype=input_dtype,
        memory_config=mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], ttnn.MeshShape(NUM_ROWS, NUM_COLS)),
        ),
    )
    sender = ttnn.MeshCoordinate(r_star, 0)
    receiver = ttnn.MeshCoordinate(r_star, 1)
    # print("shaoe if point to point input: ", tt_tp_input)
    tt_tp_input = ttnn.point_to_point(tt_tp_input, sender, receiver, topology=ttnn.Topology.Linear)
    # print("shape after p2p: ", tt_tp_input)

    # Step 2: MM for Q-projection on both TP ranks in row r*
    Q_proj = torch.randn((HIDDEN_SIZE, Q_DIM), dtype=torch.bfloat16)
    tt_Q_proj = ttnn.from_torch(
        Q_proj, device=mesh_device, layout=layout, dtype=input_dtype, memory_config=mem_config, mesh_mapper=None
    )
    Qs = []
    # Get the input for matmul for (r*,0): input to p2p
    input_c0 = tp_input_mesh[r_star, 0].unsqueeze(0)  # (1, 1536)
    tt_input_c0 = ttnn.from_torch(
        input_c0,
        device=mesh_device,
        layout=layout,
        dtype=input_dtype,
        memory_config=mem_config,
        mesh_mapper=None,
    )
    tt_Q_c0 = ttnn.matmul(tt_input_c0, tt_Q_proj)
    # Get the input for matmul for (r*,1): output of p2p
    # Convert tt_tp_input to torch and slice out the correct device's chunk
    tp_input_mesh_torch = ttnn.to_torch(tt_tp_input, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    # Each device chunk is (1, 1536), so for (r*,1): index = r_star * NUM_COLS + 1
    idx_1 = r_star * NUM_COLS + 1
    input_c1 = tp_input_mesh_torch[idx_1, 0]  # (1, 1536)
    tt_input_c1 = ttnn.from_torch(
        input_c1.unsqueeze(0),
        device=mesh_device,
        layout=layout,
        dtype=input_dtype,
        memory_config=mem_config,
        mesh_mapper=None,
    )
    tt_Q_c1 = ttnn.matmul(tt_input_c1, tt_Q_proj)
    # print("shape of Q_c0: ", tt_Q_c0)
    # print("shape of Q_c1: ", tt_Q_c1)
    print("data of Q_c0: ", tt_Q_c0)

    Q_c0 = ttnn.to_torch(tt_Q_c0, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    Q_c1 = ttnn.to_torch(tt_Q_c1, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    Qs.append(Q_c0)
    Qs.append(Q_c1)
    # Step 3: Broadcast each Q_c down its column
    # For each column, create and map the data separately for broadcast
    for c in range(NUM_COLS):
        mesh_data = torch.zeros((NUM_ROWS, NUM_COLS, 1, Q_DIM), dtype=torch.bfloat16)
        mesh_data[r_star, c, 0] = Qs[c][0]
        # Mesh mapping: shard along row axis, replicate along column axis
        mesh_mapper = ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementReplicate()], ttnn.MeshShape(NUM_ROWS, 1)),
        )
        # Place mesh_data only on sender device in column c
        # Use mesh_mapper for vertical sharding (row axis only)
        tt_mesh_data = ttnn.from_torch(
            mesh_data[:, c : c + 1],  # select column c only
            device=mesh_device,
            layout=layout,
            dtype=input_dtype,
            memory_config=mem_config,
            mesh_mapper=mesh_mapper,
        )
        sender_coord = ttnn.MeshCoordinate(r_star, 0)  # sender is (r*, 0) in this column's mesh
        tt_Qs_mesh = ttnn.broadcast(
            tt_mesh_data,
            sender_coord=ttnn.MeshCoordinate(r_star, 0),
            num_links=1,
            memory_config=mem_config,
            topology=ttnn.Topology.Linear,
            cluster_axis=0,
        )
        print("broadcast op output shape: ", tt_Qs_mesh)
        print("broadcast op output data: ", tt_Qs_mesh)
