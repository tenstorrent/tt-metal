# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Post-Decode CCL Broadcast + Mcast + Matmul Op Test

In multi-device mode: CCL broadcasts input_a [1, 7168] from sender device to all
devices, then on each device the sender core multicasts to 101 matmul cores.
Each matmul core holds a weight shard [7168, N_per_core] and computes
[1, 7168] x [7168, N_per_core] -> [1, N_per_core].
Output stays width-sharded across matmul cores.

In single-device mode (skip_ccl=True): CCL is skipped and the input is used directly.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.fused_ops.post_decode.op import PostDecode


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "sender_row, sender_col",
    [
        (1, 0),
    ],
)
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("secondary_cluster_axis", [1])
@pytest.mark.parametrize("using_persistent_buffers", [True])
@pytest.mark.parametrize("mesh_rows, mesh_cols, skip_ccl", [(4, 2, False), (1, 1, True)])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_post_decode(
    bh_2d_mesh_device,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    use_fp32,
    cluster_axis,
    secondary_cluster_axis,
    using_persistent_buffers,
    skip_ccl,
):
    """Test CCL broadcast + mcast + matmul for post-decode vocab projection.

    Core layout (per device):
        (max_x, 9) = sender core (holds input_a, multicasts to matmul cores)
        (0,0)..(9,9) + (10,0) = 101 matmul cores (hold weight shards)
    Each matmul core computes [1, K] x [K, N_per_core] -> [1, N_per_core].
    Output stays width-sharded across matmul cores.
    """

    num_devices = mesh_rows * mesh_cols

    # Validate mesh size
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    # Create submesh used by the test
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))

    # Configure a single worker sub-device covering the full compute grid
    device_grid_size = submesh.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    submesh.load_sub_device_manager(submesh.create_sub_device_manager([worker_sub_device], 0))
    submesh.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

    # ========================================================================
    # Configuration
    # ========================================================================
    M = 1
    K = 7168
    input_shape = (1, 7168)

    num_matmul_cores = 101
    N_per_core = 128  # TODO: change to 160, when matmul supports odd number of tiles
    N_total = N_per_core * num_matmul_cores
    vocab_shape = (7168, N_total)
    output_shape = (1, N_total)

    # Tile dimensions
    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    # Core layout
    mcast_core_x = device_grid_size.x - 1  # Last column
    mcast_core_y = 9

    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])

    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )

    num_tiles_k = K // a_tile.tile_shape[1]
    num_tiles_n_per_core = N_per_core // b_tile.tile_shape[1]

    fp32_str = " (fp32 acc)" if use_fp32 else ""
    logger.info(
        f"Testing post-decode{fp32_str} with shape [{M}, {K}] x [{K}, {N_total}] "
        f"({num_matmul_cores} cores, {N_per_core} per core), in0=bfloat16, in1=bfloat8_b, "
        f"mesh={mesh_rows}x{mesh_cols}, skip_ccl={skip_ccl}"
    )
    logger.info(f"Tiles: K={num_tiles_k}, N_per_core={num_tiles_n_per_core}")

    # ========================================================================
    # Create PyTorch tensors
    # ========================================================================
    torch.manual_seed(0)
    torch_a = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_b = torch.randn(vocab_shape, dtype=torch.bfloat16)

    # Compute reference output (all devices should produce same result)
    torch_expected = PostDecode.golden(torch_a.float(), torch_b.float()).bfloat16()

    # ========================================================================
    # Shard specs and memory configs
    # ========================================================================
    # Input A: HEIGHT_SHARDED on sender core (single core)
    input_a_shard_spec = ttnn.ShardSpec(
        mcast_core_grid,
        input_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_a_shard_spec
    )

    # Input B: WIDTH_SHARDED across matmul cores
    input_b_shard_shape = (K, N_per_core)
    input_b_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        input_b_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_b_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_b_shard_spec
    )

    # Output: WIDTH_SHARDED across matmul cores
    output_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        (M, N_per_core),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    # ========================================================================
    # Create mesh tensors for input and intermediate (CCL broadcast destination)
    # ========================================================================
    sender_coord = ttnn.MeshCoordinate(sender_row, sender_col)
    device_tensors = []
    intermediate_tensors = []
    for row in range(mesh_rows):
        for col in range(mesh_cols):
            if skip_ccl:
                # Single-device mode: all devices have the input
                device_tensors.append(torch_a)
            elif row == sender_row and col == sender_col:
                # Only sender device has actual input data
                device_tensors.append(torch_a)
            else:
                # All other devices start with zeros
                device_tensors.append(torch.zeros_like(torch_a))
            intermediate_tensors.append(torch.zeros_like(torch_a))

    mesh_tensor_torch = torch.cat(device_tensors, dim=0)
    intermediate_mesh_tensor_torch = torch.cat(intermediate_tensors, dim=0)

    input_tensor_mesh = ttnn.from_torch(
        mesh_tensor_torch,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=0),
    )

    intermediate_tensor_mesh = ttnn.from_torch(
        intermediate_mesh_tensor_torch,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=0),
    )

    logger.info(
        f"Created input mesh tensor with shard shape ({M}, {K}) on sender core ({mcast_core_x}, {mcast_core_y})"
    )

    # Vocab weights: replicated across mesh, width-sharded across matmul cores
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_b_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    logger.info(f"Created vocab weights with shard shape {input_b_shard_shape} on {num_matmul_cores} matmul cores")

    # Output: replicated across mesh, width-sharded across matmul cores
    ttnn_output = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    logger.info(f"Created output tensor with shard shape ({M}, {N_per_core}) on {num_matmul_cores} matmul cores")

    # ========================================================================
    # Semaphores (for CCL broadcast synchronization)
    # ========================================================================
    num_cores = device_grid_size.x * device_grid_size.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, device_grid_size, row_wise=True)

    out_ready_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    barrier_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    secondary_sync_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    semaphores = [out_ready_semaphore, barrier_semaphore, secondary_sync_semaphore]

    # ========================================================================
    # Run post-decode operation
    # ========================================================================
    logger.info(
        f"Running post-decode{fp32_str} operation: "
        f"sender=({sender_row},{sender_col}), mesh={mesh_rows}x{mesh_cols}, skip_ccl={skip_ccl}"
    )
    ttnn_result = PostDecode.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_b,
        ttnn_output,
        sender_coord,
        semaphores=semaphores,
        cluster_axis=cluster_axis,
        secondary_cluster_axis=secondary_cluster_axis,
        using_persistent_buffers=using_persistent_buffers,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=skip_ccl,
    )

    ttnn.synchronize_device(submesh)

    # ========================================================================
    # Verify output — every device should produce the same matmul result
    # ========================================================================
    output_tensor_torch = ttnn.to_torch(ttnn_result, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))

    pcc_threshold = 0.99
    slice_size = output_shape[0]
    for device_idx in range(num_devices):
        start = device_idx * slice_size
        end = start + slice_size
        received = output_tensor_torch[start:end, :]

        assert (
            received.shape == torch_expected.shape
        ), f"Shape mismatch at device {device_idx}: expected {torch_expected.shape}, got {received.shape}"

        passing, pcc_message = comp_pcc(torch_expected, received, pcc_threshold)
        logger.info(f"Device {device_idx}: {pcc_message}")

        assert passing, f"Device {device_idx} failed: {pcc_message}"

    # Cleanup
    submesh.reset_sub_device_stall_group()
    submesh.clear_loaded_sub_device_manager()

    logger.info(
        f"Post-decode{fp32_str} test passed! "
        f"({num_matmul_cores} cores, [{M}, {K}] x [{K}, {N_total}], "
        f"mesh={mesh_rows}x{mesh_cols}, skip_ccl={skip_ccl})"
    )
