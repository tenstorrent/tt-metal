# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ReduceToAllB1 operation (Ring + Cross-Column algorithm).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_wormhole_b0
from models.demos.deepseek_v3_b1.micro_ops.reduce_to_all_b1.op import ReduceToAllB1


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def setup_reduce_to_all_test(mesh_device):
    """Common setup for reduce_to_all tests. Returns test configuration."""
    logger.info(f"mesh_device shape: {mesh_device.shape}")
    logger.info(f"mesh_device num_devices: {mesh_device.get_num_devices()}")

    mesh_rows, mesh_cols = mesh_device.shape
    if mesh_rows * mesh_cols < 8:
        pytest.skip(f"Need at least 8 devices, got {mesh_rows * mesh_cols}")
    logger.info(f"Mesh is {mesh_rows}x{mesh_cols} = {mesh_rows * mesh_cols} devices")

    num_devices = 8
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    logger.info(f"Created submesh with shape: {submesh_device.shape}")
    assert submesh_device.shape == ttnn.MeshShape((4, 2))

    tensor_shape = [1, 7168]
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    tile = ttnn.Tile((1, 32))

    compute_cores = submesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_shard_cores = 8
    shard_cores_list = compute_cores[:num_shard_cores]
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in shard_cores_list})

    shard_shape = [1, 896]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec)

    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], submesh_device.shape)
    mesh_mapper = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config)

    # Intermediate tensor: 3x shard width (R1/R2/R3 receive buffers)
    intermediate_shard_shape = [1, shard_shape[1] * 3]
    intermediate_tensor_shape = [1, tensor_shape[1] * 3]
    intermediate_shard_spec = ttnn.ShardSpec(shard_grid, intermediate_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    intermediate_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, intermediate_shard_spec
    )
    intermediate_data = torch.zeros([4, 2] + intermediate_tensor_shape, dtype=torch.bfloat16)
    intermediate_tensor = ttnn.from_torch(
        intermediate_data,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=intermediate_mem_config,
        mesh_mapper=mesh_mapper,
    )

    output_data = torch.zeros([4, 2] + tensor_shape, dtype=torch.bfloat16)
    output_tensor = ttnn.from_torch(
        output_data,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config,
        mesh_mapper=mesh_mapper,
    )

    # Random input data
    data_per_device = []
    torch.manual_seed(42)
    for _ in range(num_devices):
        data = torch.randn(tensor_shape, dtype=torch.bfloat16)
        data_per_device.append(data)

    data_all = torch.stack(data_per_device, dim=0).reshape(4, 2, *tensor_shape)
    input_tensor = ttnn.from_torch(
        data_all,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config,
        mesh_mapper=mesh_mapper,
    )

    ref_output = ReduceToAllB1.golden(data_per_device)

    # Create global semaphores for round receive (trace-safe — created before trace capture)
    compute_grid = submesh_device.compute_with_storage_grid_size()
    available_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))]
    )
    ttnn.synchronize_device(submesh_device)

    # 3 round receive semaphores (global — incremented by remote fabric packets)
    semaphores = [ttnn.create_global_semaphore(submesh_device, available_cores, 0) for _ in range(3)]

    ttnn.synchronize_device(submesh_device)

    return {
        "submesh_device": submesh_device,
        "input_tensor": input_tensor,
        "intermediate_tensor": intermediate_tensor,
        "output_tensor": output_tensor,
        "ref_output": ref_output,
        "semaphores": semaphores,
    }


def verify_output(output_tensor, submesh_device, ref_output):
    """Verify output matches reference on ALL devices."""
    output_torch = ttnn.to_torch(output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))

    num_devices = submesh_device.shape[0] * submesh_device.shape[1]
    rtol = 0.01
    atol = 0.05
    ref_flat = ref_output.flatten()

    all_match = True
    for device_idx in range(num_devices):
        output_device = output_torch[device_idx].flatten()
        match = torch.allclose(output_device, ref_flat, rtol=rtol, atol=atol)

        row = device_idx // submesh_device.shape[1]
        col = device_idx % submesh_device.shape[1]
        if not match:
            diff = torch.abs(output_device - ref_flat)
            logger.warning(
                f"Device ({row},{col}) idx={device_idx}: MISMATCH  max_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}"
            )
            logger.warning(f"  ref[:8]   = {ref_flat[:8]}")
            logger.warning(f"  got[:8]   = {output_device[:8]}")
            all_match = False
        else:
            logger.info(f"Device ({row},{col}) idx={device_idx}: OK")

    return all_match


def _call_op(config):
    """Helper to call ReduceToAllB1.op with the config dict."""
    return ReduceToAllB1.op(
        config["input_tensor"],
        config["intermediate_tensor"],
        config["output_tensor"],
        config["semaphores"],
    )


def run_reduce_to_all(mesh_device, num_iterations=1):
    """Run reduce_to_all test."""
    logger.info(f"Testing reduce_to_all (num_iterations={num_iterations})")
    config = setup_reduce_to_all_test(mesh_device)

    logger.info(f"Running reduce_to_all with {num_iterations} iterations...")
    output_tensor = ReduceToAllB1.op(
        config["input_tensor"],
        config["intermediate_tensor"],
        config["output_tensor"],
        config["semaphores"],
        num_iterations=num_iterations,
    )
    ttnn.synchronize_device(config["submesh_device"])

    logger.info("Verifying output on all devices...")
    match = verify_output(output_tensor, config["submesh_device"], config["ref_output"])
    assert match, "Output tensor does not match reference on one or more devices"
    logger.info("Test passed — all 8 devices hold the correct sum!")


# === Tests ===
@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
                "fabric_router_config": create_fabric_router_config(15232),
            }
        )
    ],
    indirect=["device_params"],
    ids=["fabric_2d_torus_x"],
)
def test_reduce_to_all_2d(bh_2d_mesh_device):
    """Test reduce_to_all with 2D torus-X fabric (ring wrap-around in column direction)."""
    run_reduce_to_all(bh_2d_mesh_device)


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
                "fabric_router_config": create_fabric_router_config(15232),
            }
        )
    ],
    indirect=["device_params"],
    ids=["fabric_2d_torus_x"],
)
def test_reduce_to_all_2d_multi_iter(bh_2d_mesh_device):
    """Test reduce_to_all with 2D torus-X fabric and multiple iterations."""
    run_reduce_to_all(bh_2d_mesh_device, num_iterations=100)
