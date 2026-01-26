# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.common.utility_functions import skip_for_wormhole_b0


def compute_reference_reduce_to_one(data_per_device):
    """
    Compute the reference output for reduce_to_one operation.
    Simple sum of all device tensors.
    """
    result = data_per_device[0].clone()
    for i in range(1, len(data_per_device)):
        result = result + data_per_device[i]
    return result


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=["device_params"],
    ids=["fabric_1d_linear"],
)
def test_reduce_to_one(bh_2d_mesh_device):
    """Test reduce_to_one operation on a 2x4 mesh."""

    # Log mesh device info
    logger.info(f"bh_2d_mesh_device shape: {bh_2d_mesh_device.shape}")
    logger.info(f"bh_2d_mesh_device num_devices: {bh_2d_mesh_device.get_num_devices()}")

    # Validate mesh has enough devices for 2x4 submesh
    mesh_rows, mesh_cols = bh_2d_mesh_device.shape
    assert mesh_rows * mesh_cols >= 8, f"Need at least 8 devices, got {mesh_rows * mesh_cols}"
    logger.info(f"Mesh is {mesh_rows}x{mesh_cols} = {mesh_rows * mesh_cols} devices")

    # Setup - create 2x4 submesh
    num_devices = 8
    root_coord = (1, 1)  # b1 is the final root

    topology = ttnn.Topology.Linear
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    logger.info(f"Created submesh with shape: {submesh_device.shape}")

    assert submesh_device.shape == ttnn.MeshShape((4, 2)), f"Expected 4x2 mesh, got {submesh_device.shape}"

    # Tensor shape: (1, 7168) sharded across 8 cores
    # Each core gets 7168/8 = 896 elements
    tensor_shape = [1, 7168]
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    tile = ttnn.Tile((1, 32))  # Tiny tile for (1, N) tensors

    # Shard config - 8 cores in 2 columns x 4 rows
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3)),
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 3)),
        }
    )

    # Each core gets 7168/8 = 896 elements width
    shard_shape = [1, 896]
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec)

    # Mesh mapper - shard across devices
    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], submesh_device.shape)
    mesh_mapper = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config)

    # Create intermediate tensor (same shape as input, used as receive buffer)
    # Use same shape [4, 2, 1, 7168] and same mesh_mapper as input
    intermediate_data = torch.zeros([4, 2] + tensor_shape, dtype=torch.bfloat16)
    intermediate_tensor = ttnn.from_torch(
        intermediate_data,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config,
        mesh_mapper=mesh_mapper,
    )

    print("\n=== Testing reduce_to_one ===")

    # Generate test data - different values for each device
    torch.manual_seed(42)
    data_per_device = []

    for device_idx in range(num_devices):
        # Each device gets unique data
        data = torch.randn(tensor_shape, dtype=torch.bfloat16) * 0.5 + device_idx
        data_per_device.append(data)

    # Reshape data to match mesh shape (4 rows, 2 cols)
    # data_per_device has 8 tensors, reshape to [4, 2, 1, 7168] for mesh mapping
    data_all = torch.stack(data_per_device, dim=0)  # [8, 1, 7168]
    data_all = data_all.reshape(4, 2, *tensor_shape)  # [4, 2, 1, 7168]

    # Create input tensor on mesh
    input_tensor = ttnn.from_torch(
        data_all,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config,
        mesh_mapper=mesh_mapper,
    )

    # Compute reference output (simple sum)
    ref_output = compute_reference_reduce_to_one(data_per_device)

    # Run reduce_to_one
    print("Running reduce_to_one...")
    output_tensor = ttnn.reduce_to_one(
        input_tensor,
        root_coord=ttnn.MeshCoordinate(root_coord),
        topology=topology,
        intermediate_tensor=intermediate_tensor,
    )
    ttnn.synchronize_device(submesh_device)

    # Verify output
    print("\nVerifying output...")
    output_torch = ttnn.to_torch(output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))

    # Get output from root device (linearized index in 2x4 mesh)
    root_device_idx = root_coord[0] * 4 + root_coord[1]
    output_root = output_torch[root_device_idx]

    # Tolerances for bfloat16
    rtol = 0.01
    atol = 0.05

    # Check output matches reference
    match = torch.allclose(output_root, ref_output, rtol=rtol, atol=atol)

    if not match:
        print(f"Output mismatch!")
        print(f"Reference:\n{ref_output[:1, :8]}")
        print(f"Output:\n{output_root[:1, :8]}")
        diff = torch.abs(output_root - ref_output)
        print(f"Max diff: {diff.max()}, Mean diff: {diff.mean()}")

    assert match, "Output tensor does not match reference"
    print("Test passed!")
