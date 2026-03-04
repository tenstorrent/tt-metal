"""Shared CCL unit-test helpers for DeepSeek B1 tests."""

from dataclasses import dataclass
from typing import Any

import torch

import ttnn


@dataclass(frozen=True)
class BroadcastTestInputs:
    input_tensor_torch: torch.Tensor
    input_tensor_mesh: Any
    output_tensor_mesh: Any
    semaphores: list[Any]


def create_fabric_router_config(max_payload_size):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def build_broadcast_test_inputs(
    *,
    mesh_device,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    bcast_core,
    num_links=1,
    input_tensor_torch=None,
    create_output_tensor_mesh=True,
    create_semaphores=True,
):
    """
    Build common broadcast test inputs with root-only source placement.

    Returns:
        BroadcastTestInputs with sender torch tensor, input/output mesh tensors, and semaphores.
    """
    if input_tensor_torch is None:
        input_tensor_torch = torch.rand(output_shape, dtype=torch.bfloat16)

    if not isinstance(bcast_core, ttnn.CoreCoord):
        raise TypeError(f"bcast_core must be ttnn.CoreCoord, got {type(bcast_core)}")

    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(bcast_core, bcast_core)})
    input_shard_spec = ttnn.ShardSpec(input_shard_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)

    device_tensors = []
    for row in range(mesh_rows):
        for col in range(mesh_cols):
            if row == sender_row and col == sender_col:
                device_tensors.append(input_tensor_torch)
            else:
                device_tensors.append(torch.zeros_like(input_tensor_torch))

    input_tensor_mesh = ttnn.from_torch(
        torch.cat(device_tensors, dim=0),
        device=mesh_device,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    output_tensor_mesh = None
    if create_output_tensor_mesh:
        output_tensor_mesh = ttnn.from_torch(
            torch.zeros(output_shape, dtype=torch.bfloat16),
            device=mesh_device,
            layout=layout,
            tile=ttnn.Tile((1, 32)),
            dtype=input_dtype,
            memory_config=input_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    semaphores = []
    if create_semaphores:
        compute_grid_size = mesh_device.compute_with_storage_grid_size()
        num_cores = compute_grid_size.x * compute_grid_size.y
        available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, row_wise=True)
        semaphores = [ttnn.create_global_semaphore(mesh_device, available_cores, 0) for _ in range(num_links)]
        ttnn.synchronize_device(mesh_device)

    return BroadcastTestInputs(
        input_tensor_torch=input_tensor_torch,
        input_tensor_mesh=input_tensor_mesh,
        output_tensor_mesh=output_tensor_mesh,
        semaphores=semaphores,
    )
