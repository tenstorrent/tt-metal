# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pure Python implementation of tt-mlir wrappers of TT-NN CCL ops.
"""

import ttnn
from enum import Enum
from typing import List, Optional


class MeshShardDirection(Enum):
    """Direction for mesh sharding operations."""

    FullToShard = 0
    ShardToFull = 1


class MeshShardType(Enum):
    """Type of sharding strategy for mesh operations."""

    Identity = 0
    Replicate = 1
    Maximal = 2
    Devices = 3


def mesh_shard(
    input: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    shard_direction: MeshShardDirection,
    shard_type: MeshShardType,
    shard_shape: List[int],
    shard_dims: List[int],
) -> ttnn.Tensor:
    """
    Shard or aggregate a tensor across a mesh device.

    Args:
        input: The input tensor to shard or aggregate
        mesh_device: The mesh device to distribute across
        shard_direction: Direction of sharding (FullToShard or ShardToFull)
        shard_type: Type of sharding (Identity, Replicate, Maximal, or Devices)
        shard_shape: The shape of each shard
        shard_dims: The dimensions to shard over

    Returns:
        The sharded or aggregated tensor
    """
    # Identity type - just return the input as-is
    if shard_type == MeshShardType.Identity:
        assert (
            input.storage_type() == ttnn.StorageType.DEVICE
        ), "Input of mesh_shard with identity shard_type must be Device Storage."
        return input

    # For non-identity operations, input should be on host
    assert (
        input.storage_type() == ttnn.StorageType.HOST
    ), "Input of mesh_shard should be host tensor for replicate and devices operations."

    mesh_shape = mesh_device.shape

    if shard_direction == MeshShardDirection.FullToShard:
        # Distribute tensor from host to devices using mesh mapper
        # Initialize all placements as replicate by default
        placements = [ttnn.PlacementReplicate() for _ in range(mesh_shape.dims())]

        if shard_type == MeshShardType.Devices:
            # Override with shard placements where specified
            for i, shard_dim in enumerate(shard_dims):
                if shard_dim >= 0 and i < len(placements):
                    placements[i] = ttnn.PlacementShard(shard_dim)

        # Create mesh mapper configuration with placements
        mesh_mapper_config = ttnn.MeshMapperConfig(placements=placements)

        # Create mapper and distribute tensor
        mesh_mapper = ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config)
        output = ttnn.distribute_tensor(input, mesh_mapper)

    else:  # ShardToFull
        # Aggregate tensor from devices to host using mesh composer
        if shard_type == MeshShardType.Replicate:
            # For replicate, just concat along dimension 0 with mesh shape override
            dims = [0]
            mesh_shape_override = ttnn.MeshShape([1])
            mesh_composer_config = ttnn.MeshComposerConfig(
                dims=dims, mesh_shape_override=mesh_shape_override
            )
        elif shard_type == MeshShardType.Devices:
            # For devices, setup dimensions and mesh shape for concatenation
            input_rank = len(input.shape)

            # Helper to find non-overlapping dimension
            dims = []

            def get_non_overlapping_dim():
                for d in range(input_rank - 1, -1, -1):
                    if d not in shard_dims and d not in dims:
                        return d
                raise ValueError(
                    "All dimensions are overlapping, cannot find non-overlapping dimension"
                )

            target_sub_mesh_shape = []
            for dim_idx, dim in enumerate(shard_dims):
                if dim >= 0:
                    dims.append(dim)
                    target_sub_mesh_shape.append(mesh_shape[dim_idx])
                else:
                    dims.append(get_non_overlapping_dim())
                    target_sub_mesh_shape.append(1)

            mesh_composer_config = ttnn.MeshComposerConfig(
                dims=dims, mesh_shape_override=ttnn.MeshShape(target_sub_mesh_shape)
            )
        else:
            raise ValueError(f"Unsupported shard_type: {shard_type}")

        # Create composer and aggregate tensor
        mesh_composer = ttnn.create_mesh_composer(mesh_device, mesh_composer_config)
        output = ttnn.aggregate_tensor(input, mesh_composer)

    return output


def collective_permute(
    input: ttnn.Tensor, source_target_pairs: List[int]
) -> ttnn.Tensor:
    """
    Perform collective permute operation to remap tensor shards across devices.

    Redistributes tensor data by sending shards from source devices to target devices.
    Devices not participating receive zeros.

    Args:
        input: The input device tensor
        source_target_pairs: Flat list of [source_id, target_id, ...] pairs
                            Must have even length

    Returns:
        The permuted tensor with remapped shards
    """
    mesh_device = input.device()
    assert mesh_device is not None, "Tensor must belong to a mesh device"
    assert (
        len(source_target_pairs) % 2 == 0
    ), "Expected source_target_pairs to have size multiple of 2"
    assert (
        input.storage_type() == ttnn.StorageType.DEVICE
    ), "Input of collective_permute must be device storage."

    # Get individual device tensors
    host_tensors = ttnn.get_device_tensors(ttnn.from_device(input))
    num_devices = len(host_tensors)

    # Initialize output tensors
    new_host_tensors = [None] * num_devices
    found_dest_devices = [False] * num_devices

    # Process source-target pairs
    for i in range(0, len(source_target_pairs), 2):
        src = source_target_pairs[i]
        dest = source_target_pairs[i + 1]

        assert 0 <= src < num_devices, f"Source device id {src} is out of bounds!"
        assert (
            0 <= dest < num_devices
        ), f"Destination device id {dest} is out of bounds!"

        new_host_tensors[dest] = host_tensors[src]
        found_dest_devices[dest] = True

    # Zero out tensors for devices that didn't participate
    for i in range(num_devices):
        if not found_dest_devices[i]:
            # Create zero tensor with same shape as source
            new_host_tensors[i] = ttnn.zeros_like(host_tensors[i])
            found_dest_devices[i] = True

    # Combine all host tensor shards and send back to device
    output = ttnn.from_host_shards(new_host_tensors, mesh_device.shape)
    output = ttnn.to_device(output, mesh_device, input.memory_config())

    return output
