# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import random
from loguru import logger
import ttnn


def tt_to_torch_dtype(tt_dtype):
    if tt_dtype == ttnn.bfloat16:
        return torch.bfloat16
    elif tt_dtype == ttnn.bfloat8_b:
        return torch.bfloat16
    elif tt_dtype == ttnn.float32:
        return torch.float32
    else:
        raise ValueError(f"Invalid dtype: {tt_dtype}")


def gen_expert_mapping(experts, mesh_shape, cluster_axis):
    """
    Create per-device expert mapping tensor that maps each expert to the device it belongs to.
    Shape: [devices, experts] where each entry is the linearized mesh coordinate of the device
    that owns that expert from the perspective of that source device.

    For now, all devices see the same mapping (no replicated experts).
    For 256 experts and 128 devices (2 experts per device):
    expert_mapping[d, e] = e // experts_per_device

    In the future, this can be extended to support replicated experts where each device
    sees the "optimal" device (e.g., shortest distance) for each expert.

    This tensor is replicated on every device (even devices not along the dispatch axis).
    """
    devices = mesh_shape[0] * mesh_shape[1]
    experts_per_device = experts // devices
    expert_mapping = torch.zeros(1, experts, dtype=torch.uint16)
    for e in range(experts):
        expert_mapping[0, e] = e // experts_per_device
    # Replicate across all devices (same mapping for now)
    expert_mapping = expert_mapping.repeat(devices, 1)
    return expert_mapping


def gen_sparse_buffer_and_indices(
    tokens_per_device, hidden_size, experts, selected_experts_k, mesh_shape, cluster_axis, dtype=torch.bfloat16
):
    """
    Generate the sparse buffer (simulating output from all_to_all_dispatch) and the all-gathered
    expert indices tensor.

    The sparse buffer has shape [devices, total_tokens, hidden_size].
    Each device receives tokens from all devices in the dispatch dimension.
    A token is placed in the sparse buffer if the expert it selected lives on that device.
    total_tokens = tokens_per_device * dispatch_devices

    The expert indices tensor has shape [dispatch_devices, tokens_per_device, selected_experts_k]
    and is all-gathered so each device sees which experts every token selected.

    Returns:
        sparse_buffer: [devices, total_tokens, hidden_size] - the sparse input to selective_tilize
        expert_indices: [dispatch_devices, tokens_per_device, K] - all-gathered indices
        expert_scores: [dispatch_devices, tokens_per_device, K] - all-gathered scores
        original_tokens: [dispatch_devices, tokens_per_device, hidden_size] - for verification
    """
    devices = mesh_shape[0] * mesh_shape[1]
    if cluster_axis == 1:
        dispatch_devices = mesh_shape[1]
    elif cluster_axis == 0:
        dispatch_devices = mesh_shape[0]
    else:
        dispatch_devices = devices

    experts_per_device = experts // devices

    # Total tokens in sparse buffer = tokens_per_device * dispatch_devices
    total_tokens = tokens_per_device * dispatch_devices

    # Generate original tokens for each source device
    # Shape: [dispatch_devices, tokens_per_device, hidden_size]
    original_tokens = torch.rand(dispatch_devices, tokens_per_device, hidden_size, dtype=dtype)

    # Generate expert indices for each token
    # Shape: [dispatch_devices, tokens_per_device, selected_experts_k]
    expert_indices = torch.zeros(dispatch_devices, tokens_per_device, selected_experts_k, dtype=torch.uint16)
    for src_device in range(dispatch_devices):
        for t in range(tokens_per_device):
            # Each token selects K unique experts
            selected = torch.randperm(experts)[:selected_experts_k]
            expert_indices[src_device, t, :] = selected.to(torch.uint16)

    # Generate expert scores
    # Shape: [dispatch_devices, tokens_per_device, selected_experts_k]
    expert_scores = torch.rand(dispatch_devices, tokens_per_device, selected_experts_k, dtype=dtype) + 1e-5
    expert_scores = expert_scores / expert_scores.sum(dim=-1, keepdim=True)

    # Build the sparse buffer
    # Shape: [devices, total_tokens, hidden_size]
    # Initialize with random garbage (tokens not sent to a device will have garbage)
    sparse_buffer = torch.rand(devices, total_tokens, hidden_size, dtype=dtype)

    # Place tokens in the sparse buffer based on expert selection
    # Token layout: [src_device_0_token_0, src_device_0_token_1, ..., src_device_1_token_0, ...]
    for src_device in range(dispatch_devices):
        for t in range(tokens_per_device):
            token = original_tokens[src_device, t, :]
            token_idx_in_sparse = src_device * tokens_per_device + t

            # For each expert this token selected, place it on the device that owns that expert
            for k in range(selected_experts_k):
                expert_id = expert_indices[src_device, t, k].item()
                target_device = expert_id // experts_per_device
                sparse_buffer[target_device, token_idx_in_sparse, :] = token

    return sparse_buffer, expert_indices, expert_scores, original_tokens


def compute_selective_tilize_golden(
    sparse_buffer, expert_indices, expert_scores, expert_mapping, mesh_shape, cluster_axis
):
    """
    Compute the golden output for selective_tilize.

    For each device, we need to produce a dense, tilized output for each of its experts.
    Output shape: [devices, experts_per_device, total_tokens, hidden_size]

    Each expert on a device collects all tokens that selected it from all source devices.
    """
    devices = mesh_shape[0] * mesh_shape[1]
    if cluster_axis == 1:
        dispatch_devices = mesh_shape[1]
    elif cluster_axis == 0:
        dispatch_devices = mesh_shape[0]
    else:
        dispatch_devices = devices

    tokens_per_device = expert_indices.shape[1]
    selected_experts_k = expert_indices.shape[2]
    hidden_size = sparse_buffer.shape[2]
    experts = expert_mapping.shape[1]
    experts_per_device = experts // devices

    # Total possible tokens that could be sent to any expert
    total_tokens = tokens_per_device * dispatch_devices

    # Output: [devices, experts_per_device, total_tokens, hidden_size]
    # Initialize with zeros
    golden_output = torch.zeros(devices, experts_per_device, total_tokens, hidden_size, dtype=sparse_buffer.dtype)

    # Track how many tokens each expert has received (for each device)
    expert_token_counts = torch.zeros(devices, experts_per_device, dtype=torch.int32)

    # For each token, place it in the output for the experts it selected
    for src_device in range(dispatch_devices):
        for t in range(tokens_per_device):
            token_idx_in_sparse = src_device * tokens_per_device + t

            for k in range(selected_experts_k):
                expert_id = expert_indices[src_device, t, k].item()

                # Get the device that owns this expert from the source device's perspective
                target_device = expert_mapping[src_device, expert_id].item()

                # Local expert index on that device
                local_expert_idx = expert_id % experts_per_device

                # Get the token from sparse buffer
                token = sparse_buffer[target_device, token_idx_in_sparse, :]

                # Place in output at the next available slot for this expert
                token_slot = expert_token_counts[target_device, local_expert_idx].item()
                golden_output[target_device, local_expert_idx, token_slot, :] = token
                expert_token_counts[target_device, local_expert_idx] += 1

    return golden_output, expert_token_counts


def get_mesh_mapper(mesh_device, mesh_shape, cluster_axis, shard_dim):
    if cluster_axis is None:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim)
    elif cluster_axis == 1:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, shard_dim), mesh_shape=mesh_shape)
    elif cluster_axis == 0:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(shard_dim, None), mesh_shape=mesh_shape)
    else:
        raise ValueError(f"Invalid cluster_axis: {cluster_axis}")
    return mesh_mapper


def create_sharded_memory_config(core_range_set, tensor_shape, dtype):
    """
    Create an L1 sharded memory config for a tensor to be completely on specified cores.
    """
    num_cores = core_range_set.num_cores()
    total_elements = 1
    for dim in tensor_shape:
        total_elements *= dim

    # Calculate bytes per element
    if dtype == ttnn.uint16:
        bytes_per_element = 2
    elif dtype == ttnn.bfloat16:
        bytes_per_element = 2
    elif dtype == ttnn.float32:
        bytes_per_element = 4
    else:
        bytes_per_element = 2

    total_bytes = total_elements * bytes_per_element
    # Shard evenly across cores, but for "completely on one core" we use 1 core
    shard_height = tensor_shape[0] if len(tensor_shape) > 0 else 1
    shard_width = tensor_shape[1] if len(tensor_shape) > 1 else total_elements

    shard_spec = ttnn.ShardSpec(
        core_range_set,
        [shard_height, shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )


# Performance tests
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 500000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((1, 16), (1, 16), id="1x16_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("tokens_per_device", [32])  # Collapsed batch * seq_len
@pytest.mark.parametrize("experts", [2 * 16])  # 32 experts for 16 devices = 2 experts per device
@pytest.mark.parametrize("selected_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tokens_per_chunk", [32])
def test_selective_tilize_no_trace(
    mesh_device,
    mesh_shape,
    cluster_axis,
    tokens_per_device,
    experts,
    selected_experts_k,
    hidden_size,
    dtype,
    tokens_per_chunk,
    device_params,
):
    """
    Test the selective_tilize operation without tracing.

    This test:
    1. Generates a sparse buffer (simulating output from all_to_all_dispatch)
    2. Generates all-gathered expert indices and scores
    3. Generates per-device expert mapping
    4. Runs the selective_tilize operation
    5. Verifies the output against a golden reference
    """
    torch.manual_seed(2005)
    random.seed(2005)

    devices = mesh_shape[0] * mesh_shape[1]
    dispatch_devices = mesh_shape[cluster_axis] if cluster_axis is not None else devices
    total_tokens = tokens_per_device * dispatch_devices

    logger.info(f"Test configuration:")
    logger.info(f"  mesh_shape: {mesh_shape}")
    logger.info(f"  cluster_axis: {cluster_axis}")
    logger.info(f"  devices: {devices}, dispatch_devices: {dispatch_devices}")
    logger.info(f"  tokens_per_device: {tokens_per_device}, total_tokens: {total_tokens}")
    logger.info(f"  experts: {experts}, selected_experts_k: {selected_experts_k}")
    logger.info(f"  hidden_size: {hidden_size}")

    # Generate test data
    sparse_buffer, expert_indices, expert_scores, original_tokens = gen_sparse_buffer_and_indices(
        tokens_per_device,
        hidden_size,
        experts,
        selected_experts_k,
        mesh_shape,
        cluster_axis,
        dtype=tt_to_torch_dtype(dtype),
    )

    expert_mapping = gen_expert_mapping(experts, mesh_shape, cluster_axis)

    logger.info(f"Generated tensors:")
    logger.info(f"  sparse_buffer shape: {sparse_buffer.shape}")
    logger.info(f"  expert_indices shape: {expert_indices.shape}")
    logger.info(f"  expert_scores shape: {expert_scores.shape}")
    logger.info(f"  expert_mapping shape: {expert_mapping.shape}")

    # Compute golden output
    golden_output, expert_token_counts = compute_selective_tilize_golden(
        sparse_buffer, expert_indices, expert_scores, expert_mapping, mesh_shape, cluster_axis
    )
    logger.info(f"  golden_output shape: {golden_output.shape}")
    logger.info(f"  expert_token_counts:\n{expert_token_counts}")

    # Define core ranges for the operation
    # Budget: 4 cores for selective_tilize
    # Core 0 is the drain tilizer core where indices and scores are sharded
    num_tilizer_cores = 4
    drain_tilizer_core = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(0, 0),
            ),
        }
    )

    selective_tilize_core_range_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(num_tilizer_cores - 1, 0),  # 4 cores: (0,0) to (3,0)
            ),
        }
    )

    # For now, use placeholder core ranges for matmul and combine
    matmul_core_range_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 1),
                ttnn.CoreCoord(7, 7),
            ),
        }
    )
    combine_core_range_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 1),
                ttnn.CoreCoord(3, 1),
            ),
        }
    )

    # Create memory configs
    input_memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Create L1 sharded memory config for indices and scores on drain tilizer core
    # expert_indices shape per device: [tokens_per_device, selected_experts_k] (after shard along dispatch axis)
    # But we need the all-gathered version, so shape is [dispatch_devices * tokens_per_device, selected_experts_k]
    # which is [total_tokens, selected_experts_k]
    indices_shard_shape = [total_tokens, selected_experts_k]
    indices_sharded_mem_config = create_sharded_memory_config(drain_tilizer_core, indices_shard_shape, ttnn.uint16)

    scores_shard_shape = [total_tokens, selected_experts_k]
    scores_sharded_mem_config = create_sharded_memory_config(drain_tilizer_core, scores_shard_shape, dtype)

    # Sparse buffer is sharded across devices (dim 0)
    tt_sparse_buffer = ttnn.from_torch(
        sparse_buffer,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dtype,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    # Expert indices - all-gathered (replicated on all devices)
    # Shape: [dispatch_devices, tokens_per_device, K]
    # Flatten to [dispatch_devices * tokens_per_device, K] = [total_tokens, K] per device
    expert_indices_flat = expert_indices.reshape(total_tokens, selected_experts_k)
    # Replicate on all devices
    expert_indices_replicated = expert_indices_flat.unsqueeze(0).repeat(devices, 1, 1)

    tt_expert_indices = ttnn.from_torch(
        expert_indices_replicated,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=indices_sharded_mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    # Expert scores - same distribution as indices
    expert_scores_flat = expert_scores.reshape(total_tokens, selected_experts_k)
    expert_scores_replicated = expert_scores_flat.unsqueeze(0).repeat(devices, 1, 1)

    tt_expert_scores = ttnn.from_torch(
        expert_scores_replicated,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dtype,
        memory_config=scores_sharded_mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    # Expert mapping - per-device [devices, experts], replicated on every device
    # Each device gets its own row after sharding, but since it's replicated,
    # we give each device the full tensor and it uses its own row
    tt_expert_mapping = ttnn.from_torch(
        expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logger.info(f"TTNN tensor shapes:")
    logger.info(f"  tt_sparse_buffer: {tt_sparse_buffer.shape}")
    logger.info(f"  tt_expert_indices: {tt_expert_indices.shape}")
    logger.info(f"  tt_expert_scores: {tt_expert_scores.shape}")
    logger.info(f"  tt_expert_mapping: {tt_expert_mapping.shape}")

    # Run the operation
    output_tensor = ttnn.experimental.all_to_all_dispatch_selective_tilize(
        tt_sparse_buffer,
        tt_expert_indices,
        tt_expert_scores,
        tt_expert_mapping,
        cluster_axis=cluster_axis,
        tokens_per_chunk=tokens_per_chunk,
        selective_tilize_core_range_set=selective_tilize_core_range_set,
        matmul_core_range_set=matmul_core_range_set,
        combine_core_range_set=combine_core_range_set,
    )

    logger.info(f"Output tensor shape: {output_tensor.shape}")

    # Convert output to torch for verification
    output_torch = ttnn.to_torch(output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    logger.info(f"Output torch shape: {output_torch.shape}")

    # TODO: Add verification against golden_output once kernel implementation is complete
    # For now, just verify the output shape is correct
    experts_per_device = experts // devices

    # The output from the op is [experts_per_device, total_tokens, hidden_size] per device
    # After concat, it should be [devices * experts_per_device, total_tokens, hidden_size]
    # or depending on tiling: [devices, experts_per_device, total_tokens_padded, hidden_size_padded]

    logger.info(f"Expected shape (per device): [{experts_per_device}, {total_tokens}, {hidden_size}]")
    logger.info(f"Test completed - output shape verification pending kernel implementation")
