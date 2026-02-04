# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import random
from loguru import logger
import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.test_moe import (
    create_torch_w0,
    create_torch_w1,
    create_torch_w2,
    prepare_w0_w1_tensor,
    prepare_w2_tensor,
)


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
    Shape: [num_devices, experts] where each entry is the linearized mesh coordinate of the device
    that owns that expert from the perspective of that source device.

    For now, all devices see the same mapping (no replicated experts).
    For 256 experts and 128 devices (2 experts per device):
    expert_mapping[d, e] = e // experts_per_device

    In the future, this can be extended to support replicated experts where each device
    sees the "optimal" device (e.g., shortest distance) for each expert.

    This tensor is replicated on every device (even devices not along the dispatch axis).
    """
    num_devices = mesh_shape[0] * mesh_shape[1]
    experts_per_device = experts // num_devices
    expert_mapping = torch.zeros(1, experts, dtype=torch.uint16)
    for e in range(experts):
        expert_mapping[0, e] = e // experts_per_device
    # Replicate across all devices (same mapping for now)
    expert_mapping = expert_mapping.repeat(num_devices, 1)
    return expert_mapping


def gen_sparse_buffer_and_indices(
    tokens_per_device, hidden_size, experts, selected_experts_k, mesh_shape, cluster_axis, dtype=torch.bfloat16
):
    """
    Generate the sparse buffer (simulating output from all_to_all_dispatch) and the all-gathered
    expert indices tensor.

    The sparse buffer has shape [num_devices, total_tokens, hidden_size].
    Each device receives tokens from all devices in the dispatch dimension.
    A token is placed in the sparse buffer if the expert it selected lives on that device.
    total_tokens = tokens_per_device * num_dispatch_devices

    The expert indices tensor has shape [num_dispatch_devices, tokens_per_device, selected_experts_k]
    and is all-gathered so each device sees which experts every token selected.

    Returns:
        sparse_buffer: [num_devices, total_tokens, hidden_size] - the sparse input to selective_tilize
        expert_indices: [num_dispatch_devices, tokens_per_device, K] - all-gathered indices
        expert_scores: [num_dispatch_devices, tokens_per_device, K] - all-gathered scores
        original_tokens: [num_dispatch_devices, tokens_per_device, hidden_size] - for verification
    """
    num_devices = mesh_shape[0] * mesh_shape[1]
    if cluster_axis == 1:
        num_dispatch_devices = mesh_shape[1]
    elif cluster_axis == 0:
        num_dispatch_devices = mesh_shape[0]
    else:
        num_dispatch_devices = num_devices

    experts_per_device = experts // num_devices

    # Total tokens in sparse buffer = tokens_per_device * num_dispatch_devices
    total_tokens = tokens_per_device * num_dispatch_devices

    # Generate original tokens for each source device
    # Shape: [num_dispatch_devices, tokens_per_device, hidden_size]
    original_tokens = torch.rand(num_dispatch_devices, tokens_per_device, hidden_size, dtype=dtype)

    # Generate expert indices for each token
    # Shape: [num_dispatch_devices, tokens_per_device, selected_experts_k]
    expert_indices = torch.zeros(num_dispatch_devices, tokens_per_device, selected_experts_k, dtype=torch.uint16)
    for src_device in range(num_dispatch_devices):
        for t in range(tokens_per_device):
            # Each token selects K unique experts
            selected = torch.randperm(experts)[:selected_experts_k]
            expert_indices[src_device, t, :] = selected.to(torch.uint16)

    # Generate expert scores
    # Shape: [num_dispatch_devices, tokens_per_device, selected_experts_k]
    expert_scores = torch.rand(num_dispatch_devices, tokens_per_device, selected_experts_k, dtype=dtype) + 1e-5
    expert_scores = expert_scores / expert_scores.sum(dim=-1, keepdim=True)

    # Build the sparse buffer
    # Shape: [num_devices, total_tokens, hidden_size]
    # Initialize with random garbage (tokens not sent to a device will have garbage)
    sparse_buffer = torch.rand(num_devices, total_tokens, hidden_size, dtype=dtype)

    # Place tokens in the sparse buffer based on expert selection
    # Token layout: [src_device_0_token_0, src_device_0_token_1, ..., src_device_1_token_0, ...]
    for src_device in range(num_dispatch_devices):
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
    Output shape: [num_devices, experts_per_device, total_tokens, hidden_size]

    Each expert on a device collects all tokens that selected it from all source devices.
    """
    num_devices = mesh_shape[0] * mesh_shape[1]
    if cluster_axis == 1:
        num_dispatch_devices = mesh_shape[1]
    elif cluster_axis == 0:
        num_dispatch_devices = mesh_shape[0]
    else:
        num_dispatch_devices = num_devices

    tokens_per_device = expert_indices.shape[1]
    selected_experts_k = expert_indices.shape[2]
    hidden_size = sparse_buffer.shape[2]
    experts = expert_mapping.shape[1]
    experts_per_device = experts // num_devices

    # Total possible tokens that could be sent to any expert
    total_tokens = tokens_per_device * num_dispatch_devices

    # Output: [devices, experts_per_device, total_tokens, hidden_size]
    # Initialize with zeros
    golden_output = torch.zeros(num_devices, experts_per_device, total_tokens, hidden_size, dtype=sparse_buffer.dtype)

    # Track how many tokens each expert has received (for each device)
    expert_token_counts = torch.zeros(num_devices, experts_per_device, dtype=torch.int32)

    # For each token, place it in the output for the experts it selected
    for src_device in range(num_dispatch_devices):
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


def compute_expert_activation_golden(expert_indices, expert_scores, expert_mapping, mesh_shape, cluster_axis):
    """
    Compute the golden expert_activation tensor for each device.

    For each device, the expert_activation tensor contains rows for tokens that selected
    at least one expert on that device. Each row format:
    [token_id, k_idx_0, k_idx_1, ..., score_0, score_1, ...]

    Where:
    - token_id: global token index (src_device * tokens_per_device + local_token_idx)
    - k_idx_e: which of the K selected experts (0..K-1) maps to local expert e, or -1 if not selected
    - score_e: the score for local expert e (as bfloat16 bits in uint32), or 0 if not selected

    The last row is a sentinel with token_id = -1 (0xFFFFFFFF as uint32).

    Returns:
        golden_activation: dict[device_idx] -> list of activation row dicts
        Each row dict has: token_id, k_indices (list), scores (list)
    """
    num_devices = mesh_shape[0] * mesh_shape[1]
    if cluster_axis == 1:
        num_dispatch_devices = mesh_shape[1]
    elif cluster_axis == 0:
        num_dispatch_devices = mesh_shape[0]
    else:
        num_dispatch_devices = num_devices

    tokens_per_device = expert_indices.shape[1]
    selected_experts_k = expert_indices.shape[2]
    experts = expert_mapping.shape[1]
    experts_per_device = experts // num_devices

    # Build activation rows for each device
    # golden_activation[device] = list of (token_id, k_indices, scores)
    golden_activation = {d: [] for d in range(num_devices)}

    for src_device in range(num_dispatch_devices):
        for t in range(tokens_per_device):
            global_token_id = src_device * tokens_per_device + t

            # Track which local experts this token activated on each device
            # device -> {local_expert_idx: (k, score)}
            device_activations = {d: {} for d in range(num_devices)}

            for k in range(selected_experts_k):
                expert_id = expert_indices[src_device, t, k].item()
                target_device = expert_mapping[src_device, expert_id].item()
                local_expert_idx = expert_id % experts_per_device
                score = expert_scores[src_device, t, k].item()

                # Store the k-index and score for this local expert
                device_activations[target_device][local_expert_idx] = (k, score)

            # For each device that has at least one activation, add a row
            for device_idx in range(num_devices):
                if device_activations[device_idx]:
                    k_indices = [-1] * experts_per_device  # -1 means not activated
                    scores = [0.0] * experts_per_device

                    for local_exp_idx, (k, score) in device_activations[device_idx].items():
                        k_indices[local_exp_idx] = k
                        scores[local_exp_idx] = score

                    golden_activation[device_idx].append(
                        {
                            "token_id": global_token_id,
                            "k_indices": k_indices,
                            "scores": scores,
                        }
                    )

    return golden_activation, experts_per_device


def compute_e_t_golden(expert_indices, expert_mapping, mesh_shape, cluster_axis):
    """
    Compute the golden e_t (expert-to-token) tensor for each device.

    For each device and each local expert, this builds a list of token IDs that
    activate that expert. The list is stored in sequential order as tokens are
    processed, and is terminated with a -1 sentinel.

    Returns:
        golden_e_t: dict[device_idx] -> dict[local_expert_idx] -> list of token_ids
    """
    num_devices = mesh_shape[0] * mesh_shape[1]
    if cluster_axis == 1:
        num_dispatch_devices = mesh_shape[1]
    elif cluster_axis == 0:
        num_dispatch_devices = mesh_shape[0]
    else:
        num_dispatch_devices = num_devices

    tokens_per_device = expert_indices.shape[1]
    selected_experts_k = expert_indices.shape[2]
    experts = expert_mapping.shape[1]
    experts_per_device = experts // num_devices

    # Build e_t lists for each device and each local expert
    # golden_e_t[device][local_expert] = [token_id_0, token_id_1, ...]
    golden_e_t = {d: {e: [] for e in range(experts_per_device)} for d in range(num_devices)}

    for src_device in range(num_dispatch_devices):
        for t in range(tokens_per_device):
            global_token_id = src_device * tokens_per_device + t

            for k in range(selected_experts_k):
                expert_id = expert_indices[src_device, t, k].item()
                target_device = expert_mapping[src_device, expert_id].item()
                local_expert_idx = expert_id % experts_per_device

                # Each token selects K unique experts, so no duplicate tracking needed
                golden_e_t[target_device][local_expert_idx].append(global_token_id)

    return golden_e_t, experts_per_device


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
def test_moe(
    mesh_device,
    mesh_shape,
    cluster_axis,
    tokens_per_device,
    experts,
    selected_experts_k,
    hidden_size,
    dtype,
    device_params,
):
    """
    Test the moe operation without tracing.

    This test:
    1. Generates a sparse buffer (simulating output from all_to_all_dispatch)
    2. Generates all-gathered expert indices and scores
    3. Generates per-device expert mapping
    4. Runs the moe operation
    5. Verifies the output against a golden reference
    """
    torch.manual_seed(2005)
    random.seed(2005)

    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = mesh_shape[cluster_axis] if cluster_axis is not None else num_devices
    total_tokens = tokens_per_device * num_dispatch_devices

    logger.info(f"Test configuration:")
    logger.info(f"  mesh_shape: {mesh_shape}")
    logger.info(f"  cluster_axis: {cluster_axis}")
    logger.info(f"  num_devices: {num_devices}, num_dispatch_devices: {num_dispatch_devices}")
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

    # Compute golden expert_activation
    golden_activation, experts_per_device_check = compute_expert_activation_golden(
        expert_indices, expert_scores, expert_mapping, mesh_shape, cluster_axis
    )
    for d in range(num_devices):
        logger.info(f"  Device {d} activated tokens: {len(golden_activation[d])}")

    # Drain tilize core is core (5,0) where indices and scores are sharded
    tilize_drain_core = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(5, 0),
                ttnn.CoreCoord(5, 0),
            ),
        }
    )

    # Create memory configs
    tilize_input_memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Create L1 sharded memory config for indices and scores on drain tilizer core
    # expert_indices shape per device: [tokens_per_device, selected_experts_k] (after shard along dispatch axis)
    # But we need the all-gathered version, so shape is [num_dispatch_devices * tokens_per_device, selected_experts_k]
    # which is [total_tokens, selected_experts_k]
    indices_shard_shape = [total_tokens, selected_experts_k]
    indices_sharded_mem_config = create_sharded_memory_config(tilize_drain_core, indices_shard_shape, ttnn.uint16)

    scores_shard_shape = [total_tokens, selected_experts_k]
    scores_sharded_mem_config = create_sharded_memory_config(tilize_drain_core, scores_shard_shape, dtype)

    # Sparse buffer is sharded across devices (dim 0)
    tt_sparse_buffer = ttnn.from_torch(
        sparse_buffer,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dtype,
        memory_config=tilize_input_memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    # Expert indices - all-gathered (replicated on all devices)
    # Shape: [num_dispatch_devices, tokens_per_device, K]
    # Flatten to [num_dispatch_devices * tokens_per_device, K] = [total_tokens, K] per device
    expert_indices_flat = expert_indices.reshape(total_tokens, selected_experts_k)
    # Replicate on all devices
    expert_indices_replicated = expert_indices_flat.unsqueeze(0).repeat(num_devices, 1, 1)

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
    expert_scores_replicated = expert_scores_flat.unsqueeze(0).repeat(num_devices, 1, 1)

    tt_expert_scores = ttnn.from_torch(
        expert_scores_replicated,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dtype,
        memory_config=scores_sharded_mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    # Expert mapping - per-device [num_devices, experts], replicated on every device
    # Each device gets its own row after sharding, but since it's replicated,
    # we give each device the full tensor and it uses its own row
    tt_expert_mapping = ttnn.from_torch(
        expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=tilize_input_memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logger.info(f"TTNN tensor shapes:")
    logger.info(f"  tt_sparse_buffer: {tt_sparse_buffer.shape}")
    logger.info(f"  tt_expert_indices: {tt_expert_indices.shape}")
    logger.info(f"  tt_expert_scores: {tt_expert_scores.shape}")
    logger.info(f"  tt_expert_mapping: {tt_expert_mapping.shape}")

    #################################
    ###### START: MATMUL SETUP ######
    #################################

    SHAPE2TIME = {
        (32, 7168, 2048, 2, 1): 225.0,
        # (32, 7168, 2048, 3, 1): 329.0,
    }

    ((M, K, N, E, L),) = SHAPE2TIME.keys()
    layer_id = 0

    in0_dtype = ttnn.bfloat16
    w0_dtype = ttnn.bfloat4_b

    # --------------------------------------------------------------------------
    # Shard grid
    # --------------------------------------------------------------------------
    PAD_CORES = {2, 3, 4, 5, 6, 7, 10, 11}

    in0_core_coords = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(mesh_device, 0)
    core2dram = {}
    for dram_bank_id, core_coords in enumerate(in0_core_coords):
        core2dram[core_coords] = dram_bank_id

    in0_num_cores = len(in0_core_coords)

    # Make a new list of core coords that are sorted in decreasing order by y coordinate and then x coordinate.
    in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)

    ring2cores = {}
    for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
        # key: ring_pos, value: (core_coord, dram_bank_id, pad_flag)
        ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in PAD_CORES else 0)

    in0_core_range = [ttnn.CoreRange(ring2cores[i][0], ring2cores[i][0]) for i in range(in0_num_cores)]
    in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)

    num_dram_banks = 12

    dram_core_coords = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(in0_num_cores)]
    dram_core_range = [ttnn.CoreRange(dram_core_coord, dram_core_coord) for dram_core_coord in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    torch_w0 = create_torch_w0(L, E, K, N)
    torch_w1 = create_torch_w1(L, E, K, N)
    torch_w2 = create_torch_w2(L, E, N, K)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w0_w1
    # Tensor shape: (L, E, K, 4608) -> padded and reordered to (12, L, E, 6, K, 64)
    # ------------------------------------------------------------------------
    w0_w1_shard_height = L * E * 3 * K
    w0_w1_shard_width = 4 * ttnn.TILE_SIZE

    w0_w1_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w0_w1_shard_height, w0_w1_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w0_w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w0_w1_shard_spec)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w2
    # Tensor shape: (L, E, N, K) -> padded and reordered to (12, L, E, 5, N + 192, 128)
    # ------------------------------------------------------------------------
    w2_shard_height = L * E * 5 * (N + 192)
    w2_shard_width = 4 * ttnn.TILE_SIZE

    w2_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w2_shard_height, w2_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w2_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)

    # ------------------------------------------------------------------------
    # Prepare w0_w1 tensor (interleaved, padded, and reordered)
    torch_w0_w1_reordered = prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, K, N, ring2cores)

    # Create tt_w0_w1 tensor with DRAM sharding
    tt_w0_w1 = ttnn.from_torch(
        torch_w0_w1_reordered,
        dtype=w0_dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w0_w1_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # ------------------------------------------------------------------------
    # Prepare w2 tensor (padded and reordered)
    torch_w2_reordered = prepare_w2_tensor(torch_w2, L, E, N, K, ring2cores)

    # Create tt_w2 tensor with DRAM sharding
    tt_w2 = ttnn.from_torch(
        torch_w2_reordered,
        dtype=w0_dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w2_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    ###############################
    ###### END: MATMUL SETUP ######
    ###############################

    # TODO: (GR)
    # Run the operation
    (
        per_expert_total_tokens_output_tensor,
        expert_activation_output_tensor,
        e_t_output_tensor,
        matmul_output_tensor,
    ) = ttnn.experimental.moe_compute(
        tt_sparse_buffer,
        tt_expert_indices,
        tt_expert_scores,
        tt_expert_mapping,
        tt_w0_w1,
        tt_w2,
        layer_id=layer_id,
        cluster_axis=cluster_axis,
    )

    # logger.info(f"Output tensor shape: {output_tensor.shape}")
    logger.info(f"Expert activation tensor shape: {expert_activation_output_tensor.shape}")

    # Convert output to torch for verification
    # output_torch = ttnn.to_torch(output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    # logger.info(f"Output torch shape: {output_torch.shape}")

    experts_per_device = experts // num_devices

    # The output from the op is [experts_per_device, total_tokens, hidden_size] per device (TILE_LAYOUT)
    # After concat along dim=0, shape is [num_devices * experts_per_device, total_tokens, hidden_size]
    # We need to reshape to [num_devices, experts_per_device, total_tokens, hidden_size] for comparison

    # Handle potential tile padding in the output
    # output_total_tokens = output_torch.shape[1]
    # output_hidden_size = output_torch.shape[2]

    # logger.info(f"Expected shape (per device): [{experts_per_device}, {total_tokens}, {hidden_size}]")
    # logger.info(f"Actual output (concatenated): {output_torch.shape}")

    # Reshape concatenated output to [num_devices, experts_per_device, total_tokens, hidden_size]
    # output_reshaped = output_torch.reshape(num_devices, experts_per_device, output_total_tokens, output_hidden_size)

    # Remove tile padding if present (output may be padded to tile boundaries)
    # if output_total_tokens > total_tokens:
    #     output_reshaped = output_reshaped[:, :, :total_tokens, :]
    # if output_hidden_size > hidden_size:
    #     output_reshaped = output_reshaped[:, :, :, :hidden_size]

    # logger.info(f"Output reshaped (after removing padding): {output_reshaped.shape}")
    # logger.info(f"Golden output shape: {golden_output.shape}")

    # Verify output against golden
    # The output is dense: for each expert, the first N tokens are valid (where N = expert_token_counts)
    # and the remaining tokens are garbage. We only compare the valid tokens.
    # all_passed = True
    # total_comparisons = 0
    # total_mismatches = 0

    # for device_idx in range(num_devices):
    #     for expert_idx in range(experts_per_device):
    #         num_valid_tokens = expert_token_counts[device_idx, expert_idx].item()

    #         if num_valid_tokens == 0:
    #             continue

    #         # Extract valid tokens from output and golden
    #         output_slice = output_reshaped[device_idx, expert_idx, :num_valid_tokens, :]
    #         golden_slice = golden_output[device_idx, expert_idx, :num_valid_tokens, :]

    #         # Convert to float32 for comparison (bfloat16 comparison can be tricky)
    #         output_slice_f32 = output_slice.to(torch.float32)
    #         golden_slice_f32 = golden_slice.to(torch.float32)

    #         # Compute PCC (Pearson Correlation Coefficient)
    #         output_flat = output_slice_f32.flatten()
    #         golden_flat = golden_slice_f32.flatten()

    #         if golden_flat.numel() > 1:
    #             pcc = torch.corrcoef(torch.stack([output_flat, golden_flat]))[0, 1].item()
    #         else:
    #             pcc = 1.0 if torch.allclose(output_flat, golden_flat, rtol=1e-2, atol=1e-2) else 0.0

    #         # Check for exact match (with tolerance for bfloat16)
    #         is_close = torch.allclose(output_slice_f32, golden_slice_f32, rtol=1e-2, atol=1e-2)

    #         # Count mismatches
    #         mismatches = (~torch.isclose(output_slice_f32, golden_slice_f32, rtol=1e-2, atol=1e-2)).sum().item()
    #         total_comparisons += output_slice_f32.numel()
    #         total_mismatches += mismatches

    #         if not is_close or pcc < 0.99:
    #             logger.warning(
    #                 f"Device {device_idx}, Expert {expert_idx}: "
    #                 f"PCC={pcc:.6f}, is_close={is_close}, "
    #                 f"mismatches={mismatches}/{output_slice_f32.numel()}"
    #             )
    #             all_passed = False
    #         else:
    #             logger.info(
    #                 f"Device {device_idx}, Expert {expert_idx}: " f"PCC={pcc:.6f}, PASSED ({num_valid_tokens} tokens)"
    #             )

    # # Summary for tilized output
    # accuracy = (total_comparisons - total_mismatches) / total_comparisons * 100 if total_comparisons > 0 else 100
    # logger.info(f"\nTilized Output Verification Summary:")
    # logger.info(f"  Total comparisons: {total_comparisons}")
    # logger.info(f"  Total mismatches: {total_mismatches}")
    # logger.info(f"  Accuracy: {accuracy:.2f}%")
    # logger.info(f"  Overall result: {'PASSED' if all_passed else 'FAILED'}")

    # ========== Expert Activation Tensor Validation ==========
    logger.info(f"\n========== Expert Activation Tensor Validation ==========")

    # Convert expert_activation tensor to torch
    # Shape per device: [1, total_bytes / 4] as uint32
    expert_activation_torch = ttnn.to_torch(
        expert_activation_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )
    logger.info(f"Expert activation torch shape: {expert_activation_torch.shape}")

    # Row size in uint32 elements (aligned to 16 bytes = 4 uint32s)
    row_elements_unaligned = 2 * experts_per_device + 1  # token_id + k_indices + scores
    row_bytes_unaligned = row_elements_unaligned * 4
    aligned_row_bytes = ((row_bytes_unaligned + 15) // 16) * 16
    aligned_row_elements = aligned_row_bytes // 4

    activation_all_passed = True
    for device_idx in range(num_devices):
        golden_rows = golden_activation[device_idx]
        num_expected_rows = len(golden_rows)

        # Extract this device's activation data
        # The tensor is flattened, so we need to parse rows
        device_activation = expert_activation_torch[device_idx].flatten().to(torch.int64)
        max_rows = len(device_activation) // aligned_row_elements

        logger.info(
            f"Device {device_idx}: expecting {num_expected_rows} activated tokens, tensor has space for {max_rows} rows"
        )

        # Validate each row in sequential order (kernel preserves global token order)
        for row_idx, golden_row in enumerate(golden_rows):
            if row_idx >= max_rows:
                logger.warning(f"  Device {device_idx}: row {row_idx} out of bounds (max {max_rows})")
                activation_all_passed = False
                break

            row_start = row_idx * aligned_row_elements

            # Extract token_id
            actual_token_id = device_activation[row_start].item()
            expected_token_id = golden_row["token_id"]

            if actual_token_id != expected_token_id:
                logger.warning(
                    f"  Device {device_idx}, Row {row_idx}: token_id mismatch - "
                    f"expected {expected_token_id}, got {actual_token_id}"
                )
                activation_all_passed = False
                continue

            # Validate k_indices and scores for each local expert
            for local_exp_idx in range(experts_per_device):
                expected_k = golden_row["k_indices"][local_exp_idx]
                expected_score = golden_row["scores"][local_exp_idx]

                if expected_k >= 0:  # This expert was selected
                    actual_k = device_activation[row_start + 1 + local_exp_idx].item()
                    actual_score_bits = device_activation[row_start + 1 + experts_per_device + local_exp_idx].item()

                    # Convert score bits back to bfloat16 then float
                    # The score is stored as uint16 in the lower bits of uint32
                    actual_score_bf16 = torch.tensor([actual_score_bits & 0xFFFF], dtype=torch.int16).view(
                        torch.bfloat16
                    )
                    actual_score = actual_score_bf16.item()

                    if actual_k != expected_k:
                        logger.warning(
                            f"  Device {device_idx}, Row {row_idx}, Expert {local_exp_idx}: "
                            f"k_index mismatch - expected {expected_k}, got {actual_k}"
                        )
                        activation_all_passed = False

                    # Compare scores with tolerance (bfloat16 precision)
                    if abs(actual_score - expected_score) > 1e-2:
                        logger.warning(
                            f"  Device {device_idx}, Row {row_idx}, Expert {local_exp_idx}: "
                            f"score mismatch - expected {expected_score:.4f}, got {actual_score:.4f}"
                        )
                        activation_all_passed = False

        # Validate sentinel row (token_id = -1 = 0xFFFFFFFF as uint32)
        sentinel_row_idx = num_expected_rows
        if sentinel_row_idx >= max_rows:
            logger.warning(f"  Device {device_idx}: sentinel row {sentinel_row_idx} out of bounds")
            activation_all_passed = False
        else:
            sentinel_row_start = sentinel_row_idx * aligned_row_elements
            sentinel_token_id = device_activation[sentinel_row_start].item()
            # -1 as uint32 (0xFFFFFFFF) becomes -1 when sign-extended to int64
            is_sentinel = (sentinel_token_id == -1) or (sentinel_token_id == 0xFFFFFFFF)

            if not is_sentinel:
                logger.warning(
                    f"  Device {device_idx}: sentinel row token_id mismatch - " f"expected -1, got {sentinel_token_id}"
                )
                activation_all_passed = False
            else:
                logger.info(f"  Device {device_idx}: {num_expected_rows} tokens validated, sentinel PASSED")

    logger.info(f"\nExpert Activation Verification: {'PASSED' if activation_all_passed else 'FAILED'}")

    # ========== Per Expert Total Tokens Tensor Validation ==========
    logger.info(f"\n========== Per Expert Total Tokens Tensor Validation ==========")

    # L1 alignment constant (16 bytes)
    l1_alignment = 16

    # Convert per_expert_total_tokens tensor to torch
    # Shape per device: [1, aligned_elements] as uint32
    per_expert_total_tokens_torch = ttnn.to_torch(
        per_expert_total_tokens_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )
    logger.info(f"Per expert total tokens torch shape: {per_expert_total_tokens_torch.shape}")

    # Validate shape: [num_devices, aligned_row_elements]
    # Row is experts_per_device uint32s, aligned to 16 bytes
    per_expert_row_bytes = ((experts_per_device * 4 + l1_alignment - 1) // l1_alignment) * l1_alignment
    per_expert_row_elements = per_expert_row_bytes // 4
    expected_per_expert_shape = (num_devices, per_expert_row_elements)
    assert per_expert_total_tokens_torch.shape == expected_per_expert_shape, (
        f"per_expert_total_tokens shape mismatch: expected {expected_per_expert_shape}, "
        f"got {per_expert_total_tokens_torch.shape}"
    )

    per_expert_tokens_all_passed = True
    for device_idx in range(num_devices):
        device_counts = per_expert_total_tokens_torch[device_idx].flatten()

        for local_exp_idx in range(experts_per_device):
            expected_count = expert_token_counts[device_idx, local_exp_idx].item()
            actual_count = device_counts[local_exp_idx].item()

            if actual_count != expected_count:
                logger.warning(
                    f"  Device {device_idx}, Expert {local_exp_idx}: "
                    f"count mismatch - expected {expected_count}, got {actual_count}"
                )
                per_expert_tokens_all_passed = False
            else:
                logger.info(f"  Device {device_idx}, Expert {local_exp_idx}: count={actual_count} PASSED")

    logger.info(f"\nPer Expert Total Tokens Verification: {'PASSED' if per_expert_tokens_all_passed else 'FAILED'}")

    # ========== E-T (Expert-to-Token) Tensor Validation ==========
    logger.info(f"\n========== E-T (Expert-to-Token) Tensor Validation ==========")

    # Compute golden e_t
    golden_e_t, _ = compute_e_t_golden(expert_indices, expert_mapping, mesh_shape, cluster_axis)

    # Convert e_t tensor to torch
    # Shape per device: [experts_per_device, e_t_row_elements] as uint32
    e_t_torch = ttnn.to_torch(e_t_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    logger.info(f"E-T torch shape: {e_t_torch.shape}")

    # Each entry is 16B aligned (4 uint32s per token ID)
    e_t_entry_size_bytes = ((4 + l1_alignment - 1) // l1_alignment) * l1_alignment  # align sizeof(uint32_t) to 16B
    e_t_entry_elements = e_t_entry_size_bytes // 4  # elements per entry (4 for 16B alignment)

    # Validate shape: [num_devices * experts_per_device, e_t_row_elements]
    # Each expert has (total_tokens + 1) entries (tokens + sentinel), each entry is 16B aligned
    e_t_row_elements = (total_tokens + 1) * e_t_entry_elements
    expected_e_t_shape = (num_devices * experts_per_device, e_t_row_elements)
    assert (
        e_t_torch.shape == expected_e_t_shape
    ), f"e_t shape mismatch: expected {expected_e_t_shape}, got {e_t_torch.shape}"

    e_t_all_passed = True
    for device_idx in range(num_devices):
        for local_exp_idx in range(experts_per_device):
            expected_tokens = golden_e_t[device_idx][local_exp_idx]
            num_expected_tokens = len(expected_tokens)

            # Get the row for this expert from this device
            # Each device has experts_per_device rows in the tensor
            row_idx = device_idx * experts_per_device + local_exp_idx
            device_expert_row = e_t_torch[row_idx].flatten().to(torch.int64)

            # Validate each token in the e_t list
            tokens_match = True
            for i, expected_token_id in enumerate(expected_tokens):
                # Each entry is at 16B (e_t_entry_elements) offset
                actual_token_id = device_expert_row[i * e_t_entry_elements].item()

                if actual_token_id != expected_token_id:
                    logger.warning(
                        f"  Device {device_idx}, Expert {local_exp_idx}, Entry {i}: "
                        f"token_id mismatch - expected {expected_token_id}, got {actual_token_id}"
                    )
                    tokens_match = False
                    e_t_all_passed = False

            # Validate sentinel (-1) at end of list
            sentinel_idx = num_expected_tokens * e_t_entry_elements
            if sentinel_idx < len(device_expert_row):
                sentinel_value = device_expert_row[sentinel_idx].item()
                is_sentinel = (sentinel_value == -1) or (sentinel_value == 0xFFFFFFFF)

                if not is_sentinel:
                    logger.warning(
                        f"  Device {device_idx}, Expert {local_exp_idx}: "
                        f"sentinel mismatch - expected -1, got {sentinel_value}"
                    )
                    e_t_all_passed = False
                elif tokens_match:
                    logger.info(
                        f"  Device {device_idx}, Expert {local_exp_idx}: "
                        f"{num_expected_tokens} tokens validated, sentinel PASSED"
                    )
            else:
                logger.warning(
                    f"  Device {device_idx}, Expert {local_exp_idx}: " f"sentinel index {sentinel_idx} out of bounds"
                )
                e_t_all_passed = False

    logger.info(f"\nE-T Tensor Verification: {'PASSED' if e_t_all_passed else 'FAILED'}")

    # assert all_passed, f"Tilized output verification failed! Accuracy: {accuracy:.2f}%"
    assert activation_all_passed, "Expert activation tensor verification failed!"
    assert per_expert_tokens_all_passed, "Per expert total tokens tensor verification failed!"
    assert e_t_all_passed, "E-T tensor verification failed!"
