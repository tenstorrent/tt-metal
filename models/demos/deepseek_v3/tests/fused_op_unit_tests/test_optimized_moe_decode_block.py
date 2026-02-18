# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import random

import pytest
import torch
from loguru import logger

import ttnn
from tests.nightly.t3000.ccl.test_all_to_all_dispatch import get_metadata_tensor, get_output_tensor
from tests.nightly.tg.ccl.moe.test_moe_compute_6U.py import (
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


def gen_torch_expert_mapping_tensor(scheme, devices, experts, experts_per_device, dtype):
    if scheme == "random":
        perm = torch.randperm(experts)
        assignment = torch.empty(experts, dtype=tt_to_torch_dtype(dtype))
        for d in range(devices):
            assignment[perm[d * experts_per_device : (d + 1) * experts_per_device]] = d
    else:
        assignment = torch.arange(experts) // experts_per_device

    return assignment.unsqueeze(0).repeat(devices, 1)


def gen_torch_dispatch_input_tensor(scheme, batch, seq, hidden_size, dtype):
    tokens = []
    factor = 1
    for _ in range(batch):
        for _ in range(seq):
            if scheme == "sequential":
                tokens.append(torch.ones(1, 1, 1, hidden_size, dtype=dtype) * factor)
                factor += 1
            else:
                tokens.append(torch.rand(1, 1, 1, hidden_size, dtype=dtype))
    res = torch.cat(tokens, dim=0)
    return res.reshape(batch, 1, seq, hidden_size)


def gen_torch_dispatch_input_expert_indices_tensor(
    scheme,
    devices,
    experts,
    total_tokens,
    experts_per_device,
    batches_per_device,
    batch,
    seq,
    selected_experts_k,
    dtype,
):
    expert_indices = torch.ones(batch, 1, seq, selected_experts_k, dtype=tt_to_torch_dtype(dtype)) * -1
    current_expert = 0

    # For avg_perf scheme, track how many tokens are assigned to each expert
    if scheme == "avg_perf":
        # Use ceiling division to ensure we have enough capacity, with minimum of 1
        max_tokens_per_expert = max(1, (total_tokens * selected_experts_k + experts - 1) // experts)
        expert_token_count = {e: 0 for e in range(experts)}

    token = 0
    for b in range(batch):
        for s in range(seq):
            token += 1
            for k in range(selected_experts_k):
                if scheme == "sequential":
                    expert_indices[b, 0, s, k] = current_expert % experts
                    current_expert += 1 + (k % 2)
                elif scheme == "random" or scheme == "random_sequential_experts":
                    # need to ensure a set of unique indices
                    current_indices = expert_indices[b, 0, s, :].tolist()
                    expert_indices[b, 0, s, k] = random.choice(
                        list(filter(lambda e: e not in current_indices, range(experts)))
                    )
                elif scheme == "avg_perf":
                    # Random selection but each expert is capped at max_tokens_per_expert
                    # to simulate average performance case
                    current_indices = expert_indices[b, 0, s, :].tolist()
                    # First, try to find experts under the cap
                    available_experts = [
                        e
                        for e in range(experts)
                        if e not in current_indices and expert_token_count[e] < max_tokens_per_expert
                    ]
                    # Fallback: if all capped experts are used, pick any expert not in current token
                    if not available_experts:
                        available_experts = [e for e in range(experts) if e not in current_indices]
                    chosen_expert = random.choice(available_experts)
                    expert_indices[b, 0, s, k] = chosen_expert
                    expert_token_count[chosen_expert] += 1
                elif scheme == "worst_perf":  # worst perf is when the expert index is always on the last device
                    expert_indices[b, 0, s, k] = (
                        experts - 1
                    )  # technically each expert index should be different, but we're sending to the same device regardless
                elif scheme == "worst_congestion":
                    # Worst case link congestion: each token selects experts on consecutive
                    # devices in one direction, maximizing traffic through intermediate links.
                    # From device D, select experts on devices D+1, D+2, ..., D+k
                    # This creates a cascade where link i→i+1 carries traffic from all
                    # devices 0..i to all devices i+1..n, maximizing link utilization.

                    # Determine source device for this token (batch is sharded in chunks across devices)
                    # Device 0 gets batch 0 to batch/devices-1, device 1 gets next chunk, etc.
                    src_device = b // batches_per_device

                    # Target device is src_device + k + 1 (wrapping around)
                    target_device = (src_device + 1 + k) % devices

                    # Expert offset: if we wrap around and visit a device multiple times,
                    # pick a different expert on that device each time
                    expert_offset = k // devices
                    expert_id = target_device * experts_per_device + (expert_offset % experts_per_device)

                    expert_indices[b, 0, s, k] = expert_id
                elif scheme == "best_congestion":
                    # Best case for congestion: tokens prefer local experts first, then nearest
                    # neighbors, minimizing average hop distance and spreading traffic.
                    # Algorithm:
                    # 1. Fill local device first (up to experts_per_device)
                    # 2. Alternate between CCW and CW neighbors, picking 1 expert at a time
                    # 3. When a device is full, move to the next device in that direction
                    #
                    # Example with k=8, experts_per_device=2:
                    # - 2 local
                    # - 2 from CCW neighbor (1-hop)
                    # - 2 from CW neighbor (1-hop)
                    # - 1 from CW 2-hop neighbor
                    # - 1 from CCW 2-hop neighbor
                    # Batch is sharded in chunks across devices (not round-robin)
                    # Device 0 gets batch 0 to batch/devices-1, device 1 gets next chunk, etc.
                    src_device = b // batches_per_device

                    # Build list of (device, local_expert_idx) for all k experts
                    if k == 0:
                        # First expert slot - start building the selection list
                        # We need to compute all k experts at once for this token
                        picked = []
                        remaining = selected_experts_k

                        # Step 1: Fill local device
                        local_count = min(remaining, experts_per_device)
                        for i in range(local_count):
                            picked.append((src_device, i))
                        remaining -= local_count

                        # Step 2: Alternate CCW and CW
                        ccw_hop = 1
                        cw_hop = 1
                        ccw_count = 0  # experts picked from current CCW device
                        cw_count = 0  # experts picked from current CW device
                        use_ccw = True  # alternate starting with CCW

                        while remaining > 0:
                            if use_ccw:
                                ccw_device = (src_device - ccw_hop) % devices
                                if ccw_count < experts_per_device:
                                    picked.append((ccw_device, ccw_count))
                                    ccw_count += 1
                                    remaining -= 1
                                else:
                                    # Move to next CCW device
                                    ccw_hop += 1
                                    ccw_count = 0
                                    continue  # Don't switch direction yet, retry with new device
                            else:
                                cw_device = (src_device + cw_hop) % devices
                                if cw_count < experts_per_device:
                                    picked.append((cw_device, cw_count))
                                    cw_count += 1
                                    remaining -= 1
                                else:
                                    # Move to next CW device
                                    cw_hop += 1
                                    cw_count = 0
                                    continue  # Don't switch direction yet, retry with new device
                            use_ccw = not use_ccw  # Alternate direction

                        # Store the selection in a way we can access for subsequent k values
                        # We use a simple approach: just compute all k at once and store in tensor
                        for idx, (device, local_idx) in enumerate(picked):
                            expert_id = device * experts_per_device + local_idx
                            expert_indices[b, 0, s, idx] = expert_id
                    # For k > 0, the values were already set when k == 0
                elif scheme == "worst_congestion_descending":
                    # Worst case for sparse multicast: send to furthest device first (antipode),
                    # then progressively closer devices. This maximizes the hop distance for
                    # each packet, as opposed to worst_congestion (ascending) which starts
                    # from the nearest neighbor.
                    #
                    # For a 16-device ring from src_device:
                    # - k=0: hop 8 (antipode, maximum distance)
                    # - k=1: hop 7
                    # - k=2: hop 6
                    # - ...
                    # - k=7: hop 1 (nearest neighbor)
                    # - k=8: hop 0 (local, if k > devices/2)
                    #
                    # This is the worst case for sparse multicast because:
                    # 1. Each packet travels maximum distance before hitting any destination
                    # 2. No benefit from early delivery along the path
                    # 3. Maximum total hop-distance across all packets

                    # Batch is sharded in chunks across devices
                    src_device = b // batches_per_device

                    # Start from antipode (devices // 2 hops away) and decrement
                    antipode_hop = devices // 2
                    hop_distance = max(0, antipode_hop - k)  # Clamp to 0 for local when k > antipode
                    target_device = (src_device + hop_distance) % devices

                    # Expert offset: if we wrap around and visit a device multiple times,
                    # pick a different expert on that device each time
                    expert_offset = k // devices
                    expert_id = target_device * experts_per_device + (expert_offset % experts_per_device)

                    expert_indices[b, 0, s, k] = expert_id
                else:
                    raise ValueError(f"Invalid scheme: {scheme}")

    return expert_indices


def gen_torch_dispatch_input_export_scores_tensor(dispatch_input_expert_indices_shape, dtype):
    # Generate expert scores (same shape as expert_indices)
    # Normalize scores so they sum to 1 per token (softmax-like)
    torch_dispatch_input_expert_scores_tensor = torch.rand(dispatch_input_expert_indices_shape, dtype=torch.float32).to(
        tt_to_torch_dtype(dtype)
    )
    torch_dispatch_input_expert_scores_tensor = (
        torch_dispatch_input_expert_scores_tensor / torch_dispatch_input_expert_scores_tensor.sum(dim=-1, keepdim=True)
    )

    return torch_dispatch_input_expert_scores_tensor


def gen_compute_matmul_cores(mesh_device):
    MATMUL_FULL_CORES = {0, 1, 8, 9}
    MATMUL_PAD_CORES = {2, 3, 4, 5, 6, 7, 10, 11}

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
        ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in MATMUL_PAD_CORES else 0)

    dram_core_coords = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(in0_num_cores)]
    dram_core_range = [ttnn.CoreRange(dram_core_coord, dram_core_coord) for dram_core_coord in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    return ring2cores, dram_core_range_set


def gen_torch_compute_matmul_weight_tensors(ring2cores, num_layers, experts_per_device, hidden_size, N):
    torch_w0 = create_torch_w0(num_layers, experts_per_device, hidden_size, N)
    torch_w1 = create_torch_w1(num_layers, experts_per_device, hidden_size, N)
    torch_w2 = create_torch_w2(num_layers, experts_per_device, N, hidden_size)

    # Prepare w0_w1 tensor (interleaved, padded, and reordered)
    torch_w0_w1_reordered = prepare_w0_w1_tensor(
        torch_w0, torch_w1, num_layers, experts_per_device, hidden_size, N, ring2cores
    )

    # Prepare w2 tensor (padded and reordered)
    torch_w2_reordered = prepare_w2_tensor(torch_w2, num_layers, experts_per_device, N, hidden_size, ring2cores)

    return torch_w0_w1_reordered, torch_w2_reordered


def gen_dispatch_reference(
    torch_expert_mapping,
    torch_dispatch_input_tensor,
    torch_dispatch_input_expert_indices,
    mesh_shape,
    devices,
    experts,
    seq,
    tokens_dtype,
):
    # get_output_tensor and get_metadata_tensor expect old expert mapping version
    torch_old_expert_mapping = torch.zeros(1, 1, experts, devices, dtype=torch_expert_mapping.dtype)
    for e in range(experts):
        device_id = torch_expert_mapping[0, e].item()
        torch_old_expert_mapping[0, 0, e, device_id] = 1

    torch_dispatch_output_sparse_buffer = get_output_tensor(
        torch_dispatch_input_tensor,
        torch_dispatch_input_expert_indices,
        torch_old_expert_mapping,
        seq,
        mesh_shape,
        tt_to_torch_dtype(tokens_dtype),
    )
    torch_dispatch_output_expert_indices = get_metadata_tensor(
        torch_dispatch_input_expert_indices, torch_old_expert_mapping, mesh_shape
    )

    return (torch_dispatch_output_sparse_buffer, torch_dispatch_output_expert_indices)


def gen_compute_reference(
    torch_expert_mapping,
    torch_dispatch_output_sparse_buffer,
    torch_dispatch_output_expert_indices,
    torch_w0,
    torch_w1,
    torch_w2,
):
    # TODO: (GR) tilized output, which needs to use metadata and tokens from dispatch
    torch_input_ref = ()

    # (L, D, E/D, T, H) -> (L, E, T, H)
    torch_input_ref = torch_input_ref.reshape(num_layers, experts, total_tokens, hidden_size)

    # in the test setup the expert weights are duplicated over devices, do so here
    # (L, E/D, K, N) -> (L, E, K, N)
    torch_w0 = torch_w0.repeat([1, devices, 1, 1])
    # (L, E/D, K, N) -> (L, E, K, N)
    torch_w1 = torch_w1.repeat([1, devices, 1, 1])
    # (L, E/D, N, K) -> (L, E, N, K)
    torch_w2 = torch_w2.repeat([1, devices, 1, 1])

    # Compute gate activations for each expert
    # (L, E, T, K) @ (L, E, K, N) -> (L, E, T, N)
    torch_w0_output_ref = torch_input_ref @ torch_w0
    torch_silu_output_ref = torch.nn.functional.silu(torch_w0_output_ref)
    # (L, E, T, K) @ (L, E, K, N) -> (L, E, T, N)
    torch_w1_output_ref = torch_input_ref @ torch_w1
    torch_intermediate_ref = torch_silu_output_ref * torch_w1_output_ref  # (L, E, T, N)

    # (L, E, T, N) @ (L, E, N, K) -> (L, E, T, K)
    torch_output_ref = torch_intermediate_ref @ torch_w2

    # pull device dim back out for comparison
    # (L, E, T, H) -> (L, D, E/D, T, H)
    # TODO: (GR) possibly remove this, output needs to be applied to combine
    torch_output_ref = torch_output_ref.reshape(num_layers, devices, experts // devices, total_tokens, hidden_size)

    torch_compute_output_token_counts = ()
    torch_compute_output_dense_expert_activation = ()
    torch_compute_output = ()
    return (torch_compute_output_token_counts, torch_compute_output_dense_expert_activation, torch_compute_output)


def gen_combine_reference(
    torch_expert_mapping,  # TODO: (GR) is this the correct shape/tensor
    torch_compute_output_token_counts,
    torch_compute_output_dense_expert_activation,
    torch_compute_output,
):
    # TODO: (GR)
    # needs
    # expert_mapping -> (TBD from where, and if any reshape needed)
    # dense_input_contribs_tensor -> primary compute output -> some reshape in between
    # dense_metadata_tensor -> expert_activation from compute -> some reshape in between
    # active_token_counts,  -> token_counts from compute -> no reshape

    cluster_factor, cluster_size, devices = get_cluster_dims(cluster_axis, mesh_shape)

    num_local_experts = experts // devices
    hidden_size = dense_input_contribs_tensor.shape[-1]
    output_ref_tensor = torch.zeros(batch * seq * cluster_factor, experts, hidden_size).bfloat16()
    output_data_map = torch.zeros(output_ref_tensor.shape[:-1])

    token_parallel_block_size = batch // token_parallel_core_dim
    block_counts = _active_token_core_split_counts(
        token_parallel_block_size, active_token_counts, token_parallel_core_dim
    )

    batch_rep_idxr = get_batch_cluster_idxr(cluster_axis, batch)

    for m0, m1, rec_d in _device_mesh_iterator(mesh_shape):
        device_dense_idxs = [0] * num_local_experts
        device_blocked_dense_counts = [0] * num_local_experts
        for dt in range(dense_metadata_len[rec_d]):
            dense_metadata_entry = dense_metadata_tensor[rec_d, dt]
            k_entries, b, s = _unpack_dense_metadata_entry(dense_metadata_entry, num_local_experts, seq)
            global_batch = batch_rep_idxr(m0, m1, b)

            reduction_buffer = torch.zeros([hidden_size]).bfloat16() if local_reduce else None
            for local_e_idx, k in enumerate(k_entries):
                if k == select_experts_k + 1:
                    continue

                global_e_idx = rec_d * num_local_experts + local_e_idx

                if local_reduce:
                    reduction_buffer += dense_input_contribs_tensor[global_e_idx, device_dense_idxs[local_e_idx]]
                else:
                    output_ref_tensor[global_batch * seq + s, global_e_idx] = dense_input_contribs_tensor[
                        global_e_idx, device_dense_idxs[local_e_idx]
                    ]
                    output_data_map[global_batch * seq + s, global_e_idx] = 1

                if device_blocked_dense_counts[local_e_idx] == block_counts[global_e_idx][-1] - 1:
                    use_idx = device_dense_idxs[local_e_idx] or 1
                    device_dense_idxs[local_e_idx] = (
                        math.ceil(use_idx / token_parallel_block_size) * token_parallel_block_size
                    )
                    device_blocked_dense_counts[local_e_idx] = 0
                    if len(block_counts[global_e_idx]) > 1:
                        block_counts[global_e_idx].pop()
                else:
                    device_dense_idxs[local_e_idx] += 1
                    device_blocked_dense_counts[local_e_idx] += 1

            if local_reduce:
                local_reduction_k = next(
                    filter(lambda x: x != select_experts_k + 1, k_entries)
                )  # somewhat arbitrary placement
                output_ref_tensor[global_batch * seq + s, 0] = reduction_buffer
                output_data_map[global_batch * seq + s, 0] = 1

    return output_ref_tensor, output_data_map


def gen_output_reference(
    torch_expert_mapping, torch_dispatch_input_tensor, torch_dispatch_input_expert_indices, torch_w0, torch_w1, torch_w2
):
    # dispatch
    (
        torch_dispatch_output_sparse_buffer,
        torch_dispatch_output_expert_indices,
    ) = gen_dispatch_reference(
        torch_expert_mapping,
        torch_dispatch_input_tensor,
        torch_dispatch_input_expert_indices,
        mesh_shape,
        devices,
        experts,
        seq,
        tokens_dtype=ttnn.bfloat16,
    )

    # compute
    # don't need e_t tensor for combine reference generation
    (
        torch_compute_output_token_counts,
        torch_compute_output_dense_expert_activation,
        torch_compute_output,
    ) = gen_compute_reference(
        torch_expert_mapping,  # global
        torch_dispatch_output_sparse_buffer,  # from dispatch
        torch_dispatch_output_expert_indices,  # from dispatch
        torch_w0,  # global
        torch_w1,  # global
        torch_w2,  # global
    )

    # combine
    torch_output_reference_tensor, output_reference_data_map = gen_combine_reference(
        torch_expert_mapping,  # TODO: (GR) is this the correct shape/tensor
        torch_compute_output_token_counts,
        torch_compute_output_dense_expert_activation,
        torch_compute_output,
    )

    return torch_output_reference_tensor, output_reference_data_map


def verify_output(mesh_device, mesh_shape, cluster_axis, tt_output_tensor, output_reference_tensor, output_data_map):
    if cluster_axis == 0:
        # need to roll my own mesh composer here for the transposed ordering
        device_shards = [
            ttnn.to_torch(ittout, mesh_composer=None) for ittout in ttnn.get_device_tensors(tt_output_tensor)
        ]
        ordered_shards = []
        for ir in range(mesh_shape[1]):
            for ic in range(mesh_shape[0]):
                ordered_shards.append(device_shards[ic * mesh_shape[1] + ir])
        tt_output_tensor_agg = torch.cat(ordered_shards, dim=0)

    else:
        tt_output_tensor_agg = ttnn.to_torch(
            tt_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        )

    if tt_output_tensor_agg.shape != output_reference_tensor.shape:
        logger.warning(
            f"Output tensor shape mismatch - Expected: {output_reference_tensor.shape}, Got: {tt_output_tensor_agg.shape}"
        )
        return False

    iteration_passed = True
    for t in range(tt_output_tensor_agg.shape[0]):
        for k in range(tt_output_tensor_agg.shape[1]):
            if output_data_map[t, k].item() == 1:
                if not torch.equal(tt_output_tensor_agg[t, k, :], output_reference_tensor[t, k, :]):
                    iteration_passed = False
                    logger.warning(
                        f"Equal check failed for {t=}, {k=} with {tt_output_tensor_agg[t,k, :]=} and {output_reference_tensor[t,k, :]=}"
                    )

    return iteration_passed


@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((16, 8), (16, 8), id="16x8_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("num_iterations", [1])
@pytest.mark.parametrize("layer_id", [1])
@pytest.mark.parametrize("batches_per_device", [32])
@pytest.mark.parametrize("shard_dim", [0])
@pytest.mark.parametrize("experts", [256])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("matmul_N", [2048])
@pytest.mark.parametrize("scheme", ["random_sequential_experts"])
@pytest.mark.parametrize("combine_worker_core_range", [((5, 0), (6, 7))])
@pytest.mark.parametrize("combine_mux_core_range", [((3, 0), (4, 7))])
@pytest.mark.parametrize("combine_token_parallel_core_dim", [4])
@pytest.mark.parametrize("combine_data_parallel_core_dim", [4])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize("enable_performance", [False, True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        },
    ],
    ids=["fabric_1D_ring"],
    indirect=True,
)
def test_optimized_moe_decode_block(
    mesh_shape,
    mesh_device,
    cluster_axis,
    num_iterations,
    layer_id,
    batches_per_device,
    shard_dim,
    experts,
    select_experts_k,
    seq,
    hidden_size,
    matmul_N,
    scheme,
    combine_worker_core_range,
    combine_mux_core_range,
    combine_token_parallel_core_dim,
    combine_data_parallel_core_dim,
    enable_trace,
    enable_performance,
):
    mesh_device.disable_and_clear_program_cache()

    ############################################
    # initial setup
    ############################################

    devices = mesh_shape[0] * mesh_shape[1]
    dispatch_devices = mesh_shape[cluster_axis]
    batch = batches_per_device * dispatch_devices
    total_tokens = batch * seq
    tokens_per_device = batch // devices
    experts_per_device = experts // devices

    if cluster_axis == 1:
        shard_dims = (None, shard_dim)
    elif cluster_axis == 0:
        shard_dims = (shard_dim, None)
    else:
        shard_dims = shard_dim

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    worker_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    combine_worker_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in combine_worker_core_range])])
    combine_mux_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in combine_mux_core_range])])

    compute_tilize_drain_core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(6, 9), ttnn.CoreCoord(6, 9))})

    ############################################
    # create (double buffered) global semaphores
    ############################################
    # NOTE: these don't have to be double buffered, as there is a global sync in combine after reading all dispatch output, and a global sync at the end of dispatch
    dispatch_global_semaphore = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)
    combine_global_semaphore = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)

    ############################################
    # create constant input tensors
    ############################################
    logger.info(f"Begin creating constant input tensors")

    expert_mapping_dtype = ttnn.uint16
    torch_expert_mapping = gen_torch_expert_mapping_tensor(
        scheme, devices, experts, experts_per_device, expert_mapping_dtype
    )
    tt_expert_mapping = ttnn.from_torch(
        torch_expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=expert_mapping_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )

    # ------------------------------------------------------------------------
    # Matmul weights
    # ------------------------------------------------------------------------
    num_layers = layer_id + 1  # test only handles a single layer at a time, simulate all other layers being present
    ring2cores, compute_matmul_dram_core_range_set = gen_compute_matmul_cores(mesh_device)
    torch_w0_w1_reordered, torch_w2_reordered = gen_torch_compute_matmul_weight_tensors(
        ring2cores, num_layers, experts_per_device, hidden_size, matmul_N
    )

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w0_w1
    # Tensor shape: (num_layers, experts_per_device, hidden_size, 4608) -> padded and reordered to (12, num_layers, experts_per_device, 6, hidden_size, 64)
    # ------------------------------------------------------------------------
    w0_w1_shard_height = num_layers * experts_per_device * 3 * hidden_size
    w0_w1_shard_width = 4 * ttnn.TILE_SIZE
    w0_w1_shard_spec = ttnn.ShardSpec(
        compute_matmul_dram_core_range_set, (w0_w1_shard_height, w0_w1_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    w0_w1_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w0_w1_shard_spec
    )
    w0_w1_dtype = ttnn.bfloat4_b
    tt_w0_w1 = ttnn.from_torch(
        torch_w0_w1_reordered,
        dtype=w0_w1_dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w0_w1_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w2
    # Tensor shape: (num_layers, experts_per_device, N, hidden_size) -> padded and reordered to (12, num_layers, experts_per_device, 5, N + 192, 128)
    # ------------------------------------------------------------------------
    w2_shard_height = num_layers * experts_per_device * 5 * (matmul_N + 192)
    w2_shard_width = 4 * ttnn.TILE_SIZE
    w2_shard_spec = ttnn.ShardSpec(
        compute_matmul_dram_core_range_set, (w2_shard_height, w2_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    w2_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)
    w2_dtype = ttnn.bfloat4_b
    tt_w2 = ttnn.from_torch(
        torch_w2_reordered,
        dtype=w2_dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w2_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )

    logger.info(f"Done creating constant input tensors")

    ############################################
    # create dynamic input tensors & goldens
    ############################################
    logger.info(f"Begin creating dynamic input tensors and goldens")

    # ------------------------------------------------------------------------
    # Memory configs for dispatch input tensors
    # ------------------------------------------------------------------------
    dispatch_input_memory_config = ttnn.L1_MEMORY_CONFIG

    # Use L1 height sharded memory for indices and scores
    # Height sharded with 1 row per core ensures 16B alignment and optimal memory access
    num_cores_y = min(8, tokens_per_device)
    num_cores_x = (tokens_per_device + num_cores_y - 1) // num_cores_y
    dispatch_input_shard_core_range = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
    )
    dispatch_input_shard_spec = ttnn.ShardSpec(
        dispatch_input_shard_core_range,
        [1, seq * select_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    dispatch_input_expert_indices_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        dispatch_input_shard_spec,
    )

    dispatch_input_expert_scores_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        dispatch_input_shard_spec,
    )

    # ------------------------------------------------------------------------
    # How many sets of input tensors we need
    # ------------------------------------------------------------------------
    if enable_performance:
        # only a single reused input for performance runs, so we don't have to move it into L1
        num_input_sets = 1
    else:
        # need an additional set of inputs for the trace compile run (since we deallocate inputs after each iteration)
        if enable_trace:
            num_input_sets = num_iterations + 1
        else:
            num_input_sets = num_iterations

    # ------------------------------------------------------------------------
    # Generate the tensors
    # ------------------------------------------------------------------------
    tt_dispatch_input_tensors = []
    tt_dispatch_input_expert_indices_tensors = []
    tt_dispatch_input_expert_scores_tensors = []

    output_reference_tensors = []
    output_data_maps = []
    for iteration in range(num_input_sets):
        dispatch_input_dtype = ttnn.bfloat16
        dispatch_input_expert_indices_dtype = ttnn.uint16
        dispatch_input_expert_scores_dtype = ttnn.bfloat16

        torch_dispatch_input_tensor = gen_torch_dispatch_input_tensor(
            scheme, batch, seq, hidden_size, dispatch_input_dtype
        )
        torch_dispatch_input_expert_indices_tensor = gen_torch_dispatch_input_expert_indices_tensor(
            scheme,
            devices,
            experts,
            total_tokens,
            experts_per_device,
            batches_per_device,
            batch,
            seq,
            select_experts_k,
            dispatch_input_expert_indices_dtype,
        )
        torch_dispatch_input_expert_scores_tensor = gen_torch_dispatch_input_export_scores_tensor(
            torch_dispatch_input_expert_indices_tensor.shape, dispatch_input_expert_scores_dtype
        )

        tt_dispatch_input_tensor = ttnn.from_torch(
            torch_dispatch_input_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dispatch_input_dtype,
            memory_config=dispatch_input_memory_config if enable_performance else ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )
        tt_dispatch_input_tensors.append(tt_dispatch_input_tensor)

        tt_dispatch_input_expert_indices_tensor = ttnn.from_torch(
            torch_dispatch_input_expert_indices_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dispatch_input_expert_indices_dtype,
            memory_config=dispatch_input_expert_indices_memory_config
            if enable_performance
            else ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )
        tt_dispatch_input_expert_indices_tensors.append(tt_dispatch_input_expert_indices_tensor)

        tt_dispatch_input_expert_scores_tensor = ttnn.from_torch(
            torch_dispatch_input_expert_scores_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dispatch_input_expert_scores_dtype,
            memory_config=dispatch_input_expert_scores_memory_config if enable_performance else ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )
        tt_dispatch_input_expert_scores_tensors.append(tt_dispatch_input_expert_scores_tensor)

        # TODO: (GR)
        output_reference_tensor, output_reference_data_map = gen_output_reference(
            torch_expert_mapping, torch_dispatch_input_tensor, torch_dispatch_input_expert_indices_tensor
        )
        output_reference_tensors.append(output_reference_tensor)
        output_data_maps.append(output_reference_data_map)

    logger.info(f"Done creating dynamic input tensors and goldens")

    ############################################
    # create persistent dispatch output tensors
    ############################################
    logger.info(f"Begin creating persistent dispatch output tensors")

    dispatch_output_sparse_buffer_dtype = ttnn.bfloat16
    dispatch_output_sparse_buffer_shape = [devices, total_tokens, hidden_size]
    tt_preallocated_dispatch_output_sparse_buffer = ttnn.from_torch(
        torch.zeros(dispatch_output_sparse_buffer_shape, dtype=tt_to_torch_dtype(dispatch_output_sparse_buffer_dtype)),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dispatch_output_sparse_buffer_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
    )
    logger.info(
        f"preallocated_dispatch_output_sparse_bufffer shape: {tt_preallocated_dispatch_output_sparse_buffer.shape}"
    )

    # same shard spec for indices and scores
    dispatch_output_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(compute_tilize_drain_core, compute_tilize_drain_core)}),
        [total_tokens * devices, select_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    dispatch_output_expert_indices_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        dispatch_output_shard_spec,
    )
    dispatch_output_expert_indices_dtype = ttnn.uint16
    disptach_output_expert_indices_shape = [devices, total_tokens, select_experts_k]
    tt_preallocated_dispatch_output_expert_indices = ttnn.from_torch(
        torch.zeros(
            disptach_output_expert_indices_shape, dtype=tt_to_torch_dtype(dispatch_output_expert_indices_dtype)
        ),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dispatch_output_expert_indices_dtype,
        memory_config=dispatch_output_expert_indices_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
    )
    logger.info(
        f"preallocated_dispatch_output_expert_indices shape: {tt_preallocated_dispatch_output_expert_indices.shape}"
    )

    dispatch_output_expert_scores_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        dispatch_output_shard_spec,
    )
    dispatch_output_expert_scores_dtype = ttnn.bfloat16
    disptach_output_expert_scores_shape = [devices, total_tokens, select_experts_k]
    tt_preallocated_dispatch_output_expert_scores = ttnn.from_torch(
        torch.zeros(disptach_output_expert_scores_shape, dtype=tt_to_torch_dtype(dispatch_output_expert_scores_dtype)),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dispatch_output_expert_scores_dtype,
        memory_config=dispatch_output_expert_scores_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
    )
    logger.info(
        f"preallocated_dispatch_output_expert_scores shape: {tt_preallocated_dispatch_output_expert_scores.shape}"
    )

    # NOTE: these don't have to be double buffered, as there is a global sync in combine after reading all dispatch output
    tt_dispatch_preallocated_output_tensors = (
        tt_preallocated_dispatch_output_sparse_buffer,
        tt_preallocated_dispatch_output_expert_indices,
        tt_preallocated_dispatch_output_expert_scores,
    )

    logger.info(f"Done creating persistent dispatch output tensors")

    ############################################
    # run op
    ############################################
    logger.info(f"Begin running op iterations")

    def run_op(iteration):
        # move dispatch inputs into L1
        if enable_performance:
            tt_dispatch_input_tensor = tt_dispatch_input_tensors[0]
            tt_dispatch_input_expert_indices_tensor = tt_dispatch_input_expert_indices_tensors[0]
            tt_dispatch_input_expert_scores_tensor = tt_dispatch_input_expert_scores_tensors[0]
        else:
            tt_dispatch_input_tensor = ttnn.to_memory_config(
                tt_dispatch_input_tensors[iteration], memory_config=dispatch_input_memory_config
            )
            tt_dispatch_input_expert_indices_tensor = ttnn.to_memory_config(
                tt_dispatch_input_expert_indices_tensors[iteration],
                memory_config=dispatch_input_expert_indices_memory_config,
            )
            tt_dispatch_input_expert_scores_tensor = ttnn.to_memory_config(
                tt_dispatch_input_expert_scores_tensors[iteration],
                memory_config=dispatch_input_expert_scores_memory_config,
            )

        # create persistent output tensor for combine
        # runtime since it needs to be a zeroed out tensor
        # allacote before dispatch, as dispatch serves as the barrier to ensure the tensor is allocated on all devices
        # TODO: (GR) may have to create tiled, and then move to row-major
        tt_combine_output = ttnn.moreh_full(
            shape=[batch * seq, experts, hidden_size],
            fill_value=0,
            mesh_device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        (
            tt_dispatch_output_sparse_buffer,
            tt_dispatch_output_expert_indices,
            tt_dispatch_output_expert_scores,
        ) = ttnn.experimental.all_to_all_dispatch_metadata(
            tt_dispatch_input_tensor,
            tt_dispatch_input_expert_indices_tensor,
            tt_dispatch_input_expert_scores_tensor,
            tt_expert_mapping,
            cluster_axis=cluster_axis,
            num_links=4,
            drain_sync_tilizer_core=None,
            worker_mode=ttnn.WorkerMode.DIRECT,
            dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_MCAST_SHORTEST_PATH,
            output_tensors=tt_dispatch_preallocated_output_tensors,
            cross_device_semaphore=dispatch_global_semaphore,
        )

        # NOTE:
        # - deallocate inputs to dispatch that are allocated in L1
        # - needed to run multiple iterations since combine uses just about all of L1
        if not enable_performance:
            ttnn.deallocate(tt_dispatch_input_tensor)
            ttnn.deallocate(tt_dispatch_input_expert_indices_tensor)
            ttnn.deallocate(tt_dispatch_input_expert_scores_tensor)

        (
            tt_compute_output_token_counts,
            tt_compute_output_dense_expert_activation,
            tt_compute_ouput_dense_e_t,
            tt_compute_output,
        ) = ttnn.experimental.moe_compute(
            tt_dispatch_output_sparse_buffer,
            tt_dispatch_output_expert_indices,
            tt_dispatch_output_expert_scores,
            tt_expert_mapping,
            tt_w0_w1,
            tt_w2,
            layer_id=layer_id,
            cluster_axis=cluster_axis,
        )

        tt_combine_output = ttnn.experimental.selective_reduce_combine(
            tt_compute_output,
            tt_compute_output_dense_expert_activation,
            tt_compute_ouput_dense_e_t,
            tt_compute_output_token_counts,
            hidden_size,
            batch,
            seq,
            select_experts_k,
            experts,
            cluster_axis,
            topology=ttnn.Topology.Ring,
            num_links=4,
            token_parallel_core_dim=combine_token_parallel_core_dim,
            data_parallel_core_dim=combine_data_parallel_core_dim,
            worker_core_range_set=combine_worker_cores,
            mux_core_range_set=combine_mux_cores,
            output_tensor=tt_combine_output,
            optional_cross_device_semaphore=combine_global_semaphore,
        )

        return tt_combine_output

    tt_output_tensors = []
    if enable_trace:
        logger.info(f"Begin compiling op")
        # when running multiple iterations, we have to deallocate dispatch input tensors
        # so we need an additional set of input tensors for the compile run
        run_op(num_iterations)
        logger.info(f"Done compiling op")

        logger.info(f"Begin capturing trace")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for iteration in range(num_iterations):
            tt_output = run_op(iteration)
            tt_output_tensors.append(tt_output)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        logger.info(f"Done capturing trace")

        logger.info(f"Begin executing trace")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info(f"Done executing trace")
    else:
        for iteration in range(num_iterations):
            tt_output = run_op(iteration)
            tt_output_tensors.append(tt_output)

    logger.info(f"Begin synchronizing devices")
    ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
    logger.info(f"Done synchronizing devices")

    logger.info(f"Done running op iterations")

    ############################################
    # Validate output
    ############################################
    if not enable_performance:
        logger.info(f"Begin validating output")

        all_iterations_passed = True
        for iteration in range(num_iterations):
            logger.info(f"Validating iteration: {iteration}")
            if not verify_output(
                mesh_device,
                mesh_shape,
                cluster_axis,
                tt_output_tensors[iteration],
                output_reference_tensors[iteration],
                output_data_maps[iteration],
            ):
                all_iterations_passed = False

        logger.info(f"\nMoE Verification: {'PASSED' if all_iterations_passed else 'FAILED'}")
        assert all_iterations_passed, "MoE Verification Failed!"

        logger.info(f"Done validating output")
