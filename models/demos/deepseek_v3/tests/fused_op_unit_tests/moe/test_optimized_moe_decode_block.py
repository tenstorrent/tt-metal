# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import random

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from tests.nightly.tg.ccl.moe.test_moe_compute_6U import prepare_w0_w1_tensor, prepare_w2_tensor


def tt_to_torch_dtype(tt_dtype):
    if tt_dtype == ttnn.bfloat16:
        return torch.bfloat16
    elif tt_dtype == ttnn.bfloat8_b:
        return torch.bfloat16
    elif tt_dtype == ttnn.float32:
        return torch.float32
    elif tt_dtype == ttnn.uint16:
        return torch.uint16
    else:
        raise ValueError(f"Invalid dtype: {tt_dtype}")


def get_linearized_mesh_coord(num_replicated_devices, cluster_axis, expert_id, experts_per_cluster, experts_per_device):
    if cluster_axis == 0:
        cluster_id = expert_id // experts_per_cluster
        expert_id_within_cluster = expert_id % experts_per_cluster
        device_id_within_cluster = expert_id_within_cluster // experts_per_device

        return device_id_within_cluster * num_replicated_devices + cluster_id
    else:
        return expert_id // experts_per_device


def create_torch_w0_tensors(L, E, H, N):
    torch_w0_tensors = []
    for e in range(E):
        torch_w0 = torch.rand((L, 1, H, N), dtype=torch.bfloat16) - 0.5
        torch_w0_tensors.append(torch_w0)

    # [E, L, 1, H, N]
    return torch_w0_tensors


def create_torch_w1_tensors(L, E, H, N):
    torch_w1_tensors = []
    for e in range(E):
        torch_w1 = torch.rand((L, 1, H, N), dtype=torch.bfloat16) - 0.5
        torch_w1_tensors.append(torch_w1)

    # [E, L, 1, H, N]
    return torch_w1_tensors


def create_torch_w2_tensors(L, E, N, H):
    torch_w2_tensors = []
    for e in range(E):
        torch_w2 = torch.rand((L, 1, N, H), dtype=torch.bfloat16) - 0.5
        torch_w2_tensors.append(torch_w2)

    # [E, L, 1, N, H]
    return torch_w2_tensors


def determine_compute_matmul_cores(mesh_device):
    MATMUL_FULL_CORES = {0, 3, 6, 9}
    MATMUL_PAD_CORES = {1, 2, 4, 5, 7, 8, 10, 11}

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


def create_torch_prepared_compute_matmul_weight_tensors(
    torch_w0, torch_w1, torch_w2, num_layers, experts_per_device, hidden_size, N, ring2cores
):
    # Prepare w0_w1 tensor (interleaved, padded, and reordered)
    torch_w0_w1_reordered = prepare_w0_w1_tensor(
        torch_w0, torch_w1, num_layers, experts_per_device, hidden_size, N, ring2cores
    )

    # Prepare w2 tensor (padded and reordered)
    torch_w2_reordered = prepare_w2_tensor(torch_w2, num_layers, experts_per_device, N, hidden_size, ring2cores)

    return torch_w0_w1_reordered, torch_w2_reordered


def create_torch_expert_mapping_tensor(
    num_devices, num_replicated_devices, cluster_axis, experts, experts_per_cluster, experts_per_device, dtype
):
    torch_expert_mapping_tensor = torch.empty((1, experts), dtype=tt_to_torch_dtype(dtype))
    for e in range(experts):
        torch_expert_mapping_tensor[0, e] = get_linearized_mesh_coord(
            num_replicated_devices, cluster_axis, e, experts_per_cluster, experts_per_device
        )

    torch_expert_mapping_tensor = torch_expert_mapping_tensor.repeat(num_devices, 1)

    # [num_devices, experts]
    return torch_expert_mapping_tensor


def create_torch_dispatch_input_tensor(scheme, batch, seq, hidden_size, dtype):
    tokens = []
    for _ in range(batch):
        for _ in range(seq):
            tokens.append(torch.rand(1, 1, 1, hidden_size, dtype=tt_to_torch_dtype(dtype)) - 0.5)
    result = torch.cat(tokens, dim=0)

    # [batch, 1, seq, hidden_size]
    return result.reshape(batch, 1, seq, hidden_size)


def create_torch_dispatch_input_expert_indices_tensor(
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
    expert_indices = torch.ones((batch, 1, seq, selected_experts_k), dtype=tt_to_torch_dtype(dtype)) * -1
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
                elif scheme == "sliding_window":
                    expert_indices[b, 0, s, k] = ((b // experts_per_device) + k) % experts
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

    # [batch, 1, seq, selected_experts_k]
    return expert_indices


def create_torch_dispatch_input_export_scores_tensor(batch, seq, selected_experts_k, dtype):
    # Generate expert scores (same shape as expert_indices)
    # Normalize scores so they sum to 1 per token (softmax-like)
    torch_dispatch_input_expert_scores_tensor = torch.rand(
        (batch, 1, seq, selected_experts_k), dtype=tt_to_torch_dtype(dtype)
    )
    torch_dispatch_input_expert_scores_tensor = (
        torch_dispatch_input_expert_scores_tensor / torch_dispatch_input_expert_scores_tensor.sum(dim=-1, keepdim=True)
    )

    # [batch, 1, seq, selected_experts_k]
    return torch_dispatch_input_expert_scores_tensor


def gen_matmul_golden(torch_input_token, torch_w0, torch_w1, torch_w2):
    # NOTE: L hardcoded to 1 throughout this test

    # [L, 1, 1, H] @ [L, 1, H, N] -> [L, 1, 1, N]
    torch_w0_output_ref = torch_input_token @ torch_w0
    torch_silu_output_ref = torch.nn.functional.silu(torch_w0_output_ref)

    # [L, 1, 1, H] @ [L, 1, H, N] -> [L, 1, 1, N]
    torch_w1_output_ref = torch_input_token @ torch_w1
    torch_intermediate_ref = torch_silu_output_ref * torch_w1_output_ref

    # [L, 1, 1, N] @ [L, 1, N, H] -> [L, 1, 1, H]
    torch_output_ref = torch_intermediate_ref @ torch_w2

    return torch_output_ref


def device_mesh_iterator(mesh_shape):
    for m0 in range(mesh_shape[0]):
        for m1 in range(mesh_shape[1]):
            device = m0 * mesh_shape[1] + m1
            yield m0, m1, device


def get_batch_cluster_idxr(cluster_axis, batch, batch_per_device):
    def _idxr(m0, m1, b):
        if cluster_axis == 0:
            return m1 * batch + m0 * batch_per_device + b
        elif cluster_axis == 1:
            return m0 * batch + m1 * batch_per_device + b
        else:
            return b

    return _idxr


def gen_combine_golden(
    mesh_shape,
    cluster_axis,
    num_devices,
    num_dispatch_devices,
    num_replicated_devices,
    torch_dispatch_input_tensor,
    torch_w0_tensors,
    torch_w1_tensors,
    torch_w2_tensors,
    torch_dispatch_input_expert_indices,
    experts_per_device,
    batch,
    batches_per_device,
    hidden_size,
    select_experts_k,
):
    torch_dispatch_input_tensor = torch_dispatch_input_tensor.repeat([num_replicated_devices, 1, 1, 1])
    torch_dispatch_input_expert_indices = torch_dispatch_input_expert_indices.repeat([num_replicated_devices, 1, 1, 1])

    torch_combine_ref_tensor = torch.zeros(select_experts_k, batch * num_replicated_devices, hidden_size).bfloat16()
    batch_rep_idxr = get_batch_cluster_idxr(cluster_axis, batch, batches_per_device)
    for m0, m1, d in device_mesh_iterator(mesh_shape):
        if cluster_axis == 0:
            cluster_id = m1
        else:
            cluster_id = m0

        first_expert_on_cluster = num_dispatch_devices * cluster_id * experts_per_device
        last_expert_on_cluster = num_dispatch_devices * (cluster_id + 1) * experts_per_device - 1

        for b in range(batches_per_device):
            global_b = batch_rep_idxr(m0, m1, b)
            token = torch_dispatch_input_tensor[global_b, :, :, :]
            for k in range(select_experts_k):
                e = torch_dispatch_input_expert_indices[global_b, :, :, k].item()

                # only applicable if expert is on the current cluster
                if e >= first_expert_on_cluster and e <= last_expert_on_cluster:
                    contrib = gen_matmul_golden(token, torch_w0_tensors[e], torch_w1_tensors[e], torch_w2_tensors[e])
                    torch_combine_ref_tensor[k, global_b, :] = contrib[0, 0, 0, :]

    # [select_experts_k, batch * num_replicated_devices, hidden_size]
    return torch_combine_ref_tensor


def verify_combine(iteration, mesh_device, mesh_shape, cluster_axis, tt_combine_tensor, torch_combine_golden):
    PCC_THRESHOLD = 0.988
    ATOL_THRESHOLD = 650.0

    # factors in linearized_mesh_coord
    if cluster_axis == 0:
        device_tensors = ttnn.get_device_tensors(ttnn.from_device(tt_combine_tensor))
        host_tensors = [ttnn.to_torch(t) for t in device_tensors]

        torch_combine_output = []
        for m1 in range(mesh_shape[1]):
            for m0 in range(mesh_shape[0]):
                torch_combine_output.append(host_tensors[m0 * mesh_shape[1] + m1])

        torch_combine_output = torch.cat(torch_combine_output, dim=1)

    else:
        torch_combine_output = ttnn.to_torch(
            tt_combine_tensor, dtype=torch.bfloat16, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)
        )

    # check pcc
    pcc_passed, pcc_output = comp_pcc(torch_combine_output, torch_combine_golden, pcc=PCC_THRESHOLD)
    logger.info(f"Combine Output - Iteration: {iteration} - PCC: {pcc_output}")
    if not pcc_passed:
        logger.warning(f"FAILED Combine Output - Iteration: {iteration} - PCC: {pcc_output}")

    # check allclose
    allclose_passed, allclose_output = comp_allclose(
        torch_combine_golden, torch_combine_output, atol=ATOL_THRESHOLD, rtol=0
    )
    logger.info(f"Combine Output - Iteration: {iteration} - AllClose: {allclose_output}")
    if not allclose_passed:
        logger.warning(f"FAILED Combine Output - Iteration: {iteration} - AllClose: {allclose_output}")

    return pcc_passed and allclose_passed


def gen_output_golden(
    torch_dispatch_input_tensor,
    torch_dispatch_input_expert_indices,
    torch_dispatch_input_expert_scores,
    torch_w0_tensors,
    torch_w1_tensors,
    torch_w2_tensors,
    batch,
    hidden_size,
    select_experts_k,
):
    output_reference = torch.zeros((batch, 1, 1, hidden_size), dtype=torch.bfloat16)

    # loop over each token
    for token in range(batch):
        # loop over each selected expert
        for k in range(select_experts_k):
            # determine which expert to use
            expert = torch_dispatch_input_expert_indices[token, :, :, k].item()

            # get the output
            matmul_golden = gen_matmul_golden(
                torch_dispatch_input_tensor[token, :, :, :],
                torch_w0_tensors[expert],
                torch_w1_tensors[expert],
                torch_w2_tensors[expert],
            )

            output_reference[token, :, :, :] = (
                output_reference[token, :, :, :] + torch_dispatch_input_expert_scores[token, :, :, k] * matmul_golden
            )

    # [512, 1, 1, 7168]
    return output_reference


def verify_output(iteration, mesh_device, mesh_shape, tt_output_tensor, output_reference_tensor):
    PCC_THRESHOLD = 0.988
    ATOL_THRESHOLD = 270.0

    # bring to host
    # [1, 1, tokens_per_devices, hidden_size // num_replicated_devices] (per device) -> [1, 1, batch, hidden_size] (global on host)
    tt_output_tensor = ttnn.to_torch(
        tt_output_tensor,
        dtype=torch.bfloat16,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(-2, -1)),
    )

    # reshape for comparison with golden
    # [1, 1, batch, hidden_size] -> [batch, 1, 1, hidden_size]
    tt_output_tensor = tt_output_tensor.reshape(tt_output_tensor.shape[-2], 1, 1, tt_output_tensor.shape[-1])

    # check pcc
    pcc_passed, pcc_output = comp_pcc(tt_output_tensor, output_reference_tensor, pcc=PCC_THRESHOLD)
    logger.info(f"Final Output - Iteration: {iteration} - PCC: {pcc_output}")
    if not pcc_passed:
        logger.warning(f"FAILED Final Output - Iteration: {iteration} - PCC: {pcc_output}")

    # check allclose
    allclose_passed, allclose_output = comp_allclose(
        output_reference_tensor, tt_output_tensor, atol=ATOL_THRESHOLD, rtol=0
    )
    logger.info(f"Final Output - Iteration: {iteration} - AllClose: {allclose_output}")
    if not allclose_passed:
        logger.warning(f"FAILED Final Output - Iteration: {iteration} - AllClose: {allclose_output}")

    return pcc_passed and allclose_passed


@pytest.mark.requires_device(["QUAD"])
@pytest.mark.skipif(
    (os.getenv("USE_TORUS_MODE") is None),
    reason=f"Requires ring fabric",
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((16, 8), (16, 8), id="16x8_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("layer_id, num_layers", [(0, 1)])
@pytest.mark.parametrize("batches_per_device", [32])
@pytest.mark.parametrize("shard_dim", [0])
@pytest.mark.parametrize("experts", [256])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("matmul_N", [2048])
@pytest.mark.parametrize("scheme", ["random_sequential_experts"])
@pytest.mark.parametrize("compute_output_height_shard_dim", [4])
@pytest.mark.parametrize("compute_output_width_shard_dim", [4])
@pytest.mark.parametrize("combine_mux_core_range", [((3, 0), (4, 7))])
@pytest.mark.parametrize("combine_token_parallel_core_dim", [4])
@pytest.mark.parametrize("combine_data_parallel_core_dim", [4])
@pytest.mark.parametrize("enable_trace", [False])
@pytest.mark.parametrize("num_iterations", [1])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 500000,
        },
    ],
    ids=["fabric_1D_ring"],
    indirect=True,
)
def test_optimized_moe_decode_block(
    mesh_shape,
    mesh_device,
    cluster_axis,
    layer_id,
    num_layers,
    batches_per_device,
    shard_dim,
    experts,
    select_experts_k,
    seq,
    hidden_size,
    matmul_N,
    scheme,
    compute_output_height_shard_dim,
    compute_output_width_shard_dim,
    combine_mux_core_range,
    combine_token_parallel_core_dim,
    combine_data_parallel_core_dim,
    enable_trace,
    num_iterations,
):
    ############################################
    # initial setup
    ############################################

    torch.manual_seed(2005)
    random.seed(2005)

    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = mesh_shape[cluster_axis]
    num_replicated_devices = num_devices // num_dispatch_devices
    batch = batches_per_device * num_dispatch_devices
    total_tokens = batch * seq
    tokens_per_device = batch // num_dispatch_devices
    experts_per_device = experts // num_devices
    experts_per_cluster = experts // num_replicated_devices

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
    combine_mux_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in combine_mux_core_range])])

    compute_tilize_drain_core = ttnn.CoreCoord(6, 9)

    ############################################
    # create global semaphores
    ############################################
    # NOTE: these don't have to be double buffered
    # - there is a global sync in combine after reading all dispatch output (can't loop around on dispatch semaphore)
    # - there is a global sync at the end of dispatch (can't loop around on combine semaphore)
    dispatch_global_semaphore = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)
    combine_global_semaphore = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)

    ############################################
    # create constant input tensors
    ############################################
    logger.info(f"Begin creating constant input tensors")

    expert_mapping_dtype = ttnn.uint16
    torch_expert_mapping = create_torch_expert_mapping_tensor(
        num_devices,
        num_replicated_devices,
        cluster_axis,
        experts,
        experts_per_cluster,
        experts_per_device,
        expert_mapping_dtype,
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
    torch_w0_tensors = create_torch_w0_tensors(num_layers, experts, hidden_size, matmul_N)
    torch_w1_tensors = create_torch_w1_tensors(num_layers, experts, hidden_size, matmul_N)
    torch_w2_tensors = create_torch_w2_tensors(num_layers, experts, matmul_N, hidden_size)

    ring2cores, compute_matmul_dram_core_range_set = determine_compute_matmul_cores(mesh_device)

    # Merge the weight tensors that belong to different experts on the same device
    # Then reorder the merged weights into their sharded format
    # Finally, order merged weights in accordance to linearized_mesh_coord ordering
    torch_w0_w1_reordered_tensors = [None] * num_devices
    torch_w2_reordered_tensors = [None] * num_devices
    for e in range(0, experts, 2):
        torch_w0 = torch.cat([torch_w0_tensors[e], torch_w0_tensors[e + 1]], dim=1)  # [L, 1, H, N] -> [L, E/D, H, N]
        torch_w1 = torch.cat([torch_w1_tensors[e], torch_w1_tensors[e + 1]], dim=1)  # [L, 1, H, N] -> [L, E/D, H, N]
        torch_w2 = torch.cat([torch_w2_tensors[e], torch_w2_tensors[e + 1]], dim=1)  # [L, 1, N, H] -> [L, E/D, N, H]

        torch_w0_w1_reordered, torch_w2_reordered = create_torch_prepared_compute_matmul_weight_tensors(
            torch_w0, torch_w1, torch_w2, num_layers, experts_per_device, hidden_size, matmul_N, ring2cores
        )

        linearized_mesh_coord = get_linearized_mesh_coord(
            num_replicated_devices, cluster_axis, e, experts_per_cluster, experts_per_device
        )
        torch_w0_w1_reordered_tensors[linearized_mesh_coord] = torch_w0_w1_reordered
        torch_w2_reordered_tensors[linearized_mesh_coord] = torch_w2_reordered

    # concat before sending to device
    torch_w0_w1_reordered_tensor = torch.cat(torch_w0_w1_reordered_tensors, dim=0)
    torch_w2_reordered_tensor = torch.cat(torch_w2_reordered_tensors, dim=0)

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
        torch_w0_w1_reordered_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=w0_w1_dtype,
        memory_config=w0_w1_memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
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
        torch_w2_reordered_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=w2_dtype,
        memory_config=w2_memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
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
    # Generate the tensors
    # ------------------------------------------------------------------------
    tt_dispatch_input_tensors = []
    tt_dispatch_input_expert_indices_tensors = []
    tt_dispatch_input_expert_scores_tensors = []

    torch_combine_goldens = []
    torch_output_goldens = []
    for iteration in range(num_iterations):
        dispatch_input_dtype = ttnn.bfloat16
        dispatch_input_expert_indices_dtype = ttnn.uint16
        dispatch_input_expert_scores_dtype = ttnn.bfloat16

        torch_dispatch_input_tensor = create_torch_dispatch_input_tensor(
            scheme, batch, seq, hidden_size, dispatch_input_dtype
        )
        torch_dispatch_input_expert_indices_tensor = create_torch_dispatch_input_expert_indices_tensor(
            scheme,
            num_devices,
            experts,
            total_tokens,
            experts_per_device,
            batches_per_device,
            batch,
            seq,
            select_experts_k,
            dispatch_input_expert_indices_dtype,
        )
        torch_dispatch_input_expert_scores_tensor = create_torch_dispatch_input_export_scores_tensor(
            batch, seq, select_experts_k, dispatch_input_expert_scores_dtype
        )

        # [tokens_per_device, 1, seq, hidden_size] per device
        tt_dispatch_input_tensor = ttnn.from_torch(
            torch_dispatch_input_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dispatch_input_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )
        tt_dispatch_input_tensors.append(tt_dispatch_input_tensor)

        # [tokens_per_device, 1, seq, select_experts_k] per device
        tt_dispatch_input_expert_indices_tensor = ttnn.from_torch(
            torch_dispatch_input_expert_indices_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dispatch_input_expert_indices_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )
        tt_dispatch_input_expert_indices_tensors.append(tt_dispatch_input_expert_indices_tensor)

        # [tokens_per_device, 1, seq, select_experts_k] per device
        tt_dispatch_input_expert_scores_tensor = ttnn.from_torch(
            torch_dispatch_input_expert_scores_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dispatch_input_expert_scores_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )
        tt_dispatch_input_expert_scores_tensors.append(tt_dispatch_input_expert_scores_tensor)

        torch_combine_golden = gen_combine_golden(
            mesh_shape,
            cluster_axis,
            num_devices,
            num_dispatch_devices,
            num_replicated_devices,
            torch_dispatch_input_tensor,
            torch_w0_tensors,
            torch_w1_tensors,
            torch_w2_tensors,
            torch_dispatch_input_expert_indices_tensor,
            experts_per_device,
            batch,
            batches_per_device,
            hidden_size,
            select_experts_k,
        )
        torch_combine_goldens.append(torch_combine_golden)

        torch_output_golden = gen_output_golden(
            torch_dispatch_input_tensor,
            torch_dispatch_input_expert_indices_tensor,
            torch_dispatch_input_expert_scores_tensor,
            torch_w0_tensors,
            torch_w1_tensors,
            torch_w2_tensors,
            batch,
            hidden_size,
            select_experts_k,
        )
        torch_output_goldens.append(torch_output_golden)

    logger.info(f"Done creating dynamic input tensors and goldens")

    ############################################
    # create persistent dispatch output tensors
    ############################################
    logger.info(f"Begin creating persistent dispatch output tensors")

    # [1, total_tokens, hidden_size] per device
    dispatch_output_sparse_buffer_dtype = ttnn.bfloat16
    dispatch_output_sparse_buffer_shape = [num_dispatch_devices, total_tokens, hidden_size]
    tt_preallocated_dispatch_output_sparse_buffer = ttnn.from_torch(
        torch.zeros(dispatch_output_sparse_buffer_shape, dtype=tt_to_torch_dtype(dispatch_output_sparse_buffer_dtype)),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dispatch_output_sparse_buffer_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
    )

    # same shard spec for indices and scores
    dispatch_output_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(compute_tilize_drain_core, compute_tilize_drain_core)}),
        [total_tokens, select_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    # [1, total_tokens, select_experts_k] per device
    dispatch_output_expert_indices_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        dispatch_output_shard_spec,
    )
    dispatch_output_expert_indices_dtype = ttnn.uint16
    disptach_output_expert_indices_shape = [num_dispatch_devices, total_tokens, select_experts_k]
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

    # [1, total_tokens, select_experts_k] per device
    dispatch_output_expert_scores_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        dispatch_output_shard_spec,
    )
    dispatch_output_expert_scores_dtype = ttnn.bfloat16
    disptach_output_expert_scores_shape = [num_dispatch_devices, total_tokens, select_experts_k]
    tt_preallocated_dispatch_output_expert_scores = ttnn.from_torch(
        torch.zeros(disptach_output_expert_scores_shape, dtype=tt_to_torch_dtype(dispatch_output_expert_scores_dtype)),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dispatch_output_expert_scores_dtype,
        memory_config=dispatch_output_expert_scores_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
    )

    # NOTE: these don't have to be double buffered, as there is a global sync in combine after reading all dispatch output
    tt_dispatch_preallocated_output_tensors = (
        tt_preallocated_dispatch_output_sparse_buffer,
        tt_preallocated_dispatch_output_expert_indices,
        tt_preallocated_dispatch_output_expert_scores,
    )

    logger.info(f"Done creating persistent dispatch output tensors")

    ############################################
    # set post combine memory configs
    ############################################
    tilized_combine_output_memory_config = ttnn.L1_MEMORY_CONFIG

    scaled_output_memory_config = ttnn.L1_MEMORY_CONFIG

    fast_reduce_output_memory_config = ttnn.MemoryConfig(
        ttnn.BufferType.L1,
        ttnn.NdShardSpec(
            ttnn.Shape([1, 32, 128]),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 5), ttnn.CoreCoord(2, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 5), ttnn.CoreCoord(3, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(6, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 5), ttnn.CoreCoord(6, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                ]
            ),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )

    # NOTE:
    # - use DRAM here so we can run multiple iterations
    # - leaving this memory_config here as it is the one used in the model
    # rs_output_memory_config = ttnn.MemoryConfig(
    #     ttnn.BufferType.L1,
    #     ttnn.NdShardSpec(
    #         ttnn.Shape([1, 32, 32]),
    #         ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))]),
    #         ttnn.ShardOrientation.ROW_MAJOR,
    #         ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    #     ),
    # )
    rs_output_memory_config = ttnn.DRAM_MEMORY_CONFIG

    ############################################
    # run op
    ############################################
    logger.info(f"Begin running op iterations")

    def run_op(iteration):
        # move dispatch inputs into L1
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
        # runtime since it needs to be a zeroed out tensor (for each layer)
        # allocated before dispatch, as dispatch serves as the barrier to ensure the tensor is allocated on all devices
        # [select_experts_k, tokens_per_device, hidden_size] per device
        tt_preallocated_combine_output = ttnn.moreh_full(
            shape=[select_experts_k, tokens_per_device, hidden_size],
            fill_value=0,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
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
        # - needed since compute uses just about all of L1
        ttnn.deallocate(tt_dispatch_input_tensor)
        ttnn.deallocate(tt_dispatch_input_expert_indices_tensor)
        ttnn.deallocate(tt_dispatch_input_expert_scores_tensor)

        (
            tt_compute_output_token_counts,
            tt_compute_output_dense_expert_activation,
            tt_compute_ouput_dense_e_t,
            _,  # tile layout output of selective tilize (same buffer as output)
            tt_compute_output,
        ) = ttnn.experimental.moe_compute(
            tt_dispatch_output_sparse_buffer,
            tt_dispatch_output_expert_indices,
            tt_dispatch_output_expert_scores,
            tt_expert_mapping,
            tt_w0_w1,
            tt_w2,
            layer_id=layer_id,
            output_height_shard_dim=compute_output_height_shard_dim,
            output_width_shard_dim=compute_output_width_shard_dim,
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
            worker_cores=ttnn.get_moe_combine_cores(mesh_device),
            mux_core_range_set=combine_mux_cores,
            output_tensor=tt_preallocated_combine_output,
            optional_cross_device_semaphore=combine_global_semaphore,
        )

        tt_tilized_compute_output = ttnn.to_layout(
            tt_combine_output, layout=ttnn.TILE_LAYOUT, memory_config=tilized_combine_output_memory_config
        )

        # unsqueeze
        # [select_experts_k, tokens_per_device, hidden_size] -> [select_experts_k, 1, tokens_per_device, hidden_size]
        tt_unsqueezed_output = ttnn.unsqueeze(tt_tilized_compute_output, dim=1)

        # scale with scores
        # [tokens_per_device, 1, seq, select_experts_k] -> [select_experts_k, 1, tokens_per_device, seq]
        topk_experts_weights = ttnn.permute(
            tt_dispatch_input_expert_scores_tensors[iteration], (3, 1, 0, 2), memory_config=scaled_output_memory_config
        )
        topk_experts_weights = ttnn.to_layout(
            topk_experts_weights, layout=ttnn.TILE_LAYOUT, memory_config=scaled_output_memory_config
        )
        tt_scaled_output = ttnn.mul(
            tt_unsqueezed_output, topk_experts_weights, memory_config=scaled_output_memory_config
        )

        tt_fast_reduce_output_tensors = ttnn.experimental.deepseek_moe_fast_reduce_nc(
            tt_scaled_output,
            dim=0,
            split_size=int(tt_scaled_output.shape[-1] // num_replicated_devices),
            output_memory_config=fast_reduce_output_memory_config,
        )

        # [select_experts_k, tokens_per_device, hidden_size // num_replicated_devices] final per device shape
        tt_final_output = ttnn.experimental.deepseek_moe_reduce_scatter(
            tt_fast_reduce_output_tensors,
            output_memory_config=rs_output_memory_config,
            dim=-1,
            num_links=4,
            topology=ttnn.Topology.Ring,
            cluster_axis=1,
        )

        return tt_combine_output, tt_final_output

    tt_combine_tensors = []
    tt_output_tensors = []
    if enable_trace:
        logger.info(f"Begin compiling op")
        run_op(0)
        ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
        logger.info(f"Done compiling op")

        logger.info(f"Begin capturing trace")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for iteration in range(num_iterations):
            tt_combine_output, tt_output = run_op(iteration)
            tt_combine_tensors.append(tt_combine_output)
            tt_output_tensors.append(tt_output)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
        logger.info(f"Done capturing trace")

        logger.info(f"Begin executing trace")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info(f"Done executing trace")
    else:
        for iteration in range(num_iterations):
            tt_combine_output, tt_output = run_op(iteration)
            tt_combine_tensors.append(tt_combine_output)
            tt_output_tensors.append(tt_output)

    logger.info(f"Begin synchronizing devices")
    ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
    logger.info(f"Done synchronizing devices")

    logger.info(f"Done running op iterations")

    ############################################
    # Validate output
    ############################################
    logger.info(f"Begin validating output")

    all_iterations_passed = True
    for iteration in range(num_iterations):
        logger.info(f"Validating iteration: {iteration}")

        if not verify_combine(
            iteration,
            mesh_device,
            mesh_shape,
            cluster_axis,
            tt_combine_tensors[iteration],
            torch_combine_goldens[iteration],
        ):
            all_iterations_passed = False

        if not verify_output(
            iteration,
            mesh_device,
            mesh_shape,
            tt_output_tensors[iteration],
            torch_output_goldens[iteration],
        ):
            all_iterations_passed = False

    logger.info(f"\nMoE Verification: {'PASSED' if all_iterations_passed else 'FAILED'}")
    assert all_iterations_passed, "MoE Verification Failed!"

    logger.info(f"Done validating output")
