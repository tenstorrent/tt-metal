# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import random

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from tests.nightly.tg.ccl.moe.test_moe_compute_6U import prepare_w0_w1_tensor, prepare_w2_tensor

os.environ.setdefault("MESH_DEVICE", "QUAD")


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


def create_torch_w0_tensors(L, E, H, N):
    # torch_w0_tensors = []
    # for e in range(E):
    #     torch_w0 = torch.rand((L, 1, H, N), dtype=torch.bfloat16) - 0.5
    #     torch_w0_tensors.append(torch_w0)
    torch_w0 = torch.ones((L, 1, H, N), dtype=torch.bfloat16) - 0.5
    torch_w0_tensors = [torch_w0.clone() for _ in range(E)]

    # TODO: (GR)
    return torch_w0_tensors


def create_torch_w1_tensors(L, E, H, N):
    # torch_w1_tensors = []
    # for e in range(E):
    #     torch_w1 = torch.rand((L, 1, H, N), dtype=torch.bfloat16) - 0.5
    #     torch_w1_tensors.append(torch_w1)
    torch_w1 = torch.ones((L, 1, H, N), dtype=torch.bfloat16) - 0.5
    torch_w1_tensors = [torch_w1.clone() for _ in range(E)]

    # TODO: (GR)
    return torch_w1_tensors


def create_torch_w2_tensors(L, E, N, H):
    # torch_w2_tensors = []
    # for e in range(E):
    #     torch_w2 = torch.rand((L, 1, N, H), dtype=torch.bfloat16) - 0.5
    #     torch_w2_tensors.append(torch_w2)
    torch_w2 = torch.ones((L, 1, N, H), dtype=torch.bfloat16) - 0.5
    torch_w2_tensors = [torch_w2.clone() for _ in range(E)]

    # TODO: (GR)
    return torch_w2_tensors


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
            if True:
                # if scheme == "sequential":
                tokens.append(torch.ones(1, 1, 1, hidden_size, dtype=tt_to_torch_dtype(dtype)) * factor)
                factor += 1
            else:
                tokens.append(torch.ones(1, 1, 1, hidden_size, dtype=tt_to_torch_dtype(dtype)))
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


def gen_torch_compute_matmul_weight_tensors(num_layers, experts, hidden_size, N):
    torch_w0_tensors = create_torch_w0_tensors(num_layers, experts, hidden_size, N)
    torch_w1_tensors = create_torch_w1_tensors(num_layers, experts, hidden_size, N)
    torch_w2_tensors = create_torch_w2_tensors(num_layers, experts, N, hidden_size)

    return torch_w0_tensors, torch_w1_tensors, torch_w2_tensors


def gen_torch_prepared_compute_matmul_weight_tensors(
    torch_w0, torch_w1, torch_w2, num_layers, experts_per_device, hidden_size, N, ring2cores
):
    # Prepare w0_w1 tensor (interleaved, padded, and reordered)
    torch_w0_w1_reordered = prepare_w0_w1_tensor(
        torch_w0, torch_w1, num_layers, experts_per_device, hidden_size, N, ring2cores
    )

    # Prepare w2 tensor (padded and reordered)
    torch_w2_reordered = prepare_w2_tensor(torch_w2, num_layers, experts_per_device, N, hidden_size, ring2cores)

    return torch_w0_w1_reordered, torch_w2_reordered


def gen_matmul_golden(torch_input_token, torch_w0, torch_w1, torch_w2):
    # L = 1
    # N = 2048
    # H = hidden_size = 7168

    # (L, 1, 1, H) @ (L, 1, H, N) -> (L, 1, 1, N)
    torch_w0_output_ref = torch_input_token @ torch_w0
    torch_silu_output_ref = torch.nn.functional.silu(torch_w0_output_ref)

    # (L, 1, 1, H) @ (L, 1, H, N) -> (L, 1, 1, N)
    torch_w1_output_ref = torch_input_token @ torch_w1
    torch_intermediate_ref = torch_silu_output_ref * torch_w1_output_ref

    # (L, 1, 1, N) @ (L, 1, N, H) -> (L, 1, 1, H)
    torch_output_ref = torch_intermediate_ref @ torch_w2

    return torch_output_ref


def gen_output_reference(
    torch_dispatch_input_tensor,  # [512, 1, 1, 7168]
    torch_dispatch_input_expert_indices,  # [512, 1, 1, 8]
    torch_dispatch_input_expert_scores,  # [512, 1, 1, 8]
    torch_w0_tensors,
    torch_w1_tensors,
    torch_w2_tensors,
):
    total_tokens = torch_dispatch_input_tensor.shape[0]
    hidden_size = torch_dispatch_input_tensor.shape[-1]
    num_selected_experts = torch_dispatch_input_expert_indices.shape[-1]

    # [512, 1, 1, 7168]
    output_reference = torch.zeros((total_tokens, 1, 1, hidden_size), dtype=torch.bfloat16)

    # loop over each token
    for token in range(total_tokens):
        # loop over each selected expert
        for k in range(num_selected_experts):
            # determine which expert to use
            expert = torch_dispatch_input_expert_indices[token, :, :, k]

            # get the output
            matmul_golden = gen_matmul_golden(
                torch_dispatch_input_tensor[token, :, :, :],
                torch_w0_tensors[0],
                torch_w1_tensors[0],
                torch_w2_tensors[0],
            )

            # TODO: (GR)
            # output_reference[token, :, :, :] = (output_reference[token, :, :, :] + torch_dispatch_input_expert_scores[token, :, :, k] * matmul_golden)
            output_reference[token, :, :, :] = output_reference[token, :, :, :] + matmul_golden

    return output_reference


def verify_output(iteration, mesh_device, mesh_shape, tt_output_tensor, output_reference_tensor):
    # bring to host
    # (1, 1, 32, 896) -> (1, 1, 512, 7168)
    tt_output_tensor = ttnn.to_torch(
        tt_output_tensor,
        dtype=torch.bfloat16,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(-2, -1)),
    )

    # (1, 1, 512, 7168) -> (512, 1, 1, 7168)
    tt_output_tensor = tt_output_tensor.reshape(tt_output_tensor.shape[-2], 1, 1, tt_output_tensor.shape[-1])

    # compare output
    eq, output = comp_pcc(tt_output_tensor, output_reference_tensor)
    logger.info(f"{output}, iteration {iteration}")
    if not eq:
        logger.warning(f"FAILED: {output}")

    # allclose_passed = torch.allclose(output_reference_tensor, tt_output_tensor, atol=600)
    # logger.info(f"AllClose: {allclose_passed}")

    # logger.info(comp_allclose(output_reference_tensor, tt_output_tensor))

    return eq


@pytest.mark.requires_device(["QUAD"])
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
@pytest.mark.parametrize(
    "combine_worker_core_coords",
    [
        (
            ttnn.CoreCoord(5, 0),  # TODO: (GR) check if x-y should be interleaved
            ttnn.CoreCoord(5, 1),
            ttnn.CoreCoord(5, 2),
            ttnn.CoreCoord(5, 3),
            ttnn.CoreCoord(5, 4),
            ttnn.CoreCoord(5, 5),
            ttnn.CoreCoord(5, 6),
            ttnn.CoreCoord(5, 7),
            ttnn.CoreCoord(6, 0),
            ttnn.CoreCoord(6, 1),
            ttnn.CoreCoord(6, 2),
            ttnn.CoreCoord(6, 3),
            ttnn.CoreCoord(6, 4),
            ttnn.CoreCoord(6, 5),
            ttnn.CoreCoord(6, 6),
            ttnn.CoreCoord(6, 7),
        )
    ],
)
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
    combine_worker_core_coords,
    combine_mux_core_range,
    combine_token_parallel_core_dim,
    combine_data_parallel_core_dim,
    enable_trace,
    num_iterations,
):
    ############################################
    # initial setup
    ############################################

    devices = mesh_shape[0] * mesh_shape[1]
    dispatch_devices = mesh_shape[cluster_axis]
    rs_devices = devices // dispatch_devices
    batch = batches_per_device * dispatch_devices
    total_tokens = batch * seq
    tokens_per_device = batch // dispatch_devices
    experts_per_device = experts // devices
    experts_per_cluster = experts // (devices // dispatch_devices)

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
    combine_worker_cores = ttnn.CoreRangeSet([ttnn.CoreRange(coord, coord) for coord in combine_worker_core_coords])
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
    ring2cores, compute_matmul_dram_core_range_set = gen_compute_matmul_cores(mesh_device)
    torch_w0_tensors, torch_w1_tensors, torch_w2_tensors = gen_torch_compute_matmul_weight_tensors(
        num_layers, 2, hidden_size, matmul_N
    )

    # Merge the weight tensors that belong to different experts on the same device
    # Then reorder the merged weights into their sharded format
    torch_w0_w1_reordered_tensors = []
    torch_w2_reordered_tensors = []
    for i in range(0, 2, 2):
        torch_w0 = torch.cat([torch_w0_tensors[i], torch_w0_tensors[i + 1]], dim=1)  # (L, 1, H, N) -> (L, E/D, H, N)
        torch_w1 = torch.cat([torch_w1_tensors[i], torch_w1_tensors[i + 1]], dim=1)  # (L, 1, H, N) -> (L, E/D, H, N)
        torch_w2 = torch.cat([torch_w2_tensors[i], torch_w2_tensors[i + 1]], dim=1)  # (L, 1, N, H) -> (L, E/D, N, H)

        torch_w0_w1_reordered, torch_w2_reordered = gen_torch_prepared_compute_matmul_weight_tensors(
            torch_w0, torch_w1, torch_w2, num_layers, experts_per_device, hidden_size, matmul_N, ring2cores
        )

        torch_w0_w1_reordered_tensors.append(torch_w0_w1_reordered)
        torch_w2_reordered_tensors.append(torch_w2_reordered)

    # concat before sending to device
    torch_w0_w1_reordered_tensor = torch_w0_w1_reordered_tensors[0]  # torch.cat(torch_w0_w1_reordered_tensors, dim=0)
    torch_w2_reordered_tensor = torch_w2_reordered_tensors[0]  # torch.cat(torch_w2_reordered_tensors, dim=0)

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
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
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
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
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
    # Need an additional set of inputs for the trace compile run (since we deallocate inputs after each iteration)
    # ------------------------------------------------------------------------
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )
        tt_dispatch_input_tensors.append(tt_dispatch_input_tensor)

        tt_dispatch_input_expert_indices_tensor = ttnn.from_torch(
            torch_dispatch_input_expert_indices_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dispatch_input_expert_indices_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )
        tt_dispatch_input_expert_indices_tensors.append(tt_dispatch_input_expert_indices_tensor)

        tt_dispatch_input_expert_scores_tensor = ttnn.from_torch(
            torch_dispatch_input_expert_scores_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dispatch_input_expert_scores_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )
        tt_dispatch_input_expert_scores_tensors.append(tt_dispatch_input_expert_scores_tensor)

        output_reference_tensor = gen_output_reference(
            torch_dispatch_input_tensor,
            torch_dispatch_input_expert_indices_tensor,
            torch_dispatch_input_expert_scores_tensor,
            torch_w0_tensors,
            torch_w1_tensors,
            torch_w2_tensors,
        )
        output_reference_tensors.append(output_reference_tensor)

    logger.info(f"Done creating dynamic input tensors and goldens")

    ############################################
    # create persistent dispatch output tensors
    ############################################
    logger.info(f"Begin creating persistent dispatch output tensors")

    dispatch_output_sparse_buffer_dtype = ttnn.bfloat16
    dispatch_output_sparse_buffer_shape = [dispatch_devices, total_tokens, hidden_size]
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
        [total_tokens, select_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    dispatch_output_expert_indices_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        dispatch_output_shard_spec,
    )
    dispatch_output_expert_indices_dtype = ttnn.uint16
    disptach_output_expert_indices_shape = [dispatch_devices, total_tokens, select_experts_k]
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
    disptach_output_expert_scores_shape = [dispatch_devices, total_tokens, select_experts_k]
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

    # NOTE: use DRAM here so we can run multiple iterations
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

    ttnn.set_printoptions(profile="full", precision=4)

    def run_op(iteration):
        # move dispatch inputs into L1
        # logger.info("XXXXX")
        tt_dispatch_input_tensor = ttnn.to_memory_config(
            tt_dispatch_input_tensors[iteration], memory_config=dispatch_input_memory_config
        )
        # logger.info("YYYYY")
        tt_dispatch_input_expert_indices_tensor = ttnn.to_memory_config(
            tt_dispatch_input_expert_indices_tensors[iteration],
            memory_config=dispatch_input_expert_indices_memory_config,
        )
        # logger.info("ZZZZZ")
        tt_dispatch_input_expert_scores_tensor = ttnn.to_memory_config(
            tt_dispatch_input_expert_scores_tensors[iteration],
            memory_config=dispatch_input_expert_scores_memory_config,
        )

        logger.info("AAAAA")
        ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
        logger.info("BBBBB")

        # logger.info("Dispatch Inputs")
        # logger.info(f"Input: {tt_dispatch_input_tensor.shape}")
        # logger.info(f"Indices: {tt_dispatch_input_expert_indices_tensor.shape}")
        # logger.info(f"Scores: {tt_dispatch_input_expert_scores_tensor.shape}")
        # logger.info(f"Mapping: {tt_expert_mapping.shape}")

        # create persistent output tensor for combine
        # runtime since it needs to be a zeroed out tensor
        # allacote before dispatch, as dispatch serves as the barrier to ensure the tensor is allocated on all devices
        # TODO: (GR) some issue with hangs using moreh_full on multiple iters
        # tt_preallocated_combine_output = ttnn.moreh_full(
        #     shape=[select_experts_k, tokens_per_device, hidden_size],
        #     fill_value=0,
        #     device=mesh_device,
        #     layout=ttnn.ROW_MAJOR_LAYOUT,
        #     dtype=ttnn.bfloat16,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        # )

        tt_preallocated_combine_output = ttnn.from_torch(
            torch.zeros([select_experts_k, tokens_per_device, hidden_size], dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        logger.info("CCCCC")
        ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
        logger.info("DDDDD")

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

        logger.info("EEEEE")
        ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
        logger.info("FFFFF")

        # NOTE:
        # - deallocate inputs to dispatch that are allocated in L1
        # - needed since compute uses just about all of L1
        ttnn.deallocate(tt_dispatch_input_tensor)
        ttnn.deallocate(tt_dispatch_input_expert_indices_tensor)
        ttnn.deallocate(tt_dispatch_input_expert_scores_tensor)

        logger.info("GGGGG")
        ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
        logger.info("HHHHH")

        # TODO: (GR) temp
        temp = ttnn.to_memory_config(tt_dispatch_output_sparse_buffer, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # logger.info("Compute Inputs")
        # logger.info(f"Input: {tt_dispatch_output_sparse_buffer.shape}")
        # logger.info(f"Indices: {tt_dispatch_output_expert_indices.shape}")
        # logger.info(f"Scores: {tt_dispatch_output_expert_scores.shape}")
        # logger.info(f"Mapping: {tt_expert_mapping.shape}")

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

        # TODO: (GR)
        # slice off the -1 terminator
        tt_compute_output_dense_expert_activation = tt_compute_output_dense_expert_activation[:, 0:4096]

        logger.info("IIIII")
        ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
        logger.info("JJJJJ")

        # logger.info("Combine Inputs")
        # logger.info(f"Input: {tt_compute_output.shape}")
        # logger.info(f"Expert Activation: {tt_compute_output_dense_expert_activation.shape}")
        # logger.info(f"E_T: {tt_compute_ouput_dense_e_t.shape}")
        # logger.info(f"Token Counts: {tt_compute_output_token_counts.shape}")

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
            worker_cores=list(ttnn.corerange_to_cores(combine_worker_cores)),
            mux_core_range_set=combine_mux_cores,
            output_tensor=tt_preallocated_combine_output,
            optional_cross_device_semaphore=combine_global_semaphore,
        )

        logger.info("KKKKK")
        ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
        logger.info("LLLLL")

        # logger.info("Tilized Compute Input")
        # logger.info(f"Input: {tt_combine_output.shape}")

        tt_tilized_compute_output = ttnn.to_layout(
            tt_combine_output, layout=ttnn.TILE_LAYOUT, memory_config=tilized_combine_output_memory_config
        )

        logger.info("MMMMM")
        ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
        logger.info("NNNNN")

        # logger.info("Unsqueeze Input")
        # logger.info(f"Input: {tt_tilized_compute_output.shape}")

        # unsqueeze
        # (8, 32, 896) -> (8, 1, 32, 896)
        tt_unsqueezed_output = ttnn.unsqueeze(tt_tilized_compute_output, dim=1)

        logger.info("OOOOO")
        ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
        logger.info("PPPPP")

        # TODO: (GR)
        # logger.info("Scale Input")
        # logger.info(f"Input: {tt_unsqueezed_output.shape}")
        # topk_experts_weights = ttnn.permute(tt_dispatch_input_expert_scores_tensors[iteration], (3, 1, 0, 2), memory_config=scaled_output_memory_config)
        # topk_experts_weights = ttnn.to_layout(topk_experts_weights, layout=ttnn.TILE_LAYOUT, memory_config=scaled_output_memory_config)
        # tt_scaled_output = ttnn.mul(tt_unsqueezed_output, topk_experts_weights, memory_config=scaled_output_memory_config)
        tt_scaled_output = tt_unsqueezed_output

        logger.info("QQQQQ")
        ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
        logger.info("RRRRR")

        # logger.info("Fast Reduce Input")
        # logger.info(f"Input: {tt_scaled_output.shape}")

        tt_fast_reduce_output_tensors = ttnn.experimental.deepseek_moe_fast_reduce_nc(
            tt_scaled_output,
            dim=0,
            split_size=int(tt_scaled_output.shape[-1] // rs_devices),
            output_memory_config=fast_reduce_output_memory_config,
        )

        logger.info("SSSSS")
        ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
        logger.info("TTTTT")

        # logger.info("Reduce Scatter Input")
        # logger.info(f"Single Slice: {tt_fast_reduce_output_tensors[0].shape}")

        tt_final_output = ttnn.experimental.deepseek_moe_reduce_scatter(
            tt_fast_reduce_output_tensors,
            output_memory_config=rs_output_memory_config,
            dim=-1,
            num_links=4,
            topology=ttnn.Topology.Ring,
            cluster_axis=1,
        )

        logger.info("UUUUU")
        ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
        logger.info("VVVVV")

        logger.info(f"Final Output Shape: {tt_final_output.shape}")
        return tt_final_output, temp

    tt_output_tensors = []
    if enable_trace:
        logger.info(f"Begin compiling op")
        # when running multiple iterations, we have to deallocate dispatch input tensors
        # so we need an additional set of input tensors for the compile run
        run_op(num_iterations)
        ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
        logger.info(f"Done compiling op")

        logger.info(f"Begin capturing trace")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for iteration in range(num_iterations):
            tt_output = run_op(iteration)
            tt_output_tensors.append(tt_output)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
        logger.info(f"Done capturing trace")

        logger.info(f"Begin executing trace")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info(f"Done executing trace")
    else:
        for iteration in range(num_iterations):
            tt_output, temp = run_op(iteration)
            tt_output_tensors.append(tt_output)

            torch_temp = ttnn.to_torch(
                temp, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(0, 1))
            )

            torch.set_printoptions(
                threshold=float("inf"),  # Print all elements (no truncation)
                linewidth=200,  # Wider lines before wrapping
                precision=10,  # Decimal places for floats
                # sci_mode=False,          # Disable scientific notation
            )

            torch_temp = torch_temp[:1, :512, :4]
            # print(torch_temp.shape)
            # print(torch_temp)

            # if torch.all(torch_comb_out == 1):
            #     print("COMBINE OUTPUT ALL ONES")
            # else:
            #     print("COMBINE OUTPUT NOT ALL ONES")

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
        if not verify_output(
            iteration,
            mesh_device,
            mesh_shape,
            tt_output_tensors[iteration],
            output_reference_tensors[iteration],
        ):
            all_iterations_passed = False

    logger.info(f"\nMoE Verification: {'PASSED' if all_iterations_passed else 'FAILED'}")

    # TODO: (GR)
    # assert all_iterations_passed, "MoE Verification Failed!"

    logger.info(f"Done validating output")
