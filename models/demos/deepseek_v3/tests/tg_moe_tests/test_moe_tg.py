# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TG (Single Galaxy) E2E MoE Decode Block Test for Quad Galaxy Validation

This test runs the full optimized MoE decode block on a single galaxy (32 devices, 4x8 mesh)
to validate that changes won't break the quad galaxy (128 devices, 16x8 mesh) implementation.

Key Design Principles:
1. **Same Work Per Device**: Each device handles 2 experts in both configurations
   - Quad: 256 experts / 128 devices = 2 experts/device
   - TG:   64 experts / 32 devices = 2 experts/device

2. **Full Decode Path**: Tests complete optimized pipeline:
   - Dispatch → Compute → Combine → Tilize → Scale → Reduce

3. **Non-Optimized Reduce**: Uses standard reduce operations since optimized versions
   are hardcoded for 8-device setups (cluster_axis=1 with 16x8 mesh)

If this test passes on TG, we have high confidence the optimized path will work on quad.
If a change breaks quad, it should also break this test.
"""

import os
import random

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from tests.nightly.tg.ccl.moe.test_moe_compute_6U import prepare_w0_w1_tensor, prepare_w2_tensor

# TG Configuration (Single Galaxy)
TG_MESH_SHAPE = (4, 8)  # 32 devices
TG_NUM_DEVICES = 32
TG_NUM_EXPERTS = 64  # 2 per device (same as quad)
TG_BATCH = 128  # 32 batches_per_device * 4 devices on cluster_axis
TG_BATCHES_PER_DEVICE = 32


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
    return torch_w0_tensors


def create_torch_w1_tensors(L, E, H, N):
    torch_w1_tensors = []
    for e in range(E):
        torch_w1 = torch.rand((L, 1, H, N), dtype=torch.bfloat16) - 0.5
        torch_w1_tensors.append(torch_w1)
    return torch_w1_tensors


def create_torch_w2_tensors(L, E, N, H):
    torch_w2_tensors = []
    for e in range(E):
        torch_w2 = torch.rand((L, 1, N, H), dtype=torch.bfloat16) - 0.5
        torch_w2_tensors.append(torch_w2)
    return torch_w2_tensors


def determine_compute_matmul_cores(mesh_device):
    MATMUL_FULL_CORES = {0, 3, 6, 9}
    MATMUL_PAD_CORES = {1, 2, 4, 5, 7, 8, 10, 11}

    in0_core_coords = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(mesh_device, 0)
    core2dram = {}
    for dram_bank_id, core_coords in enumerate(in0_core_coords):
        core2dram[core_coords] = dram_bank_id

    in0_num_cores = len(in0_core_coords)
    in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)

    ring2cores = {}
    for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
        ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in MATMUL_PAD_CORES else 0)

    dram_core_coords = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(in0_num_cores)]
    dram_core_range = [ttnn.CoreRange(dram_core_coord, dram_core_coord) for dram_core_coord in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    return ring2cores, dram_core_range_set


def create_torch_prepared_compute_matmul_weight_tensors(
    torch_w0, torch_w1, torch_w2, num_layers, experts_per_device, hidden_size, N, ring2cores
):
    torch_w0_w1_reordered = prepare_w0_w1_tensor(
        torch_w0, torch_w1, num_layers, experts_per_device, hidden_size, N, ring2cores
    )
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
    return torch_expert_mapping_tensor


def create_torch_dispatch_input_tensor(batch, seq, hidden_size, dtype):
    tokens = []
    for _ in range(batch):
        for _ in range(seq):
            tokens.append(torch.rand(1, 1, 1, hidden_size, dtype=tt_to_torch_dtype(dtype)) - 0.5)
    result = torch.cat(tokens, dim=0)
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

    if scheme == "avg_perf":
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
                    current_indices = expert_indices[b, 0, s, :].tolist()
                    expert_indices[b, 0, s, k] = random.choice(
                        list(filter(lambda e: e not in current_indices, range(experts)))
                    )
                elif scheme == "sliding_window":
                    expert_indices[b, 0, s, k] = ((b // experts_per_device) + k) % experts
                elif scheme == "avg_perf":
                    current_indices = expert_indices[b, 0, s, :].tolist()
                    available_experts = [
                        e
                        for e in range(experts)
                        if e not in current_indices and expert_token_count[e] < max_tokens_per_expert
                    ]
                    if not available_experts:
                        available_experts = [e for e in range(experts) if e not in current_indices]
                    chosen_expert = random.choice(available_experts)
                    expert_indices[b, 0, s, k] = chosen_expert
                    expert_token_count[chosen_expert] += 1
                else:
                    raise ValueError(f"Invalid scheme: {scheme}")

    return expert_indices


def create_torch_dispatch_input_expert_scores_tensor(batch, seq, selected_experts_k, dtype):
    torch_dispatch_input_expert_scores_tensor = torch.rand(
        (batch, 1, seq, selected_experts_k), dtype=tt_to_torch_dtype(dtype)
    )
    torch_dispatch_input_expert_scores_tensor = (
        torch_dispatch_input_expert_scores_tensor / torch_dispatch_input_expert_scores_tensor.sum(dim=-1, keepdim=True)
    )
    return torch_dispatch_input_expert_scores_tensor


def gen_matmul_golden(torch_input_token, torch_w0, torch_w1, torch_w2):
    torch_w0_output_ref = torch_input_token @ torch_w0
    torch_silu_output_ref = torch.nn.functional.silu(torch_w0_output_ref)
    torch_w1_output_ref = torch_input_token @ torch_w1
    torch_intermediate_ref = torch_silu_output_ref * torch_w1_output_ref
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

                if e >= first_expert_on_cluster and e <= last_expert_on_cluster:
                    contrib = gen_matmul_golden(token, torch_w0_tensors[e], torch_w1_tensors[e], torch_w2_tensors[e])
                    torch_combine_ref_tensor[k, global_b, :] = contrib[0, 0, 0, :]

    return torch_combine_ref_tensor


def verify_combine(iteration, mesh_device, mesh_shape, cluster_axis, tt_combine_tensor, torch_combine_golden):
    PCC_THRESHOLD = 0.988
    ATOL_THRESHOLD = 650.0

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

    pcc_passed, pcc_output = comp_pcc(torch_combine_output, torch_combine_golden, pcc=PCC_THRESHOLD)
    logger.info(f"Combine Output - Iteration: {iteration} - PCC: {pcc_output}")
    if not pcc_passed:
        logger.warning(f"FAILED Combine Output - Iteration: {iteration} - PCC: {pcc_output}")

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

    for token in range(batch):
        for k in range(select_experts_k):
            expert = torch_dispatch_input_expert_indices[token, :, :, k].item()
            matmul_golden = gen_matmul_golden(
                torch_dispatch_input_tensor[token, :, :, :],
                torch_w0_tensors[expert],
                torch_w1_tensors[expert],
                torch_w2_tensors[expert],
            )
            output_reference[token, :, :, :] = (
                output_reference[token, :, :, :] + torch_dispatch_input_expert_scores[token, :, :, k] * matmul_golden
            )

    return output_reference


def verify_output(iteration, mesh_device, mesh_shape, tt_output_tensor, output_reference_tensor):
    PCC_THRESHOLD = 0.988
    ATOL_THRESHOLD = 310.0

    tt_output_tensor = ttnn.to_torch(
        tt_output_tensor,
        dtype=torch.bfloat16,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(-2, -1)),
    )

    tt_output_tensor = tt_output_tensor.reshape(tt_output_tensor.shape[-2], 1, 1, tt_output_tensor.shape[-1])

    pcc_passed, pcc_output = comp_pcc(tt_output_tensor, output_reference_tensor, pcc=PCC_THRESHOLD)
    logger.info(f"Final Output - Iteration: {iteration} - PCC: {pcc_output}")
    if not pcc_passed:
        logger.warning(f"FAILED Final Output - Iteration: {iteration} - PCC: {pcc_output}")

    allclose_passed, allclose_output = comp_allclose(
        output_reference_tensor, tt_output_tensor, atol=ATOL_THRESHOLD, rtol=0
    )
    logger.info(f"Final Output - Iteration: {iteration} - AllClose: {allclose_output}")
    if not allclose_passed:
        logger.warning(f"FAILED Final Output - Iteration: {iteration} - AllClose: {allclose_output}")

    return pcc_passed and allclose_passed


@pytest.mark.requires_device("TG")  # Only run on single galaxy
@pytest.mark.skipif(
    (os.getenv("USE_TORUS_MODE") is None),
    reason=f"Requires ring fabric",
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param(TG_MESH_SHAPE, TG_MESH_SHAPE, id="4x8_tg"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("layer_id, num_layers", [(0, 1)])
@pytest.mark.parametrize("batches_per_device", [TG_BATCHES_PER_DEVICE])
@pytest.mark.parametrize("shard_dim", [0])
@pytest.mark.parametrize("experts", [TG_NUM_EXPERTS])
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
@pytest.mark.parametrize("enable_trace", [True])
@pytest.mark.parametrize("num_iterations", [3])
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
def test_optimized_moe_decode_block_tg(
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
    """
    TG E2E MoE decode block test for quad validation.

    Tests the full optimized pipeline on 4x8 mesh (32 devices):
    Dispatch → Compute → Combine → Tilize → Scale → Reduce

    Uses non-optimized reduce operations since optimized versions are hardcoded for 8-device setups.
    """
    ############################################
    # initial setup
    ############################################

    torch.manual_seed(2003)
    random.seed(2003)

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

    logger.info("=" * 80)
    logger.info(f"TG MoE E2E Test Configuration:")
    logger.info(f"  Mesh shape: {mesh_shape} ({num_devices} devices)")
    logger.info(f"  Batch: {batch} (batches_per_device={batches_per_device})")
    logger.info(f"  Experts: {experts} (per_device={experts_per_device})")
    logger.info(f"  Selected experts K: {select_experts_k}")
    logger.info(f"  Cluster axis: {cluster_axis}")
    logger.info("=" * 80)

    ############################################
    # create global semaphores
    ############################################
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

    torch_w0_w1_reordered_tensors = [None] * num_devices
    torch_w2_reordered_tensors = [None] * num_devices
    for e in range(0, experts, 2):
        torch_w0 = torch.cat([torch_w0_tensors[e], torch_w0_tensors[e + 1]], dim=1)
        torch_w1 = torch.cat([torch_w1_tensors[e], torch_w1_tensors[e + 1]], dim=1)
        torch_w2 = torch.cat([torch_w2_tensors[e], torch_w2_tensors[e + 1]], dim=1)

        torch_w0_w1_reordered, torch_w2_reordered = create_torch_prepared_compute_matmul_weight_tensors(
            torch_w0, torch_w1, torch_w2, num_layers, experts_per_device, hidden_size, matmul_N, ring2cores
        )

        linearized_mesh_coord = get_linearized_mesh_coord(
            num_replicated_devices, cluster_axis, e, experts_per_cluster, experts_per_device
        )
        torch_w0_w1_reordered_tensors[linearized_mesh_coord] = torch_w0_w1_reordered
        torch_w2_reordered_tensors[linearized_mesh_coord] = torch_w2_reordered

    torch_w0_w1_reordered_tensor = torch.cat(torch_w0_w1_reordered_tensors, dim=0)
    torch_w2_reordered_tensor = torch.cat(torch_w2_reordered_tensors, dim=0)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w0_w1
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

    dispatch_input_memory_config = ttnn.L1_MEMORY_CONFIG

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

    tt_dispatch_input_tensors = []
    tt_dispatch_input_expert_indices_tensors = []
    tt_dispatch_input_expert_scores_tensors = []

    torch_combine_goldens = []
    torch_output_goldens = []

    for iteration in range(num_iterations):
        dispatch_input_dtype = ttnn.bfloat16
        dispatch_input_expert_indices_dtype = ttnn.uint16
        dispatch_input_expert_scores_dtype = ttnn.bfloat16

        torch_dispatch_input_tensor = create_torch_dispatch_input_tensor(batch, seq, hidden_size, dispatch_input_dtype)
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
        torch_dispatch_input_expert_scores_tensor = create_torch_dispatch_input_expert_scores_tensor(
            batch, seq, select_experts_k, dispatch_input_expert_scores_dtype
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
    dispatch_output_expert_indices_shape = [num_dispatch_devices, total_tokens, select_experts_k]
    tt_preallocated_dispatch_output_expert_indices = ttnn.from_torch(
        torch.zeros(
            dispatch_output_expert_indices_shape, dtype=tt_to_torch_dtype(dispatch_output_expert_indices_dtype)
        ),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dispatch_output_expert_indices_dtype,
        memory_config=dispatch_output_expert_indices_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
    )

    dispatch_output_expert_scores_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        dispatch_output_shard_spec,
    )
    dispatch_output_expert_scores_dtype = ttnn.bfloat16
    dispatch_output_expert_scores_shape = [num_dispatch_devices, total_tokens, select_experts_k]
    tt_preallocated_dispatch_output_expert_scores = ttnn.from_torch(
        torch.zeros(dispatch_output_expert_scores_shape, dtype=tt_to_torch_dtype(dispatch_output_expert_scores_dtype)),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dispatch_output_expert_scores_dtype,
        memory_config=dispatch_output_expert_scores_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
    )

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

    # Use standard memory config for non-optimized reduce
    fast_reduce_output_memory_config = ttnn.L1_MEMORY_CONFIG
    rs_output_memory_config = ttnn.DRAM_MEMORY_CONFIG

    ############################################
    # run op
    ############################################
    logger.info(f"Begin running op iterations")

    def run_op(iteration):
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

        ttnn.deallocate(tt_dispatch_input_tensor)
        ttnn.deallocate(tt_dispatch_input_expert_indices_tensor)
        ttnn.deallocate(tt_dispatch_input_expert_scores_tensor)

        (
            tt_compute_output_token_counts,
            tt_compute_output_dense_expert_activation,
            tt_compute_output_dense_e_t,
            _,
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

        # Get worker cores and convert to list (API expects Sequence[CoreCoord])
        combine_worker_cores = ttnn.experimental.get_moe_combine_cores(mesh_device)
        combine_worker_cores_list = list(ttnn.corerange_to_cores(combine_worker_cores))

        tt_combine_output = ttnn.experimental.selective_reduce_combine(
            tt_compute_output,
            tt_compute_output_dense_expert_activation,
            tt_compute_output_dense_e_t,
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
            worker_cores=combine_worker_cores_list,
            mux_core_range_set=combine_mux_cores,
            output_tensor=tt_preallocated_combine_output,
            optional_cross_device_semaphore=combine_global_semaphore,
        )

        tt_tilized_compute_output = ttnn.to_layout(
            tt_combine_output, layout=ttnn.TILE_LAYOUT, memory_config=tilized_combine_output_memory_config
        )

        tt_unsqueezed_output = ttnn.unsqueeze(tt_tilized_compute_output, dim=1)

        topk_experts_weights = ttnn.permute(
            tt_dispatch_input_expert_scores_tensors[iteration], (3, 1, 0, 2), memory_config=scaled_output_memory_config
        )
        topk_experts_weights = ttnn.to_layout(
            topk_experts_weights, layout=ttnn.TILE_LAYOUT, memory_config=scaled_output_memory_config
        )
        tt_scaled_output = ttnn.mul(
            tt_unsqueezed_output, topk_experts_weights, memory_config=scaled_output_memory_config
        )

        # Use standard sum reduction (non-optimized version)
        # Reduce along dim=0 to sum contributions from all selected experts (K experts)
        tt_summed_output = ttnn.sum(
            tt_scaled_output,
            dim=0,
            memory_config=fast_reduce_output_memory_config,
        )

        # Use standard reduce_scatter (non-optimized version for 4x8 mesh)
        # Reduce along the tensor parallel dimension (dim=-1, across num_replicated_devices)
        # cluster_axis=1 means reduce across the replicated devices (8 devices per cluster)
        tt_final_output = ttnn.reduce_scatter(
            tt_summed_output,
            dim=-1,
            cluster_axis=1,
            num_links=4,
            memory_config=rs_output_memory_config,
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

    logger.info(f"\nTG MoE E2E Verification: {'PASSED' if all_iterations_passed else 'FAILED'}")
    assert all_iterations_passed, "TG MoE E2E Verification Failed!"

    logger.info(f"Done validating output")
