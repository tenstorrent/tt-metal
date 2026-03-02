# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Galaxy "minimesh" test for moe_gpt op on a 4x1 column (4 devices).

The moe_gpt op runs on a single column of the full 4x8 galaxy mesh.
Each column of 4 devices forms a "minimesh" where all_to_all dispatch
and combine happen along cluster_axis=0 (the 4 rows).

This test mirrors the structure of test_moe_compute_6U.py but targets
our moe_gpt op which currently only supports the matmul phase (DRAM-sharded
weight inputs + L1-sharded activation input).

Future inputs (commented out below) will be added incrementally as the op
gains support for them:
  - sparse_buffer (token data from all_to_all_dispatch)
  - expert_indices (all-gathered top-K expert selections)
  - expert_scores (all-gathered routing weights)
  - expert_mapping (expert-to-device ownership map)

Dimensions (gpt-oss 20b):
  M = 32 (tokens per device, collapsed batch * seq_len)
  K = 2880 (hidden_size) -> 90 tiles
  N = 2880 (intermediate_size) -> 90 tiles
  E = 4 (experts per device, 128 total / 32 devices)
  E_column = 16 (experts in this column, 4 devices * 4 experts/device)
  L = 1 (layers)

  Minimesh: (4, 1) — one column of the galaxy
  cluster_axis = 0 — dispatch across the 4 row-devices
  num_dispatch_devices = 4
  total_tokens = M * 4 = 128

Activation: SwiGLU (gpt-oss variant)
  gate_clamped = clamp(gate, max=7.0)
  up_clamped   = clamp(up, min=-7.0, max=7.0)
  result       = (up_clamped + 1) * gate_clamped * sigmoid(1.702 * gate_clamped)
"""

import itertools
import math
import pytest
import random
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc, comp_allclose
from models.demos.gpt_oss.tt.experts_throughput.weights import (
    _FUSED_MAX_TILES_PER_CORE as MAX_W0_W1_TILES_PER_CORE,
    _FUSED_PAD_CORES as PAD_CORES,
    _prepare_w0_w1_tensor as prepare_w0_w1_tensor,
    _prepare_w2_tensor as prepare_w2_tensor,
)

PCC_THRESHOLD = 0.984


# ---------------------------------------------------------------------------
# Torch reference helpers
# ---------------------------------------------------------------------------


def create_torch_input(L, in0_num_cores, E, M, K):
    """Create random input tensor of shape (L, in0_num_cores, E, M, K)."""
    torch_input = torch.rand((L, E, M, K), dtype=torch.bfloat16) - 0.5
    torch_input = torch_input.unsqueeze(1).repeat(1, in0_num_cores, 1, 1, 1)
    return torch_input


def create_torch_w0(L, E, K, N):
    return torch.rand((L, E, K, N), dtype=torch.bfloat16) - 0.5


def create_torch_w1(L, E, K, N):
    return torch.rand((L, E, K, N), dtype=torch.bfloat16) - 0.5


def create_torch_w2(L, E, N, K):
    return torch.rand((L, E, N, K), dtype=torch.bfloat16) - 0.5


def swiglu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    gate_c = torch.clamp(gate, max=clamp_limit)
    up_c = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
    return (up_c + 1.0) * gate_c * torch.sigmoid(alpha * gate_c)


def prepare_output_tensor(tt_output, E, M, K, ring2cores):
    """Extract valid tiles per core from the raw output tensor."""
    each_shard = []
    for ring_pos in range(len(ring2cores)):
        (_, _, pad_flag) = ring2cores[ring_pos]
        num_tiles = 7 if pad_flag else 8
        each_shard.append(tt_output[ring_pos, :, :, : num_tiles * ttnn.TILE_SIZE])

    result = torch.cat(each_shard, dim=-1)
    assert result.shape == (E, M, K), f"Expected shape {(E, M, K)}, got {result.shape}"
    return result


def get_accuracy_metrics(torch_output, tt_output):
    _pcc_passed, pcc_val = comp_pcc(torch_output, tt_output)
    std = torch_output.std().item()
    relative_rmse_val = (torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / std) if std != 0 else 0.0
    allclose_passed, allclose_val = comp_allclose(torch_output, tt_output)
    return {
        "pcc": pcc_val,
        "relative_rmse": relative_rmse_val,
        "allclose": allclose_passed,
        "allclose_val": allclose_val,
    }


# ---------------------------------------------------------------------------
# Future input helpers (moe_compute-style, for when the op supports them)
# ---------------------------------------------------------------------------

# def gen_expert_mapping(experts, num_devices):
#     """
#     Per-device expert mapping: expert_mapping[d, e] = e // experts_per_device.
#     Shape: [num_devices, experts], dtype uint16, replicated on all devices.
#     """
#     experts_per_device = experts // num_devices
#     expert_mapping = torch.zeros(1, experts, dtype=torch.uint16)
#     for e in range(experts):
#         expert_mapping[0, e] = e // experts_per_device
#     expert_mapping = expert_mapping.repeat(num_devices, 1)
#     return expert_mapping


# def gen_sparse_buffer_and_indices(
#     tokens_per_device, hidden_size, experts, selected_experts_k,
#     num_devices, num_dispatch_devices, dtype=torch.bfloat16,
# ):
#     """
#     Generate sparse buffer (simulating all_to_all_dispatch output) and
#     all-gathered expert indices/scores.
#
#     Returns:
#         sparse_buffer: [num_devices, total_tokens, hidden_size]
#         expert_indices: [num_dispatch_devices, tokens_per_device, K] (uint16)
#         expert_scores:  [num_dispatch_devices, tokens_per_device, K] (bfloat16)
#     """
#     experts_per_device = experts // num_devices
#     total_tokens = tokens_per_device * num_dispatch_devices
#
#     expert_indices = torch.zeros(num_dispatch_devices, tokens_per_device, selected_experts_k, dtype=torch.uint16)
#     for src_device in range(num_dispatch_devices):
#         for t in range(tokens_per_device):
#             selected = torch.randperm(experts)[:selected_experts_k]
#             expert_indices[src_device, t, :] = selected.to(torch.uint16)
#
#     expert_scores = torch.rand(num_dispatch_devices, tokens_per_device, selected_experts_k, dtype=dtype) + 1e-5
#     expert_scores = expert_scores / expert_scores.sum(dim=-1, keepdim=True)
#
#     sparse_buffer = torch.rand(num_devices, total_tokens, hidden_size, dtype=dtype)
#     for src_device in range(num_dispatch_devices):
#         for t in range(tokens_per_device):
#             token_idx = src_device * tokens_per_device + t
#             for k in range(selected_experts_k):
#                 expert_id = expert_indices[src_device, t, k].item()
#                 target_device = expert_id // experts_per_device
#                 sparse_buffer[target_device, token_idx, :] = torch.rand(hidden_size, dtype=dtype) - 0.5
#
#     return sparse_buffer, expert_indices, expert_scores


# def create_sharded_memory_config_single_core(core_coord, tensor_shape, dtype):
#     """Create L1 HEIGHT_SHARDED memory config that pins a tensor to a single core."""
#     core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(core_coord, core_coord)})
#     shard_height = tensor_shape[0]
#     shard_width = tensor_shape[1] if len(tensor_shape) > 1 else 1
#     shard_spec = ttnn.ShardSpec(
#         core_range_set,
#         [shard_height, shard_width],
#         ttnn.ShardOrientation.ROW_MAJOR,
#     )
#     return ttnn.MemoryConfig(
#         ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#         ttnn.BufferType.L1,
#         shard_spec,
#     )


# ---------------------------------------------------------------------------
# Main test body
# ---------------------------------------------------------------------------


def run_test_moe_gpt_galaxy(
    mesh_device,
    mesh_shape,
    cluster_axis,
    M,  # tokens per device
    K,  # hidden_size
    N,  # intermediate_size
    E,  # experts per device
    L,  # layers
    # selected_experts_k,  # top-K (future)
):
    torch.manual_seed(42)
    random.seed(42)

    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = mesh_shape[cluster_axis]
    total_tokens = M * num_dispatch_devices
    experts_total = E * num_devices

    logger.info(f"Test configuration:")
    logger.info(f"  mesh_shape: {mesh_shape}, cluster_axis: {cluster_axis}")
    logger.info(f"  num_devices: {num_devices}, num_dispatch_devices: {num_dispatch_devices}")
    logger.info(f"  tokens_per_device (M): {M}, total_tokens: {total_tokens}")
    logger.info(f"  experts_per_device (E): {E}, experts_total: {experts_total}")
    logger.info(f"  hidden_size (K): {K}, intermediate_size (N): {N}")
    logger.info(f"  layers (L): {L}")

    # ------------------------------------------------------------------
    # Per-device shard grid (identical on every device in the mesh)
    # ------------------------------------------------------------------
    in0_core_coords = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(0)
    core2dram = {core_coords: dram_bank_id for dram_bank_id, core_coords in enumerate(in0_core_coords)}
    in0_num_cores = len(in0_core_coords)

    in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)
    ring2cores = {}
    for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
        ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in PAD_CORES else 0)

    in0_core_range = [ttnn.CoreRange(ring2cores[i][0], ring2cores[i][0]) for i in range(in0_num_cores)]
    in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)

    dram_core_coords = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(in0_num_cores)]
    dram_core_range = [ttnn.CoreRange(c, c) for c in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    in0_dtype = ttnn.bfloat16
    w_dtype = ttnn.bfloat4_b
    num_dram_banks = 12

    # ------------------------------------------------------------------
    # Input activation memory config (L1 HEIGHT_SHARDED across ring cores)
    # ------------------------------------------------------------------
    input_shape = (in0_num_cores, E, M, K)

    in0_shard_spec = ttnn.ShardSpec(
        grid=in0_core_range_set,
        shard_shape=(E * M, K),
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec
    )

    # ------------------------------------------------------------------
    # W0/W1 memory config (DRAM HEIGHT_SHARDED across DRAM banks)
    # ------------------------------------------------------------------
    groups_per_core = MAX_W0_W1_TILES_PER_CORE // 2
    w0_w1_shard_height = L * E * groups_per_core * K
    w0_w1_shard_width = 4 * ttnn.TILE_SIZE

    w0_w1_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w0_w1_shard_height, w0_w1_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    w0_w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w0_w1_shard_spec)

    # ------------------------------------------------------------------
    # W2 memory config (DRAM HEIGHT_SHARDED across DRAM banks)
    # ------------------------------------------------------------------
    w2_shard_height = L * E * 2 * N
    w2_shard_width = 4 * ttnn.TILE_SIZE

    w2_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w2_shard_height, w2_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    w2_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)

    # ------------------------------------------------------------------
    # Future: tilize-phase input memory configs
    # These will be needed when the op gains selective-tilize support.
    # ------------------------------------------------------------------

    # # Drain tilize core for indices/scores (pinned to a single core)
    # tilize_drain_core = ttnn.CoreCoord(6, 9)

    # # Sparse buffer: per-device [total_tokens, K], interleaved L1
    # # Sharded across mesh dim-0, each device gets its [total_tokens, K] slice
    # sparse_mem_config = ttnn.L1_MEMORY_CONFIG

    # # Expert indices: [total_tokens, selected_experts_k], L1 pinned to drain core
    # # expert_indices_mem_config = create_sharded_memory_config_single_core(
    # #     tilize_drain_core, [total_tokens, selected_experts_k], ttnn.uint16
    # # )

    # # Expert scores: [total_tokens, selected_experts_k], L1 pinned to drain core
    # # expert_scores_mem_config = create_sharded_memory_config_single_core(
    # #     tilize_drain_core, [total_tokens, selected_experts_k], in0_dtype
    # # )

    # # Expert mapping: [num_devices, experts_total], interleaved L1, replicated
    # # expert_mapping_mem_config = ttnn.L1_MEMORY_CONFIG

    # ------------------------------------------------------------------
    # Create torch tensors
    # ------------------------------------------------------------------
    torch_input = create_torch_input(L, in0_num_cores, E, M, K)
    torch_w0 = create_torch_w0(L, E, K, N)
    torch_w1 = create_torch_w1(L, E, K, N)
    torch_w2 = create_torch_w2(L, E, N, K)

    # ------------------------------------------------------------------
    # Prepare weight tensors (replicated on all devices in the mesh)
    # ------------------------------------------------------------------
    torch_w0_w1_reordered = prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, K, N, ring2cores)
    tt_w0_w1 = ttnn.from_torch(
        torch_w0_w1_reordered,
        dtype=w_dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w0_w1_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    torch_w2_reordered = prepare_w2_tensor(torch_w2, L, E, N, K, ring2cores)
    tt_w2 = ttnn.from_torch(
        torch_w2_reordered,
        dtype=w_dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w2_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # ------------------------------------------------------------------
    # Future: create tilize-phase input tensors
    # ------------------------------------------------------------------

    # # Expert mapping (constant, replicated on every device)
    # expert_mapping = gen_expert_mapping(experts_total, num_devices)
    # tt_expert_mapping = ttnn.from_torch(
    #     expert_mapping,
    #     device=mesh_device,
    #     layout=ttnn.ROW_MAJOR_LAYOUT,
    #     dtype=ttnn.uint16,
    #     memory_config=expert_mapping_mem_config,
    #     mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    # )

    # # Per-layer sparse buffer, indices, scores
    # for layer_id in range(L):
    #     sparse_buffer, expert_indices, expert_scores = gen_sparse_buffer_and_indices(
    #         M, K, experts_total, selected_experts_k, num_devices, num_dispatch_devices,
    #     )
    #
    #     # Sparse buffer: sharded across mesh dim-0
    #     tt_sparse_buffer = ttnn.from_torch(
    #         sparse_buffer,
    #         device=mesh_device,
    #         layout=ttnn.ROW_MAJOR_LAYOUT,
    #         dtype=in0_dtype,
    #         memory_config=sparse_mem_config,
    #         mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    #     )
    #
    #     # Expert indices: flattened [total_tokens, K], replicated via dim-0 shard
    #     expert_indices_flat = expert_indices.reshape(total_tokens, selected_experts_k)
    #     expert_indices_replicated = expert_indices_flat.unsqueeze(0).repeat(num_devices, 1, 1)
    #     tt_expert_indices = ttnn.from_torch(
    #         expert_indices_replicated,
    #         device=mesh_device,
    #         layout=ttnn.ROW_MAJOR_LAYOUT,
    #         dtype=ttnn.uint16,
    #         memory_config=expert_indices_mem_config,
    #         mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    #     )
    #
    #     # Expert scores: same distribution as indices
    #     expert_scores_flat = expert_scores.reshape(total_tokens, selected_experts_k)
    #     expert_scores_replicated = expert_scores_flat.unsqueeze(0).repeat(num_devices, 1, 1)
    #     tt_expert_scores = ttnn.from_torch(
    #         expert_scores_replicated,
    #         device=mesh_device,
    #         layout=ttnn.ROW_MAJOR_LAYOUT,
    #         dtype=in0_dtype,
    #         memory_config=expert_scores_mem_config,
    #         mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    #     )

    # ------------------------------------------------------------------
    # Run moe_gpt on every device in the mesh
    # ------------------------------------------------------------------
    all_accuracy_metrics = {}

    for layer_id in range(L):
        tt_input = ttnn.from_torch(
            torch_input[layer_id],
            dtype=in0_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=input_sharded_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        _tt_output = ttnn.experimental.moe_gpt(
            tt_input,
            w0_w1_tensor=tt_w0_w1,
            w2_tensor=tt_w2,
            output_tensor=tt_input,
            num_experts=E,
            layer_id=layer_id,
            # Future kwargs (uncomment as op gains support):
            # sparse_buffer=tt_sparse_buffer,
            # expert_indices=tt_expert_indices,
            # expert_scores=tt_expert_scores,
            # expert_mapping=tt_expert_mapping,
            # output_height_shard_dim=4,
            # output_width_shard_dim=4,
            # cluster_axis=cluster_axis,
        )

        # Validate on device 0 (all devices get identical inputs/weights)
        tt_raw_output = ttnn.to_torch(ttnn.get_device_tensors(_tt_output)[0])
        tt_to_torch_output = prepare_output_tensor(tt_raw_output, E, M, K, ring2cores)

        with torch.no_grad():
            torch_input_ref = torch_input[layer_id, 0, ...]  # (E, M, K)
            torch_w0_output_ref = torch_input_ref @ torch_w0[layer_id]
            torch_w1_output_ref = torch_input_ref @ torch_w1[layer_id]
            torch_intermediate_ref = swiglu_reference(torch_w0_output_ref, torch_w1_output_ref)
            torch_output_ref = torch_intermediate_ref @ torch_w2[layer_id]

        for expert_id in range(E):
            torch_layer_output = torch_output_ref[expert_id, :, :]
            tt_layer_output = tt_to_torch_output[expert_id, :, :]
            layer_metrics = get_accuracy_metrics(torch_layer_output, tt_layer_output)
            all_accuracy_metrics[(layer_id, expert_id)] = layer_metrics

    return all_accuracy_metrics


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 500000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((4, 8), id="4x8_galaxy"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("M", [32])
@pytest.mark.parametrize("K, N", [(2880, 2880)])
@pytest.mark.parametrize("E", [4])
@pytest.mark.parametrize("L", [1])
def test_moe_gpt_galaxy(
    mesh_device,
    cluster_axis,
    M,
    K,
    N,
    E,
    L,
    device_params,
):
    # Create a 4x1 column submesh from the full 4x8 galaxy.
    # The moe_gpt op runs on this "minimesh" — one column where
    # all_to_all dispatch/combine happens along cluster_axis=0.
    submesh = mesh_device.create_submesh(ttnn.MeshShape(4, 1), ttnn.MeshCoordinate(0, 0))
    submesh_shape = (4, 1)

    accuracy_metrics = run_test_moe_gpt_galaxy(
        mesh_device=submesh,
        mesh_shape=submesh_shape,
        cluster_axis=cluster_axis,
        M=M,
        K=K,
        N=N,
        E=E,
        L=L,
    )

    passing = True
    for (layer_id, expert_id), metrics in accuracy_metrics.items():
        if metrics["pcc"] < PCC_THRESHOLD:
            passing = False
            logger.warning(
                f"Layer {layer_id}, Expert {expert_id}: PCC={metrics['pcc']:.6f} "
                f"RMSE={metrics['relative_rmse']:.6f}"
            )
        else:
            logger.info(
                f"Layer {layer_id}, Expert {expert_id}: PCC={metrics['pcc']:.6f} "
                f"RMSE={metrics['relative_rmse']:.6f} (Passed)"
            )

    assert passing, "Some experts in some layers did not pass the PCC check"


# ---------------------------------------------------------------------------
# Tilize verification helpers
# ---------------------------------------------------------------------------


def gen_expert_mapping(experts_total, num_devices):
    """
    Expert-to-device ownership: mapping[d, e] = e // experts_per_device.
    Shape: [num_devices, experts_total], dtype uint16, replicated on all devices.
    """
    experts_per_device = experts_total // num_devices
    row = torch.zeros(experts_total, dtype=torch.int32)
    for e in range(experts_total):
        row[e] = e // experts_per_device
    return row.unsqueeze(0).repeat(num_devices, 1)


def gen_sparse_buffer_and_indices(
    tokens_per_device,
    hidden_size,
    experts_total,
    selected_experts_k,
    num_dispatch_devices,
):
    """
    Generate sparse buffer and all-gathered expert indices/scores.

    Returns:
        sparse_buffer: [total_tokens, hidden_size] bfloat16
        expert_indices: [total_tokens, selected_experts_k] int32 (converted to uint16 later)
        expert_scores:  [total_tokens, selected_experts_k] bfloat16
    """
    total_tokens = tokens_per_device * num_dispatch_devices

    expert_indices = torch.zeros(total_tokens, selected_experts_k, dtype=torch.int32)
    for t in range(total_tokens):
        selected = torch.randperm(experts_total)[:selected_experts_k]
        expert_indices[t, :] = selected.to(torch.int32)

    expert_scores = torch.rand(total_tokens, selected_experts_k, dtype=torch.bfloat16) + 1e-5
    expert_scores = expert_scores / expert_scores.sum(dim=-1, keepdim=True)

    sparse_buffer = torch.rand(total_tokens, hidden_size, dtype=torch.bfloat16) - 0.5

    return sparse_buffer, expert_indices, expert_scores


def tilize_reference(sparse_buffer, expert_indices, expert_mapping, device_idx, experts_per_device, tokens_per_chunk):
    """
    Torch reference for what the tilize kernels should produce.

    For each local expert, gather the tokens routed to it (in order of token index),
    and lay them out in [experts_per_device * total_tokens, hidden_size] format where
    expert e's data starts at row e * total_tokens.

    Returns:
        output: [experts_per_device * total_tokens, hidden_size] bfloat16
        per_expert_counts: [experts_per_device] int
    """
    total_tokens, hidden_size = sparse_buffer.shape
    experts_total = expert_mapping.shape[-1]

    output = torch.zeros(experts_per_device * total_tokens, hidden_size, dtype=sparse_buffer.dtype)
    per_expert_counts = [0] * experts_per_device

    local_expert_global_ids = []
    for e in range(experts_total):
        if expert_mapping[device_idx, e].item() == device_idx:
            local_expert_global_ids.append(e)
            if len(local_expert_global_ids) >= experts_per_device:
                break

    for local_idx, global_id in enumerate(local_expert_global_ids):
        count = 0
        for t in range(total_tokens):
            for k in range(expert_indices.shape[-1]):
                if expert_indices[t, k].item() == global_id:
                    row = local_idx * total_tokens + count
                    output[row, :] = sparse_buffer[t, :]
                    count += 1
                    break
        per_expert_counts[local_idx] = count

    return output, per_expert_counts


# ---------------------------------------------------------------------------
# Tilize test body
# ---------------------------------------------------------------------------


def run_test_moe_gpt_tilize(
    mesh_device,
    mesh_shape,
    cluster_axis,
    M,
    K,
    N,
    E,
    selected_experts_k,
):
    torch.manual_seed(42)
    random.seed(42)

    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = mesh_shape[cluster_axis]
    total_tokens = M * num_dispatch_devices
    experts_total = E * num_devices

    logger.info(f"Tilize test configuration:")
    logger.info(f"  mesh_shape: {mesh_shape}, cluster_axis: {cluster_axis}")
    logger.info(f"  num_devices: {num_devices}, num_dispatch_devices: {num_dispatch_devices}")
    logger.info(f"  tokens_per_device (M): {M}, total_tokens: {total_tokens}")
    logger.info(f"  experts_per_device (E): {E}, experts_total: {experts_total}")
    logger.info(f"  hidden_size (K): {K}, selected_experts_k: {selected_experts_k}")

    tokens_per_chunk = 32
    max_chunks_per_expert = total_tokens // tokens_per_chunk

    # ------------------------------------------------------------------
    # Matmul infrastructure (still needed by the op even though we only verify tilize)
    # ------------------------------------------------------------------
    in0_core_coords = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(0)
    core2dram = {core_coords: dram_bank_id for dram_bank_id, core_coords in enumerate(in0_core_coords)}
    in0_num_cores = len(in0_core_coords)

    in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)
    ring2cores = {}
    for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
        ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in PAD_CORES else 0)

    in0_core_range = [ttnn.CoreRange(ring2cores[i][0], ring2cores[i][0]) for i in range(in0_num_cores)]
    in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)

    dram_core_coords = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(in0_num_cores)]
    dram_core_range = [ttnn.CoreRange(c, c) for c in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    in0_dtype = ttnn.bfloat16
    w_dtype = ttnn.bfloat4_b
    L = 1

    # Input activation (matmul path, still needed)
    input_shape = (in0_num_cores, E, M, K)
    in0_shard_spec = ttnn.ShardSpec(
        grid=in0_core_range_set,
        shard_shape=(E * M, K),
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec
    )

    # Weight configs
    groups_per_core = MAX_W0_W1_TILES_PER_CORE // 2
    w0_w1_shard_height = L * E * groups_per_core * K
    w0_w1_shard_width = 4 * ttnn.TILE_SIZE

    w0_w1_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w0_w1_shard_height, w0_w1_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    w0_w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w0_w1_shard_spec)

    w2_shard_height = L * E * 2 * N
    w2_shard_width = 4 * ttnn.TILE_SIZE
    w2_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w2_shard_height, w2_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    w2_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)

    # Create torch tensors for matmul path (dummy, not verified in this test)
    torch_input = create_torch_input(L, in0_num_cores, E, M, K)
    torch_w0 = create_torch_w0(L, E, K, N)
    torch_w1 = create_torch_w1(L, E, K, N)
    torch_w2 = create_torch_w2(L, E, N, K)

    torch_w0_w1_reordered = prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, K, N, ring2cores)
    tt_w0_w1 = ttnn.from_torch(
        torch_w0_w1_reordered,
        dtype=w_dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w0_w1_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    torch_w2_reordered = prepare_w2_tensor(torch_w2, L, E, N, K, ring2cores)
    tt_w2 = ttnn.from_torch(
        torch_w2_reordered,
        dtype=w_dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w2_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Matmul input
    tt_input = ttnn.from_torch(
        torch_input[0],
        dtype=in0_dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_sharded_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # ------------------------------------------------------------------
    # Tilize inputs
    # ------------------------------------------------------------------
    expert_mapping_torch = gen_expert_mapping(experts_total, num_devices)
    sparse_torch, indices_torch, scores_torch = gen_sparse_buffer_and_indices(
        M,
        K,
        experts_total,
        selected_experts_k,
        num_dispatch_devices,
    )

    logger.info(f"  sparse_buffer shape: {sparse_torch.shape}")
    logger.info(f"  expert_indices shape: {indices_torch.shape}")
    logger.info(f"  expert_scores shape: {scores_torch.shape}")
    logger.info(f"  expert_mapping shape: {expert_mapping_torch.shape}")

    tt_sparse = ttnn.from_torch(
        sparse_torch,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_indices = ttnn.from_torch(
        indices_torch,
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_scores = ttnn.from_torch(
        scores_torch,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_mapping = ttnn.from_torch(
        expert_mapping_torch,
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # ------------------------------------------------------------------
    # Tilize DRAM output (pre-allocated)
    # ------------------------------------------------------------------
    tilize_out_shape = (E * total_tokens, K)
    dram_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    tilize_out_torch = torch.zeros(tilize_out_shape, dtype=torch.bfloat16)
    tt_tilize_out = ttnn.from_torch(
        tilize_out_torch,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # ------------------------------------------------------------------
    # Run the op (matmul + tilize in parallel, verify tilize output)
    # ------------------------------------------------------------------
    _tt_output = ttnn.experimental.moe_gpt(
        tt_input,
        w0_w1_tensor=tt_w0_w1,
        w2_tensor=tt_w2,
        output_tensor=tt_input,
        num_experts=E,
        layer_id=0,
        sparse_buffer=tt_sparse,
        expert_indices=tt_indices,
        expert_scores=tt_scores,
        expert_mapping=tt_mapping,
        tilize_output=tt_tilize_out,
        cluster_axis=cluster_axis,
    )

    # ------------------------------------------------------------------
    # Verify tilize output on device 0
    # ------------------------------------------------------------------
    device_idx = 0
    tt_tilize_result = ttnn.to_torch(ttnn.get_device_tensors(tt_tilize_out)[device_idx])
    tt_tilize_result = tt_tilize_result.reshape(E * total_tokens, K)

    ref_output, per_expert_counts = tilize_reference(
        sparse_torch,
        indices_torch,
        expert_mapping_torch,
        device_idx,
        E,
        tokens_per_chunk,
    )

    logger.info(f"  Per-expert token counts: {per_expert_counts}")

    all_passing = True
    for e in range(E):
        count = per_expert_counts[e]
        if count == 0:
            logger.info(f"  Expert {e}: no tokens routed, skipping")
            continue

        start_row = e * total_tokens
        end_row = start_row + count
        ref_slice = ref_output[start_row:end_row, :]
        tt_slice = tt_tilize_result[start_row:end_row, :]

        _passed, pcc_val = comp_pcc(ref_slice, tt_slice)
        if pcc_val < 0.999:
            all_passing = False
            logger.warning(f"  Expert {e}: PCC={pcc_val:.6f} ({count} tokens) FAILED")
        else:
            logger.info(f"  Expert {e}: PCC={pcc_val:.6f} ({count} tokens) Passed")

    return all_passing


# ---------------------------------------------------------------------------
# Tilize pytest entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 500000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((4, 8), id="4x8_galaxy"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("M", [32])
@pytest.mark.parametrize("K, N", [(2880, 2880)])
@pytest.mark.parametrize("E", [4])
@pytest.mark.parametrize("selected_experts_k", [4])
def test_moe_gpt_tilize_galaxy(
    mesh_device,
    cluster_axis,
    M,
    K,
    N,
    E,
    selected_experts_k,
    device_params,
):
    submesh = mesh_device.create_submesh(ttnn.MeshShape(4, 1), ttnn.MeshCoordinate(0, 0))
    submesh_shape = (4, 1)

    passing = run_test_moe_gpt_tilize(
        mesh_device=submesh,
        mesh_shape=submesh_shape,
        cluster_axis=cluster_axis,
        M=M,
        K=K,
        N=N,
        E=E,
        selected_experts_k=selected_experts_k,
    )

    assert passing, "Tilize output PCC check failed for one or more experts"
