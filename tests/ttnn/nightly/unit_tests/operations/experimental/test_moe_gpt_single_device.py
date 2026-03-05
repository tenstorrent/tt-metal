# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Single-device test for moe_gpt op.

This mirrors test_moe_gpt_galaxy.py but runs on a single device, removing the
need for a galaxy mesh.  Useful for iterating on individual kernel phases
(selective tilize, combine writer) before testing multi-device CCL integration.

Currently supported phases:
  - Matmul phase: DRAM-sharded weights + L1-sharded activation input
  - Tilize phase: selective tilize from a sparse token buffer (simulating
                  all_to_all_dispatch output) to a dense DRAM buffer for
                  correctness verification

Future phases (commented out, included for completeness):
  - all_to_all CCL dispatch/combine
  - Tilize-to-matmul integration (tilized chunks sent directly to matmul
    cores via L1 multicast instead of writing to DRAM)

Dimensions (gpt-oss 20b, single device):
  Matmul test:
    M = 32 (tokens per device)
    K = 2880 (hidden_size) -> 90 tiles
    N = 2880 (intermediate_size) -> 90 tiles
    E = 4 (experts on this device)
    L = 1 (layers)

  Tilize test:
    M = 32 (tokens per device)
    experts_total = 16 (across 4 simulated devices)
    E = 4 (experts on this device = experts_total / 4)
    total_tokens = M * (experts_total / E) = 128  (sparse buffer rows)
    K = 2880 (hidden_size)
    selected_experts_k = 4

    The sparse buffer has 128 rows but only ~32 per-expert tokens are
    routed to this device (k=4 out of 16 experts -> ~25% per device).

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
# Future input helpers (for when the op supports CCL dispatch/combine)
# ---------------------------------------------------------------------------

# def gen_expert_mapping_multidevice(experts, num_devices):
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


# def gen_sparse_buffer_and_indices_multidevice(
#     tokens_per_device, hidden_size, experts, selected_experts_k,
#     num_devices, num_dispatch_devices, dtype=torch.bfloat16,
# ):
#     """
#     Generate sparse buffer (simulating all_to_all_dispatch output) and
#     all-gathered expert indices/scores for multi-device.
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
# Main matmul test body
# ---------------------------------------------------------------------------


def run_test_moe_gpt_matmul(
    device,
    M,  # tokens per device
    K,  # hidden_size
    N,  # intermediate_size
    E,  # experts per device
    L,  # layers
):
    torch.manual_seed(42)
    random.seed(42)

    num_devices = 1
    num_dispatch_devices = 1
    total_tokens = M * num_dispatch_devices
    experts_total = E * num_devices

    logger.info(f"Matmul test configuration:")
    logger.info(f"  num_devices: {num_devices}")
    logger.info(f"  tokens_per_device (M): {M}, total_tokens: {total_tokens}")
    logger.info(f"  experts_per_device (E): {E}, experts_total: {experts_total}")
    logger.info(f"  hidden_size (K): {K}, intermediate_size (N): {N}")
    logger.info(f"  layers (L): {L}")

    # ------------------------------------------------------------------
    # Per-device shard grid
    # ------------------------------------------------------------------
    in0_core_coords = device.get_optimal_dram_bank_to_logical_worker_assignment(0)
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
    # These will be needed when the op gains selective-tilize support
    # integrated with the matmul cores.
    # ------------------------------------------------------------------

    # # Drain tilize core for indices/scores (pinned to a single core)
    # tilize_drain_core = ttnn.CoreCoord(6, 9)

    # # Sparse buffer: [total_tokens, K], interleaved L1
    # sparse_mem_config = ttnn.L1_MEMORY_CONFIG

    # # Expert indices: [total_tokens, selected_experts_k], L1 pinned to drain core
    # # expert_indices_mem_config = create_sharded_memory_config_single_core(
    # #     tilize_drain_core, [total_tokens, selected_experts_k], ttnn.uint16
    # # )

    # # Expert scores: [total_tokens, selected_experts_k], L1 pinned to drain core
    # # expert_scores_mem_config = create_sharded_memory_config_single_core(
    # #     tilize_drain_core, [total_tokens, selected_experts_k], in0_dtype
    # # )

    # # Expert mapping: [num_devices, experts_total], interleaved L1
    # # expert_mapping_mem_config = ttnn.L1_MEMORY_CONFIG

    # ------------------------------------------------------------------
    # Create torch tensors
    # ------------------------------------------------------------------
    torch_input = create_torch_input(L, in0_num_cores, E, M, K)
    torch_w0 = create_torch_w0(L, E, K, N)
    torch_w1 = create_torch_w1(L, E, K, N)
    torch_w2 = create_torch_w2(L, E, N, K)

    # ------------------------------------------------------------------
    # Prepare weight tensors (single device, no mesh_mapper)
    # ------------------------------------------------------------------
    torch_w0_w1_reordered = prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, K, N, ring2cores)
    tt_w0_w1 = ttnn.from_torch(
        torch_w0_w1_reordered,
        dtype=w_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w0_w1_mem_config,
    )

    torch_w2_reordered = prepare_w2_tensor(torch_w2, L, E, N, K, ring2cores)
    tt_w2 = ttnn.from_torch(
        torch_w2_reordered,
        dtype=w_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w2_mem_config,
    )

    # ------------------------------------------------------------------
    # Future: create tilize-phase input tensors (multi-device CCL path)
    # ------------------------------------------------------------------

    # # Expert mapping (constant)
    # expert_mapping = gen_expert_mapping_multidevice(experts_total, num_devices)
    # tt_expert_mapping = ttnn.from_torch(
    #     expert_mapping,
    #     device=device,
    #     layout=ttnn.ROW_MAJOR_LAYOUT,
    #     dtype=ttnn.uint16,
    #     memory_config=expert_mapping_mem_config,
    # )

    # # Per-layer sparse buffer, indices, scores
    # for layer_id in range(L):
    #     sparse_buffer, expert_indices, expert_scores = gen_sparse_buffer_and_indices_multidevice(
    #         M, K, experts_total, selected_experts_k, num_devices, num_dispatch_devices,
    #     )
    #
    #     tt_sparse_buffer = ttnn.from_torch(
    #         sparse_buffer,
    #         device=device,
    #         layout=ttnn.ROW_MAJOR_LAYOUT,
    #         dtype=in0_dtype,
    #         memory_config=sparse_mem_config,
    #     )
    #
    #     expert_indices_flat = expert_indices.reshape(total_tokens, selected_experts_k)
    #     tt_expert_indices = ttnn.from_torch(
    #         expert_indices_flat,
    #         device=device,
    #         layout=ttnn.ROW_MAJOR_LAYOUT,
    #         dtype=ttnn.uint16,
    #         memory_config=expert_indices_mem_config,
    #     )
    #
    #     expert_scores_flat = expert_scores.reshape(total_tokens, selected_experts_k)
    #     tt_expert_scores = ttnn.from_torch(
    #         expert_scores_flat,
    #         device=device,
    #         layout=ttnn.ROW_MAJOR_LAYOUT,
    #         dtype=in0_dtype,
    #         memory_config=expert_scores_mem_config,
    #     )

    # ------------------------------------------------------------------
    # Run moe_gpt
    # ------------------------------------------------------------------
    all_accuracy_metrics = {}

    for layer_id in range(L):
        tt_input = ttnn.from_torch(
            torch_input[layer_id],
            dtype=in0_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=input_sharded_mem_config,
        )

        _tt_output = ttnn.experimental.moe_gpt(
            tt_input,
            w0_w1_tensor=tt_w0_w1,
            w2_tensor=tt_w2,
            output_tensor=tt_input,
            num_experts=E,
            # Future kwargs (uncomment as op gains support):
            # sparse_buffer=tt_sparse_buffer,
            # expert_indices=tt_expert_indices,
            # expert_scores=tt_expert_scores,
            # expert_mapping=tt_expert_mapping,
            # output_height_shard_dim=4,
            # output_width_shard_dim=4,
            # cluster_axis=0,
        )

        tt_raw_output = ttnn.to_torch(_tt_output)
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
# Matmul pytest entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("M", [32])
@pytest.mark.parametrize("K, N", [(2880, 2880)])
@pytest.mark.parametrize("E", [4])
@pytest.mark.parametrize("L", [1])
def test_moe_gpt_matmul_single_device(
    device,
    M,
    K,
    N,
    E,
    L,
    device_params,
):
    accuracy_metrics = run_test_moe_gpt_matmul(
        device=device,
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
    Shape: [num_devices, experts_total], dtype int32.

    All rows are identical (static mapping).  The reader kernel selects the
    row corresponding to the source device of each token, but on single-device
    tests all tokens see the same row.
    """
    experts_per_device = experts_total // num_devices
    row = torch.zeros(experts_total, dtype=torch.int32)
    for e in range(experts_total):
        row[e] = e // experts_per_device
    return row.unsqueeze(0).repeat(num_devices, 1)


def gen_sparse_buffer_and_indices(
    total_tokens,
    hidden_size,
    experts_total,
    selected_experts_k,
    expert_mapping,
    device_idx,
):
    """
    Generate a sparse token buffer and all-gathered expert indices/scores.

    Simulates the all_to_all_dispatch output: the sparse buffer has
    ``total_tokens`` rows, but only rows for tokens that route to at least
    one expert on ``device_idx`` contain valid data.  The rest are zeros
    (garbage in practice, but zeros make debugging easier).

    Returns:
        sparse_buffer: [total_tokens, hidden_size] bfloat16
        expert_indices: [total_tokens, selected_experts_k] int32
        expert_scores:  [total_tokens, selected_experts_k] bfloat16
        active_token_ids: set of token indices that have valid sparse data
    """
    expert_indices = torch.zeros(total_tokens, selected_experts_k, dtype=torch.int32)
    for t in range(total_tokens):
        selected = torch.randperm(experts_total)[:selected_experts_k]
        expert_indices[t, :] = selected.to(torch.int32)

    expert_scores = torch.rand(total_tokens, selected_experts_k, dtype=torch.bfloat16) + 1e-5
    expert_scores = expert_scores / expert_scores.sum(dim=-1, keepdim=True)

    # Only populate sparse buffer rows for tokens routed to this device
    sparse_buffer = torch.zeros(total_tokens, hidden_size, dtype=torch.bfloat16)
    active_token_ids = set()
    for t in range(total_tokens):
        for k in range(selected_experts_k):
            eid = expert_indices[t, k].item()
            if expert_mapping[device_idx, eid].item() == device_idx:
                active_token_ids.add(t)
                sparse_buffer[t, :] = torch.rand(hidden_size, dtype=torch.bfloat16) - 0.5
                break

    return sparse_buffer, expert_indices, expert_scores, active_token_ids


def tilize_reference(sparse_buffer, expert_indices, expert_mapping, device_idx, experts_per_device, tokens_per_chunk):
    """
    Torch reference for what the tilize kernels should produce.

    For each local expert, gather the tokens routed to it (in token-index
    order) and lay them out in a dense ``[experts_per_device * total_tokens,
    hidden_size]`` buffer.  Expert ``e``'s data starts at row
    ``e * total_tokens``.  Only the first ``per_expert_counts[e]`` rows in
    each expert's section contain valid data; the rest are zero-padding.

    Returns:
        output: [experts_per_device * total_tokens, hidden_size] bfloat16
        per_expert_counts: list[int] of length experts_per_device
    """
    total_tokens, hidden_size = sparse_buffer.shape
    experts_total = expert_mapping.shape[-1]

    output = torch.zeros(experts_per_device * total_tokens, hidden_size, dtype=sparse_buffer.dtype)
    per_expert_counts = [0] * experts_per_device

    # Identify which global expert IDs live on this device
    local_expert_global_ids = []
    for e in range(experts_total):
        if expert_mapping[device_idx, e].item() == device_idx:
            local_expert_global_ids.append(e)
            if len(local_expert_global_ids) >= experts_per_device:
                break

    # For each local expert, gather activated tokens in token-index order
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
    device,
    M,
    K,
    N,
    E,
    selected_experts_k,
    experts_total,
):
    """
    Run the moe_gpt tilize-to-DRAM verification test.

    The sparse buffer has ``total_tokens = experts_total // E * M`` rows
    (simulating the all_to_all dispatch output across multiple devices).
    Only tokens whose expert selections include a local expert are actually
    read and tilized.
    """
    torch.manual_seed(42)
    random.seed(42)

    num_devices = experts_total // E
    total_tokens = M * num_devices

    logger.info(f"Tilize test configuration (single device, simulating {num_devices}-device routing):")
    logger.info(f"  num_devices (simulated): {num_devices}")
    logger.info(f"  tokens_per_device (M): {M}, total_tokens: {total_tokens}")
    logger.info(f"  experts_per_device (E): {E}, experts_total: {experts_total}")
    logger.info(f"  hidden_size (K): {K}, selected_experts_k: {selected_experts_k}")

    tokens_per_chunk = 32
    cluster_axis = 0

    # ------------------------------------------------------------------
    # Matmul infrastructure (still needed by the op even though we only verify tilize)
    # ------------------------------------------------------------------
    in0_core_coords = device.get_optimal_dram_bank_to_logical_worker_assignment(0)
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

    # Create torch tensors for matmul path
    torch_input = create_torch_input(L, in0_num_cores, E, M, K)
    torch_w0 = create_torch_w0(L, E, K, N)
    torch_w1 = create_torch_w1(L, E, K, N)
    torch_w2 = create_torch_w2(L, E, N, K)

    torch_w0_w1_reordered = prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, K, N, ring2cores)
    tt_w0_w1 = ttnn.from_torch(
        torch_w0_w1_reordered,
        dtype=w_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w0_w1_mem_config,
    )

    torch_w2_reordered = prepare_w2_tensor(torch_w2, L, E, N, K, ring2cores)
    tt_w2 = ttnn.from_torch(
        torch_w2_reordered,
        dtype=w_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w2_mem_config,
    )

    # Matmul input
    tt_input = ttnn.from_torch(
        torch_input[0],
        dtype=in0_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_sharded_mem_config,
    )

    # ------------------------------------------------------------------
    # Tilize inputs
    # ------------------------------------------------------------------
    device_idx = 0
    expert_mapping_torch = gen_expert_mapping(experts_total, num_devices)
    sparse_torch, indices_torch, scores_torch, active_token_ids = gen_sparse_buffer_and_indices(
        total_tokens,
        K,
        experts_total,
        selected_experts_k,
        expert_mapping_torch,
        device_idx,
    )

    logger.info(f"  sparse_buffer shape: {sparse_torch.shape}")
    logger.info(f"  active tokens on device {device_idx}: {len(active_token_ids)} / {total_tokens}")
    logger.info(f"  expert_indices shape: {indices_torch.shape}")
    logger.info(f"  expert_scores shape: {scores_torch.shape}")
    logger.info(f"  expert_mapping shape: {expert_mapping_torch.shape}")

    tt_sparse = ttnn.from_torch(
        sparse_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_indices = ttnn.from_torch(
        indices_torch,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_scores = ttnn.from_torch(
        scores_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_mapping = ttnn.from_torch(
        expert_mapping_torch,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
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
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_mem_config,
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
        sparse_buffer=tt_sparse,
        expert_indices=tt_indices,
        expert_scores=tt_scores,
        expert_mapping=tt_mapping,
        tilize_output=tt_tilize_out,
        cluster_axis=cluster_axis,
    )

    # ------------------------------------------------------------------
    # Verify tilize output
    # ------------------------------------------------------------------
    tt_tilize_result = ttnn.to_torch(tt_tilize_out)
    tt_tilize_result = tt_tilize_result.reshape(E * total_tokens, K)

    ref_output, per_expert_counts = tilize_reference(
        sparse_torch,
        indices_torch,
        expert_mapping_torch,
        device_idx,
        E,
        tokens_per_chunk,
    )

    logger.info(f"  Per-expert token counts: {per_expert_counts} (total_tokens={total_tokens})")

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
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("M", [32])
@pytest.mark.parametrize("K, N", [(2880, 2880)])
@pytest.mark.parametrize("E, experts_total", [(4, 16)])
@pytest.mark.parametrize("selected_experts_k", [4])
def test_moe_gpt_tilize_single_device(
    device,
    M,
    K,
    N,
    E,
    experts_total,
    selected_experts_k,
    device_params,
):
    passing = run_test_moe_gpt_tilize(
        device=device,
        M=M,
        K=K,
        N=N,
        E=E,
        selected_experts_k=selected_experts_k,
        experts_total=experts_total,
    )

    assert passing, "Tilize output PCC check failed for one or more experts"


# ---------------------------------------------------------------------------
# Tilize-to-matmul integration helpers
# ---------------------------------------------------------------------------


def gen_controlled_sparse_buffer(
    total_tokens,
    hidden_size,
    experts_per_device,
    experts_total,
    selected_experts_k,
    device_idx,
):
    """
    Generate sparse buffer with exactly total_tokens / experts_per_device
    tokens per local expert (one chunk each).

    Token t is assigned to local expert (t % experts_per_device).  This
    guarantees even distribution and exactly 1 chunk per expert when
    tokens_per_chunk == total_tokens / experts_per_device.
    """
    expert_indices = torch.zeros(total_tokens, selected_experts_k, dtype=torch.int32)
    non_local_experts = [e for e in range(experts_total) if e // experts_per_device != device_idx]

    for t in range(total_tokens):
        local_expert = t % experts_per_device
        global_expert = device_idx * experts_per_device + local_expert

        expert_indices[t, 0] = global_expert
        others = random.sample(non_local_experts, selected_experts_k - 1)
        for k, eid in enumerate(others):
            expert_indices[t, k + 1] = eid

    expert_scores = torch.rand(total_tokens, selected_experts_k, dtype=torch.bfloat16) + 1e-5
    expert_scores = expert_scores / expert_scores.sum(dim=-1, keepdim=True)

    sparse_buffer = torch.rand(total_tokens, hidden_size, dtype=torch.bfloat16) - 0.5

    return sparse_buffer, expert_indices, expert_scores


# ---------------------------------------------------------------------------
# Tilize-to-matmul integration test body
# ---------------------------------------------------------------------------


def run_test_moe_gpt_tilize_matmul(
    device,
    M,
    K,
    N,
    E,
    selected_experts_k,
    experts_total,
):
    """
    Integration test: tilize → W0/W1 matmul → SwiGLU → A2A → W2 → combine (fused mode).

    Invokes the op with tilize inputs but WITHOUT tilize_output, triggering
    the fused path where tilized chunks are multicast directly to matmul
    cores.  The full pipeline output (W2 matmul result) is written to
    BLOCK_SHARDED combine cores.

    Verifies the combine output against a torch reference:
      tilize → per-expert W0/W1 matmul → SwiGLU → W2 matmul
    """
    torch.manual_seed(42)
    random.seed(42)

    num_devices = experts_total // E
    total_tokens = M * num_devices

    logger.info(f"Tilize-matmul integration test configuration:")
    logger.info(f"  num_devices (simulated): {num_devices}")
    logger.info(f"  tokens_per_device (M): {M}, total_tokens: {total_tokens}")
    logger.info(f"  experts_per_device (E): {E}, experts_total: {experts_total}")
    logger.info(f"  hidden_size (K): {K}, intermediate_size (N): {N}")

    tokens_per_chunk = 32
    cluster_axis = 0

    # ------------------------------------------------------------------
    # Matmul core infrastructure (same as matmul/tilize tests)
    # ------------------------------------------------------------------
    in0_core_coords = device.get_optimal_dram_bank_to_logical_worker_assignment(0)
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

    # Input activation memory config
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

    # ------------------------------------------------------------------
    # Create torch tensors
    # ------------------------------------------------------------------
    torch_input = create_torch_input(L, in0_num_cores, E, M, K)
    torch_w0 = create_torch_w0(L, E, K, N)
    torch_w1 = create_torch_w1(L, E, K, N)
    torch_w2 = create_torch_w2(L, E, N, K)

    # Prepare weight tensors
    torch_w0_w1_reordered = prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, K, N, ring2cores)
    tt_w0_w1 = ttnn.from_torch(
        torch_w0_w1_reordered,
        dtype=w_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w0_w1_mem_config,
    )

    torch_w2_reordered = prepare_w2_tensor(torch_w2, L, E, N, K, ring2cores)
    tt_w2 = ttnn.from_torch(
        torch_w2_reordered,
        dtype=w_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w2_mem_config,
    )

    # Matmul input (still needed for the op's output_tensor parameter)
    tt_input = ttnn.from_torch(
        torch_input[0],
        dtype=in0_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_sharded_mem_config,
    )

    # ------------------------------------------------------------------
    # Tilize inputs (controlled routing: 32 tokens per expert)
    # ------------------------------------------------------------------
    device_idx = 0
    expert_mapping_torch = gen_expert_mapping(experts_total, num_devices)
    sparse_torch, indices_torch, scores_torch = gen_controlled_sparse_buffer(
        total_tokens,
        K,
        E,
        experts_total,
        selected_experts_k,
        device_idx,
    )

    logger.info(f"  sparse_buffer shape: {sparse_torch.shape}")
    logger.info(f"  expert_indices shape: {indices_torch.shape}")
    logger.info(f"  expert_mapping shape: {expert_mapping_torch.shape}")

    tt_sparse = ttnn.from_torch(
        sparse_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_indices = ttnn.from_torch(
        indices_torch,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_scores = ttnn.from_torch(
        scores_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_mapping = ttnn.from_torch(
        expert_mapping_torch,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # ------------------------------------------------------------------
    # Pre-allocate BLOCK_SHARDED combine output on combine cores
    # Combine grid: 3x4 at CoreRange({1,0},{3,3})
    # Shard shape: [E * M / height_shard_dim, K / width_shard_dim] = [32, 960]
    # ------------------------------------------------------------------
    combine_width_shard_dim = 3
    combine_height_shard_dim = 4
    combine_shard_h = E * M // combine_height_shard_dim  # 4*32/4 = 32
    combine_shard_w = K // combine_width_shard_dim  # 2880/3 = 960

    # Dynamically find a 3x4 combine core range that avoids matmul (DRAM-bank) cores
    # Must match the algorithm in moe_gpt_program_factory.cpp
    bank_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.RISCV_0_default)
    matmul_core_set = {(c.x, c.y) for c in bank_cores}
    worker_grid = device.compute_with_storage_grid_size()
    combine_core_range = None
    for sy in range(worker_grid.y - combine_height_shard_dim + 1):
        for sx in range(worker_grid.x - combine_width_shard_dim + 1):
            valid = all(
                (sx + dx, sy + dy) not in matmul_core_set
                for dy in range(combine_height_shard_dim)
                for dx in range(combine_width_shard_dim)
            )
            if valid:
                combine_core_range = ttnn.CoreRange(
                    ttnn.CoreCoord(sx, sy),
                    ttnn.CoreCoord(sx + combine_width_shard_dim - 1, sy + combine_height_shard_dim - 1),
                )
                break
        if combine_core_range is not None:
            break
    assert combine_core_range is not None, "Could not find 3x4 combine core range avoiding matmul cores"
    logger.info(f"Selected combine core range: {combine_core_range}")
    combine_core_range_set = ttnn.CoreRangeSet([combine_core_range])

    combine_shard_spec = ttnn.ShardSpec(
        grid=combine_core_range_set,
        shard_shape=(combine_shard_h, combine_shard_w),
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    combine_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, combine_shard_spec
    )

    # Create the output tensor: [E * M, K] = [128, 2880] ROW_MAJOR BLOCK_SHARDED
    combine_out_torch = torch.zeros((E * M, K), dtype=torch.bfloat16)
    tt_combine_output = ttnn.from_torch(
        combine_out_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=combine_mem_config,
    )

    # ------------------------------------------------------------------
    # Run the op (fused mode: tilize inputs present, NO tilize_output)
    # ------------------------------------------------------------------
    _tt_output = ttnn.experimental.moe_gpt(
        tt_input,
        w0_w1_tensor=tt_w0_w1,
        w2_tensor=tt_w2,
        output_tensor=tt_input,
        num_experts=E,
        enable_dram_output=True,
        dram_output_tensor=tt_combine_output,
        sparse_buffer=tt_sparse,
        expert_indices=tt_indices,
        expert_scores=tt_scores,
        expert_mapping=tt_mapping,
        cluster_axis=cluster_axis,
    )

    # ------------------------------------------------------------------
    # Read back BLOCK_SHARDED combine output
    # ------------------------------------------------------------------
    tt_combine_result = ttnn.to_torch(tt_combine_output)
    tt_combine_result = tt_combine_result.reshape(E * M, K)

    # ------------------------------------------------------------------
    # Torch reference: tilize → per-expert W0/W1 matmul → SwiGLU → W2
    # ------------------------------------------------------------------
    ref_tilized, per_expert_counts = tilize_reference(
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

        # Gather tilized tokens for this expert
        start_row = e * total_tokens
        expert_input = ref_tilized[start_row : start_row + count, :]  # [count, K]

        # Pad to tile height (32) for comparison with hardware output
        padded_input = torch.zeros(M, K, dtype=torch.bfloat16)
        padded_input[:count, :] = expert_input

        # Torch reference: W0/W1 matmul + SwiGLU + W2 matmul
        with torch.no_grad():
            gate = padded_input @ torch_w0[0, e]  # [M, N]
            up = padded_input @ torch_w1[0, e]  # [M, N]
            swiglu_out = swiglu_reference(gate, up)  # [M, N]
            reference = swiglu_out @ torch_w2[0, e]  # [M, K]

        # Extract device result for this expert from combine output
        # Each expert occupies M rows: [e*M : (e+1)*M, :]
        tt_expert_result = tt_combine_result[e * M : (e + 1) * M, :]  # [M, K]

        metrics = get_accuracy_metrics(reference, tt_expert_result)
        pcc = metrics["pcc"]
        rmse = metrics["relative_rmse"]

        if pcc < PCC_THRESHOLD:
            all_passing = False
            logger.warning(f"  Expert {e}: PCC={pcc:.6f} RMSE={rmse:.6f} ({count} tokens) FAILED")
        else:
            logger.info(f"  Expert {e}: PCC={pcc:.6f} RMSE={rmse:.6f} ({count} tokens) Passed")

    return all_passing


# ---------------------------------------------------------------------------
# Tilize-matmul integration pytest entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("M", [32])
@pytest.mark.parametrize("K, N", [(2880, 2880)])
@pytest.mark.parametrize("E, experts_total", [(4, 16)])
@pytest.mark.parametrize("selected_experts_k", [4])
def test_moe_gpt_tilize_matmul_single_device(
    device,
    M,
    K,
    N,
    E,
    experts_total,
    selected_experts_k,
    device_params,
):
    passing = run_test_moe_gpt_tilize_matmul(
        device=device,
        M=M,
        K=K,
        N=N,
        E=E,
        selected_experts_k=selected_experts_k,
        experts_total=experts_total,
    )

    assert passing, "Tilize-matmul integration PCC check failed for one or more experts"
