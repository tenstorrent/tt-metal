# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Single-device test for moe_gpt op (fused tilize → matmul → combine pipeline).

Tests the full fused path:
  tilize → W0/W1 matmul → SwiGLU → A2A ring → W2 matmul → untilize → combine

Dimensions (gpt-oss 20b, single device):
    M = 32 (tokens per device)
    K = 2880 (hidden_size) -> 90 tiles
    N = 2880 (intermediate_size) -> 90 tiles
    E = 4 (experts on this device)
    experts_total = 16 (across 4 simulated devices)
    total_tokens = M * (experts_total / E) = 128 (sparse buffer rows)
    selected_experts_k = 4

Activation: SwiGLU (gpt-oss variant)
  gate_clamped = clamp(gate, max=7.0)
  up_clamped   = clamp(up, min=-7.0, max=7.0)
  result       = (up_clamped + 1) * gate_clamped * sigmoid(1.702 * gate_clamped)
"""

import random
import pytest
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
# Tilize helpers
# ---------------------------------------------------------------------------


def gen_expert_mapping(experts_total, num_devices):
    """
    Expert-to-device ownership: mapping[d, e] = e // experts_per_device.
    Shape: [num_devices, experts_total], dtype int32.
    """
    experts_per_device = experts_total // num_devices
    row = torch.zeros(experts_total, dtype=torch.int32)
    for e in range(experts_total):
        row[e] = e // experts_per_device
    return row.unsqueeze(0).repeat(num_devices, 1)


def gen_controlled_sparse_buffer(
    total_tokens,
    hidden_size,
    experts_per_device,
    experts_total,
    selected_experts_k,
    device_idx,
    tokens_per_expert=None,
    all_local_experts=False,
):
    """
    Generate sparse buffer with controlled token-to-expert routing.

    Modes:
      - tokens_per_expert=None, all_local_experts=False:
          Even distribution, 1 local expert per token.
      - tokens_per_expert=[...], all_local_experts=False:
          Uneven distribution, 1 local expert per token. Sum must equal total_tokens.
      - all_local_experts=True:
          Every token routes to ALL local experts (tokens_per_expert ignored).
          Each k-slot is filled with a local expert. Requires selected_experts_k >= experts_per_device.
    """
    expert_indices = torch.zeros(total_tokens, selected_experts_k, dtype=torch.int32)
    non_local_experts = [e for e in range(experts_total) if e // experts_per_device != device_idx]

    if all_local_experts:
        assert selected_experts_k >= experts_per_device
        local_globals = [device_idx * experts_per_device + le for le in range(experts_per_device)]
        for t in range(total_tokens):
            for k in range(experts_per_device):
                expert_indices[t, k] = local_globals[k]
            # Fill remaining slots with non-local experts (if selected_experts_k > experts_per_device)
            remaining = selected_experts_k - experts_per_device
            if remaining > 0:
                others = random.sample(non_local_experts, remaining)
                for k, eid in enumerate(others):
                    expert_indices[t, experts_per_device + k] = eid
    else:
        if tokens_per_expert is None:
            tokens_per_expert = [total_tokens // experts_per_device] * experts_per_device

        assert len(tokens_per_expert) == experts_per_device
        assert sum(tokens_per_expert) == total_tokens

        # Build assignment: token t -> local expert based on tokens_per_expert distribution
        token_to_local_expert = []
        for local_e, count in enumerate(tokens_per_expert):
            token_to_local_expert.extend([local_e] * count)

        for t in range(total_tokens):
            local_expert = token_to_local_expert[t]
            global_expert = device_idx * experts_per_device + local_expert

            expert_indices[t, 0] = global_expert
            others = random.sample(non_local_experts, selected_experts_k - 1)
            for k, eid in enumerate(others):
                expert_indices[t, k + 1] = eid

    expert_scores = torch.rand(total_tokens, selected_experts_k, dtype=torch.bfloat16) + 1e-5
    expert_scores = expert_scores / expert_scores.sum(dim=-1, keepdim=True)

    sparse_buffer = torch.rand(total_tokens, hidden_size, dtype=torch.bfloat16) - 0.5

    return sparse_buffer, expert_indices, expert_scores


def tilize_reference(sparse_buffer, expert_indices, expert_mapping, device_idx, experts_per_device, tokens_per_chunk):
    """
    Torch reference for the tilize phase output.

    For each local expert, gather tokens routed to it (in token-index order).
    Returns:
        output: [experts_per_device * total_tokens, hidden_size] bfloat16
        per_expert_counts: list[int] of length experts_per_device
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
# Test body
# ---------------------------------------------------------------------------


def run_test_moe_gpt_tilize_matmul(
    device,
    M,
    K,
    N,
    E,
    selected_experts_k,
    experts_total,
    tokens_per_expert=None,
    all_local_experts=False,
):
    """
    Integration test: tilize → W0/W1 matmul → SwiGLU → A2A → W2 → combine (fused mode).

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
    # Matmul core infrastructure
    # ------------------------------------------------------------------
    in0_core_coords = device.get_optimal_dram_bank_to_logical_worker_assignment(0)
    core2dram = {core_coords: dram_bank_id for dram_bank_id, core_coords in enumerate(in0_core_coords)}
    in0_num_cores = len(in0_core_coords)

    in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)
    ring2cores = {}
    for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
        ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in PAD_CORES else 0)

    dram_core_coords = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(in0_num_cores)]
    dram_core_range = [ttnn.CoreRange(c, c) for c in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    w_dtype = ttnn.bfloat4_b
    L = 1

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
        tokens_per_expert=tokens_per_expert,
        all_local_experts=all_local_experts,
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
    # Run the op (returns vector of 5 tensors)
    # ------------------------------------------------------------------
    outputs = ttnn.experimental.moe_gpt(
        tt_sparse,
        expert_indices=tt_indices,
        expert_scores=tt_scores,
        expert_mapping=tt_mapping,
        w0_w1_tensor=tt_w0_w1,
        w2_tensor=tt_w2,
        cluster_axis=cluster_axis,
    )

    # Output 4 is the ROW_MAJOR re-perceived output
    tt_output = outputs[4]

    # ------------------------------------------------------------------
    # Read back output
    # ------------------------------------------------------------------
    tt_output_result = ttnn.to_torch(tt_output)
    # Reshape from sharded to flat: [E * M, K]
    tt_output_result = tt_output_result.reshape(-1, K)[: E * M, :]

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

        # Multi-chunk: combine cores only hold the LAST chunk's output (each chunk
        # overwrites the same shard).  Extract the last chunk's tokens for reference.
        num_chunks = (count + tokens_per_chunk - 1) // tokens_per_chunk
        last_chunk_start = (num_chunks - 1) * tokens_per_chunk
        last_chunk_tokens = min(count - last_chunk_start, tokens_per_chunk)

        # Pad to tile height (32) for comparison with hardware output
        padded_input = torch.zeros(M, K, dtype=torch.bfloat16)
        padded_input[:last_chunk_tokens, :] = expert_input[last_chunk_start : last_chunk_start + last_chunk_tokens, :]

        # Torch reference: W0/W1 matmul + SwiGLU + W2 matmul
        with torch.no_grad():
            gate = padded_input @ torch_w0[0, e]  # [M, N]
            up = padded_input @ torch_w1[0, e]  # [M, N]
            swiglu_out = swiglu_reference(gate, up)  # [M, N]
            reference = swiglu_out @ torch_w2[0, e]  # [M, K]

        # Extract device result for this expert from output
        # Each expert occupies M rows: [e*M : (e+1)*M, :]
        tt_expert_result = tt_output_result[e * M : (e + 1) * M, :]  # [M, K]

        metrics = get_accuracy_metrics(reference, tt_expert_result)
        pcc = metrics["pcc"]
        rmse = metrics["relative_rmse"]

        chunks_str = f"{num_chunks} chunk{'s' if num_chunks > 1 else ''}"
        if pcc < PCC_THRESHOLD:
            all_passing = False
            logger.warning(f"  Expert {e}: PCC={pcc:.6f} RMSE={rmse:.6f} ({count} tokens, {chunks_str}) FAILED")
        else:
            logger.info(f"  Expert {e}: PCC={pcc:.6f} RMSE={rmse:.6f} ({count} tokens, {chunks_str}) Passed")

    return all_passing


# ---------------------------------------------------------------------------
# Pytest entry point
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
@pytest.mark.parametrize(
    "tokens_per_expert",
    [
        [64, 32, 32, 0],  # Expert 0 gets 2 chunks, expert 3 gets none
        [96, 32, 0, 0],  # Expert 0 gets 3 chunks, experts 2-3 get none
    ],
    ids=["2chunks_1_1_0", "3chunks_1_0_0"],
)
def test_moe_gpt_tilize_matmul_multi_chunk(
    device,
    M,
    K,
    N,
    E,
    experts_total,
    selected_experts_k,
    tokens_per_expert,
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
        tokens_per_expert=tokens_per_expert,
    )

    assert passing, "Multi-chunk tilize-matmul PCC check failed for one or more experts"


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
@pytest.mark.parametrize("selected_experts_k", [4])
@pytest.mark.parametrize(
    "E, experts_total",
    [
        (4, 8),  # 2 devices -> 64 total tokens, all to all 4 local -> 64/expert (2 chunks)
        (4, 12),  # 3 devices -> 96 total tokens, all to all 4 local -> 96/expert (3 chunks)
    ],
    ids=["64tok_all_local", "96tok_all_local"],
)
def test_moe_gpt_tilize_matmul_all_local(
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
        all_local_experts=True,
    )

    assert passing, "All-local-experts tilize-matmul PCC check failed for one or more experts"
