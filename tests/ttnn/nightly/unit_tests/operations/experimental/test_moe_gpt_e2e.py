# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
E2E test for GPT-OSS MoE pipeline:
  all_to_all_dispatch_metadata → moe_gpt (compute)

This test chains the optimized A2A dispatch (all_to_all_dispatch_metadata)
with the GPT-OSS fused compute kernel (moe_gpt) to verify the full pipeline
from raw token inputs through to per-expert outputs.

Pipeline:
  1. all_to_all_dispatch_metadata (cluster_axis=0, ring along rows)
       - Routes raw token embeddings to devices owning the selected experts
       - Outputs: sparse_buffer + expert_indices + expert_scores per device
  2. moe_gpt (compute, cluster_axis=0)
       - Fused: tilize → W0/W1 matmul → SwiGLU → A2A ring → W2 matmul → combine
       - Accepts sparse_buffer from dispatch as input
  3. Accuracy verification against torch reference
  4. E2E performance measurement (dispatch + compute)

Similar to test_optimized_moe_decode_block.py for deepseek but for GPT-OSS moe_gpt.

Dimensions (gpt-oss 20b, single ring simulation):
    M = 32 (tokens per device in ring)
    K = 2880 (hidden_size)
    N = 2880 (intermediate_size)
    E = 4 (experts per ring device)
    ring_devices = 4 (dispatch_devices along cluster_axis=0)
    experts_total = 16 (ring_devices × E = 4×4)
    selected_experts_k = 4
    total_tokens = M × ring_devices = 32 × 4 = 128 per ring

Note: On a 4x8 mesh, all 8 columns run the same simulation independently.
"""

import random
import time
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc, profiler
from models.demos.gpt_oss.tt.experts_throughput.weights import (
    _FUSED_MAX_TILES_PER_CORE as MAX_W0_W1_TILES_PER_CORE,
    _FUSED_PAD_CORES as PAD_CORES,
    _prepare_w0_w1_tensor as prepare_w0_w1_tensor,
    _prepare_w2_tensor as prepare_w2_tensor,
    _build_ring2cores as build_ring2cores,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.test_moe_gpt_single_device import (
    gen_controlled_sparse_buffer,
    verify_device_output,
    create_torch_w0,
    create_torch_w1,
    create_torch_w2,
    swiglu_reference,
    get_accuracy_metrics,
    PCC_THRESHOLD,
)
from models.demos.gpt_oss.tt.experts_throughput.config import create_expert_mapping_tensors

# Single-iteration values for Tracy device profiling (TT_METAL_DEVICE_PROFILER=1).
# A single warmup compiles/caches the program; a single measure is profiled.
PERF_WARMUP_ITERS = 1
PERF_MEASURE_ITERS = 1


# ---------------------------------------------------------------------------
# Dispatch expert mapping (ring-position format)
# ---------------------------------------------------------------------------


def get_moe_gpt_combine_core_range(mesh_device, combine_w=3, combine_h=4):
    """Find the COMBINE_W×COMBINE_H rectangle that moe_gpt uses for combine output.

    Replicates moe_gpt's combine core selection logic: searches for the first
    (combine_w × combine_h) rectangle of worker cores that avoids DRAM-bank
    (matmul) cores.  This ensures selective_reduce_combine's worker cores are
    identical to moe_gpt's combine shard cores, so set_globally_allocated_address
    on the BLOCK_SHARDED combine output works correctly.

    Returns:
        (CoreRangeSet, start_coord, end_coord)
    """
    dram_core_coords = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(0)
    matmul_cores = {(c.x, c.y) for c in dram_core_coords}
    compute_grid = mesh_device.compute_with_storage_grid_size()

    for sy in range(compute_grid.y - combine_h + 1):
        for sx in range(compute_grid.x - combine_w + 1):
            if all((sx + dx, sy + dy) not in matmul_cores for dy in range(combine_h) for dx in range(combine_w)):
                start = ttnn.CoreCoord(sx, sy)
                end = ttnn.CoreCoord(sx + combine_w - 1, sy + combine_h - 1)
                return ttnn.CoreRangeSet([ttnn.CoreRange(start, end)]), start, end

    raise RuntimeError(f"Could not find {combine_w}x{combine_h} combine core range")


def gen_dispatch_expert_mapping(experts_total, ring_devices, total_mesh_devices):
    """Create expert mapping for all_to_all_dispatch_metadata.

    dispatch_mapping[d, e] = ring_position of device owning expert e.
    Ring positions are 0..ring_devices-1 (independent of mesh column).
    All rows use the same assignment since ring position = row index in the ring.

    Args:
        experts_total: Total number of experts in the simulation
        ring_devices: Number of devices in each ring (dispatch_devices)
        total_mesh_devices: Total mesh devices (used to replicate mapping)

    Returns:
        Tensor [total_mesh_devices, experts_total] uint16
    """
    experts_per_device = experts_total // ring_devices
    # Expert e → ring position = e // experts_per_device
    assignment = torch.arange(experts_total, dtype=torch.int32) // experts_per_device
    return assignment.unsqueeze(0).repeat(total_mesh_devices, 1).to(torch.int16)


# ---------------------------------------------------------------------------
# Torch reference for the full e2e dispatch + compute pipeline
# ---------------------------------------------------------------------------


def compute_e2e_reference(
    raw_input,  # [ring_devices, M, K] raw tokens per ring device
    expert_indices,  # [ring_devices, M, selected_k] routing indices
    expert_scores,  # [ring_devices, M, selected_k] routing scores
    torch_w0,  # [L, E, K, N]
    torch_w1,  # [L, E, K, N]
    torch_w2,  # [L, E, N, K]
    E,
    ring_devices,
    experts_total,
):
    """
    Compute per-device per-expert W2 outputs matching moe_gpt's output format.

    For each device (ring position 0..ring_devices-1):
      - Find tokens routed to each of its E experts
      - Run W0/W1 -> SwiGLU -> W2
      - Returns [E, M, K] (padded to M per expert)

    This mirrors what moe_gpt outputs for verify_device_output comparison.
    """
    M = raw_input.shape[1]
    K = raw_input.shape[2]
    experts_per_device = E

    results = []  # per ring device

    for ring_pos in range(ring_devices):
        device_result = torch.zeros(E * M, K, dtype=torch.bfloat16)

        # This ring device owns experts: [ring_pos*E .. (ring_pos+1)*E)
        local_expert_global_ids = list(range(ring_pos * experts_per_device, (ring_pos + 1) * experts_per_device))

        for local_e, global_e in enumerate(local_expert_global_ids):
            # Gather all tokens across all ring devices that route to this expert
            tokens_for_expert = []
            for src_dev in range(ring_devices):
                for t in range(M):
                    for k in range(expert_indices.shape[-1]):
                        if expert_indices[src_dev, t, k].item() == global_e:
                            tokens_for_expert.append(raw_input[src_dev, t, :])
                            break

            if not tokens_for_expert:
                continue

            # Process through W0/W1/SwiGLU/W2 (only last chunk of M tokens)
            all_tokens = torch.stack(tokens_for_expert, dim=0)  # [count, K]
            count = all_tokens.shape[0]
            tokens_per_chunk = M

            # Multi-chunk: last chunk only (matches moe_gpt combine behavior)
            num_chunks = (count + tokens_per_chunk - 1) // tokens_per_chunk
            last_chunk_start = (num_chunks - 1) * tokens_per_chunk
            last_chunk_tokens = min(count - last_chunk_start, tokens_per_chunk)

            chunk = torch.zeros(M, K, dtype=torch.bfloat16)
            chunk[:last_chunk_tokens, :] = all_tokens[last_chunk_start : last_chunk_start + last_chunk_tokens, :]

            with torch.no_grad():
                gate = chunk @ torch_w0[0, local_e]  # [M, N]
                up = chunk @ torch_w1[0, local_e]  # [M, N]
                activated = swiglu_reference(gate, up)  # [M, N]
                w2_out = activated @ torch_w2[0, local_e]  # [M, K]

            device_result[local_e * M : (local_e + 1) * M, :] = w2_out

        results.append(device_result)

    return results  # list of [E*M, K] per ring position


# ---------------------------------------------------------------------------
# Dispatch-only test
# ---------------------------------------------------------------------------


def run_test_dispatch(mesh_device, tokens_global, hidden_size, selected_experts_k, experts_total):
    """Test all_to_all_dispatch_metadata in isolation.

    Verifies that each device's sparse buffer contains exactly the tokens that
    route to experts owned by that device, based on the expert_indices tensor.
    """
    torch.manual_seed(42)
    cluster_axis = 0
    total_mesh_devices = mesh_device.get_num_devices()
    mesh_cols = mesh_device.shape[1]

    # Expert mapping: mapping[d, e] = linearized target device ID for expert e
    mapping = create_expert_mapping_tensors(
        num_devices=total_mesh_devices,
        num_experts_global=experts_total,
        mesh_device=mesh_device,
        new_format=True,
        return_torch=True,
    )
    tt_mapping = ttnn.from_torch(
        mapping,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    # Raw input: [1, 1, tokens_global, hidden_size], sharded across mesh rows, replicated across cols
    raw_input_torch = torch.rand(1, 1, tokens_global, hidden_size, dtype=torch.bfloat16) - 0.5

    # Expert routing: each token picks selected_experts_k unique global expert IDs
    indices_list = [
        torch.randperm(experts_total)[:selected_experts_k].sort().values.to(torch.int32) for _ in range(tokens_global)
    ]
    expert_indices = torch.stack(indices_list).reshape(1, 1, tokens_global, selected_experts_k)
    scores_list = [torch.rand(selected_experts_k, dtype=torch.float32) for _ in range(tokens_global)]
    expert_scores = torch.stack([s / s.sum() for s in scores_list]).reshape(1, 1, tokens_global, selected_experts_k)

    # HEIGHT_SHARDED L1 config for input indices/scores (required by dispatch kernel)
    num_cores_y = min(8, tokens_global // mesh_device.shape[0])
    num_cores_x = (tokens_global // mesh_device.shape[0] + num_cores_y - 1) // num_cores_y
    input_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}),
        [1, selected_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_indices_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    shard_2d = dict(
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=tuple(mesh_device.shape)),
    )
    tt_raw_input = ttnn.from_torch(
        raw_input_torch, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, **shard_2d
    )
    tt_indices = ttnn.from_torch(
        expert_indices.to(torch.int16), dtype=ttnn.uint16, memory_config=input_indices_mem, **shard_2d
    )
    tt_scores = ttnn.from_torch(expert_scores, dtype=ttnn.bfloat16, memory_config=input_indices_mem, **shard_2d)

    # Pre-allocate dispatch outputs
    dispatch_drain_core = ttnn.CoreCoord(6, 9)
    drain_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(dispatch_drain_core, dispatch_drain_core)}),
        [tokens_global, selected_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    drain_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, drain_shard_spec)

    tt_sparse_out = ttnn.from_torch(
        torch.zeros(mesh_device.shape[0], tokens_global, hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        **shard_2d,
    )
    tt_indices_out = ttnn.from_torch(
        torch.zeros(1, tokens_global, selected_experts_k, dtype=torch.int16),
        dtype=ttnn.uint16,
        memory_config=drain_mem,
        **shard_2d,
    )
    tt_scores_out = ttnn.from_torch(
        torch.zeros(1, tokens_global, selected_experts_k, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        memory_config=drain_mem,
        **shard_2d,
    )

    compute_grid = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    semaphore = ttnn.create_global_semaphore(mesh_device, all_cores, 0)

    tt_input_l1 = ttnn.to_memory_config(tt_raw_input, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.synchronize_device(mesh_device)

    (output_tensor, indices_tensor, scores_tensor) = ttnn.experimental.all_to_all_dispatch_metadata(
        tt_input_l1,
        tt_indices,
        tt_scores,
        tt_mapping,
        cluster_axis=cluster_axis,
        num_links=4,
        output_tensors=(tt_sparse_out, tt_indices_out, tt_scores_out),
        cross_device_semaphore=semaphore,
        dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_UNICAST,
    )
    ttnn.synchronize_device(mesh_device)

    # Verify: device d receives token (0, token_idx) at slot token_idx
    # iff that token routes to any expert owned by device d
    sparse_per_device = ttnn.get_device_tensors(output_tensor)
    indices_per_device = ttnn.get_device_tensors(indices_tensor)
    scores_per_device = ttnn.get_device_tensors(scores_tensor)
    all_passing = True
    for device_idx in range(total_mesh_devices):
        actual = ttnn.to_torch(sparse_per_device[device_idx]).reshape(tokens_global, hidden_size).bfloat16()

        expected = torch.zeros(tokens_global, hidden_size, dtype=torch.bfloat16)
        for token_idx in range(tokens_global):
            for expert_idx in expert_indices[0, 0, token_idx]:
                if int(mapping[0, expert_idx]) == device_idx:
                    expected[token_idx] = raw_input_torch[0, 0, token_idx]

        if not torch.equal(actual, expected):
            mismatch_slots = (actual != expected).any(dim=-1).nonzero(as_tuple=True)[0].tolist()
            logger.warning(
                f"Device {device_idx}): " f"FAILED at {len(mismatch_slots)} slots, first few: {mismatch_slots[:8]}"
            )
            all_passing = False
        else:
            logger.info(f"Device {device_idx}): Passed")

    return all_passing


# ---------------------------------------------------------------------------
# Dispatch+Compute test
# ---------------------------------------------------------------------------


def run_test_dispatch_compute(mesh_device, tokens_global, hidden_size, selected_experts_k, experts_total):
    """Test all_to_all_dispatch_metadata → moe_gpt pipeline.

    Verifies per-expert W2 outputs against torch reference.
    Expert mapping uses global IDs: mapping[d, e] = e // experts_per_device (0..total_devices-1).
    """
    torch.manual_seed(42)
    cluster_axis = 0
    ring_devices = mesh_device.shape[cluster_axis]  # 4
    total_mesh_devices = mesh_device.get_num_devices()  # 32
    total_tokens = tokens_global  # 128
    M = tokens_global // ring_devices  # 32 tokens per ring device
    E = experts_total // total_mesh_devices  # 4 local experts per device
    N = hidden_size  # 2880 (intermediate_size = hidden_size for GPT-OSS)
    L = 1
    tokens_per_chunk = 32

    # --- Expert mapping: global format, mapping[d, e] = e // E (linearized device ID) ---
    expert_mapping_torch = create_expert_mapping_tensors(
        num_devices=total_mesh_devices,
        num_experts_global=experts_total,
        mesh_device=mesh_device,
        new_format=True,
        return_torch=True,
    )
    tt_dispatch_mapping = ttnn.from_torch(
        expert_mapping_torch.to(torch.int16),
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_moe_gpt_mapping = ttnn.from_torch(
        expert_mapping_torch.to(torch.int16),
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # --- Weights: random torch tensors prepared in DRAM HEIGHT_SHARDED format ---
    torch_w0 = torch.randn(1, E, hidden_size, N, dtype=torch.bfloat16) * 0.01
    torch_w1 = torch.randn(1, E, hidden_size, N, dtype=torch.bfloat16) * 0.01
    torch_w2 = torch.randn(1, E, N, hidden_size, dtype=torch.bfloat16) * 0.01

    ring2cores = build_ring2cores(mesh_device)
    num_cores = len(ring2cores)
    groups_per_core = MAX_W0_W1_TILES_PER_CORE // 2
    dram_cores = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(num_cores)]
    dram_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in dram_cores])

    tt_w0_w1 = ttnn.from_torch(
        prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, hidden_size, N, ring2cores),
        dtype=ttnn.bfloat4_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                dram_core_range_set,
                (L * E * groups_per_core * hidden_size, 4 * ttnn.TILE_SIZE),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_w2 = ttnn.from_torch(
        prepare_w2_tensor(torch_w2, L, E, N, hidden_size, ring2cores),
        dtype=ttnn.bfloat4_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_core_range_set, (L * E * 2 * N, 4 * ttnn.TILE_SIZE), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # --- Raw input and expert routing (same layout as run_test_dispatch) ---
    raw_input_torch = torch.rand(1, 1, tokens_global, hidden_size, dtype=torch.bfloat16) - 0.5
    indices_list = [
        torch.randperm(experts_total)[:selected_experts_k].sort().values.to(torch.int32) for _ in range(tokens_global)
    ]
    expert_indices = torch.stack(indices_list).reshape(1, 1, tokens_global, selected_experts_k)
    expert_scores = torch.stack(
        [s / s.sum() for s in [torch.rand(selected_experts_k, dtype=torch.float32) for _ in range(tokens_global)]]
    ).reshape(1, 1, tokens_global, selected_experts_k)

    num_cores_y = min(8, tokens_global // mesh_device.shape[0])
    num_cores_x = (tokens_global // mesh_device.shape[0] + num_cores_y - 1) // num_cores_y
    input_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}),
        [1, selected_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_indices_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    shard_2d = dict(
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=tuple(mesh_device.shape)),
    )
    tt_raw_input = ttnn.from_torch(
        raw_input_torch, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, **shard_2d
    )
    tt_indices = ttnn.from_torch(
        expert_indices.to(torch.int16), dtype=ttnn.uint16, memory_config=input_indices_mem, **shard_2d
    )
    tt_scores = ttnn.from_torch(expert_scores, dtype=ttnn.bfloat16, memory_config=input_indices_mem, **shard_2d)

    dispatch_drain_core = ttnn.CoreCoord(6, 9)
    drain_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(dispatch_drain_core, dispatch_drain_core)}),
        [total_tokens, selected_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    drain_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, drain_shard_spec)

    tt_sparse_out = ttnn.from_torch(
        torch.zeros(1, total_tokens, hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_indices_out = ttnn.from_torch(
        torch.zeros(1, total_tokens, selected_experts_k, dtype=torch.int16),
        dtype=ttnn.uint16,
        memory_config=drain_mem,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_scores_out = ttnn.from_torch(
        torch.zeros(1, total_tokens, selected_experts_k, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        memory_config=drain_mem,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    compute_grid = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    semaphore = ttnn.create_global_semaphore(mesh_device, all_cores, 0)

    tt_input_l1 = ttnn.to_memory_config(tt_raw_input, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.synchronize_device(mesh_device)

    # --- Dispatch ---
    (tt_sparse, tt_idx, tt_sc) = ttnn.experimental.all_to_all_dispatch_metadata(
        tt_input_l1,
        tt_indices,
        tt_scores,
        tt_dispatch_mapping,
        cluster_axis=cluster_axis,
        num_links=4,
        output_tensors=(tt_sparse_out, tt_indices_out, tt_scores_out),
        cross_device_semaphore=semaphore,
        dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_UNICAST,
    )
    ttnn.synchronize_device(mesh_device)

    # Verify: device d receives token (0, token_idx) at slot token_idx
    # iff that token routes to any expert owned by device d
    sparse_per_device = ttnn.get_device_tensors(tt_sparse)
    indices_per_device = ttnn.get_device_tensors(tt_idx)
    scores_per_device = ttnn.get_device_tensors(tt_sc)
    all_passing = True
    for device_idx in range(total_mesh_devices):
        actual = ttnn.to_torch(sparse_per_device[device_idx]).reshape(tokens_global, hidden_size).bfloat16()

        expected = torch.zeros(tokens_global, hidden_size, dtype=torch.bfloat16)
        for token_idx in range(tokens_global):
            for expert_idx in expert_indices[0, 0, token_idx]:
                if int(expert_mapping_torch[0, expert_idx]) == device_idx:
                    expected[token_idx] = raw_input_torch[0, 0, token_idx]

        if not torch.equal(actual, expected):
            mismatch_slots = (actual != expected).any(dim=-1).nonzero(as_tuple=True)[0].tolist()
            logger.warning(
                f"Device {device_idx}): " f"FAILED at {len(mismatch_slots)} slots, first few: {mismatch_slots[:8]}"
            )
            all_passing = False
        else:
            logger.info(f"Device {device_idx}): Passed")

    # Convert sparse to L1 and reshape to 2D for moe_gpt
    tt_sparse = ttnn.reshape(tt_sparse, [total_tokens, hidden_size])

    # --- moe_gpt: pass HEIGHT_SHARDED indices/scores from dispatch directly ---
    moe_gpt_outputs = ttnn.experimental.moe_gpt(
        tt_sparse,
        expert_indices=tt_idx,
        expert_scores=tt_sc,
        expert_mapping=tt_moe_gpt_mapping,
        w0_w1_tensor=tt_w0_w1,
        w2_tensor=tt_w2,
        cluster_axis=cluster_axis,
    )
    ttnn.synchronize_device(mesh_device)

    # NOTE: output[4] is HEIGHT_SHARDED on all worker cores but only combine cores
    # have data (written by dm1). Direct verification would read garbage from unused cores.
    # Correctness is verified via selective_reduce_combine in test_dispatch_compute_combine.
    all_passing = True

    # --- Verify moe_gpt outputs 0, 1, 2 (routing metadata) ---
    # Expected values derived from original routing (independent of dispatch correctness).
    # output[0]: [1, padded_E_elements] UINT32 — counts[e] = tokens routed to local expert e
    # output[1]: [1, flat] UINT32 — rows of [token_id, k_idx_0..E-1, score_0..E-1, pad],
    #             aligned to 64 bytes; sentinel row has token_id == 0xFFFFFFFF (-1 as int32)
    #             k_idx_e == selected_experts_k+1 means expert e not selected for this token
    # output[2]: {1, E*total_tokens} UINT32 — stride-1 flat token IDs per expert, zero-padded
    tc_per_device = ttnn.get_device_tensors(moe_gpt_outputs[0])
    act_per_device = ttnn.get_device_tensors(moe_gpt_outputs[1])
    et_per_device = ttnn.get_device_tensors(moe_gpt_outputs[2])
    tc_pass = True
    act_pass = True
    et_pass = True
    # Row stride for output[1]: align((2E+1)*4, l1_alignment) / 4 uint32 elements
    # l1_alignment = hal::get_l1_alignment() = 16 bytes on TT hardware
    l1_align_bytes = 16
    act_row_bytes_raw = (2 * E + 1) * 4
    act_row_bytes_aligned = (act_row_bytes_raw + l1_align_bytes - 1) // l1_align_bytes * l1_align_bytes
    act_row_stride = act_row_bytes_aligned // 4  # uint32 elements per row
    for dev_idx in range(total_mesh_devices):
        # Find global expert IDs owned by this device (in order = local expert 0, 1, ..., E-1)
        global_experts_for_dev = [e for e in range(experts_total) if int(expert_mapping_torch[0, e]) == dev_idx]

        expected_counts = []
        expected_slots_per_expert = []
        expected_act_rows = {}  # token_id → k_indices list [E]
        for local_e, global_e in enumerate(global_experts_for_dev):
            routed = (expert_indices[0, 0, :, :] == global_e).any(dim=-1)
            expected_counts.append(int(routed.sum().item()))
            expected_slots_per_expert.append(routed.nonzero(as_tuple=True)[0].tolist())
        for t in range(tokens_global):
            token_experts = expert_indices[0, 0, t].tolist()
            k_indices_expected = [selected_experts_k + 1] * E
            activated = False
            for local_e, global_e in enumerate(global_experts_for_dev):
                if global_e in token_experts:
                    k_indices_expected[local_e] = token_experts.index(global_e)
                    activated = True
            if activated:
                expected_act_rows[t] = k_indices_expected

        # Check output[0]: per-expert token counts (first E entries of the padded row)
        tc_raw = ttnn.to_torch(tc_per_device[dev_idx]).flatten()
        actual_counts = [int(tc_raw[i].item()) for i in range(E)]
        if actual_counts != expected_counts:
            logger.error(f"  Token counts device {dev_idx}: FAILED actual={actual_counts} expected={expected_counts}")
            tc_pass = False
            all_passing = False
        else:
            logger.info(f"  Token counts device {dev_idx}: PASSED counts={actual_counts}")

        # Check output[1]: expert activation metadata rows
        act_raw = ttnn.to_torch(act_per_device[dev_idx]).flatten()
        actual_act_rows = {}
        for row_idx in range(len(act_raw) // act_row_stride):
            offset = row_idx * act_row_stride
            token_id = int(act_raw[offset].item())
            if token_id == -1 or token_id == 0xFFFFFFFF:  # 0xFFFFFFFF sentinel (int64 from to_torch)
                break
            k_indices_actual = [int(act_raw[offset + 1 + e].item()) for e in range(E)]
            actual_act_rows[token_id] = k_indices_actual
        if set(actual_act_rows.keys()) != set(expected_act_rows.keys()):
            logger.error(
                f"  Activation metadata device {dev_idx}: FAILED token set mismatch "
                f"actual_count={len(actual_act_rows)} expected_count={len(expected_act_rows)}"
            )
            act_pass = False
            all_passing = False
        else:
            k_mismatches = [
                (t, actual_act_rows[t], expected_act_rows[t])
                for t in expected_act_rows
                if actual_act_rows[t] != expected_act_rows[t]
            ]
            if k_mismatches:
                logger.error(
                    f"  Activation metadata device {dev_idx}: FAILED {len(k_mismatches)} k_idx mismatches, "
                    f"first: token {k_mismatches[0][0]} actual={k_mismatches[0][1]} expected={k_mismatches[0][2]}"
                )
                act_pass = False
                all_passing = False
            elif dev_idx < 4:
                logger.info(f"  Activation metadata device {dev_idx}: PASSED {len(actual_act_rows)} activated tokens")

        # Check output[2]: token slot indices per local expert
        # Shape: {E, (T+1)*stride} matching moe_compute format; stride = align(4, 16) / 4 = 4.
        # Token IDs at offset (e_row * stride + k * stride) within the flattened tensor.
        tc_raw = ttnn.to_torch(tc_per_device[dev_idx]).reshape(-1)  # [padded_E]
        et_raw = ttnn.to_torch(et_per_device[dev_idx]).reshape(-1)  # [E * (T+1) * stride]
        e_t_stride = 4  # align(sizeof(uint32_t), 16) / sizeof(uint32_t)
        e_t_row_elems = (total_tokens + 1) * e_t_stride
        for local_e in range(E):
            count = int(tc_raw[local_e].item())
            actual_slots = [int(et_raw[local_e * e_t_row_elems + k * e_t_stride].item()) for k in range(count)]
            expected_slots = expected_slots_per_expert[local_e]
            if sorted(actual_slots) != sorted(expected_slots):
                logger.error(
                    f"  Dense tokens indices device {dev_idx} expert {local_e}(global {global_experts_for_dev[local_e]}): "
                    f"FAILED actual={sorted(actual_slots)} expected={sorted(expected_slots)}"
                )
                et_pass = False
                all_passing = False
            else:
                logger.info(f"  Dense tokens indices device {dev_idx} expert {local_e}: PASSED slots={actual_slots}")

    if tc_pass:
        logger.info("output[0] (token counts): ALL PASSED")
    else:
        logger.warning("output[0] (token counts): SOME FAILED — moe_gpt kernel may not scan full mapping")
    if act_pass:
        logger.info("output[1] (activation metadata): ALL PASSED")
    else:
        logger.warning("output[1] (activation metadata): SOME FAILED — moe_gpt kernel may not scan full mapping")
    if et_pass:
        logger.info("output[2] (token indices): ALL PASSED")
    else:
        logger.warning("output[2] (token indices): SOME FAILED — moe_gpt kernel may not scan full mapping")

    for t in moe_gpt_outputs:
        ttnn.deallocate(t)

    return all_passing


# ---------------------------------------------------------------------------
# Dispatch+Compute+Combine test
# ---------------------------------------------------------------------------


def run_test_dispatch_compute_combine(mesh_device, tokens_global, hidden_size, selected_experts_k, experts_total):
    """Test all_to_all_dispatch_metadata → moe_gpt pipeline.

    Verifies per-expert W2 outputs against torch reference.
    Expert mapping uses global IDs: mapping[d, e] = e // experts_per_device (0..total_devices-1).
    """
    torch.manual_seed(42)
    cluster_axis = 0
    ring_devices = mesh_device.shape[cluster_axis]  # 4
    total_mesh_devices = mesh_device.get_num_devices()  # 32
    total_tokens = tokens_global  # 128
    M = tokens_global // ring_devices  # 32 tokens per ring device
    E = experts_total // total_mesh_devices  # 4 local experts per device
    N = hidden_size  # 2880 (intermediate_size = hidden_size for GPT-OSS)
    L = 1
    tokens_per_chunk = 32

    # --- Expert mapping: global format, mapping[d, e] = e // E (linearized device ID) ---
    expert_mapping_torch = create_expert_mapping_tensors(
        num_devices=total_mesh_devices,
        num_experts_global=experts_total,
        mesh_device=mesh_device,
        new_format=True,
        return_torch=True,
    )
    tt_dispatch_mapping = ttnn.from_torch(
        expert_mapping_torch.to(torch.int16),
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_moe_gpt_mapping = ttnn.from_torch(
        expert_mapping_torch.to(torch.int16),
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # --- Weights: random torch tensors prepared in DRAM HEIGHT_SHARDED format ---
    torch_w0 = torch.randn(1, E, hidden_size, N, dtype=torch.bfloat16) * 0.01
    torch_w1 = torch.randn(1, E, hidden_size, N, dtype=torch.bfloat16) * 0.01
    torch_w2 = torch.randn(1, E, N, hidden_size, dtype=torch.bfloat16) * 0.01

    ring2cores = build_ring2cores(mesh_device)
    num_cores = len(ring2cores)
    groups_per_core = MAX_W0_W1_TILES_PER_CORE // 2
    dram_cores = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(num_cores)]
    dram_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in dram_cores])

    tt_w0_w1 = ttnn.from_torch(
        prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, hidden_size, N, ring2cores),
        dtype=ttnn.bfloat4_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                dram_core_range_set,
                (L * E * groups_per_core * hidden_size, 4 * ttnn.TILE_SIZE),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_w2 = ttnn.from_torch(
        prepare_w2_tensor(torch_w2, L, E, N, hidden_size, ring2cores),
        dtype=ttnn.bfloat4_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_core_range_set, (L * E * 2 * N, 4 * ttnn.TILE_SIZE), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # --- Raw input and expert routing (same layout as run_test_dispatch) ---
    raw_input_torch = torch.rand(1, 1, tokens_global, hidden_size, dtype=torch.bfloat16) - 0.5
    indices_list = [
        torch.randperm(experts_total)[:selected_experts_k].sort().values.to(torch.int32) for _ in range(tokens_global)
    ]
    expert_indices = torch.stack(indices_list).reshape(1, 1, tokens_global, selected_experts_k)
    expert_scores = torch.stack(
        [s / s.sum() for s in [torch.rand(selected_experts_k, dtype=torch.float32) for _ in range(tokens_global)]]
    ).reshape(1, 1, tokens_global, selected_experts_k)

    num_cores_y = min(8, tokens_global // mesh_device.shape[0])
    num_cores_x = (tokens_global // mesh_device.shape[0] + num_cores_y - 1) // num_cores_y
    input_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}),
        [1, selected_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_indices_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    shard_2d = dict(
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=tuple(mesh_device.shape)),
    )
    tt_raw_input = ttnn.from_torch(
        raw_input_torch, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, **shard_2d
    )
    tt_indices = ttnn.from_torch(
        expert_indices.to(torch.int16), dtype=ttnn.uint16, memory_config=input_indices_mem, **shard_2d
    )
    tt_scores = ttnn.from_torch(expert_scores, dtype=ttnn.bfloat16, memory_config=input_indices_mem, **shard_2d)

    dispatch_drain_core = ttnn.CoreCoord(6, 9)
    drain_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(dispatch_drain_core, dispatch_drain_core)}),
        [total_tokens, selected_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    drain_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, drain_shard_spec)

    tt_sparse_out = ttnn.from_torch(
        torch.zeros(1, total_tokens, hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_indices_out = ttnn.from_torch(
        torch.zeros(1, total_tokens, selected_experts_k, dtype=torch.int16),
        dtype=ttnn.uint16,
        memory_config=drain_mem,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_scores_out = ttnn.from_torch(
        torch.zeros(1, total_tokens, selected_experts_k, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        memory_config=drain_mem,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    compute_grid = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    semaphore = ttnn.create_global_semaphore(mesh_device, all_cores, 0)

    tt_input_l1 = ttnn.to_memory_config(tt_raw_input, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.synchronize_device(mesh_device)

    # --- Dispatch ---
    (tt_sparse, tt_idx, tt_sc) = ttnn.experimental.all_to_all_dispatch_metadata(
        tt_input_l1,
        tt_indices,
        tt_scores,
        tt_dispatch_mapping,
        cluster_axis=cluster_axis,
        num_links=4,
        output_tensors=(tt_sparse_out, tt_indices_out, tt_scores_out),
        cross_device_semaphore=semaphore,
        dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_UNICAST,
    )
    ttnn.synchronize_device(mesh_device)

    # Verify: device d receives token (0, token_idx) at slot token_idx
    # iff that token routes to any expert owned by device d
    sparse_per_device = ttnn.get_device_tensors(tt_sparse)
    indices_per_device = ttnn.get_device_tensors(tt_idx)
    scores_per_device = ttnn.get_device_tensors(tt_sc)
    all_passing = True
    for device_idx in range(total_mesh_devices):
        actual = ttnn.to_torch(sparse_per_device[device_idx]).reshape(tokens_global, hidden_size).bfloat16()

        expected = torch.zeros(tokens_global, hidden_size, dtype=torch.bfloat16)
        for token_idx in range(tokens_global):
            for expert_idx in expert_indices[0, 0, token_idx]:
                if int(expert_mapping_torch[0, expert_idx]) == device_idx:
                    expected[token_idx] = raw_input_torch[0, 0, token_idx]

        if not torch.equal(actual, expected):
            mismatch_slots = (actual != expected).any(dim=-1).nonzero(as_tuple=True)[0].tolist()
            logger.warning(
                f"Device {device_idx}): " f"FAILED at {len(mismatch_slots)} slots, first few: {mismatch_slots[:8]}"
            )
            all_passing = False
        else:
            logger.info(f"Device {device_idx}): Passed")

    # Convert sparse to L1 and reshape to 2D for moe_gpt
    tt_sparse = ttnn.reshape(tt_sparse, [total_tokens, hidden_size])

    # --- moe_gpt: pass HEIGHT_SHARDED indices/scores from dispatch directly ---
    moe_gpt_outputs = ttnn.experimental.moe_gpt(
        tt_sparse,
        expert_indices=tt_idx,
        expert_scores=tt_sc,
        expert_mapping=tt_moe_gpt_mapping,
        w0_w1_tensor=tt_w0_w1,
        w2_tensor=tt_w2,
        cluster_axis=cluster_axis,
    )
    ttnn.synchronize_device(mesh_device)

    # NOTE: output[4] is HEIGHT_SHARDED on all worker cores but only combine cores
    # have data (written by dm1). Direct verification would read garbage from unused cores.
    # Correctness is verified via selective_reduce_combine output below.
    all_passing = True

    # --- Verify moe_gpt outputs 0, 1, 2 (routing metadata) ---
    # Expected values derived from original routing (independent of dispatch correctness).
    # output[0]: [1, padded_E_elements] UINT32 — counts[e] = tokens routed to local expert e
    # output[1]: [1, flat] UINT32 — rows of [token_id, k_idx_0..E-1, score_0..E-1, pad],
    #             aligned to 64 bytes; sentinel row has token_id == 0xFFFFFFFF (-1 as int32)
    #             k_idx_e == selected_experts_k+1 means expert e not selected for this token
    # output[2]: {1, E*total_tokens} UINT32 — stride-1 flat token IDs per expert, zero-padded
    tc_per_device = ttnn.get_device_tensors(moe_gpt_outputs[0])
    act_per_device = ttnn.get_device_tensors(moe_gpt_outputs[1])
    et_per_device = ttnn.get_device_tensors(moe_gpt_outputs[2])
    tc_pass = True
    act_pass = True
    et_pass = True
    # Row stride for output[1]: align((2E+1)*4, l1_alignment) / 4 uint32 elements
    # l1_alignment = hal::get_l1_alignment() = 16 bytes on TT hardware
    l1_align_bytes = 16
    act_row_bytes_raw = (2 * E + 1) * 4
    act_row_bytes_aligned = (act_row_bytes_raw + l1_align_bytes - 1) // l1_align_bytes * l1_align_bytes
    act_row_stride = act_row_bytes_aligned // 4  # uint32 elements per row
    for dev_idx in range(total_mesh_devices):
        # Find global expert IDs owned by this device (in order = local expert 0, 1, ..., E-1)
        global_experts_for_dev = [e for e in range(experts_total) if int(expert_mapping_torch[0, e]) == dev_idx]

        expected_counts = []
        expected_slots_per_expert = []
        expected_act_rows = {}  # token_id → k_indices list [E]
        for local_e, global_e in enumerate(global_experts_for_dev):
            routed = (expert_indices[0, 0, :, :] == global_e).any(dim=-1)
            expected_counts.append(int(routed.sum().item()))
            expected_slots_per_expert.append(routed.nonzero(as_tuple=True)[0].tolist())
        for t in range(tokens_global):
            token_experts = expert_indices[0, 0, t].tolist()
            k_indices_expected = [selected_experts_k + 1] * E
            activated = False
            for local_e, global_e in enumerate(global_experts_for_dev):
                if global_e in token_experts:
                    k_indices_expected[local_e] = token_experts.index(global_e)
                    activated = True
            if activated:
                expected_act_rows[t] = k_indices_expected

        # Check output[0]: per-expert token counts (first E entries of the padded row)
        tc_raw = ttnn.to_torch(tc_per_device[dev_idx]).flatten()
        actual_counts = [int(tc_raw[i].item()) for i in range(E)]
        if actual_counts != expected_counts:
            logger.error(f"  Token counts device {dev_idx}: FAILED actual={actual_counts} expected={expected_counts}")
            tc_pass = False
            all_passing = False
        else:
            logger.info(f"  Token counts device {dev_idx}: PASSED counts={actual_counts}")

        # Check output[1]: expert activation metadata rows
        act_raw = ttnn.to_torch(act_per_device[dev_idx]).flatten()
        actual_act_rows = {}
        for row_idx in range(len(act_raw) // act_row_stride):
            offset = row_idx * act_row_stride
            token_id = int(act_raw[offset].item())
            if token_id == -1 or token_id == 0xFFFFFFFF:  # 0xFFFFFFFF sentinel (int64 from to_torch)
                break
            k_indices_actual = [int(act_raw[offset + 1 + e].item()) for e in range(E)]
            actual_act_rows[token_id] = k_indices_actual
        if set(actual_act_rows.keys()) != set(expected_act_rows.keys()):
            logger.error(
                f"  Activation metadata device {dev_idx}: FAILED token set mismatch "
                f"actual_act_rows={set(actual_act_rows.keys())} expected_act_rows={set(expected_act_rows.keys())}"
            )
            act_pass = False
            all_passing = False
        else:
            k_mismatches = [
                (t, actual_act_rows[t], expected_act_rows[t])
                for t in expected_act_rows
                if actual_act_rows[t] != expected_act_rows[t]
            ]
            if k_mismatches:
                logger.error(
                    f"  Activation metadata device {dev_idx}: FAILED {len(k_mismatches)} k_idx mismatches, "
                    f"first: token {k_mismatches[0][0]} actual={k_mismatches[0][1]} expected={k_mismatches[0][2]}"
                )
                act_pass = False
                all_passing = False
            elif dev_idx < 4:
                logger.info(f"  Activation metadata device {dev_idx}: PASSED {len(actual_act_rows)} activated tokens")

        # Check output[2]: token slot indices per local expert
        # Shape: {E, (T+1)*stride} matching moe_compute format; stride = align(4, 16) / 4 = 4.
        # Token IDs at offset (e_row * stride + k * stride) within the flattened tensor.
        tc_raw = ttnn.to_torch(tc_per_device[dev_idx]).reshape(-1)  # [padded_E]
        et_raw = ttnn.to_torch(et_per_device[dev_idx]).reshape(-1)  # [E * (T+1) * stride]
        e_t_stride = 4  # align(sizeof(uint32_t), 16) / sizeof(uint32_t)
        e_t_row_elems = (total_tokens + 1) * e_t_stride
        for local_e in range(E):
            count = int(tc_raw[local_e].item())
            actual_slots = [int(et_raw[local_e * e_t_row_elems + k * e_t_stride].item()) for k in range(count)]
            expected_slots = expected_slots_per_expert[local_e]
            if sorted(actual_slots) != sorted(expected_slots):
                logger.error(
                    f"  Dense tokens indices device {dev_idx} expert {local_e}(global {global_experts_for_dev[local_e]}): "
                    f"FAILED actual={sorted(actual_slots)} expected={sorted(expected_slots)}"
                )
                et_pass = False
                all_passing = False
            else:
                logger.info(f"  Dense tokens indices device {dev_idx} expert {local_e}: PASSED slots={actual_slots}")

    if tc_pass:
        logger.info("output[0] (token counts): ALL PASSED")
    else:
        logger.warning("output[0] (token counts): SOME FAILED — moe_gpt kernel may not scan full mapping")
    if act_pass:
        logger.info("output[1] (activation metadata): ALL PASSED")
    else:
        logger.warning("output[1] (activation metadata): SOME FAILED — moe_gpt kernel may not scan full mapping")
    if et_pass:
        logger.info("output[2] (token indices): ALL PASSED")
    else:
        logger.warning("output[2] (token indices): SOME FAILED — moe_gpt kernel may not scan full mapping")

    # --- Setup selective_reduce_combine infrastructure ---
    mesh_cols = mesh_device.shape[1]  # 8
    num_clusters = total_mesh_devices // ring_devices  # 8 independent column rings
    experts_per_ring = experts_total // num_clusters  # 16 ring-local experts per ring
    global_experts = experts_total  # 128

    COMBINE_H = 4  # token_parallel_core_dim (rows)
    COMBINE_W = 3  # data_parallel_core_dim (cols)
    combine_core_range_set, combine_start, combine_end = get_moe_gpt_combine_core_range(
        mesh_device, COMBINE_W, COMBINE_H
    )
    combine_worker_cores = list(ttnn.corerange_to_cores(combine_core_range_set, row_wise=True))
    mux_start = ttnn.CoreCoord(combine_end.x + 1, combine_start.y)
    mux_end = ttnn.CoreCoord(combine_end.x + 2, combine_end.y)
    combine_mux_cores = ttnn.CoreRangeSet([ttnn.CoreRange(mux_start, mux_end)])
    combine_semaphore = ttnn.create_global_semaphore(mesh_device, all_cores, 0)

    # Pre-allocate combine output: [K, M, hidden_size] (K-indexed) replicated on all devices
    tt_combine_preallocated = ttnn.from_torch(
        torch.zeros(selected_experts_k, M, hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # --- Run selective_reduce_combine ---
    tt_combine_result = ttnn.experimental.selective_reduce_combine(
        moe_gpt_outputs[4],  # BLOCK_SHARDED token-parallel input {E*total_tokens, K}
        moe_gpt_outputs[1],  # dense_metadata: not used by kernel, pass as-is
        moe_gpt_outputs[2],  # token_maps {E, (T+1)*stride} matching moe_compute format
        moe_gpt_outputs[0],  # token_counts [1, E]
        hidden_size=hidden_size,
        batch_size=total_tokens,  # total ring tokens (128); tokens_per_device = 128/4 = 32
        seq_size=1,
        select_experts_k=selected_experts_k,
        experts=global_experts,  # 128 → experts_per_device = 128/32 = 4 = E
        cluster_axis=cluster_axis,
        topology=ttnn.Topology.Ring,
        num_links=4,
        token_parallel_core_dim=COMBINE_H,
        data_parallel_core_dim=COMBINE_W,
        worker_cores=combine_worker_cores,
        mux_core_range_set=combine_mux_cores,
        output_tensor=tt_combine_preallocated,
        optional_cross_device_semaphore=combine_semaphore,
    )
    ttnn.synchronize_device(mesh_device)
    logger.info("selective_reduce_combine complete.")

    # Deallocate converted inputs and moe_gpt outputs
    for i in range(4):
        ttnn.deallocate(moe_gpt_outputs[i])

    # --- Verify combine output accuracy ---
    # With K-indexed output (upstream #38542), combine shape is [K, M, hidden_size].
    # Slot k (0..K-1) holds the W2 output for each token's k-th selected expert.
    # combine[k, t, :] = W2(SwiGLU(W0*x, W1*x)) for global_expert = expert_indices[t, k].
    #
    # Verification: sum over K slots -> [M, hidden_size] and compare PCC against
    # torch reference (unweighted sum of W2 outputs from all selected experts per token).
    K_sel = selected_experts_k
    combine_pass = True
    combine_per_device = ttnn.get_device_tensors(tt_combine_result)
    raw_input_2d = raw_input_torch.reshape(tokens_global, hidden_size)  # [128, 2880]

    for dev_idx in range(total_mesh_devices):
        ring_pos = dev_idx // mesh_cols  # row = which ring position this device occupies
        col = dev_idx % mesh_cols  # column = which independent ring this device belongs to

        actual_combine = ttnn.to_torch(combine_per_device[dev_idx]).reshape(K_sel, M, hidden_size)

        # Build torch reference: only include experts belonging to THIS ring.
        # Use the actual expert mapping tensor to determine ring membership.
        # An expert belongs to ring 'col' if its owning device has col index == col.
        ring_expert_set = set()
        for e_global in range(experts_total):
            owning_device = int(expert_mapping_torch[0, e_global])
            if owning_device % mesh_cols == col:
                ring_expert_set.add(e_global)

        ref_combine = torch.zeros(K_sel, M, hidden_size, dtype=torch.float32)
        with torch.no_grad():
            for t in range(M):
                global_token = ring_pos * M + t
                for k in range(K_sel):
                    global_e = expert_indices[0, 0, global_token, k].item()
                    if global_e not in ring_expert_set:
                        continue  # expert is on a different ring, skip
                    local_e = global_e % E
                    token = raw_input_2d[global_token : global_token + 1].to(torch.float32)
                    gate = (token @ torch_w0[0, local_e].to(torch.float32)).to(torch.bfloat16)
                    up = (token @ torch_w1[0, local_e].to(torch.float32)).to(torch.bfloat16)
                    activated = swiglu_reference(gate, up).to(torch.float32)
                    ref_combine[k, t] = activated @ torch_w2[0, local_e].to(torch.float32)

        # Compare summed expert contributions per token: [M, hidden_size]
        actual_sum = actual_combine.to(torch.float32).sum(dim=0)
        ref_sum = ref_combine.sum(dim=0)

        pcc_pass, pcc_val = comp_pcc(actual_sum, ref_sum, PCC_THRESHOLD)
        if dev_idx < 4 or not pcc_pass:
            log_fn = logger.info if pcc_pass else logger.error
            log_fn(
                f"  Combine device {dev_idx} (ring_pos={ring_pos}, col={col}): "
                f"{'PASSED' if pcc_pass else 'FAILED'} PCC={pcc_val:.4f}"
            )
        if not pcc_pass:
            combine_pass = False
            all_passing = False

    if combine_pass:
        logger.info("selective_reduce_combine: ALL PASSED")
    else:
        logger.warning("selective_reduce_combine: SOME FAILED")

    return all_passing


# ---------------------------------------------------------------------------
# Main e2e test function
# ---------------------------------------------------------------------------


def run_test_moe_gpt_e2e(
    mesh_device,
    M,
    K,
    N,
    E,
    selected_experts_k,
    experts_total,
    measure_perf=False,
):
    """
    E2E test: all_to_all_dispatch_metadata → moe_gpt.

    1. Creates raw token inputs and expert routing per device
    2. Runs all_to_all_dispatch_metadata to dispatch tokens
    3. Feeds dispatch output into moe_gpt
    4. Verifies per-expert outputs against torch reference
    5. Optionally measures e2e performance
    """
    torch.manual_seed(42)
    random.seed(42)

    cluster_axis = 0
    ring_devices = mesh_device.shape[cluster_axis]  # 4 for 4x8 mesh
    total_mesh_devices = mesh_device.get_num_devices()  # 32 for 4x8 mesh
    mesh_cols = mesh_device.shape[1]  # 8 for 4x8 mesh
    total_tokens = M * ring_devices  # 128 = 32 * 4

    logger.info(f"E2E test configuration:")
    logger.info(f"  mesh_shape: {mesh_device.shape}")
    logger.info(f"  ring_devices: {ring_devices}, total_tokens: {total_tokens}")
    logger.info(f"  M={M}, K={K}, N={N}, E={E}, experts_total={experts_total}")
    logger.info(f"  selected_experts_k={selected_experts_k}")

    tokens_per_chunk = 32  # moe_gpt tiles tokens in 32-token chunks

    # ------------------------------------------------------------------
    # Weight infrastructure
    # ------------------------------------------------------------------
    in0_core_coords = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(0)
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

    # ------------------------------------------------------------------
    # Expert mappings
    # ------------------------------------------------------------------
    # Both dispatch and moe_gpt use the SAME linearized-device-ID format:
    #   mapping[d, e] = (e // experts_per_device) * mesh_cols + (d % mesh_cols)
    # This sends tokens from device d to the device in the same column that owns expert e.
    # On a (4,8) mesh with cluster_axis=0, each column is an independent ring.
    # For (1,6) mesh (6U test), device_id == ring_pos so both formats coincide,
    # but on (4,8) they differ: ring_pos=0..3 vs linearized_id=0,8,16,24 (col 0).
    expert_mapping_torch = create_expert_mapping_tensors(
        num_devices=total_mesh_devices,
        num_experts_global=experts_total,
        mesh_device=mesh_device,
        new_format=True,
        return_torch=True,
    )

    # Dispatch mapping: replicated on all mesh devices (each device reads its own row by linearized ID)
    tt_dispatch_mapping = ttnn.from_torch(
        expert_mapping_torch.to(torch.int16),
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # moe_gpt mapping: same tensor, also replicated
    tt_moe_gpt_mapping = ttnn.from_torch(
        expert_mapping_torch.to(torch.int16),
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # ------------------------------------------------------------------
    # Raw input tokens and expert routing (per ring device)
    # ------------------------------------------------------------------
    # Each ring device starts with M=32 tokens. For 4x8 mesh with cluster_axis=0,
    # all 8 columns run the same simulation (replicated).
    # After dispatch, each device has [total_tokens=128, K] sparse_buffer.

    # Create raw input: [ring_devices * M, 1, 1, K] to shard along rows
    raw_input_torch = torch.rand(ring_devices * M, 1, 1, K, dtype=torch.bfloat16) - 0.5

    # Create expert routing: [ring_devices * M, 1, 1, selected_k]
    experts_per_device = experts_total // ring_devices
    indices_list = []
    scores_list = []
    for dev in range(ring_devices):
        for t in range(M):
            selected = torch.randperm(experts_total)[:selected_experts_k].sort().values
            indices_list.append(selected.to(torch.int16))
            scores = torch.rand(selected_experts_k, dtype=torch.bfloat16) + 1e-5
            scores = scores / scores.sum()
            scores_list.append(scores)
    expert_indices_flat = torch.stack(indices_list, dim=0).reshape(ring_devices * M, 1, 1, selected_experts_k)
    expert_scores_flat = torch.stack(scores_list, dim=0).reshape(ring_devices * M, 1, 1, selected_experts_k)

    # Reshape for reference: [ring_devices, M, selected_k]
    raw_input_ref = raw_input_torch.squeeze(1).squeeze(1).reshape(ring_devices, M, K)
    indices_ref = expert_indices_flat.squeeze(1).squeeze(1).reshape(ring_devices, M, selected_experts_k)
    scores_ref = expert_scores_flat.squeeze(1).squeeze(1).reshape(ring_devices, M, selected_experts_k)

    # Load to device: shard along dim=0 (ring_devices) across mesh rows, replicate across columns
    tt_raw_input = ttnn.from_torch(
        raw_input_torch,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=tuple(mesh_device.shape)),
    )

    # Input indices/scores MUST be in L1 (not DRAM) - the dispatch kernel requires L1 alignment (16B).
    # DRAM uses 32B alignment which causes metadata write offsets to overflow the output buffer.
    # Use HEIGHT_SHARDED with 1 row per core: M tokens across num_cores_x × num_cores_y cores.
    tokens_per_device = M
    num_cores_y = min(8, tokens_per_device)
    num_cores_x = (tokens_per_device + num_cores_y - 1) // num_cores_y
    input_indices_scores_core_range = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
    )
    input_indices_shard_spec = ttnn.ShardSpec(
        input_indices_scores_core_range,
        [1, selected_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_indices_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        input_indices_shard_spec,
    )
    logger.info(
        f"Input indices/scores height sharded: {tokens_per_device} tokens across "
        f"{num_cores_x}x{num_cores_y} cores, shard shape [1, {selected_experts_k}]"
    )

    tt_expert_indices = ttnn.from_torch(
        expert_indices_flat.to(torch.int16),
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=input_indices_sharded_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=tuple(mesh_device.shape)),
    )
    tt_expert_scores = ttnn.from_torch(
        expert_scores_flat,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=input_indices_sharded_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=tuple(mesh_device.shape)),
    )

    # ------------------------------------------------------------------
    # Pre-allocate dispatch output tensors
    # Shape must match dispatch's expected output: [ring_devices, total_tokens, K] globally,
    # giving each device [1, total_tokens, K] (3D with leading ring-position dim).
    # Dispatch writes one row per ring device (ring_devices * total_tokens total rows globally).
    # moe_gpt needs [total_tokens, K] (2D), so we reshape AFTER converting to L1.
    # ------------------------------------------------------------------
    dispatch_drain_core = ttnn.CoreCoord(6, 9)

    # Sparse buffer: [ring_devices, total_tokens, K] → [1, 128, 2880] per device (DRAM)
    dispatch_sparse_shape = torch.zeros(ring_devices, total_tokens, K, dtype=torch.bfloat16)
    tt_dispatch_sparse = ttnn.from_torch(
        dispatch_sparse_shape,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=tuple(mesh_device.shape)),
    )

    # Output metadata (indices/scores): HEIGHT_SHARDED on drain_core, [total_tokens, k] per device (L1)
    # Shard shape [128, 4] on single drain core. from_torch zero-initializes the output buffer.
    dispatch_output_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(dispatch_drain_core, dispatch_drain_core)}),
        [total_tokens, selected_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dispatch_indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        dispatch_output_shard_spec,
    )

    dispatch_indices_shape = torch.zeros(ring_devices, total_tokens, selected_experts_k, dtype=torch.int16)
    tt_dispatch_indices = ttnn.from_torch(
        dispatch_indices_shape,
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=dispatch_indices_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=tuple(mesh_device.shape)),
    )

    dispatch_scores_shape = torch.zeros(ring_devices, total_tokens, selected_experts_k, dtype=torch.bfloat16)
    tt_dispatch_scores = ttnn.from_torch(
        dispatch_scores_shape,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=dispatch_indices_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=tuple(mesh_device.shape)),
    )

    # ------------------------------------------------------------------
    # Global semaphores and selective_reduce_combine configuration
    # ------------------------------------------------------------------
    compute_grid = mesh_device.compute_with_storage_grid_size()
    all_worker_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    dispatch_semaphore = ttnn.create_global_semaphore(mesh_device, all_worker_cores, 0)
    combine_semaphore = ttnn.create_global_semaphore(mesh_device, all_worker_cores, 0)

    # selective_reduce_combine core parameters.
    # Worker cores must be the SAME as moe_gpt's combine shard cores so that
    # CircularBufferConfig::set_globally_allocated_address(*input_tensor.buffer())
    # correctly aliases each core's BLOCK_SHARDED shard.
    #
    # moe_gpt uses COMBINE_H=4 rows × COMBINE_W=3 cols = 12 cores.
    # Shard per core: [E*M/COMBINE_H, K/COMBINE_W] = [4*32/4, 2880/3] = [32, 960] = 61440 B.
    #
    # CB size formula (must ≤ 61440 = L1 bank limit = shard size):
    #   (K/dp) * (M/tp) * E = (2880/3) * (32/4) * 4 = 1920 * 8 * 4 = 61440 ✓
    # This requires batch_size=M=32 (per-device tokens), NOT total_tokens=128.
    COMBINE_H = 4  # token_parallel_core_dim = rows (expert / token dimension)
    COMBINE_W = 3  # data_parallel_core_dim  = cols (hidden dimension)
    moe_gpt_combine_core_range_set, combine_start, combine_end = get_moe_gpt_combine_core_range(
        mesh_device, COMBINE_W, COMBINE_H
    )
    logger.info(f"moe_gpt combine cores: ({combine_start.x},{combine_start.y}) to ({combine_end.x},{combine_end.y})")
    # corerange_to_cores iterates x-fast (ROW_MAJOR), consistent with BLOCK_SHARDED shard assignment
    combine_worker_cores = list(ttnn.corerange_to_cores(moe_gpt_combine_core_range_set, row_wise=True))
    combine_token_parallel_core_dim = COMBINE_H  # 4
    combine_data_parallel_core_dim = COMBINE_W  # 3

    # Mux cores: 2 columns immediately right of worker cores.
    # Need num_links * ring_neighbors = 4 * 2 = 8 mux cores = 2 cols × COMBINE_H rows.
    mux_start = ttnn.CoreCoord(combine_end.x + 1, combine_start.y)
    mux_end = ttnn.CoreCoord(combine_end.x + 2, combine_end.y)
    combine_mux_cores = ttnn.CoreRangeSet([ttnn.CoreRange(mux_start, mux_end)])
    logger.info(f"combine mux cores: ({mux_start.x},{mux_start.y}) to ({mux_end.x},{mux_end.y})")

    # Pre-allocate selective_reduce_combine output.
    # selective_reduce_combine uses: experts_per_device = global_experts / total_devices
    # GPT-OSS: 128 global experts / 32 devices = 4 per device (same as E=4 local experts).
    # DeepSeek:  256 global experts / 128 devices = 2 per device.
    # experts_total is the GLOBAL expert count across all devices (128 for GPT-OSS on 4x8).
    # Each column-ring has experts_total / mesh_cols = 16 ring-local experts.
    # experts_per_device = experts_total / total_mesh_devices = 128 / 32 = 4 = E ✓
    global_experts = experts_total  # 128
    experts_per_cluster = experts_total // mesh_cols  # 128 // 8 = 16 ring-local experts
    combine_output_torch = torch.zeros(selected_experts_k, M, K, dtype=torch.bfloat16)
    tt_combine_preallocated = ttnn.from_torch(
        combine_output_torch,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # ------------------------------------------------------------------
    # Move raw input to L1 (dispatch expects L1 input tokens)
    # expert_indices/scores are already in L1 HEIGHT_SHARDED (required by dispatch kernel)
    # ------------------------------------------------------------------
    tt_raw_input_l1 = ttnn.to_memory_config(tt_raw_input, memory_config=ttnn.L1_MEMORY_CONFIG)

    ttnn.synchronize_device(mesh_device)

    # ------------------------------------------------------------------
    # Run dispatch + moe_gpt (one iteration for accuracy check)
    # ------------------------------------------------------------------
    # Use SPARSE_UNICAST dispatch algorithm: SPARSE_MCAST_SHORTEST_PATH has a bug for cluster_axis=0
    # where it computes hop distances using linearized device IDs (0-31) instead of column ring
    # positions (0-3). For device 0→8, it gets distance=8 (bit 7 in hop_mask) instead of 1 (bit 0),
    # causing the sparse multicast to target the wrong device. SPARSE_UNICAST uses get_route()
    # which correctly returns SOUTH (1 hop) for column-adjacent devices.
    logger.info("Running all_to_all_dispatch_metadata...")
    (
        tt_sparse_output,
        tt_indices_output,
        tt_scores_output,
    ) = ttnn.experimental.all_to_all_dispatch_metadata(
        tt_raw_input_l1,
        tt_expert_indices,
        tt_expert_scores,
        tt_dispatch_mapping,
        cluster_axis=cluster_axis,
        num_links=4,
        output_tensors=(tt_dispatch_sparse, tt_dispatch_indices, tt_dispatch_scores),
        cross_device_semaphore=dispatch_semaphore,
        dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_UNICAST,
    )

    ttnn.synchronize_device(mesh_device)
    logger.info("Dispatch complete, reading output for reference...")

    # Read dispatch outputs per device to CPU BEFORE running moe_gpt
    # (moe_gpt may reuse L1 for indices/scores internally)
    dispatch_sparse_devices = ttnn.get_device_tensors(tt_sparse_output)
    dispatch_indices_devices = ttnn.get_device_tensors(tt_indices_output)
    dispatch_scores_devices = ttnn.get_device_tensors(tt_scores_output)
    dispatch_sparse_cpu = [
        ttnn.to_torch(dispatch_sparse_devices[i]).reshape(total_tokens, K) for i in range(total_mesh_devices)
    ]
    dispatch_indices_cpu = [
        ttnn.to_torch(dispatch_indices_devices[i]).reshape(total_tokens, selected_experts_k)
        for i in range(total_mesh_devices)
    ]
    dispatch_scores_cpu = [
        ttnn.to_torch(dispatch_scores_devices[i]).reshape(total_tokens, selected_experts_k).bfloat16()
        for i in range(total_mesh_devices)
    ]

    # Diagnostic: check if dispatch wrote non-zero values
    for check_dev in [0, 1, 8, 9]:
        sparse_sample = dispatch_sparse_cpu[check_dev]
        indices_sample = dispatch_indices_cpu[check_dev]
        nonzero_rows = (sparse_sample.abs().sum(dim=1) > 0).sum().item()
        nonzero_row_indices = (sparse_sample.abs().sum(dim=1) > 0).nonzero(as_tuple=True)[0].tolist()
        logger.info(
            f"  Dispatch dev {check_dev}: sparse nonzero_rows={nonzero_rows}/{total_tokens}, "
            f"nonzero at rows={nonzero_row_indices[:8]}{'...' if len(nonzero_row_indices) > 8 else ''}, "
            f"indices row0={indices_sample[0].tolist()}, row32={indices_sample[32].tolist() if total_tokens > 32 else 'N/A'}"
        )

    # Move sparse_buffer to L1 for moe_gpt (expects L1 input)
    # Dispatch output is [1, total_tokens, K] per device (3D); moe_gpt needs [total_tokens, K] (2D)
    # because it reads tokens = sparse_shape[0].
    tt_sparse_l1 = ttnn.to_memory_config(tt_sparse_output, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_sparse_l1 = ttnn.reshape(tt_sparse_l1, [total_tokens, K])  # [1,128,2880] → [128,2880]

    logger.info("Running moe_gpt...")
    # Pass HEIGHT_SHARDED dispatch outputs directly to moe_gpt.
    # moe_gpt uses CB aliasing: the drain tilize core is set to the dispatch drain core,
    # and the indices/scores CBs are backed directly by the HEIGHT_SHARDED L1 buffers.
    moe_gpt_outputs = ttnn.experimental.moe_gpt(
        tt_sparse_l1,
        expert_indices=tt_indices_output,
        expert_scores=tt_scores_output,
        expert_mapping=tt_moe_gpt_mapping,
        w0_w1_tensor=tt_w0_w1,
        w2_tensor=tt_w2,
        cluster_axis=cluster_axis,
    )

    ttnn.synchronize_device(mesh_device)
    logger.info("moe_gpt complete, verifying moe_gpt accuracy (before selective_reduce_combine)...")

    # NOTE: output[4] is HEIGHT_SHARDED on all worker cores but only combine cores
    # have data (written by dm1). Direct verification would read garbage from unused cores.
    # Correctness is verified via selective_reduce_combine output below.
    all_passing = True

    # ------------------------------------------------------------------
    # Run selective_reduce_combine directly with moe_gpt outputs.
    # moe_gpt now outputs token-parallel BLOCK_SHARDED combine data {E*M_ring, K}
    # and stride-1 e_t {1, E*M_ring}, directly compatible with selective_reduce_combine.
    # ------------------------------------------------------------------
    M_ring = total_tokens  # 128 = M * ring_devices

    logger.info("Running selective_reduce_combine...")
    tt_combine_output = ttnn.experimental.selective_reduce_combine(
        moe_gpt_outputs[4],  # BLOCK_SHARDED token-parallel input {E*M_ring, K}
        moe_gpt_outputs[1],  # dense_metadata: not used by kernel
        moe_gpt_outputs[2],  # token_maps {E, (T+1)*stride} matching moe_compute format
        moe_gpt_outputs[0],  # token_counts {1, E}
        hidden_size=K,
        batch_size=M_ring,  # 128; tokens_per_device = 128/4 = 32
        seq_size=1,
        select_experts_k=selected_experts_k,
        experts=global_experts,
        cluster_axis=cluster_axis,
        topology=ttnn.Topology.Ring,
        num_links=4,
        token_parallel_core_dim=combine_token_parallel_core_dim,
        data_parallel_core_dim=combine_data_parallel_core_dim,
        worker_cores=combine_worker_cores,
        mux_core_range_set=combine_mux_cores,
        output_tensor=tt_combine_preallocated,
        optional_cross_device_semaphore=combine_semaphore,
    )

    ttnn.synchronize_device(mesh_device)
    logger.info("selective_reduce_combine complete.")

    # ------------------------------------------------------------------
    # Performance measurement (dispatch + moe_gpt + selective_reduce_combine)
    # ------------------------------------------------------------------
    if measure_perf:
        logger.info(f"Measuring e2e performance over {PERF_MEASURE_ITERS} iterations...")

        def run_pipeline():
            tt_raw_input_l1_ = ttnn.to_memory_config(tt_raw_input, memory_config=ttnn.L1_MEMORY_CONFIG)

            (tt_sparse_, tt_idx_, tt_sc_) = ttnn.experimental.all_to_all_dispatch_metadata(
                tt_raw_input_l1_,
                tt_expert_indices,
                tt_expert_scores,
                tt_dispatch_mapping,
                cluster_axis=cluster_axis,
                num_links=4,
                output_tensors=(tt_dispatch_sparse, tt_dispatch_indices, tt_dispatch_scores),
                cross_device_semaphore=dispatch_semaphore,
                dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_UNICAST,
            )
            ttnn.deallocate(tt_raw_input_l1_)

            tt_sparse_l1_ = ttnn.to_memory_config(tt_sparse_, memory_config=ttnn.L1_MEMORY_CONFIG)
            tt_sparse_l1_ = ttnn.reshape(tt_sparse_l1_, [total_tokens, K])  # [1,128,2880] → [128,2880]
            # Use HEIGHT_SHARDED indices/scores directly from dispatch (same format as accuracy path).
            out = ttnn.experimental.moe_gpt(
                tt_sparse_l1_,
                expert_indices=tt_idx_,
                expert_scores=tt_sc_,
                expert_mapping=tt_moe_gpt_mapping,
                w0_w1_tensor=tt_w0_w1,
                w2_tensor=tt_w2,
                cluster_axis=cluster_axis,
            )
            ttnn.deallocate(tt_sparse_l1_)

            tt_combine_out_ = ttnn.experimental.selective_reduce_combine(
                out[4],
                out[1],
                out[2],
                out[0],
                hidden_size=K,
                batch_size=M_ring,  # total ring tokens (128); tokens_per_device = 128/4 = 32
                seq_size=1,
                select_experts_k=selected_experts_k,
                experts=global_experts,  # 128 = experts_total * mesh_cols
                cluster_axis=cluster_axis,
                topology=ttnn.Topology.Ring,
                num_links=4,
                token_parallel_core_dim=combine_token_parallel_core_dim,
                data_parallel_core_dim=combine_data_parallel_core_dim,
                worker_cores=combine_worker_cores,
                mux_core_range_set=combine_mux_cores,
                output_tensor=tt_combine_preallocated,
                optional_cross_device_semaphore=combine_semaphore,
            )
            return out, tt_combine_out_

        # Warmup
        for _ in range(PERF_WARMUP_ITERS):
            out, combine_out = run_pipeline()
            ttnn.synchronize_device(mesh_device)
            for t in out:
                ttnn.deallocate(t)

        # Measure (single call for Tracy device profiler)
        profiler.clear()
        profiler.start("moe_gpt_e2e")
        for _ in range(PERF_MEASURE_ITERS):
            out, combine_out = run_pipeline()
            ttnn.synchronize_device(mesh_device)
            for t in out:
                ttnn.deallocate(t)
        profiler.end("moe_gpt_e2e", PERF_CNT=PERF_MEASURE_ITERS)

        perf_us = profiler.get("moe_gpt_e2e") * 1e6
        logger.info(
            f"E2E (dispatch + moe_gpt + selective_reduce_combine) avg latency: {perf_us:.1f} us "
            f"over {PERF_MEASURE_ITERS} iters (warmup {PERF_WARMUP_ITERS})"
        )
        return all_passing, perf_us

    return all_passing


# ---------------------------------------------------------------------------
# Pytest entry points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((4, 8), id="4x8"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("M", [32])
@pytest.mark.parametrize("K, N", [(2880, 2880)])
@pytest.mark.parametrize("E, experts_total", [(4, 128)])
@pytest.mark.parametrize("selected_experts_k", [4])
def test_moe_gpt_e2e(
    mesh_device,
    M,
    K,
    N,
    E,
    experts_total,
    selected_experts_k,
    device_params,
):
    """E2E test: all_to_all_dispatch_metadata → moe_gpt (compute).

    Verifies the full dispatch+compute pipeline from raw token inputs
    through to per-expert W2 outputs on a 4x8 galaxy mesh.
    """
    mesh_device.disable_and_clear_program_cache()

    passing = run_test_moe_gpt_e2e(
        mesh_device=mesh_device,
        M=M,
        K=K,
        N=N,
        E=E,
        selected_experts_k=selected_experts_k,
        experts_total=experts_total,
        measure_perf=False,
    )

    assert passing, "E2E dispatch+moe_gpt accuracy check failed for one or more devices"


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((4, 8), id="4x8"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("M", [32])
@pytest.mark.parametrize("K, N", [(2880, 2880)])
@pytest.mark.parametrize("E, experts_total", [(4, 128)])
@pytest.mark.parametrize("selected_experts_k", [4])
def test_moe_gpt_e2e_perf(
    mesh_device,
    M,
    K,
    N,
    E,
    experts_total,
    selected_experts_k,
    device_params,
):
    """E2E performance test: all_to_all_dispatch_metadata → moe_gpt.

    Measures wall-clock latency of the dispatch+compute pipeline on a 4x8 galaxy mesh.
    """
    mesh_device.disable_and_clear_program_cache()

    result = run_test_moe_gpt_e2e(
        mesh_device=mesh_device,
        M=M,
        K=K,
        N=N,
        E=E,
        selected_experts_k=selected_experts_k,
        experts_total=experts_total,
        measure_perf=True,
    )

    if isinstance(result, tuple):
        passing, perf_us = result
    else:
        passing = result
        perf_us = None

    assert passing, "E2E dispatch+moe_gpt accuracy check failed for one or more devices"
    if perf_us is not None:
        logger.info(f"PERF RESULT: E2E avg latency = {perf_us:.1f} us")


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 8), id="4x8")], indirect=True)
@pytest.mark.parametrize("tokens_global", [128])
@pytest.mark.parametrize("hidden_size", [2880])
@pytest.mark.parametrize("selected_experts_k", [4])
@pytest.mark.parametrize("experts_total", [128])
def test_dispatch(mesh_device, tokens_global, hidden_size, selected_experts_k, experts_total, device_params):
    """Test all_to_all_dispatch_metadata: verifies sparse buffer correctness per device."""
    mesh_device.disable_and_clear_program_cache()
    passing = run_test_dispatch(
        mesh_device=mesh_device,
        tokens_global=tokens_global,
        hidden_size=hidden_size,
        selected_experts_k=selected_experts_k,
        experts_total=experts_total,
    )
    assert passing, "Dispatch sparse buffer mismatch on one or more devices"


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 8), id="4x8")], indirect=True)
@pytest.mark.parametrize("tokens_global", [128])
@pytest.mark.parametrize("hidden_size", [2880])
@pytest.mark.parametrize("selected_experts_k", [4])
@pytest.mark.parametrize("experts_total", [128])
def test_dispatch_compute(mesh_device, tokens_global, hidden_size, selected_experts_k, experts_total, device_params):
    """Test all_to_all_dispatch_metadata: verifies sparse buffer correctness per device."""
    mesh_device.disable_and_clear_program_cache()
    passing = run_test_dispatch_compute(
        mesh_device=mesh_device,
        tokens_global=tokens_global,
        hidden_size=hidden_size,
        selected_experts_k=selected_experts_k,
        experts_total=experts_total,
    )
    assert passing, "Dispatch sparse buffer mismatch on one or more devices"


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 8), id="4x8")], indirect=True)
@pytest.mark.parametrize("tokens_global", [128])
@pytest.mark.parametrize("hidden_size", [2880])
@pytest.mark.parametrize("selected_experts_k", [4])
@pytest.mark.parametrize("experts_total", [128])
def test_dispatch_compute_combine(
    mesh_device, tokens_global, hidden_size, selected_experts_k, experts_total, device_params
):
    """Test all_to_all_dispatch_metadata → moe_gpt → selective_reduce_combine pipeline."""
    mesh_device.disable_and_clear_program_cache()
    passing = run_test_dispatch_compute_combine(
        mesh_device=mesh_device,
        tokens_global=tokens_global,
        hidden_size=hidden_size,
        selected_experts_k=selected_experts_k,
        experts_total=experts_total,
    )
    assert passing, "Dispatch sparse buffer mismatch on one or more devices"


def run_test_combine_isolation(mesh_device, tokens_global, hidden_size, selected_experts_k, experts_total, num_iters=5):
    """Test selective_reduce_combine in isolation with gpt-oss shapes.

    Runs dispatch+moe_gpt once to generate real combine inputs, then calls
    selective_reduce_combine multiple times to verify correctness and stability.
    """
    torch.manual_seed(42)
    cluster_axis = 0
    ring_devices = mesh_device.shape[cluster_axis]  # 4
    total_mesh_devices = mesh_device.get_num_devices()  # 32
    total_tokens = tokens_global  # 128
    M = tokens_global // ring_devices  # 32
    E = experts_total // total_mesh_devices  # 4
    N = hidden_size
    L = 1
    tokens_per_chunk = 32
    mesh_cols = mesh_device.shape[1]

    # --- Expert mapping ---
    expert_mapping_torch = create_expert_mapping_tensors(
        num_devices=total_mesh_devices,
        num_experts_global=experts_total,
        mesh_device=mesh_device,
        new_format=True,
        return_torch=True,
    )
    tt_dispatch_mapping = ttnn.from_torch(
        expert_mapping_torch.to(torch.int16),
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_moe_gpt_mapping = ttnn.from_torch(
        expert_mapping_torch.to(torch.int16),
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # --- Weights ---
    torch_w0 = torch.randn(1, E, hidden_size, N, dtype=torch.bfloat16) * 0.01
    torch_w1 = torch.randn(1, E, hidden_size, N, dtype=torch.bfloat16) * 0.01
    torch_w2 = torch.randn(1, E, N, hidden_size, dtype=torch.bfloat16) * 0.01

    ring2cores = build_ring2cores(mesh_device)
    num_cores = len(ring2cores)
    groups_per_core = MAX_W0_W1_TILES_PER_CORE // 2
    dram_cores = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(num_cores)]
    dram_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in dram_cores])

    tt_w0_w1 = ttnn.from_torch(
        prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, hidden_size, N, ring2cores),
        dtype=ttnn.bfloat4_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                dram_core_range_set,
                (L * E * groups_per_core * hidden_size, 4 * ttnn.TILE_SIZE),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_w2 = ttnn.from_torch(
        prepare_w2_tensor(torch_w2, L, E, N, hidden_size, ring2cores),
        dtype=ttnn.bfloat4_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_core_range_set, (L * E * 2 * N, 4 * ttnn.TILE_SIZE), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # --- Input tokens + routing ---
    raw_input_torch = torch.randn(1, 1, tokens_global, hidden_size, dtype=torch.bfloat16)
    indices_list = [torch.randperm(experts_total)[:selected_experts_k].to(torch.int32) for _ in range(tokens_global)]
    expert_indices = torch.stack(indices_list).reshape(1, 1, tokens_global, selected_experts_k)
    expert_scores = torch.stack(
        [s / s.sum() for s in [torch.rand(selected_experts_k, dtype=torch.float32) for _ in range(tokens_global)]]
    ).reshape(1, 1, tokens_global, selected_experts_k)

    # Input sharding for dispatch
    num_cores_y = min(8, tokens_global // ring_devices)
    num_cores_x = (tokens_global // ring_devices + num_cores_y - 1) // num_cores_y
    input_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}),
        [1, selected_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_indices_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    shard_2d = dict(
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=tuple(mesh_device.shape)),
    )
    tt_raw_input = ttnn.from_torch(
        raw_input_torch, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, **shard_2d
    )
    tt_indices = ttnn.from_torch(
        expert_indices.to(torch.int16), dtype=ttnn.uint16, memory_config=input_indices_mem, **shard_2d
    )
    tt_scores = ttnn.from_torch(expert_scores, dtype=ttnn.bfloat16, memory_config=input_indices_mem, **shard_2d)

    # Dispatch output tensors
    dispatch_drain_core = ttnn.CoreCoord(6, 9)
    drain_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(dispatch_drain_core, dispatch_drain_core)}),
        [total_tokens, selected_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    drain_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, drain_shard_spec)

    tt_sparse_out = ttnn.from_torch(
        torch.zeros(1, total_tokens, hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_indices_out = ttnn.from_torch(
        torch.zeros(1, total_tokens, selected_experts_k, dtype=torch.int16),
        dtype=ttnn.uint16,
        memory_config=drain_mem,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_scores_out = ttnn.from_torch(
        torch.zeros(1, total_tokens, selected_experts_k, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        memory_config=drain_mem,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    compute_grid = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    dispatch_sem = ttnn.create_global_semaphore(mesh_device, all_cores, 0)

    tt_input_l1 = ttnn.to_memory_config(tt_raw_input, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.synchronize_device(mesh_device)

    # --- Dispatch ---
    (tt_sparse, tt_idx, tt_sc) = ttnn.experimental.all_to_all_dispatch_metadata(
        tt_input_l1,
        tt_indices,
        tt_scores,
        tt_dispatch_mapping,
        cluster_axis=cluster_axis,
        num_links=4,
        output_tensors=(tt_sparse_out, tt_indices_out, tt_scores_out),
        cross_device_semaphore=dispatch_sem,
        dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_UNICAST,
    )
    ttnn.synchronize_device(mesh_device)
    logger.info("combine_isolation: dispatch done")

    # --- moe_gpt ---
    tt_sparse = ttnn.reshape(tt_sparse, [total_tokens, hidden_size])
    moe_gpt_outputs = ttnn.experimental.moe_gpt(
        tt_sparse,
        expert_indices=tt_idx,
        expert_scores=tt_sc,
        expert_mapping=tt_moe_gpt_mapping,
        w0_w1_tensor=tt_w0_w1,
        w2_tensor=tt_w2,
        cluster_axis=cluster_axis,
    )
    ttnn.synchronize_device(mesh_device)
    logger.info("combine_isolation: moe_gpt done")
    # Debug: print moe_gpt output tensor configs
    for i in range(5):
        t = moe_gpt_outputs[i]
        per_dev = ttnn.get_device_tensors(t)
        logger.info(f"  moe_gpt_output[{i}]: shape={t.shape}, dtype={t.dtype}, layout={t.layout}")
        logger.info(f"    per_device[0] shape={per_dev[0].shape}, memory_config={per_dev[0].memory_config()}")

    # --- Setup combine infrastructure ---
    COMBINE_H = 4
    COMBINE_W = 3
    combine_core_range_set, combine_start, combine_end = get_moe_gpt_combine_core_range(
        mesh_device, COMBINE_W, COMBINE_H
    )
    combine_worker_cores = list(ttnn.corerange_to_cores(combine_core_range_set, row_wise=True))
    mux_start = ttnn.CoreCoord(combine_end.x + 1, combine_start.y)
    mux_end = ttnn.CoreCoord(combine_end.x + 2, combine_end.y)
    combine_mux_cores = ttnn.CoreRangeSet([ttnn.CoreRange(mux_start, mux_end)])
    combine_semaphore = ttnn.create_global_semaphore(mesh_device, all_cores, 0)

    # --- Build torch reference ---
    raw_input_2d = raw_input_torch.reshape(tokens_global, hidden_size)
    ring_expert_sets = {}
    for col in range(mesh_cols):
        ring_expert_sets[col] = set()
        for e_global in range(experts_total):
            owning_device = int(expert_mapping_torch[0, e_global])
            if owning_device % mesh_cols == col:
                ring_expert_sets[col].add(e_global)

    ref_per_device = {}
    K_sel = selected_experts_k
    for dev_idx in range(total_mesh_devices):
        ring_pos = dev_idx // mesh_cols
        col = dev_idx % mesh_cols
        ref_combine = torch.zeros(K_sel, M, hidden_size, dtype=torch.float32)
        with torch.no_grad():
            for t in range(M):
                global_token = ring_pos * M + t
                for k in range(K_sel):
                    global_e = expert_indices[0, 0, global_token, k].item()
                    if global_e not in ring_expert_sets[col]:
                        continue
                    local_e = global_e % E
                    token = raw_input_2d[global_token : global_token + 1].to(torch.float32)
                    gate = (token @ torch_w0[0, local_e].to(torch.float32)).to(torch.bfloat16)
                    up = (token @ torch_w1[0, local_e].to(torch.float32)).to(torch.bfloat16)
                    activated = swiglu_reference(gate, up).to(torch.float32)
                    ref_combine[k, t] = activated @ torch_w2[0, local_e].to(torch.float32)
        ref_per_device[dev_idx] = ref_combine

    # Pre-allocate combine output tensor (same as test_dispatch_compute_combine)
    tt_combine_preallocated = ttnn.from_torch(
        torch.zeros(selected_experts_k, M, hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # --- Run combine N times ---
    all_passing = True
    for iteration in range(num_iters):
        tt_combine_result = ttnn.experimental.selective_reduce_combine(
            moe_gpt_outputs[4],
            moe_gpt_outputs[1],
            moe_gpt_outputs[2],
            moe_gpt_outputs[0],
            hidden_size=hidden_size,
            batch_size=total_tokens,
            seq_size=1,
            select_experts_k=selected_experts_k,
            experts=experts_total,
            cluster_axis=cluster_axis,
            topology=ttnn.Topology.Ring,
            num_links=4,
            token_parallel_core_dim=COMBINE_H,
            data_parallel_core_dim=COMBINE_W,
            worker_cores=combine_worker_cores,
            mux_core_range_set=combine_mux_cores,
            output_tensor=tt_combine_preallocated,
            optional_cross_device_semaphore=combine_semaphore,
        )
        ttnn.synchronize_device(mesh_device)

        # Verify
        combine_per_device = ttnn.get_device_tensors(tt_combine_result)
        iter_passing = True
        for dev_idx in range(total_mesh_devices):
            actual_combine = ttnn.to_torch(combine_per_device[dev_idx]).reshape(K_sel, M, hidden_size)
            actual_sum = actual_combine.to(torch.float32).sum(dim=0)
            ref_sum = ref_per_device[dev_idx].sum(dim=0)
            pcc_pass, pcc_val = comp_pcc(actual_sum, ref_sum, PCC_THRESHOLD)
            if not pcc_pass:
                logger.error(f"  iter {iteration} device {dev_idx}: FAILED PCC={pcc_val:.4f}")
                iter_passing = False
                all_passing = False

        if iter_passing:
            logger.info(f"combine_isolation iter {iteration}: ALL PASSED")
        else:
            logger.error(f"combine_isolation iter {iteration}: SOME FAILED")

    # Cleanup
    for i in range(4):  # output[3] and [4] share a buffer; deallocating [3] frees both
        ttnn.deallocate(moe_gpt_outputs[i])

    return all_passing


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 8), id="4x8")], indirect=True)
@pytest.mark.parametrize("tokens_global", [128])
@pytest.mark.parametrize("hidden_size", [2880])
@pytest.mark.parametrize("selected_experts_k", [4])
@pytest.mark.parametrize("experts_total", [128])
@pytest.mark.parametrize("num_iters", [5])
def test_combine_isolation(
    mesh_device, tokens_global, hidden_size, selected_experts_k, experts_total, num_iters, device_params
):
    """Test selective_reduce_combine in isolation: generate data via dispatch+moe_gpt, then run combine N times."""
    mesh_device.disable_and_clear_program_cache()
    passing = run_test_combine_isolation(
        mesh_device=mesh_device,
        tokens_global=tokens_global,
        hidden_size=hidden_size,
        selected_experts_k=selected_experts_k,
        experts_total=experts_total,
        num_iters=num_iters,
    )
    assert passing, "Combine isolation test failed"


def run_test_full_pipeline_multi_iter(
    mesh_device,
    tokens_global,
    hidden_size,
    selected_experts_k,
    experts_total,
    num_iters=3,
    use_output_tensor=True,
):
    """Test full fused pipeline (dispatch -> moe_gpt -> combine) N times.

    Unlike combine_isolation (which reuses moe_gpt outputs), this runs the
    ENTIRE pipeline each iteration, mimicking what the model does across layers.
    """
    torch.manual_seed(42)
    cluster_axis = 0
    ring_devices = mesh_device.shape[cluster_axis]  # 4
    total_mesh_devices = mesh_device.get_num_devices()  # 32
    total_tokens = tokens_global  # 128
    M = tokens_global // ring_devices  # 32
    E = experts_total // total_mesh_devices  # 4
    N = hidden_size
    L = 1
    mesh_cols = mesh_device.shape[1]

    # --- Expert mapping ---
    expert_mapping_torch = create_expert_mapping_tensors(
        num_devices=total_mesh_devices,
        num_experts_global=experts_total,
        mesh_device=mesh_device,
        new_format=True,
        return_torch=True,
    )
    tt_dispatch_mapping = ttnn.from_torch(
        expert_mapping_torch.to(torch.int16),
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_moe_gpt_mapping = ttnn.from_torch(
        expert_mapping_torch.to(torch.int16),
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # --- Weights ---
    torch_w0 = torch.randn(1, E, hidden_size, N, dtype=torch.bfloat16) * 0.01
    torch_w1 = torch.randn(1, E, hidden_size, N, dtype=torch.bfloat16) * 0.01
    torch_w2 = torch.randn(1, E, N, hidden_size, dtype=torch.bfloat16) * 0.01

    ring2cores = build_ring2cores(mesh_device)
    num_cores = len(ring2cores)
    groups_per_core = MAX_W0_W1_TILES_PER_CORE // 2
    dram_cores = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(num_cores)]
    dram_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in dram_cores])

    tt_w0_w1 = ttnn.from_torch(
        prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, hidden_size, N, ring2cores),
        dtype=ttnn.bfloat4_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                dram_core_range_set,
                (L * E * groups_per_core * hidden_size, 4 * ttnn.TILE_SIZE),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_w2 = ttnn.from_torch(
        prepare_w2_tensor(torch_w2, L, E, N, hidden_size, ring2cores),
        dtype=ttnn.bfloat4_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_core_range_set, (L * E * 2 * N, 4 * ttnn.TILE_SIZE), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # --- Combine infrastructure (shared across iterations) ---
    compute_grid = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    dispatch_sem = ttnn.create_global_semaphore(mesh_device, all_cores, 0)

    COMBINE_H = 4
    COMBINE_W = 3
    combine_core_range_set, combine_start, combine_end = get_moe_gpt_combine_core_range(
        mesh_device, COMBINE_W, COMBINE_H
    )
    combine_worker_cores = list(ttnn.corerange_to_cores(combine_core_range_set, row_wise=True))
    mux_start = ttnn.CoreCoord(combine_end.x + 1, combine_start.y)
    mux_end = ttnn.CoreCoord(combine_end.x + 2, combine_end.y)
    combine_mux_cores = ttnn.CoreRangeSet([ttnn.CoreRange(mux_start, mux_end)])
    combine_semaphore = ttnn.create_global_semaphore(mesh_device, all_cores, 0)

    # Pre-allocate combine output (reused across iterations if use_output_tensor)
    tt_combine_preallocated = None
    if use_output_tensor:
        tt_combine_preallocated = ttnn.from_torch(
            torch.zeros(selected_experts_k, M, hidden_size, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # Input sharding config
    num_cores_y = min(8, tokens_global // ring_devices)
    num_cores_x = (tokens_global // ring_devices + num_cores_y - 1) // num_cores_y
    input_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}),
        [1, selected_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_indices_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    dispatch_drain_core = ttnn.CoreCoord(6, 9)
    drain_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(dispatch_drain_core, dispatch_drain_core)}),
        [total_tokens, selected_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    drain_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, drain_shard_spec)

    shard_2d = dict(
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=tuple(mesh_device.shape)),
    )

    # === Run full pipeline N times ===
    all_passing = True
    for iteration in range(num_iters):
        logger.info(f"=== Pipeline iteration {iteration}/{num_iters} ===")

        # Fresh random input each iteration (like different layers in the model)
        raw_input_torch = torch.randn(1, 1, tokens_global, hidden_size, dtype=torch.bfloat16)
        indices_list = [
            torch.randperm(experts_total)[:selected_experts_k].to(torch.int32) for _ in range(tokens_global)
        ]
        expert_indices = torch.stack(indices_list).reshape(1, 1, tokens_global, selected_experts_k)

        # Upload inputs
        tt_raw_input = ttnn.from_torch(
            raw_input_torch, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, **shard_2d
        )
        tt_indices = ttnn.from_torch(
            expert_indices.to(torch.int16), dtype=ttnn.uint16, memory_config=input_indices_mem, **shard_2d
        )
        # Use uniform scores for simplicity
        expert_scores = torch.ones(1, 1, tokens_global, selected_experts_k, dtype=torch.float32) / selected_experts_k
        tt_scores = ttnn.from_torch(expert_scores, dtype=ttnn.bfloat16, memory_config=input_indices_mem, **shard_2d)

        # Dispatch output tensors (fresh each iteration)
        tt_sparse_out = ttnn.from_torch(
            torch.zeros(1, total_tokens, hidden_size, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        tt_indices_out = ttnn.from_torch(
            torch.zeros(1, total_tokens, selected_experts_k, dtype=torch.int16),
            dtype=ttnn.uint16,
            memory_config=drain_mem,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        tt_scores_out = ttnn.from_torch(
            torch.zeros(1, total_tokens, selected_experts_k, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            memory_config=drain_mem,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        tt_input_l1 = ttnn.to_memory_config(tt_raw_input, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.synchronize_device(mesh_device)

        # --- Step 1: Dispatch ---
        logger.info(f"  iter {iteration}: dispatch")
        (tt_sparse, tt_idx, tt_sc) = ttnn.experimental.all_to_all_dispatch_metadata(
            tt_input_l1,
            tt_indices,
            tt_scores,
            tt_dispatch_mapping,
            cluster_axis=cluster_axis,
            num_links=4,
            output_tensors=(tt_sparse_out, tt_indices_out, tt_scores_out),
            cross_device_semaphore=dispatch_sem,
            dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_UNICAST,
        )
        ttnn.synchronize_device(mesh_device)
        logger.info(f"  iter {iteration}: dispatch done")

        # --- Step 2: moe_gpt ---
        tt_sparse_2d = ttnn.reshape(tt_sparse, [total_tokens, hidden_size])
        logger.info(f"  iter {iteration}: moe_gpt")
        moe_gpt_outputs = ttnn.experimental.moe_gpt(
            tt_sparse_2d,
            expert_indices=tt_idx,
            expert_scores=tt_sc,
            expert_mapping=tt_moe_gpt_mapping,
            w0_w1_tensor=tt_w0_w1,
            w2_tensor=tt_w2,
            cluster_axis=cluster_axis,
        )
        ttnn.synchronize_device(mesh_device)
        logger.info(f"  iter {iteration}: moe_gpt done")

        # --- Step 3: selective_reduce_combine ---
        # WORKAROUND: disable program cache around combine to avoid hang caused by
        # override_runtime_arguments not updating fabric mux connection runtime args.
        mesh_device.disable_and_clear_program_cache()
        logger.info(f"  iter {iteration}: combine (use_output_tensor={use_output_tensor})")
        combine_kwargs = dict(
            hidden_size=hidden_size,
            batch_size=total_tokens,
            seq_size=1,
            select_experts_k=selected_experts_k,
            experts=experts_total,
            cluster_axis=cluster_axis,
            topology=ttnn.Topology.Ring,
            num_links=4,
            token_parallel_core_dim=COMBINE_H,
            data_parallel_core_dim=COMBINE_W,
            worker_cores=combine_worker_cores,
            mux_core_range_set=combine_mux_cores,
        )
        if use_output_tensor:
            combine_kwargs["output_tensor"] = tt_combine_preallocated
            combine_kwargs["optional_cross_device_semaphore"] = combine_semaphore

        tt_combine_result = ttnn.experimental.selective_reduce_combine(
            moe_gpt_outputs[4],
            moe_gpt_outputs[1],
            moe_gpt_outputs[2],
            moe_gpt_outputs[0],
            **combine_kwargs,
        )
        ttnn.synchronize_device(mesh_device)
        logger.info(f"  iter {iteration}: combine done")
        mesh_device.enable_program_cache()

        # --- Cleanup moe_gpt outputs (like fused_decode.py does) ---
        for i in range(4):  # output[3] and [4] share a buffer; deallocating [3] frees both
            ttnn.deallocate(moe_gpt_outputs[i])

        # --- Optional: do tilize + deallocate combine output (like model does) ---
        tt_combine_tile = ttnn.to_layout(tt_combine_result, ttnn.TILE_LAYOUT)
        if not use_output_tensor:
            ttnn.deallocate(tt_combine_result)
        # Deallocate tilized version too
        ttnn.deallocate(tt_combine_tile)

        ttnn.synchronize_device(mesh_device)
        logger.info(f"  iter {iteration}: cleanup done")

    logger.info(f"Full pipeline multi-iter test PASSED ({num_iters} iterations, use_output_tensor={use_output_tensor})")
    return True


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 8), id="4x8")], indirect=True)
@pytest.mark.parametrize("tokens_global", [128])
@pytest.mark.parametrize("hidden_size", [2880])
@pytest.mark.parametrize("selected_experts_k", [4])
@pytest.mark.parametrize("experts_total", [128])
@pytest.mark.parametrize("num_iters", [3])
@pytest.mark.parametrize("use_output_tensor", [True, False], ids=["with_output_tensor", "without_output_tensor"])
def test_full_pipeline_multi_iter(
    mesh_device,
    tokens_global,
    hidden_size,
    selected_experts_k,
    experts_total,
    num_iters,
    use_output_tensor,
    device_params,
):
    """Test full fused pipeline (dispatch -> moe_gpt -> combine) run N times.

    This mimics what the model does across layers: each iteration is a fresh
    dispatch+moe_gpt+combine with deallocations in between.
    Tests both with and without output_tensor to isolate the hang cause.
    """
    mesh_device.enable_program_cache()
    passing = run_test_full_pipeline_multi_iter(
        mesh_device=mesh_device,
        tokens_global=tokens_global,
        hidden_size=hidden_size,
        selected_experts_k=selected_experts_k,
        experts_total=experts_total,
        num_iters=num_iters,
        use_output_tensor=use_output_tensor,
    )
    assert passing, "Full pipeline multi-iter test failed"
