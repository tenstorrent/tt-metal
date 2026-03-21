# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for moe_gpt op (fused tilize → matmul → combine pipeline).

Tests the full fused path:
  tilize → W0/W1 matmul → SwiGLU → A2A ring → W2 matmul → untilize → combine

Runs on a 1x1 mesh (single device) or larger meshes (e.g. 4x8 galaxy).
On single device, all inputs are replicated (identical). On multi-device,
each device processes its own local experts based on linearized_mesh_coord.

Dimensions (gpt-oss 20b):
    M = 32 (tokens per device)
    K = 2880 (hidden_size) -> 90 tiles
    N = 2880 (intermediate_size) -> 90 tiles
    E = 4 (experts per device)
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
    _prepare_w0_b0_w1_b1_tensor as prepare_w0_b0_w1_b1_tensor,
    _prepare_w2_b2_tensor as prepare_w2_b2_tensor,
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


def gen_expert_mapping(experts_total, num_devices, mesh_cols=1):
    """
    Expert-to-device ownership mapping.

    The kernel uses row-major linearization: linearized_coord = row * mesh_cols + col.
    Python get_device_tensors uses the same row-major ordering.

    For cluster_axis=0 (ring along rows), source_device values seen by the kernel are:
        source_device = device_in_group * mesh_cols + col
    where device_in_group ranges 0..num_devices-1 and col ranges 0..mesh_cols-1.
    Max source_device = (num_devices - 1) * mesh_cols + (mesh_cols - 1) = num_devices * mesh_cols - 1.
    So the mapping must have num_devices * mesh_cols rows.

    For device d (linearized row-major) and expert e:
        mapping[d, e] = (e // experts_per_device) * mesh_cols + (d % mesh_cols)

    This gives the linearized_mesh_coord of the device that owns expert e in the
    same ring (column) as device d.

    For 1x1 mesh (mesh_cols=1): 4 rows, mapping[d, e] = e // E (ring position, same for all rows)
    For 4x8 mesh (mesh_cols=8): 32 rows, mapping[d, e] = (e // E) * 8 + col
        where col = d % 8.

    Expert ownership check: mapping[d, e] == d is true when e // E == row (ring position).
    """
    mapping_rows = num_devices * mesh_cols
    experts_per_device = experts_total // num_devices
    result = torch.zeros(mapping_rows, experts_total, dtype=torch.int32)
    for d in range(mapping_rows):
        col = d % mesh_cols
        for e in range(experts_total):
            result[d, e] = (e // experts_per_device) * mesh_cols + col
    return result


def gen_controlled_sparse_buffer(
    total_tokens,
    hidden_size,
    experts_per_device,
    experts_total,
    selected_experts_k,
    num_devices,
    tokens_per_expert=None,
    all_local_experts=False,
):
    """
    Generate sparse buffer with controlled token-to-expert routing.

    Routes tokens across ALL devices so every device has work to do.

    Modes:
      - tokens_per_expert=None, all_local_experts=False:
          Even distribution across all experts globally. Each token routes to
          one expert per device (round-robin), giving each expert
          total_tokens / experts_total tokens.
      - tokens_per_expert=[...], all_local_experts=False:
          Uneven distribution on device 0 only. Sum must equal total_tokens.
          Other devices get 0 tokens (single-device stress test mode).
      - all_local_experts=True:
          Every token routes to ALL E local experts on one device, cycling
          through devices. Each device gets total_tokens/num_devices tokens
          per expert. Requires selected_experts_k >= experts_per_device.
    """
    expert_indices = torch.zeros(total_tokens, selected_experts_k, dtype=torch.int32)

    if all_local_experts:
        assert selected_experts_k >= experts_per_device
        # Each token routes to all E experts on one device, cycling through devices.
        # Token t -> device (t % num_devices) -> all E experts on that device.
        for t in range(total_tokens):
            target_device = t % num_devices
            local_globals = [target_device * experts_per_device + le for le in range(experts_per_device)]
            for k in range(experts_per_device):
                expert_indices[t, k] = local_globals[k]
            # Fill remaining k-slots with experts from other devices
            remaining = selected_experts_k - experts_per_device
            if remaining > 0:
                other_experts = [e for e in range(experts_total) if e // experts_per_device != target_device]
                others = random.sample(other_experts, remaining)
                for k, eid in enumerate(others):
                    expert_indices[t, experts_per_device + k] = eid

    elif tokens_per_expert is not None:
        # Uneven distribution: device 0 only (single-device stress test mode)
        device_idx = 0
        assert len(tokens_per_expert) == experts_per_device
        assert sum(tokens_per_expert) == total_tokens

        non_local_experts = [e for e in range(experts_total) if e // experts_per_device != device_idx]

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

    else:
        # Default: even distribution across all devices.
        # Each token routes to one expert per device (round-robin across local experts).
        # With total_tokens=128, experts_total=16: each expert gets 32 tokens.
        all_experts = list(range(experts_total))
        for t in range(total_tokens):
            # Pick one expert per device, cycling through local expert indices
            selected = []
            for d in range(num_devices):
                local_e = t % experts_per_device
                global_e = d * experts_per_device + local_e
                selected.append(global_e)
            # If selected_experts_k < num_devices, truncate; if >, fill with random extras
            if len(selected) >= selected_experts_k:
                selected = selected[:selected_experts_k]
            else:
                remaining_pool = [e for e in all_experts if e not in selected]
                extra = random.sample(remaining_pool, selected_experts_k - len(selected))
                selected.extend(extra)
            for k, eid in enumerate(selected):
                expert_indices[t, k] = eid

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


def verify_device_output(
    tt_output_result,
    device_idx,
    sparse_torch,
    indices_torch,
    expert_mapping_torch,
    torch_w0,
    torch_w1,
    torch_w2,
    E,
    M,
    K,
    total_tokens,
    tokens_per_chunk,
    height_shard_dim=4,
):
    """
    Verify a single device's combine output against the torch reference.

    The combine kernel distributes tokens across height_shard_dim height shards
    using floor+remainder (matching selective_reduce_combine). This function
    gathers tokens from all shards before comparing with the reference.

    tt_output_result must be the FULL output tensor {E * total_tokens, K},
    not just the first height shard.

    Returns (passing, per_expert_counts).
    """
    ref_tilized, per_expert_counts = tilize_reference(
        sparse_torch,
        indices_torch,
        expert_mapping_torch,
        device_idx,
        E,
        tokens_per_chunk,
    )

    shard_rows = E * total_tokens // height_shard_dim
    expert_rows_per_shard = total_tokens // height_shard_dim  # rows reserved per expert per shard

    all_passing = True
    for e in range(E):
        count = per_expert_counts[e]
        if count == 0:
            logger.info(f"  Device {device_idx} Expert {e}: no tokens routed, skipping")
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

        # Gather tokens from all height shards. The combine kernel distributes
        # tokens for expert e across height_shard_dim shards using floor+remainder:
        #   chunk = active_tokens // H, rem = active_tokens % H
        #   shard k gets (chunk+1) tokens if k < rem, else chunk tokens
        # Within each shard, expert e's data starts at row e * expert_rows_per_shard.
        chunk_size = last_chunk_tokens // height_shard_dim
        remainder = last_chunk_tokens % height_shard_dim
        expert_base_row = e * expert_rows_per_shard
        gathered_rows = []
        for hs in range(height_shard_dim):
            n_toks = chunk_size + (1 if hs < remainder else 0)
            if n_toks == 0:
                break
            global_row = hs * shard_rows + expert_base_row
            gathered_rows.append(tt_output_result[global_row : global_row + n_toks, :])
        tt_expert_result = torch.cat(gathered_rows, dim=0)  # [last_chunk_tokens, K]

        metrics = get_accuracy_metrics(reference[:last_chunk_tokens, :], tt_expert_result[:last_chunk_tokens, :])
        pcc = metrics["pcc"]
        rmse = metrics["relative_rmse"]

        chunks_str = f"{num_chunks} chunk{'s' if num_chunks > 1 else ''}"
        if pcc < PCC_THRESHOLD:
            all_passing = False
            logger.warning(
                f"  Device {device_idx} Expert {e}: PCC={pcc:.6f} RMSE={rmse:.6f} "
                f"({count} tokens, {chunks_str}) FAILED"
            )
        else:
            logger.info(
                f"  Device {device_idx} Expert {e}: PCC={pcc:.6f} RMSE={rmse:.6f} "
                f"({count} tokens, {chunks_str}) Passed"
            )

    return all_passing, per_expert_counts


# ---------------------------------------------------------------------------
# Test body
# ---------------------------------------------------------------------------


def run_test_moe_gpt_tilize_matmul(
    mesh_device,
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

    num_devices = experts_total // E  # logical ring size = mesh_device.shape[cluster_axis]
    total_tokens = M * num_devices
    num_mesh_devices = mesh_device.get_num_devices()
    mesh_cols = mesh_device.shape[1]  # number of columns (independent rings)

    logger.info(f"Tilize-matmul integration test configuration:")
    logger.info(f"  num_devices (simulated): {num_devices}")
    logger.info(f"  mesh_devices: {num_mesh_devices}")
    logger.info(f"  tokens_per_device (M): {M}, total_tokens: {total_tokens}")
    logger.info(f"  experts_per_device (E): {E}, experts_total: {experts_total}")
    logger.info(f"  hidden_size (K): {K}, intermediate_size (N): {N}")

    tokens_per_chunk = 32
    cluster_axis = 0

    # ------------------------------------------------------------------
    # Matmul core infrastructure (queried from first device, same on all)
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

    # Weight configs
    groups_per_core = MAX_W0_W1_TILES_PER_CORE // 2
    K_bias = K + 32  # K + 1 bias tile row
    w0_w1_shard_height = L * E * groups_per_core * K_bias
    w0_w1_shard_width = 4 * ttnn.TILE_SIZE

    w0_w1_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w0_w1_shard_height, w0_w1_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    w0_w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w0_w1_shard_spec)

    N_bias = N + 32  # N + 1 bias tile row
    w2_shard_height = L * E * 2 * N_bias
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

    # Prepare weight tensors (replicated on all mesh devices)
    # Create zero biases (1 tile row = 32 rows)
    torch_b0 = torch.zeros(1, E, 32, N, dtype=torch.bfloat16)
    torch_b1 = torch.zeros(1, E, 32, N, dtype=torch.bfloat16)
    torch_b2 = torch.zeros(1, E, 32, K, dtype=torch.bfloat16)

    torch_w0_w1_reordered = prepare_w0_b0_w1_b1_tensor(torch_w0, torch_b0, torch_w1, torch_b1, L, E, K, N, ring2cores)
    tt_w0_w1 = ttnn.from_torch(
        torch_w0_w1_reordered,
        dtype=w_dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w0_w1_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    torch_w2_reordered = prepare_w2_b2_tensor(torch_w2, torch_b2, L, E, N, K, ring2cores)
    tt_w2 = ttnn.from_torch(
        torch_w2_reordered,
        dtype=w_dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w2_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # ------------------------------------------------------------------
    # Tilize inputs (routing distributed across all devices)
    # All devices get the same replicated inputs; each processes its own local experts.
    # ------------------------------------------------------------------
    expert_mapping_torch = gen_expert_mapping(experts_total, num_devices, mesh_cols)
    sparse_torch, indices_torch, scores_torch = gen_controlled_sparse_buffer(
        total_tokens,
        K,
        E,
        experts_total,
        selected_experts_k,
        num_devices,
        tokens_per_expert=tokens_per_expert,
        all_local_experts=all_local_experts,
    )

    logger.info(f"  sparse_buffer shape: {sparse_torch.shape}")
    logger.info(f"  expert_indices shape: {indices_torch.shape}")
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
    # Verify output per device
    # ------------------------------------------------------------------
    device_tensors = ttnn.get_device_tensors(tt_output)
    all_passing = True

    # Python get_device_tensors uses row-major ordering matching the kernel's
    # get_linearized_index: linearized = row * mesh_cols + col.
    # For cluster_axis=0, the ring goes along rows, so ring_pos = row = dev_idx // mesh_cols.
    # logical_dev_idx = dev_idx (Python index = kernel linearized_mesh_coord).

    for dev_idx in range(len(device_tensors)):
        ring_pos = dev_idx // mesh_cols  # row index = ring position
        col = dev_idx % mesh_cols

        tt_output_result = ttnn.to_torch(device_tensors[dev_idx])
        # Reshape from sharded to flat: [E * M, K]
        tt_output_result = tt_output_result.reshape(-1, K)[: E * M, :]

        passing, per_expert_counts = verify_device_output(
            tt_output_result,
            dev_idx,  # logical_dev_idx = kernel linearized_mesh_coord = Python dev_idx
            sparse_torch,
            indices_torch,
            expert_mapping_torch,
            torch_w0,
            torch_w1,
            torch_w2,
            E,
            M,
            K,
            total_tokens,
            tokens_per_chunk,
        )

        logger.info(f"  Device {dev_idx} (ring_pos={ring_pos}, col={col}) per-expert token counts: {per_expert_counts}")
        if not passing:
            all_passing = False

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
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((1, 1), id="1x1"),
        pytest.param((4, 8), id="4x8"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("M", [32])
@pytest.mark.parametrize("K, N", [(2880, 2880)])
@pytest.mark.parametrize("E, experts_total", [(4, 16)])
@pytest.mark.parametrize("selected_experts_k", [4])
def test_moe_gpt_tilize_matmul_single_device(
    mesh_device,
    M,
    K,
    N,
    E,
    experts_total,
    selected_experts_k,
    device_params,
):
    passing = run_test_moe_gpt_tilize_matmul(
        mesh_device=mesh_device,
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
@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((1, 1), id="1x1")],
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
    mesh_device,
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
        mesh_device=mesh_device,
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
@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((1, 1), id="1x1")],
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
        (4, 16),  # 4 devices -> 128 total tokens, all to all 4 local -> 128/expert (4 chunks)
    ],
    ids=["64tok_all_local", "96tok_all_local", "128tok_all_local"],
)
def test_moe_gpt_tilize_matmul_all_local(
    mesh_device,
    M,
    K,
    N,
    E,
    experts_total,
    selected_experts_k,
    device_params,
):
    passing = run_test_moe_gpt_tilize_matmul(
        mesh_device=mesh_device,
        M=M,
        K=K,
        N=N,
        E=E,
        selected_experts_k=selected_experts_k,
        experts_total=experts_total,
        all_local_experts=True,
    )

    assert passing, "All-local-experts tilize-matmul PCC check failed for one or more experts"
