# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for GPT-OSS fused MOE compute kernel (moe_gpt_fused).

Input is WIDTH_SHARDED ROW_MAJOR on 3 tilize cores.
Output is BLOCK_SHARDED ROW_MAJOR on 12 combine cores.

Dimensions:
  H = 2880 (hidden_size) -> 90 tiles
  N = 2880 (intermediate_size) -> 90 tiles
  E = 4 (experts per device)
  tokens = 32 (1 chunk, 1 tile row)

Activation: SwiGLU
  gate_clamped = clamp(gate, max=7.0)
  up_clamped   = clamp(up, min=-7.0, max=7.0)
  result       = (up_clamped + 1) * gate_clamped * sigmoid(1.702 * gate_clamped)
"""

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


def swiglu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    """GPT-OSS SwiGLU activation reference."""
    gate_c = torch.clamp(gate, max=clamp_limit)
    up_c = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
    return (up_c + 1.0) * gate_c * torch.sigmoid(alpha * gate_c)


def run_test_moe_gpt_fused(device, total_tokens, H, N, E, check_accuracy):
    """
    Run the fused MoE GPT test.

    In this initial version, all tokens go to all experts (no sparse routing).
    Input is WIDTH_SHARDED ROW_MAJOR on 3 tilize cores.
    Output is BLOCK_SHARDED ROW_MAJOR on 12 combine cores.
    """
    logger.info(
        f"Running test_moe_gpt_fused with total_tokens={total_tokens}, H={H}, N={N}, E={E}, "
        f"check_accuracy={check_accuracy}"
    )

    L = 1  # Single layer
    top_k = 2

    # --------------------------------------------------------------------------
    # Core assignments (same as moe_gpt)
    # --------------------------------------------------------------------------
    in0_core_coords = device.get_optimal_dram_bank_to_logical_worker_assignment(0)
    core2dram = {}
    for dram_bank_id, core_coords in enumerate(in0_core_coords):
        core2dram[core_coords] = dram_bank_id

    in0_num_cores = len(in0_core_coords)
    in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)

    ring2cores = {}
    for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
        ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in PAD_CORES else 0)

    num_dram_banks = 12
    dram_core_coords = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(in0_num_cores)]
    dram_core_range = [ttnn.CoreRange(dram_core_coord, dram_core_coord) for dram_core_coord in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    # --------------------------------------------------------------------------
    # Create input tokens
    # --------------------------------------------------------------------------
    input_tokens = torch.rand((total_tokens, H), dtype=torch.bfloat16) - 0.5

    # Create dummy routing (not used by kernel, but required by API)
    expert_indices = torch.zeros((total_tokens, top_k), dtype=torch.int16)
    expert_scores = torch.ones((total_tokens, top_k), dtype=torch.bfloat16) * 0.5

    logger.info(f"Input shape: [{total_tokens}, {H}], all tokens to all {E} experts")

    # --------------------------------------------------------------------------
    # Create weight tensors (same format as moe_gpt)
    # --------------------------------------------------------------------------
    torch_w0 = torch.rand((L, E, H, N), dtype=torch.bfloat16) - 0.5
    torch_w1 = torch.rand((L, E, H, N), dtype=torch.bfloat16) - 0.5
    torch_w2 = torch.rand((L, E, N, H), dtype=torch.bfloat16) - 0.5

    groups_per_core = MAX_W0_W1_TILES_PER_CORE // 2
    w0_w1_shard_height = L * E * groups_per_core * H
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

    torch_w0_w1_reordered = prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, H, N, ring2cores)
    tt_w0_w1 = ttnn.from_torch(
        torch_w0_w1_reordered,
        dtype=ttnn.bfloat4_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w0_w1_mem_config,
    )

    torch_w2_reordered = prepare_w2_tensor(torch_w2, L, E, N, H, ring2cores)
    tt_w2 = ttnn.from_torch(
        torch_w2_reordered,
        dtype=ttnn.bfloat4_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w2_mem_config,
    )

    # --------------------------------------------------------------------------
    # Create input tensor: WIDTH_SHARDED ROW_MAJOR on 3 tilize cores
    # Tilize cores at CoreRange({5,0},{5,2}): 3 cores in a row
    # Shard shape: [32, 960] (total_tokens, H/3)
    # --------------------------------------------------------------------------
    num_tilize_cores = 3
    tilize_core_range = ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(5, 2))

    input_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([tilize_core_range]),
        [total_tokens, H // num_tilize_cores],  # [32, 960]
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    # Create on DRAM first, then reshard to L1
    tt_input_dram = ttnn.from_torch(
        input_tokens,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_input = ttnn.to_memory_config(tt_input_dram, input_mem_config)
    ttnn.deallocate(tt_input_dram)

    logger.info(f"Input memory: {tt_input.memory_config()}")

    # Create dummy expert indices/scores tensors (required by API)
    tt_expert_indices = ttnn.from_torch(
        expert_indices.unsqueeze(0).unsqueeze(0).to(torch.int16),
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    tt_expert_scores = ttnn.from_torch(
        expert_scores.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # --------------------------------------------------------------------------
    # Run the fused operation
    # --------------------------------------------------------------------------
    tt_outputs = ttnn.experimental.moe_gpt_fused(
        tt_input,
        expert_indices=tt_expert_indices,
        expert_scores=tt_expert_scores,
        w0_w1_tensor=tt_w0_w1,
        w2_tensor=tt_w2,
        num_experts=E,
        layer_id=0,
        experts_per_device=E,
    )

    # --------------------------------------------------------------------------
    # Verify accuracy
    # --------------------------------------------------------------------------
    accuracy_metrics = {}

    if check_accuracy:
        # Output is BLOCK_SHARDED L1 ROW_MAJOR: [E * total_tokens, H] = [128, 2880]
        tt_output_torch = ttnn.to_torch(tt_outputs[0])
        logger.info(f"Output shape: {tt_output_torch.shape}, memory: {tt_outputs[0].memory_config()}")

        # Reshape to [E, total_tokens, H] for per-expert comparison
        tt_output_torch = tt_output_torch.reshape(E, total_tokens, H)

        for e in range(E):
            x = input_tokens.float()
            gate = x @ torch_w0[0, e].float()
            up = x @ torch_w1[0, e].float()
            intermediate = swiglu_reference(gate, up)
            ref_output = (intermediate @ torch_w2[0, e].float()).bfloat16()

            expert_output = tt_output_torch[e, :total_tokens, :]

            _pcc_passed, pcc_val = comp_pcc(ref_output, expert_output)
            allclose_passed, allclose_val = comp_allclose(ref_output, expert_output)

            logger.info(f"Expert {e}: PCC={pcc_val:.6f}")

            accuracy_metrics[e] = {
                "pcc": pcc_val,
                "allclose": allclose_passed,
                "allclose_val": allclose_val,
            }

    return accuracy_metrics


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            },
            id="dispatch_row",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("total_tokens", [32])
@pytest.mark.parametrize("check_accuracy", [True, False], ids=["check_accuracy_True", "check_accuracy_False"])
def test_moe_gpt_fused(device, total_tokens, check_accuracy):
    """
    Test the fused MoE GPT operation.

    Parametrized over:
        - total_tokens: 32 (1 chunk)
        - check_accuracy: Whether to verify PCC against torch reference
    """
    H = 2880
    N = 2880
    E = 4

    accuracy_metrics = run_test_moe_gpt_fused(device, total_tokens, H, N, E, check_accuracy)

    if check_accuracy:
        passing = True
        for expert_id, metrics in accuracy_metrics.items():
            if metrics["pcc"] < PCC_THRESHOLD:
                passing = False
                logger.warning(f"Expert {expert_id}: PCC={metrics['pcc']:.6f} (FAILED)")
            else:
                logger.info(f"Expert {expert_id}: PCC={metrics['pcc']:.6f} (Passed)")

        assert accuracy_metrics, "No accuracy metrics computed"
        assert passing, "Some experts did not pass the PCC check"
