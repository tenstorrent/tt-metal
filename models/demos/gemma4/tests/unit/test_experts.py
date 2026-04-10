# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 routed experts — on-device via sparse_matmul."""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.experts import Gemma4ExpertConfig, Gemma4Experts

from ...tests.test_factory import TestFactory, compare_tensors, parametrize_batch_seq, skip_if_not_moe


@skip_if_not_moe
@parametrize_batch_seq(configs=[(1, 1), (1, 32)], ids=["decode", "prefill_32"])
def test_experts(batch_size, seq_len, device):
    """Test on-device experts (decode + prefill) against HF Gemma4TextExperts."""
    num_experts = 8
    top_k = 4
    hf_text_config = TestFactory.create_hf_text_config(num_experts=num_experts, top_k=top_k)
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_experts = hf_layer.experts

    state_dict = {
        "gate_up_proj": hf_experts.gate_up_proj.data.clone(),
        "down_proj": hf_experts.down_proj.data.clone(),
    }

    hf_config = TestFactory.create_hf_config()
    hf_config.num_experts = num_experts
    hf_config.top_k_experts = top_k
    expert_config = Gemma4ExpertConfig(hf_config)

    tt_experts = Gemma4Experts(
        mesh_device=device,
        config=expert_config,
        state_dict=state_dict,
        ccl_manager=None,
        mesh_config=None,
        program_config=None,
        weight_dtype=ttnn.bfloat16,
    )

    # Create input + dense routing on device
    x = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    # Create random routing: pick top_k experts per token (float32 to avoid bf16 zeros)
    hf_indices = torch.zeros(seq_len, top_k, dtype=torch.int64)
    hf_weights = torch.zeros(seq_len, top_k, dtype=torch.float32)
    routing = torch.zeros(1, 1, seq_len, num_experts, dtype=torch.float32)
    for s in range(seq_len):
        experts_selected = torch.randperm(num_experts)[:top_k]
        weights = torch.rand(top_k) + 0.1  # avoid zeros
        weights = weights / weights.sum()
        hf_indices[s] = experts_selected
        hf_weights[s] = weights
        routing[0, 0, s, experts_selected] = weights
    routing = routing.to(torch.bfloat16)

    # HF reference
    x_flat = x.reshape(-1, hf_config.hidden_size).float()

    with torch.no_grad():
        ref_output = hf_experts(x_flat, hf_indices, hf_weights)

    # TT forward
    x_tt = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    routing_tt = ttnn.from_torch(routing, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_output = tt_experts(x_tt, routing_tt)
    tt_output_torch = ttnn.to_torch(tt_output).reshape(-1, hf_config.hidden_size).float()

    # Trim to seq_len (may be padded to 32)
    tt_output_torch = tt_output_torch[:seq_len]

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.90)
    assert passing, f"Experts PCC too low: {pcc_msg}"


@skip_if_not_moe
@parametrize_batch_seq(configs=[(1, 1), (1, 32)], ids=["decode", "prefill_32"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_experts_t3k(batch_size, seq_len, mesh_device):
    """Test MoE experts on T3K (TP=8) with real A4B dimensions.

    Reproduces the exact sparse_matmul configuration from the full model demo:
    128 experts, hidden=2816, intermediate=704, TP=8 → 96 per device (padded from 88).
    Uses the HF Gemma4TextExperts module as reference.
    """
    logger.info(f"[test_experts_t3k] START batch_size={batch_size} seq_len={seq_len} mesh={mesh_device.shape}")

    logger.info("[test_experts_t3k] Creating HF reference layer...")
    t0 = time.perf_counter()
    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_experts = hf_layer.experts
    logger.info(f"[test_experts_t3k] HF reference layer created in {time.perf_counter()-t0:.1f}s")

    state_dict = {
        "gate_up_proj": hf_experts.gate_up_proj.data.clone(),
        "down_proj": hf_experts.down_proj.data.clone(),
    }

    hf_config = TestFactory.create_hf_config()
    num_experts = hf_config.num_experts  # 128
    top_k = hf_config.top_k_experts  # 8
    hidden_size = hf_config.hidden_size  # 2816
    intermediate_size = hf_config.moe_intermediate_size  # 704
    tp = mesh_device.shape[1]
    logger.info(
        f"[test_experts_t3k] Model dims: E={num_experts} top_k={top_k} H={hidden_size} "
        f"I={intermediate_size} tp={tp} I/tp={intermediate_size//tp} "
        f"I/tp tile-aligned={intermediate_size//tp % 32 == 0}"
    )

    expert_config = Gemma4ExpertConfig(hf_config)
    mesh_config = TestFactory.create_mesh_config(mesh_shape=mesh_device.shape)
    ccl_manager = CCLManager(mesh_device, num_links=1)

    logger.info("[test_experts_t3k] Loading expert weights to device...")
    t0 = time.perf_counter()
    tt_experts = Gemma4Experts(
        mesh_device=mesh_device,
        config=expert_config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        weight_dtype=ttnn.bfloat8_b,
    )
    logger.info(
        f"[test_experts_t3k] Weights loaded in {time.perf_counter()-t0:.1f}s — "
        f"gate_proj={tt_experts.weights.gate_proj.shape} "
        f"down_proj={tt_experts.weights.down_proj.shape} "
        f"intermediate_per_device={tt_experts.weights.intermediate_size_per_device}"
    )

    # Create input + dense routing
    x = torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16)

    hf_indices = torch.zeros(seq_len, top_k, dtype=torch.int64)
    hf_weights = torch.zeros(seq_len, top_k, dtype=torch.float32)
    routing = torch.zeros(1, 1, seq_len, num_experts, dtype=torch.float32)
    for s in range(seq_len):
        experts_selected = torch.randperm(num_experts)[:top_k]
        weights = torch.rand(top_k) + 0.1
        weights = weights / weights.sum()
        hf_indices[s] = experts_selected
        hf_weights[s] = weights
        routing[0, 0, s, experts_selected] = weights
    routing = routing.to(torch.bfloat16)

    # HF reference
    logger.info("[test_experts_t3k] Running HF reference forward...")
    t0 = time.perf_counter()
    x_flat = x.reshape(-1, hidden_size).float()
    with torch.no_grad():
        ref_output = hf_experts(x_flat, hf_indices, hf_weights)
    logger.info(f"[test_experts_t3k] HF reference done in {time.perf_counter()-t0:.3f}s")

    # TT forward
    logger.info("[test_experts_t3k] Sending inputs to device...")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    x_tt = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate)
    routing_tt = ttnn.from_torch(
        routing, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
    )
    logger.info(f"[test_experts_t3k] Inputs on device: x={x_tt.shape} routing={routing_tt.shape}")

    logger.info("[test_experts_t3k] Running TT expert forward...")
    t0 = time.perf_counter()
    tt_output = tt_experts(x_tt, routing_tt)
    logger.info(f"[test_experts_t3k] TT forward done in {time.perf_counter()-t0:.3f}s output={tt_output.shape}")

    logger.info("[test_experts_t3k] Reading output back to CPU...")
    tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]).reshape(-1, hidden_size).float()
    tt_out_torch = tt_out_torch[:seq_len]

    passing, pcc_msg = compare_tensors(tt_out_torch, ref_output, pcc_threshold=0.85)
    logger.info(f"[test_experts_t3k] DONE {'PASS' if passing else 'FAIL'}: {pcc_msg}")
    assert passing, f"T3K experts PCC too low: {pcc_msg}"
