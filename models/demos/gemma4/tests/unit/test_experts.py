# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 routed experts — on-device via sparse_matmul."""

import torch

import ttnn
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
