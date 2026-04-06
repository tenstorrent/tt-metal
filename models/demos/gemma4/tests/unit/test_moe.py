# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 MoE block — uses HF router + experts as reference."""

import torch

from models.demos.gemma4.tt.moe import MoEBlock

from ...tests.test_factory import TestFactory, compare_tensors, parametrize_batch_seq


@parametrize_batch_seq()
def test_moe(batch_size, seq_len, device):
    """
    Test MoE: use HF routing for both HF and TT experts to isolate expert accuracy.

    This tests that TT experts produce the same output as HF experts given
    identical routing decisions, avoiding false failures from bf16 routing divergence.
    """
    num_experts = 8
    top_k = 4
    hf_text_config = TestFactory.create_hf_text_config(num_experts=num_experts, top_k=top_k)
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_router = hf_layer.router
    hf_experts = hf_layer.experts

    state_dict = {
        "router.scale": hf_router.scale.data.clone(),
        "router.proj.weight": hf_router.proj.weight.data.clone(),
        "router.per_expert_scale": hf_router.per_expert_scale.data.clone(),
        "experts.gate_up_proj": hf_experts.gate_up_proj.data.clone(),
        "experts.down_proj": hf_experts.down_proj.data.clone(),
    }

    hf_config = TestFactory.create_hf_config()
    hf_config.num_experts = num_experts
    hf_config.top_k_experts = top_k

    moe = MoEBlock(
        mesh_device=device,
        hf_config=hf_config,
        state_dict=state_dict,
        ccl_manager=None,
        mesh_config=TestFactory.create_mesh_config((1, 1)),
    )

    x_torch = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)
    x_flat = x_torch.reshape(-1, hf_config.hidden_size)

    # Get routing from HF (ground truth)
    with torch.no_grad():
        _, ref_weights, ref_indices = hf_router(x_flat.float())
        # HF experts with HF routing
        ref_output = hf_experts(x_flat.float(), ref_indices, ref_weights)

    # TT experts with same HF routing (bypass TT router)
    tt_output = moe.experts(x_flat, ref_indices, ref_weights.to(torch.bfloat16))

    passing, pcc_msg = compare_tensors(tt_output.float(), ref_output.float(), pcc_threshold=0.95)
    assert passing, f"MoE experts PCC too low: {pcc_msg}"
