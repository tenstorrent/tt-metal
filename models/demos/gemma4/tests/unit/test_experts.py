# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 routed experts — uses HF Gemma4TextExperts as reference."""

import torch

from models.demos.gemma4.tt.experts import Gemma4ExpertConfig, Gemma4Experts

from ...tests.test_factory import TestFactory, compare_tensors, parametrize_batch_seq


@parametrize_batch_seq()
def test_experts(batch_size, seq_len):
    """Test experts forward (CPU) against HF Gemma4TextExperts."""
    num_experts = 8
    top_k = 4
    hf_text_config = TestFactory.create_hf_text_config(num_experts=num_experts, top_k=top_k)
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_experts = hf_layer.experts  # Gemma4TextExperts

    state_dict = {
        "gate_up_proj": hf_experts.gate_up_proj.data.clone(),
        "down_proj": hf_experts.down_proj.data.clone(),
    }

    hf_config = TestFactory.create_hf_config()
    hf_config.num_experts = num_experts
    hf_config.top_k_experts = top_k
    expert_config = Gemma4ExpertConfig(hf_config)

    tt_experts = Gemma4Experts(
        mesh_device=None,
        config=expert_config,
        state_dict=state_dict,
        ccl_manager=None,
        mesh_config=None,
        program_config=None,
    )

    # Create input + routing
    x = torch.randn(seq_len, hf_config.hidden_size, dtype=torch.bfloat16)
    top_k_indices = torch.randint(0, num_experts, (seq_len, top_k))
    top_k_weights = torch.rand(seq_len, top_k, dtype=torch.bfloat16)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    # HF reference
    with torch.no_grad():
        ref_output = hf_experts(x.float(), top_k_indices, top_k_weights.float())

    # TT forward (CPU-based)
    tt_output = tt_experts(x, top_k_indices, top_k_weights)

    passing, pcc_msg = compare_tensors(tt_output.float(), ref_output.float(), pcc_threshold=0.999)
    assert passing, f"Experts PCC too low: {pcc_msg}"
