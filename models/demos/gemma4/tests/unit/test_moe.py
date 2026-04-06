# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 MoE block — fully on device."""

import torch

import ttnn
from models.demos.gemma4.tt.moe import MoEBlock

from ...tests.test_factory import TestFactory, compare_tensors, parametrize_batch_seq


@parametrize_batch_seq(configs=[(1, 32)], ids=["prefill_32"])
def test_moe(batch_size, seq_len, device):
    """
    Test MoE end-to-end on device.

    Uses HF routing for the reference, TT router+experts for the test.
    Since router may select different experts due to bf16 precision,
    we use a relaxed PCC threshold.
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
        dtype=ttnn.bfloat16,
    )

    x_torch = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    # HF reference: router → experts
    x_flat = x_torch.reshape(-1, hf_config.hidden_size).float()
    with torch.no_grad():
        _, ref_weights, ref_indices = hf_router(x_flat)
        ref_output = hf_experts(x_flat, ref_indices, ref_weights)

    # TT forward: fully on device
    x_tt = ttnn.from_torch(x_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    # Router input = same as expert input for this test
    tt_output = moe(x_tt, x_tt)
    tt_output_torch = ttnn.to_torch(tt_output).reshape(-1, hf_config.hidden_size).float()[:seq_len]

    # Relaxed threshold: router may pick different experts due to bf16
    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.80)
    assert passing, f"MoE PCC too low: {pcc_msg}"
