# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 Router — uses HF Gemma4TextRouter as reference."""

import torch

import ttnn
from models.demos.gemma4.tt.router import Gemma4Router

from ...tests.test_factory import TestFactory, compare_tensors, parametrize_batch_seq, skip_if_not_moe


@skip_if_not_moe
@parametrize_batch_seq()
def test_router(batch_size, seq_len, device):
    """Test Router returns dense routing weights that match HF reference."""
    hf_text_config = TestFactory.create_hf_text_config(num_experts=8, top_k=4)
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_router = hf_layer.router

    state_dict = {
        "scale": hf_router.scale.data.clone(),
        "proj.weight": hf_router.proj.weight.data.clone(),
        "per_expert_scale": hf_router.per_expert_scale.data.clone(),
    }

    hf_config = TestFactory.create_hf_config()
    hf_config.num_experts = 8
    hf_config.top_k_experts = 4

    tt_router = Gemma4Router(mesh_device=device, hf_config=hf_config, state_dict=state_dict)

    x_torch = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    # HF reference: returns (probs, top_k_weights, top_k_indices)
    x_flat = x_torch.reshape(-1, hf_config.hidden_size).float()
    with torch.no_grad():
        _, ref_weights, ref_indices = hf_router(x_flat)

    # Build dense reference: scatter weights into [S, E]
    ref_dense = torch.zeros(seq_len, 8)
    ref_dense.scatter_(-1, ref_indices.to(torch.int64), ref_weights.float())

    # TT forward: returns dense routing [1, 1, S, E] on device
    x_tt = ttnn.from_torch(x_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_dense = tt_router(x_tt)
    tt_dense_torch = ttnn.to_torch(tt_dense).squeeze(0).squeeze(0).float()  # [S, E]

    # Compare dense routing weights
    # Dense routing PCC: lower for prefill since bf16 softmax+topk may select
    # different experts, leading to different positions having non-zero weights
    pcc_thresh = 0.90 if seq_len > 1 else 0.5
    passing, pcc_msg = compare_tensors(tt_dense_torch, ref_dense, pcc_threshold=pcc_thresh)
    assert passing, f"Router dense routing PCC too low: {pcc_msg}"
