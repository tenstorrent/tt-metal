# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 Router — uses HF Gemma4TextRouter as reference."""

import torch

import ttnn
from models.demos.gemma4.tt.router import Gemma4Router

from ...tests.test_factory import TestFactory, compare_tensors, parametrize_batch_seq


@parametrize_batch_seq()
def test_router(batch_size, seq_len, device):
    """Test Router against HF Gemma4TextRouter (softmax-then-topk)."""
    hf_text_config = TestFactory.create_hf_text_config(num_experts=8, top_k=4)
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_router = hf_layer.router  # Gemma4TextRouter

    # Extract state_dict for our TT router
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

    # HF reference: expects [tokens, hidden_size]
    x_flat = x_torch.reshape(-1, hf_config.hidden_size).float()
    with torch.no_grad():
        _, ref_weights, ref_indices = hf_router(x_flat)

    # TT forward
    x_tt = ttnn.from_torch(x_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_weights, tt_indices = tt_router(x_tt)

    # Compare weights with PCC (indices may differ due to bf16 near-ties)
    pcc_thresh = 0.95 if seq_len > 1 else 0.5
    passing, pcc_msg = compare_tensors(
        tt_weights.float(), ref_weights.to(torch.bfloat16).float(), pcc_threshold=pcc_thresh
    )
    assert passing, f"Router weights PCC too low: {pcc_msg}"
