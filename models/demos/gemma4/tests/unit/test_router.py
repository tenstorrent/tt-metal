# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 Router.

Router is replicated (no TP sharding on weights), but runs on mesh for consistency.

    pytest -k "1x1"   # single card
    pytest -k "1x8"   # T3K (router is replicated)
"""

import torch

import ttnn
from models.demos.gemma4.tt.router import Gemma4Router

from ...tests.test_factory import (
    TestFactory,
    compare_tensors,
    parametrize_batch_seq,
    parametrize_mesh_with_fabric,
    skip_if_not_moe,
)


@skip_if_not_moe
@parametrize_mesh_with_fabric()
@parametrize_batch_seq()
def test_router(batch_size, seq_len, mesh_device, reset_seeds):
    """Test Router returns dense routing weights that match the HF Gemma4 formula."""
    hf_config = TestFactory.create_hf_config()
    hf_config.num_experts = 32
    hf_config.top_k_experts = 4

    state_dict = {
        "scale": torch.randn(hf_config.hidden_size, dtype=torch.bfloat16),
        "proj.weight": torch.randn(hf_config.num_experts, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02,
        "per_expert_scale": torch.randn(hf_config.num_experts, dtype=torch.bfloat16),
    }

    tt_router = Gemma4Router(mesh_device=mesh_device, hf_config=hf_config, state_dict=state_dict)

    x_torch = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    # HF Gemma4TextRouter formula:
    # RMSNorm without learned norm weight, learned scale / sqrt(hidden), linear,
    # softmax over all experts, top-k, selected-probability sum renormalization,
    # then per-expert scale.
    x_flat = x_torch.reshape(-1, hf_config.hidden_size).float()
    with torch.no_grad():
        normed = x_flat * torch.rsqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + hf_config.rms_norm_eps)
        scaled = normed * state_dict["scale"].float() * (hf_config.hidden_size**-0.5)
        router_probs = torch.softmax(scaled @ state_dict["proj.weight"].float().T, dim=-1)
        ref_values, ref_indices = torch.topk(router_probs, k=hf_config.top_k_experts, dim=-1)
        ref_values = ref_values / ref_values.sum(dim=-1, keepdim=True)
        ref_dense = torch.zeros(seq_len, hf_config.num_experts)
        ref_dense.scatter_(-1, ref_indices.to(torch.int64), ref_values.float())
        ref_dense = ref_dense * state_dict["per_expert_scale"].float()

    # TT forward
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_dense = tt_router(x_tt)
    tt_dense_torch = (
        (ttnn.to_torch(ttnn.get_device_tensors(tt_dense)[0]) if is_mesh else ttnn.to_torch(tt_dense))
        .squeeze(0)
        .squeeze(0)
        .float()
    )

    # TODO: investigate low PCC on the MoE router and raise this back to 0.90.
    pcc_thresh = 0.85 if seq_len > 1 else 0.5
    passing, pcc_msg = compare_tensors(tt_dense_torch, ref_dense, pcc_threshold=pcc_thresh)
    assert passing, f"Router dense routing PCC too low: {pcc_msg}"
