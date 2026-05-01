# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 routed experts — on-device via sparse_matmul.

    pytest -k "1x1"        # single card (reduced experts)
    pytest -k "1x8"        # T3K with TP-sharded experts + CCL
    pytest -k "decode"     # decode only
    pytest -k "prefill"    # prefill only
"""

import torch

import ttnn
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.experts import Gemma4ExpertConfig, Gemma4Experts

from ...tests.test_factory import (
    TestFactory,
    compare_tensors,
    parametrize_batch_seq,
    parametrize_mesh_with_fabric,
    skip_if_not_moe,
)


@skip_if_not_moe
@parametrize_mesh_with_fabric()
@parametrize_batch_seq(
    configs=[(1, 1), (1, 32), (1, 128), (1, 1024)], ids=["decode", "prefill_32", "prefill_128", "prefill_1024"]
)
def test_experts(batch_size, seq_len, mesh_device, reset_seeds):
    """Test MoE experts against HF reference.

    1x1: Uses reduced experts (8) and bfloat16 weights for fast single-card test.
    1x8: Uses real model dimensions with TP-sharded bfloat8_b weights.
    """
    from models.demos.gemma4.config import MeshConfig, ModeConfig

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1

    if tp > 1:
        # Multi-device: real model dims, TP-sharded, HF reference layer for weights
        hf_text_config = TestFactory.create_hf_text_config()
        hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
        hf_experts = hf_layer.experts
        state_dict = {
            "gate_up_proj": hf_experts.gate_up_proj.data.clone(),
            "down_proj": hf_experts.down_proj.data.clone(),
        }
        hf_config = TestFactory.create_hf_config()
        weight_dtype = ttnn.bfloat8_b
    else:
        # Single device: reduced experts for speed
        num_experts_override = 8
        top_k_override = 4
        hf_text_config = TestFactory.create_hf_text_config(num_experts=num_experts_override, top_k=top_k_override)
        hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
        hf_experts = hf_layer.experts
        state_dict = {
            "gate_up_proj": hf_experts.gate_up_proj.data.clone(),
            "down_proj": hf_experts.down_proj.data.clone(),
        }
        hf_config = TestFactory.create_hf_config()
        hf_config.num_experts = num_experts_override
        hf_config.top_k_experts = top_k_override
        weight_dtype = ttnn.bfloat16

    num_experts = hf_config.num_experts
    top_k = hf_config.top_k_experts
    hidden_size = hf_config.hidden_size

    expert_config = Gemma4ExpertConfig(hf_config)
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    tt_experts = Gemma4Experts(
        mesh_device=mesh_device,
        config=expert_config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        weight_dtype=weight_dtype,
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
    x_flat = x.reshape(-1, hidden_size).float()
    with torch.no_grad():
        ref_output = hf_experts(x_flat, hf_indices, hf_weights)

    # TT forward
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    x_tt = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate)
    routing_tt = ttnn.from_torch(
        routing, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
    )
    tt_output = tt_experts(x_tt, routing_tt)

    tt_out_torch = (
        (ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output))
        .reshape(-1, hidden_size)
        .float()[:seq_len]
    )

    pcc_thresh = 0.85 if tp > 1 else 0.90
    passing, pcc_msg = compare_tensors(tt_out_torch, ref_output, pcc_threshold=pcc_thresh)
    assert passing, f"Experts (tp={tp}) PCC too low: {pcc_msg}"
