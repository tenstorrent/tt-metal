# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 routed experts — on-device via sparse_matmul.

Filter by mesh shape:
    pytest -k "1x2"          # N300 / TP=2
    pytest -k "1x8"          # T3K  / TP=8
    pytest -k "decode"       # decode only
    pytest -k "prefill"      # prefill only
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
@parametrize_batch_seq(configs=[(1, 1), (1, 32)], ids=["decode", "prefill_32"])
def test_experts(batch_size, seq_len, device):
    """Test on-device experts (decode + prefill) on single device against HF reference."""
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

    x = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

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

    x_flat = x.reshape(-1, hf_config.hidden_size).float()
    with torch.no_grad():
        ref_output = hf_experts(x_flat, hf_indices, hf_weights)

    x_tt = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    routing_tt = ttnn.from_torch(routing, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_output = tt_experts(x_tt, routing_tt)
    tt_output_torch = ttnn.to_torch(tt_output).reshape(-1, hf_config.hidden_size).float()
    tt_output_torch = tt_output_torch[:seq_len]

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.90)
    assert passing, f"Experts PCC too low: {pcc_msg}"


@skip_if_not_moe
@parametrize_batch_seq(configs=[(1, 1), (1, 32)], ids=["decode", "prefill_32"])
@parametrize_mesh_with_fabric()
def test_experts_tp(batch_size, seq_len, mesh_device):
    """Test MoE experts with TP on multi-device mesh against HF reference.

    Uses real model dimensions (128 experts, 704 intermediate, 2816 hidden).
    TP-shards expert weights with tile-aligned padding.

    Filter by mesh shape:
        pytest -k "1x2"   # N300 / TP=2
        pytest -k "1x8"   # T3K  / TP=8
    """
    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_experts = hf_layer.experts

    state_dict = {
        "gate_up_proj": hf_experts.gate_up_proj.data.clone(),
        "down_proj": hf_experts.down_proj.data.clone(),
    }

    hf_config = TestFactory.create_hf_config()
    num_experts = hf_config.num_experts
    top_k = hf_config.top_k_experts
    hidden_size = hf_config.hidden_size

    expert_config = Gemma4ExpertConfig(hf_config)
    mesh_config = TestFactory.create_mesh_config(mesh_shape=mesh_device.shape)
    ccl_manager = CCLManager(mesh_device, num_links=1)

    tt_experts = Gemma4Experts(
        mesh_device=mesh_device,
        config=expert_config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        weight_dtype=ttnn.bfloat8_b,
    )

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

    x_flat = x.reshape(-1, hidden_size).float()
    with torch.no_grad():
        ref_output = hf_experts(x_flat, hf_indices, hf_weights)

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    x_tt = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate)
    routing_tt = ttnn.from_torch(
        routing, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
    )
    tt_output = tt_experts(x_tt, routing_tt)

    tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]).reshape(-1, hidden_size).float()
    tt_out_torch = tt_out_torch[:seq_len]

    passing, pcc_msg = compare_tensors(tt_out_torch, ref_output, pcc_threshold=0.85)
    assert passing, f"Experts TP PCC too low: {pcc_msg}"
