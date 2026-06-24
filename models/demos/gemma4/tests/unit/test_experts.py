# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 routed experts — on-device via sparse_matmul.

    pytest -k "1x1"        # single card (reduced experts)
    pytest -k "1x8"        # T3K with TP-sharded experts + CCL
    pytest -k "decode"     # decode only
    pytest -k "prefill"    # prefill only
    pytest -k "per_expert" # per-expert PCC sweep
"""

import torch

import ttnn
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.experts import Gemma4ExpertConfig, Gemma4Experts

from ...tests.test_factory import (
    TestFactory,
    compare_tensors,
    get_pcc_threshold,
    parametrize_batch_seq,
    parametrize_mesh_with_fabric,
    skip_if_not_moe,
)


def _build_experts(mesh_device):
    """Construct the TT experts module and the matching HF reference experts.

    1x1 (tp=1): reduced experts (8) + bfloat16 weights for a fast single-card test.
    multi-device (tp>1): real model dims with TP-sharded bfloat8_b weights.

    Returns (tt_experts, hf_experts, num_experts, top_k, hidden_size).
    """
    from models.demos.gemma4.config import MeshConfig, ModeConfig

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1

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
    return tt_experts, hf_experts, hf_config.num_experts, hf_config.top_k_experts, hf_config.hidden_size


def _to_device(x, routing, mesh_device):
    """Replicate (hidden_states, dense_routing) onto the mesh."""
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    x_tt = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate)
    routing_tt = ttnn.from_torch(
        routing, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
    )
    return x_tt, routing_tt


def _from_device(tt_output, mesh_device, hidden_size, seq_len):
    """Read the experts output back to a [seq_len, hidden_size] torch tensor."""
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    t = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output)
    return t.reshape(-1, hidden_size).float()[:seq_len]


@skip_if_not_moe
@parametrize_mesh_with_fabric()
@parametrize_batch_seq()
def test_experts(batch_size, seq_len, mesh_device, reset_seeds, request):
    """Test MoE experts against HF reference.

    1x1: Uses reduced experts (8) and bfloat16 weights for fast single-card test.
    1x8: Uses real model dimensions with TP-sharded bfloat8_b weights.
    """
    tt_experts, hf_experts, num_experts, top_k, hidden_size = _build_experts(mesh_device)
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1

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
    x_tt, routing_tt = _to_device(x, routing, mesh_device)
    tt_output = tt_experts(x_tt, routing_tt)
    tt_out_torch = _from_device(tt_output, mesh_device, hidden_size, seq_len)

    passing, pcc_msg = compare_tensors(tt_out_torch, ref_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"Experts (tp={tp}) PCC too low: {pcc_msg}"


@skip_if_not_moe
@parametrize_mesh_with_fabric()
def test_experts_per_expert_pcc(mesh_device, reset_seeds, request):
    """Per-expert PCC: route every token to a single distinct expert and check
    each expert's output individually.

    test_experts mixes every expert into one aggregate PCC, so a regression
    isolated to a single expert (a TP-sharding / weight-indexing / routing bug
    that corrupts one expert's weights or selection) can hide behind the others.
    Here token ``s`` routes 100% to expert ``s % num_experts``; we then compute
    PCC per expert and fail on the worst, surfacing exactly which expert
    regressed. One device run covers all experts (prefill path).
    """
    from models.common.utility_functions import comp_pcc

    tt_experts, hf_experts, num_experts, top_k, hidden_size = _build_experts(mesh_device)
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1

    # One run covering every expert: seq rounded up to a multiple of 32 and
    # >= num_experts so each expert is exercised by at least one token.
    seq_len = max(32, ((num_experts + 31) // 32) * 32)
    x = torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16)

    hf_indices = torch.zeros(seq_len, top_k, dtype=torch.int64)
    hf_weights = torch.zeros(seq_len, top_k, dtype=torch.float32)
    routing = torch.zeros(1, 1, seq_len, num_experts, dtype=torch.float32)
    for s in range(seq_len):
        e = s % num_experts
        hf_indices[s, 0] = e
        # Pad the remaining top_k slots with distinct valid experts at weight 0
        # so HF's gather sees a well-formed (seq, top_k) tensor; zero weight means
        # they contribute nothing — the output is 100% expert e.
        for j in range(1, top_k):
            hf_indices[s, j] = (e + j) % num_experts
        hf_weights[s, 0] = 1.0
        routing[0, 0, s, e] = 1.0
    routing = routing.to(torch.bfloat16)

    # HF reference
    x_flat = x.reshape(-1, hidden_size).float()
    with torch.no_grad():
        ref_output = hf_experts(x_flat, hf_indices, hf_weights)

    # TT forward
    x_tt, routing_tt = _to_device(x, routing, mesh_device)
    tt_output = tt_experts(x_tt, routing_tt)
    tt_out_torch = _from_device(tt_output, mesh_device, hidden_size, seq_len)

    # Per-expert PCC: group rows by the expert they routed to.
    threshold = get_pcc_threshold(request)
    worst_expert, worst_pcc = -1, 1.0
    failures = []
    for e in range(num_experts):
        rows = torch.arange(e, seq_len, num_experts)
        passing_e, pcc_e = comp_pcc(ref_output[rows], tt_out_torch[rows], threshold)
        pcc_e = float(pcc_e)
        if pcc_e < worst_pcc:
            worst_pcc, worst_expert = pcc_e, e
        if not passing_e:
            failures.append((e, pcc_e))

    from loguru import logger

    logger.info(
        f"Per-expert PCC (tp={tp}, num_experts={num_experts}, seq={seq_len}): "
        f"worst expert {worst_expert} @ {worst_pcc:.5f} (threshold={threshold})"
    )
    assert not failures, (
        f"Per-expert PCC below {threshold} (tp={tp}) for "
        + ", ".join(f"expert {e} (pcc={p:.5f})" for e, p in failures)
        + f"; worst expert {worst_expert} @ {worst_pcc:.5f}"
    )
