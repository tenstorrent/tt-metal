# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 MoE block — fully on device.

    pytest -k "1x1"   # single card
    pytest -k "1x8"   # T3K with TP-sharded experts + CCL
"""

import torch

import ttnn
from models.demos.gemma4.tt.moe import MoEBlock

from ...tests.test_factory import (
    TestFactory,
    compare_tensors,
    parametrize_batch_seq,
    parametrize_mesh_with_fabric,
    skip_if_not_moe,
)


@skip_if_not_moe
@parametrize_mesh_with_fabric()
@parametrize_batch_seq(configs=[(1, 32)], ids=["prefill_32"])
def test_moe(batch_size, seq_len, mesh_device, reset_seeds):
    """Test MoE end-to-end on device against HF reference.

    Uses HF routing for the reference, TT router+experts for the test.
    Relaxed PCC: router may select different experts due to bf16 precision.
    """
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager

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

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    moe = MoEBlock(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        dtype=ttnn.bfloat16,
    )

    x_torch = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    # HF reference
    x_flat = x_torch.reshape(-1, hf_config.hidden_size).float()
    with torch.no_grad():
        _, ref_weights, ref_indices = hf_router(x_flat)
        ref_output = hf_experts(x_flat, ref_indices, ref_weights)

    # TT forward
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_output = moe(x_tt, x_tt)
    tt_output_torch = (
        (ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output))
        .reshape(-1, hf_config.hidden_size)
        .float()[:seq_len]
    )

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.80)
    assert passing, f"MoE (tp={tp}) PCC too low: {pcc_msg}"
