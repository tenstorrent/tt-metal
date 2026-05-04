# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 SharedMLP — uses HF Gemma4TextMLP as reference.

    pytest -k "1x1"   # single card
    pytest -k "1x8"   # T3K with CCL all-reduce
"""

import torch

import ttnn
from models.demos.gemma4.tt.shared_mlp import SharedMLP

from ...tests.test_factory import TestFactory, compare_tensors, parametrize_batch_seq, parametrize_mesh_with_fabric


@parametrize_mesh_with_fabric()
@parametrize_batch_seq()
def test_shared_mlp(batch_size, seq_len, mesh_device, reset_seeds):
    """Test SharedMLP against HF Gemma4TextMLP (GeGLU)."""
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_mlp = hf_layer.mlp

    state_dict = {
        "gate_proj.weight": hf_mlp.gate_proj.weight.data.clone(),
        "up_proj.weight": hf_mlp.up_proj.weight.data.clone(),
        "down_proj.weight": hf_mlp.down_proj.weight.data.clone(),
    }

    hf_config = TestFactory.create_hf_config()
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    tt_mlp = SharedMLP(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
    )

    x_torch = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_output = hf_mlp(x_torch.squeeze(0).float()).unsqueeze(0).to(torch.bfloat16)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_output = tt_mlp(x_tt)
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output)

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.99)
    assert passing, f"SharedMLP (tp={tp}) PCC too low: {pcc_msg}"
