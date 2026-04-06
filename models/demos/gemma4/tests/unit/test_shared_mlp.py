# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 SharedMLP — uses HF Gemma4TextMLP as reference."""

import torch

import ttnn
from models.demos.gemma4.tt.shared_mlp import SharedMLP

from ...tests.test_factory import TestFactory, compare_tensors, parametrize_batch_seq


@parametrize_batch_seq()
def test_shared_mlp(batch_size, seq_len, device):
    """Test SharedMLP against HF Gemma4TextMLP (GeGLU)."""
    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_mlp = hf_layer.mlp  # Gemma4TextMLP

    state_dict = {
        "gate_proj.weight": hf_mlp.gate_proj.weight.data.clone(),
        "up_proj.weight": hf_mlp.up_proj.weight.data.clone(),
        "down_proj.weight": hf_mlp.down_proj.weight.data.clone(),
    }

    hf_config = TestFactory.create_hf_config()
    mesh_config = TestFactory.create_mesh_config((1, 1))

    tt_mlp = SharedMLP(
        mesh_device=device,
        hf_config=hf_config,
        state_dict=state_dict,
        mesh_config=mesh_config,
        dtype=ttnn.bfloat16,
    )

    x_torch = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_output = hf_mlp(x_torch.squeeze(0).float()).unsqueeze(0).to(torch.bfloat16)

    x_tt = ttnn.from_torch(x_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_output = tt_mlp(x_tt)
    tt_output_torch = ttnn.to_torch(tt_output)

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.99)
    assert passing, f"SharedMLP PCC too low: {pcc_msg}"
