# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 RMSNorm — uses HF Gemma4RMSNorm as reference.

    pytest -k "1x1"   # single card
    pytest -k "1x8"   # T3K (RMSNorm is replicated, no TP sharding)
"""

import torch

import ttnn
from models.demos.gemma4.tt.rms_norm import RMSNorm

from ...tests.test_factory import TestFactory, compare_tensors, parametrize_batch_seq, parametrize_mesh_with_fabric


@parametrize_mesh_with_fabric()
@parametrize_batch_seq()
def test_rms_norm_with_scale(batch_size, seq_len, mesh_device, reset_seeds):
    """Test RMSNorm (with_scale=True) against HF Gemma4RMSNorm."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    hf_config = TestFactory.create_hf_config()
    hidden_size = hf_config.hidden_size

    hf_norm = Gemma4RMSNorm(hidden_size, eps=hf_config.rms_norm_eps, with_scale=True)
    hf_norm.eval()
    state_dict = {"weight": hf_norm.weight.data.clone()}

    tt_norm = RMSNorm(mesh_device=mesh_device, hf_config=hf_config, state_dict=state_dict, with_scale=True)

    x_torch = torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        ref_output = hf_norm(x_torch.float()).to(torch.bfloat16)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_output = tt_norm.forward(x_tt)
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output)

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.999)
    assert passing, f"RMSNorm with_scale PCC too low: {pcc_msg}"


@parametrize_mesh_with_fabric()
@parametrize_batch_seq()
def test_rms_norm_without_scale(batch_size, seq_len, mesh_device, reset_seeds):
    """Test RMSNorm (with_scale=False) against HF Gemma4RMSNorm."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    hf_config = TestFactory.create_hf_config()
    hidden_size = hf_config.hidden_size

    hf_norm = Gemma4RMSNorm(hidden_size, eps=hf_config.rms_norm_eps, with_scale=False)
    hf_norm.eval()

    tt_norm = RMSNorm(mesh_device=mesh_device, hf_config=hf_config, state_dict={}, with_scale=False)

    x_torch = torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        ref_output = hf_norm(x_torch.float()).to(torch.bfloat16)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_output = tt_norm.forward(x_tt)
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output)

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=0.999)
    assert passing, f"RMSNorm without_scale PCC too low: {pcc_msg}"
