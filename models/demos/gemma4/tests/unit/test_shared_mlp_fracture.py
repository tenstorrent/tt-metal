# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""SharedMLP under 2D weight fracture (galaxy one-instance) vs HF GeGLU.

The (8,4) mesh holds ONE weight copy: gate_up/down shard their inter dim over
rows*cols=32 chips (shard_mapper over both mesh axes), the residual stays replicated, and the
down partials are completed by one all-reduce per mesh axis.

    pytest models/demos/gemma4/tests/unit/test_shared_mlp_fracture.py -k 8x4
"""

import torch

import ttnn
from models.demos.gemma4.tt.shared_mlp import SharedMLP

from ...tests.test_factory import (
    TestFactory,
    compare_tensors,
    get_pcc_threshold,
    parametrize_batch_seq,
    parametrize_mesh_with_fabric,
)


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@parametrize_batch_seq()
def test_shared_mlp_fracture(batch_size, seq_len, mesh_device, reset_seeds, request):
    """One weight copy over 32 chips; output PCC vs HF Gemma4TextMLP."""
    from models.demos.gemma4.config import MeshConfig, ModeConfig

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_mlp = hf_layer.mlp

    state_dict = {
        "gate_proj.weight": hf_mlp.gate_proj.weight.data.clone(),
        "up_proj.weight": hf_mlp.up_proj.weight.data.clone(),
        "down_proj.weight": hf_mlp.down_proj.weight.data.clone(),
    }

    hf_config = TestFactory.create_hf_config()
    rows, cols = tuple(mesh_device.shape)
    mesh_config = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(tp=rows),
        tp_axis=0,
        weight_fracture=True,
    )
    assert mesh_config.fracture_ways == rows * cols

    tt_mlp = SharedMLP(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        mesh_config=mesh_config,
        ccl_manager=None,  # fractured all-reduce uses the sync per-axis path
        dtype=ttnn.bfloat16,
    )
    # One weight copy across the mesh: per-chip GeGLU shard is inter/ways.
    inter = tt_mlp.intermediate_size
    assert tt_mlp._inter_per_device * mesh_config.fracture_ways >= inter
    assert tt_mlp._inter_per_device < inter // rows  # strictly smaller than plain TP=rows

    x_torch = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_output = hf_mlp(x_torch.squeeze(0).float()).unsqueeze(0).to(torch.bfloat16)

    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_output = tt_mlp(x_tt)

    # Output must be identical (replicated) on every chip; check the two mesh
    # corners plus PCC vs HF on chip 0.
    device_tensors = ttnn.get_device_tensors(tt_output)
    out0 = ttnn.to_torch(device_tensors[0])
    out_last = ttnn.to_torch(device_tensors[-1])
    assert torch.equal(out0, out_last), "fractured MLP output not replicated across the mesh"

    passing, pcc_msg = compare_tensors(out0, ref_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"SharedMLP fractured ({rows}x{cols}) PCC too low: {pcc_msg}"
