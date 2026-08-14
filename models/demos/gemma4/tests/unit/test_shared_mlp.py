# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 SharedMLP — uses HF Gemma4TextMLP as reference.

``test_shared_mlp`` runs on HF's constructor init; ``test_shared_mlp_real_weights``
runs the same comparison on the checkpoint's trained projections, where the
GeGLU intermediate has the magnitudes and outlier channels bf16/bfp8 storage
actually has to survive.

    pytest -k "1x1"   # single card
    pytest -k "1x8"   # T3K with CCL all-reduce
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.shared_mlp import SharedMLP

from ...tests.test_factory import (
    TestFactory,
    compare_tensors,
    get_pcc_threshold,
    load_real_weights_into,
    parametrize_batch_seq,
    parametrize_mesh_with_fabric,
)


def _run_shared_mlp(mesh_device, seq_len, hf_mlp, request, label):
    """Build the TT SharedMLP from ``hf_mlp``'s weights and PCC one forward."""
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager

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

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"SharedMLP {label} (tp={tp}) PCC too low: {pcc_msg}"


@parametrize_mesh_with_fabric()
@parametrize_batch_seq()
def test_shared_mlp(batch_size, seq_len, mesh_device, reset_seeds, request):
    """Test SharedMLP against HF Gemma4TextMLP (GeGLU)."""
    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    _run_shared_mlp(mesh_device, seq_len, hf_layer.mlp, request, "random-init")


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_idx", [0], ids=["layer0"])
@parametrize_batch_seq(configs=[(1, 1), (1, 128)], ids=["decode", "prefill_128"])
def test_shared_mlp_real_weights(batch_size, seq_len, layer_idx, mesh_device, reset_seeds, request):
    """Same PCC on the checkpoint's trained gate/up/down projections.

    The random-init test cannot see a weight-magnitude regression: HF's default
    Linear init is uniform and tiny, so the GeGLU intermediate never reaches the
    range where bf16 (or the shipped bfp8 override) loses mantissa. These are
    the weights the demo runs.

    Prefill capped at 128: ``test_shared_mlp``'s own 1024 bucket already fails
    on a single card for 12B (CBs want 2.60 MB against a 1.50 MB L1), and 12B
    layer 0 is double-wide — ``gate_proj`` is ``[15360, 3840]`` — so the real
    weights hit that ceiling sooner, not later.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP

    hf_text_config = TestFactory.create_hf_text_config()
    hf_mlp = Gemma4TextMLP(hf_text_config, layer_idx)
    state = load_real_weights_into(hf_mlp, f"layers.{layer_idx}.mlp")
    hf_mlp.eval()

    gate = state["gate_proj.weight"].float()
    logger.info(f"layers.{layer_idx}.mlp gate_proj {tuple(gate.shape)}: absmax={gate.abs().max():.4g}")
    _run_shared_mlp(mesh_device, seq_len, hf_mlp, request, f"real-weights layer{layer_idx}")
