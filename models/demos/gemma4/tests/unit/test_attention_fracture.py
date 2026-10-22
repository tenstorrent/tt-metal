# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention under the galaxy one-instance fracture gate vs HF: heads shard over the tp axis and are
replicated across the lane axis with no axis-1 reduction, so per-column TP=8 attention must still match HF."""

import pytest
import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4.tt.ccl import CCLManager

from ...tests.test_factory import (
    TestFactory,
    build_hf_prefill_mask,
    compare_tensors,
    find_layer_idx,
    get_pcc_threshold,
    parametrize_mesh_with_fabric,
)


def _setup_fractured_attention(mesh_device, layer_idx, max_seq_len=128):
    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)

    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    rows, _cols = tuple(mesh_device.shape)
    mesh_config = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(tp=rows),
        tp_axis=0,
        weight_fracture=True,
    )
    ccl_manager = CCLManager(mesh_device, num_links=1)

    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
        create_kv_cache=False,
        max_batch_size=1,
        max_seq_len=max_seq_len,
    )
    return hf_text_config, hf_attn, config, tt_attn, mesh_config


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "global"])
@pytest.mark.parametrize("seq_len", [128, 1024], ids=["seq128", "seq1024"])
def test_attention_fracture_prefill(layer_type, seq_len, mesh_device, reset_seeds, request):
    """Prefill PCC vs HF: heads over the tp axis, replicated across the lane axis."""
    hf_text_config = TestFactory.create_hf_text_config()
    try:
        layer_idx = find_layer_idx(hf_text_config, layer_type)
    except ValueError:
        pytest.skip(f"No {layer_type} layer in this model")

    hf_text_config, hf_attn, config, tt_attn, _ = _setup_fractured_attention(
        mesh_device, layer_idx, max_seq_len=max(seq_len, 128)
    )

    x_torch = torch.randn(1, seq_len, config.hidden_size, dtype=torch.float32)

    hf_rope = TestFactory.create_hf_rope(hf_text_config, seq_len, layer_idx)
    sliding = config.sliding_window if config.is_sliding else None
    attn_mask = build_hf_prefill_mask(seq_len, sliding_window=sliding)
    with torch.no_grad():
        ref_output, _ = hf_attn(x_torch, position_embeddings=hf_rope, attention_mask=attn_mask, shared_kv_states=None)

    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max(seq_len, 128), layer_idx)
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_output = tt_attn(x_tt, rope_mats=(cos_tt, sin_tt), is_decode=False)

    device_tensors = ttnn.get_device_tensors(tt_output)
    out0 = ttnn.to_torch(device_tensors[0])
    out_last = ttnn.to_torch(device_tensors[-1])
    assert torch.equal(out0, out_last), "fractured attention output not replicated across the mesh"

    tt_output_torch = out0.squeeze(0).float()
    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, (
        f"Fractured attention prefill (layer_type={layer_type}, layer_idx={layer_idx}, "
        f"seq={seq_len}) PCC too low: {pcc_msg}"
    )
