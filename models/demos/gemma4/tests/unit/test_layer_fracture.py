# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Dense decoder layer under 2D weight fracture (galaxy one-instance): prefill PCC vs the HF layer plus
cross-mesh replication of the output, with the MLP fractured 32-way and attention heads over axis 0."""

import pytest
import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

from ...tests.test_factory import (
    TestFactory,
    build_hf_prefill_mask,
    compare_tensors,
    find_layer_idx,
    get_pcc_threshold,
    parametrize_mesh_with_fabric,
)
from .test_layer import _create_hf_rope, _hf_state_to_tt_state


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "global"])
@pytest.mark.parametrize("seq_len", [128], ids=["seq128"])
def test_layer_fracture_prefill(layer_type, seq_len, mesh_device, reset_seeds, request):
    hf_text_config = TestFactory.create_hf_text_config()
    if getattr(hf_text_config, "num_experts", 0):
        pytest.skip("fracture layer test targets the dense (SharedMLP) variant")
    try:
        layer_idx = find_layer_idx(hf_text_config, layer_type)
    except ValueError:
        pytest.skip(f"No {layer_type} layer in this model")

    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    tt_state = _hf_state_to_tt_state(hf_layer.state_dict(), layer_idx)
    model_args = Gemma4ModelArgs.from_hf_config(hf_text_config)
    attn_cfg = Gemma4AttentionConfig(model_args, layer_idx)

    rows, _cols = tuple(mesh_device.shape)
    mesh_config = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(tp=rows),
        tp_axis=0,
        weight_fracture=True,
    )
    ccl_manager = CCLManager(mesh_device, num_links=1)

    tt_layer = Gemma4DecoderLayer(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=tt_state,
        layer_idx=layer_idx,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=seq_len,
        max_local_batch_size=1,
    )

    x_torch = torch.randn(1, seq_len, model_args.hidden_size, dtype=torch.float32)
    hf_rope = _create_hf_rope(hf_text_config, seq_len, layer_idx)
    sliding = attn_cfg.sliding_window if attn_cfg.is_sliding else None
    attn_mask = build_hf_prefill_mask(seq_len, sliding_window=sliding)
    with torch.no_grad():
        hf_output = hf_layer(x_torch, position_embeddings=hf_rope, attention_mask=attn_mask)

    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max(seq_len, 128), layer_idx)
    tt_output = tt_layer(
        x_tt,
        rope_mats=(cos_tt, sin_tt),
        position_idx=None,
        page_table=None,
        kv_cache=None,
        is_decode=False,
    )

    device_tensors = ttnn.get_device_tensors(tt_output)
    out0 = ttnn.to_torch(device_tensors[0])
    out_last = ttnn.to_torch(device_tensors[-1])
    assert torch.equal(out0, out_last), "fractured layer output not replicated across the mesh"

    tt_output_torch = out0.squeeze(0).float()
    passing, pcc_msg = compare_tensors(tt_output_torch, hf_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"Fractured layer (layer_type={layer_type}, idx={layer_idx}) PCC too low: {pcc_msg}"
