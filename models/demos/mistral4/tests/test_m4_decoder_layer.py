# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end PCC for the Mistral-Small-4 decoder layer via the model-local module classes.

Exercises TtMistral4DecoderLayer (input_layernorm -> MLA -> +residual -> post_attention_layernorm
-> MoE -> +residual) on real layer-0 weights, PCC'd vs the reference layer output golden. Feeds
the reference RoPE cos/sin (decoupled from YaRN regeneration).
"""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import capture_golden, load_m4_text_reference
from models.demos.mistral4.tt.mistral4_text import TtMistral4DecoderLayer


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_decoder_layer(mesh_device, reset_seeds):
    pcc_required = 0.99
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config

    model, _, _ = load_m4_text_reference(ckpt, n_layers=1)
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 32))
    g = capture_golden(model, ids)
    layer_sd = {k: v for k, v in model.model.layers[0].named_parameters()}
    B, S, rope = g["hidden_in"].shape[0], g["hidden_in"].shape[1], cfg.qk_rope_head_dim

    layer = TtMistral4DecoderLayer(mesh_device, layer_sd, cfg, cfg.rms_norm_eps)

    def _from(t, shape=None):
        return ttnn.from_torch(
            (t if shape is None else t.view(shape)).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    x = _from(g["hidden_in"])
    cos = _from(g["rope_cos"], (B, 1, S, rope))
    sin = _from(g["rope_sin"], (B, 1, S, rope))

    out = layer(x, cos, sin)
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:B]
    passing, msg = comp_pcc(g["layer_out"], out_t, pcc_required)
    logger.info(f"Mistral-Small-4 decoder-layer PCC: {msg}")
    assert passing, f"decoder-layer PCC below {pcc_required}: {msg}"
