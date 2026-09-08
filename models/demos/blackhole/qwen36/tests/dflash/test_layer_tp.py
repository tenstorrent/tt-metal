# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M7: one decoder layer vs ``Qwen3DFlashDecoderLayer.forward``, real weights.

Run for both a sliding layer and the full-attention layer, since they take different mask
paths through the attention inside.

Run:
    MESH_DEVICE=T3K pytest models/demos/blackhole/qwen36/tests/dflash/test_layer_tp.py -v -s
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.dflash.capture_fixtures import _drafter_hf_config, _resolve
from models.demos.blackhole.qwen36.tests.dflash.conftest import load_fixture
from models.demos.blackhole.qwen36.tests.test_factory import get_pcc_threshold, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.dflash import mask as dflash_mask
from models.demos.blackhole.qwen36.tt.dflash.ccl import ccl_topology
from models.demos.blackhole.qwen36.tt.dflash.layer import DFlashLayer
from models.demos.blackhole.qwen36.tt.dflash.rope import DFlashRoPE
from models.demos.blackhole.qwen36.tt.dflash.weights import layer_keys, load_layer_weights, read_state_dict
from models.tt_transformers.tt.ccl import TT_CCL

CTX = 128
BLOCK = 16


def _replicate(x, mesh):
    return ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("layer_idx", [0, 4], ids=["sliding", "full"])
def test_layer_tp(mesh_device, layer_idx, reset_seeds, ensure_gc, request, drafter_cfg):
    cfg = drafter_cfg
    fx = load_fixture(512)
    x = fx["noise_embedding"].float()[:, :BLOCK]
    ctx = fx["target_hidden"].float()[:, :CTX, : cfg.hidden_size]
    sd = read_state_dict(keys=layer_keys(layer_idx))

    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    from models.demos.blackhole.qwen36.reference.dflash.dflash import Qwen3DFlashDecoderLayer

    hf_cfg = _drafter_hf_config(_resolve("z-lab/Qwen3.6-27B-DFlash"))
    ref = Qwen3DFlashDecoderLayer(hf_cfg, layer_idx).eval()
    p = f"layers.{layer_idx}."
    with torch.no_grad():
        ref.self_attn.q_proj.weight.copy_(sd[p + "self_attn.q_proj.weight"])
        ref.self_attn.k_proj.weight.copy_(sd[p + "self_attn.k_proj.weight"])
        ref.self_attn.v_proj.weight.copy_(sd[p + "self_attn.v_proj.weight"])
        ref.self_attn.o_proj.weight.copy_(sd[p + "self_attn.o_proj.weight"])
        ref.self_attn.q_norm.weight.copy_(sd[p + "self_attn.q_norm.weight"])
        ref.self_attn.k_norm.weight.copy_(sd[p + "self_attn.k_norm.weight"])
        ref.input_layernorm.weight.copy_(sd[p + "input_layernorm.weight"])
        ref.post_attention_layernorm.weight.copy_(sd[p + "post_attention_layernorm.weight"])
        ref.mlp.gate_proj.weight.copy_(sd[p + "mlp.gate_proj.weight"])
        ref.mlp.up_proj.weight.copy_(sd[p + "mlp.up_proj.weight"])
        ref.mlp.down_proj.weight.copy_(sd[p + "mlp.down_proj.weight"])

    pos_emb = Qwen3RotaryEmbedding(hf_cfg)(x, torch.arange(CTX + BLOCK)[None])
    expected = ref(
        target_hidden=ctx,
        hidden_states=x,
        attention_mask=None,
        position_embeddings=pos_emb,
        past_key_value=None,
    )

    weights = load_layer_weights(mesh_device, cfg, layer_idx, state_dict=sd)
    tt_ccl = TT_CCL(mesh_device)
    layer = DFlashLayer(mesh_device, cfg, weights, layer_idx, tt_ccl, topology=ccl_topology(mesh_device))
    rope = DFlashRoPE(mesh_device, cfg, max_position=CTX + BLOCK)
    cos_k, sin_k = rope.tables_for(0, CTX + BLOCK)
    cos_q, sin_q = rope.tables_for(CTX, BLOCK)
    attn_mask = dflash_mask.to_device(cfg, layer_idx, mesh_device, ctx_len=CTX, block_len=BLOCK)

    tt_ctx = _replicate(ctx.reshape(1, 1, CTX, cfg.hidden_size), mesh_device)
    ctx_k, ctx_v = layer.project_context(tt_ctx)
    tt_out = layer.forward(
        _replicate(x.reshape(1, 1, BLOCK, cfg.hidden_size), mesh_device),
        ctx_k,
        ctx_v,
        rope,
        cos_q,
        sin_q,
        cos_k,
        sin_k,
        attn_mask,
    )

    stacked = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    actual = stacked[:1].reshape(1, BLOCK, cfg.hidden_size).float()

    passing, pcc = comp_pcc(expected, actual, get_pcc_threshold(request))
    logger.info(f"layer {layer_idx} ({'sliding' if layer_idx == 0 else 'full'}) PCC {pcc}")
    assert passing, f"layer {layer_idx} PCC {pcc}"
