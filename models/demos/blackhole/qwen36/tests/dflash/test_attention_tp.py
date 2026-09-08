# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M6: dual-source attention vs ``Qwen3DFlashAttention.forward``, real weights.

Graded against the reference module itself -- not a reimplementation -- so the oracle also
supplies the mask, the RoPE tail-slice and the k-norm-over-the-concatenation behaviour.

Parametrized over a **sliding** layer (causal + windowed) and the **full** layer
(bidirectional, no mask). Running only one would leave half the mask logic ungraded, and
the two differ in exactly the way that is easy to get backwards.

Run:
    MESH_DEVICE=T3K pytest models/demos/blackhole/qwen36/tests/dflash/test_attention_tp.py -v -s
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
from models.demos.blackhole.qwen36.tt.dflash.attention import DFlashAttention
from models.demos.blackhole.qwen36.tt.dflash.ccl import ccl_topology
from models.demos.blackhole.qwen36.tt.dflash.rope import DFlashRoPE
from models.demos.blackhole.qwen36.tt.dflash.weights import layer_keys, load_layer_weights, read_state_dict
from models.tt_transformers.tt.ccl import TT_CCL

CTX = 128
BLOCK = 16


def _replicate(x: torch.Tensor, mesh):
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
def test_attention_tp(mesh_device, layer_idx, reset_seeds, ensure_gc, request, drafter_cfg):
    cfg = drafter_cfg
    tp = mesh_device.get_num_devices()
    fx = load_fixture(512)

    # Real activations: the block is the target's real embeddings; the context stands in for
    # the encoder output at the right width and scale (graded identically on both sides).
    x = fx["noise_embedding"].float()[:, :BLOCK]  # [1, block, 5120]
    ctx = fx["target_hidden"].float()[:, :CTX, : cfg.hidden_size]  # [1, ctx, 5120]

    sd = read_state_dict(keys=layer_keys(layer_idx))

    # ---- oracle: the reference attention module with the real weights -----------------
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    from models.demos.blackhole.qwen36.reference.dflash.dflash import Qwen3DFlashAttention

    hf_cfg = _drafter_hf_config(_resolve("z-lab/Qwen3.6-27B-DFlash"))
    ref = Qwen3DFlashAttention(hf_cfg, layer_idx).eval()
    p = f"layers.{layer_idx}.self_attn."
    with torch.no_grad():
        ref.q_proj.weight.copy_(sd[p + "q_proj.weight"])
        ref.k_proj.weight.copy_(sd[p + "k_proj.weight"])
        ref.v_proj.weight.copy_(sd[p + "v_proj.weight"])
        ref.o_proj.weight.copy_(sd[p + "o_proj.weight"])
        ref.q_norm.weight.copy_(sd[p + "q_norm.weight"])
        ref.k_norm.weight.copy_(sd[p + "k_norm.weight"])

    # Sanity-check the oracle really is the layer kind this case claims to cover.
    assert ref.is_causal == cfg.is_causal(layer_idx)
    assert ref.sliding_window == cfg.window_for(layer_idx)

    positions = torch.arange(CTX + BLOCK)[None]
    pos_emb = Qwen3RotaryEmbedding(hf_cfg)(x, positions)
    expected = ref(
        hidden_states=x,
        target_hidden=ctx,
        position_embeddings=pos_emb,
        attention_mask=None,  # let the reference build its own -- that is part of the oracle
        past_key_values=None,
    )[0]

    # ---- device ----------------------------------------------------------------------
    weights = load_layer_weights(mesh_device, cfg, layer_idx, state_dict=sd)
    tt_ccl = TT_CCL(mesh_device)
    topology = ccl_topology(mesh_device)
    attn = DFlashAttention(mesh_device, cfg, weights, layer_idx, tt_ccl, topology=topology)

    rope = DFlashRoPE(mesh_device, cfg, max_position=CTX + BLOCK)
    cos_k, sin_k = rope.tables_for(0, CTX + BLOCK)
    cos_q, sin_q = rope.tables_for(CTX, BLOCK)
    attn_mask = dflash_mask.to_device(cfg, layer_idx, mesh_device, ctx_len=CTX, block_len=BLOCK)
    logger.info(f"layer {layer_idx}: mask is {'None (bidirectional)' if attn_mask is None else tuple(attn_mask.shape)}")

    tt_x = _replicate(x.reshape(1, 1, BLOCK, cfg.hidden_size), mesh_device)
    tt_ctx = _replicate(ctx.reshape(1, 1, CTX, cfg.hidden_size), mesh_device)
    ctx_k, ctx_v = attn.project_context(tt_ctx)

    tt_out = attn.forward(tt_x, ctx_k, ctx_v, rope, cos_q, sin_q, cos_k, sin_k, attn_mask)

    stacked = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    actual = stacked[:1].reshape(1, BLOCK, cfg.hidden_size).float()
    spread = max((stacked[d : d + 1].float() - stacked[:1].float()).abs().max().item() for d in range(1, tp))

    logger.info(f"layer {layer_idx} replication spread {spread:.3e}")
    passing, pcc = comp_pcc(expected, actual, get_pcc_threshold(request))
    logger.info(f"layer {layer_idx} ({'sliding' if layer_idx == 0 else 'full'}) attention PCC {pcc}")
    assert passing, f"layer {layer_idx} attention PCC {pcc}"
