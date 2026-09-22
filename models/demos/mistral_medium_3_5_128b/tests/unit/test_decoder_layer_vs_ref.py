# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decoder suite row 13: one complete decoder layer with residuals vs the torch reference.

Recipe row ``test_decoder_layer_vs_ref.py``. Everything the earlier rows measured in isolation —
both norms, attention, the dense MLP — composed, plus the two residual adds that only exist here.
Full width and the target mesh: hidden 12288, 96/8 heads, intermediate 28672, SP=8 x TP=4.

The residuals are what this row adds over the sum of its parts, and they are also where a
layout mistake would hide: the residual stream stays hidden-replicated across TP the whole way
through, so ``x + attn_out`` is a plain element-wise add. If either sub-block returned a
TP-partial instead of an all-reduced result, the block tests would still pass and only this one
would fail.

The chunked variant runs the same layer over two chunks and compares against the reference's own
chunked run, which is the layer-level rehearsal for P2.
"""

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.reference.golden import run_reference_layer
from models.demos.mistral_medium_3_5_128b.reference.modeling import REF_DTYPE, LayerWeights
from models.demos.mistral_medium_3_5_128b.tests.device_utils import assert_pcc, from_mesh_sp, to_mesh
from models.demos.mistral_medium_3_5_128b.tt.attention.kv_cache import allocate_kv_cache
from models.demos.mistral_medium_3_5_128b.tt.layer import DecoderLayer
from models.demos.mistral_medium_3_5_128b.tt.rope import build_rope_mats

SEQ, CHUNK = 2048, 1024
SP_AXIS = 0


def _state_dict(w: LayerWeights):
    """``LayerWeights`` -> the ``model.layers.N.``-relative state dict ``DecoderLayer`` takes."""
    return {
        "input_layernorm.weight": w.input_layernorm,
        "post_attention_layernorm.weight": w.post_attention_layernorm,
        "self_attn.q_proj.weight": w.q_proj,
        "self_attn.k_proj.weight": w.k_proj,
        "self_attn.v_proj.weight": w.v_proj,
        "self_attn.o_proj.weight": w.o_proj,
        "mlp.gate_proj.weight": w.gate_proj,
        "mlp.up_proj.weight": w.up_proj,
        "mlp.down_proj.weight": w.down_proj,
    }


def _build(galaxy, cfg, mesh_config, ccl, weights, max_seq_len=SEQ):
    return DecoderLayer(
        galaxy,
        cfg,
        _state_dict(weights),
        0,
        ccl,
        mesh_config,
        max_seq_len=max_seq_len,
        sequence_parallel=True,
    )


def _cache(galaxy, cfg, chunk_size, max_seq_len=SEQ):
    return allocate_kv_cache(
        galaxy,
        num_layers=1,
        max_seq_len=max_seq_len,
        chunk_size=chunk_size,
        sp_axis=SP_AXIS,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )


def test_decoder_layer_vs_ref(galaxy, mesh_config, ccl, cfg):
    """One layer, one shot, full width, against the torch reference."""
    weights = LayerWeights.random(cfg, seed=5)
    torch.manual_seed(6)
    x = (torch.randn(1, SEQ, cfg.hidden_size) * 0.1).to(REF_DTYPE)
    ref_out, _, _ = run_reference_layer(cfg, weights, x)

    layer = _build(galaxy, cfg, mesh_config, ccl, weights)
    out = from_mesh_sp(
        galaxy,
        layer(
            to_mesh(galaxy, x.unsqueeze(0), dims=[-2, None]),
            build_rope_mats(galaxy, cfg, 0, SEQ, mesh_config=mesh_config),
            kv_cache=_cache(galaxy, cfg, SEQ),
        ),
    )
    assert_pcc("decoder_layer", ref_out.unsqueeze(0), out)


@pytest.mark.parametrize("chunk", [CHUNK], ids=[f"c{CHUNK}"])
def test_decoder_layer_chunked_vs_ref(galaxy, mesh_config, ccl, cfg, chunk):
    """The same layer over two chunks, against the reference's own chunked run."""
    weights = LayerWeights.random(cfg, seed=5)
    torch.manual_seed(7)
    x = (torch.randn(1, SEQ, cfg.hidden_size) * 0.1).to(REF_DTYPE)

    ref_outs, past_k, past_v = [], None, None
    for start in range(0, SEQ, chunk):
        out, k, v = run_reference_layer(
            cfg, weights, x[:, start : start + chunk], position_offset=start, past_k=past_k, past_v=past_v
        )
        past_k = k if past_k is None else torch.cat([past_k, k], dim=2)
        past_v = v if past_v is None else torch.cat([past_v, v], dim=2)
        ref_outs.append(out)

    layer = _build(galaxy, cfg, mesh_config, ccl, weights)
    kv = _cache(galaxy, cfg, chunk)
    for i, start in enumerate(range(0, SEQ, chunk)):
        out = from_mesh_sp(
            galaxy,
            layer(
                to_mesh(galaxy, x[:, start : start + chunk].unsqueeze(0), dims=[-2, None]),
                build_rope_mats(galaxy, cfg, start, start + chunk, mesh_config=mesh_config),
                kv_cache=kv,
                cached_len=start,
            ),
        )
        assert_pcc(f"decoder_layer_chunked[chunk{i}]", ref_outs[i].unsqueeze(0), out)
