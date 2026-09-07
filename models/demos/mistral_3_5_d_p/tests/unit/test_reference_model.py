# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D1: the standalone CPU reference against the inline torch golden, on a reduced config with random
weights — so the two oracles cannot drift apart.

Pattern: ``minimax_m3/tests/unit/test_reference_model.py``.

The standalone reference is the transformers ``ministral3`` implementation (imported, not vendored);
the inline golden is a second, hand-written implementation of the same math in
``reference/model.py`` (``golden_*``). Every PCC test downstream compares the device against one of
these two, so an agreement failure here means the oracle itself is wrong and nothing below it can be
trusted. Widened to the whole model (M1's first row) at reduced depth/width.

Host only — no TTNN, no device, no checkpoint. The head geometry (96 Q / 8 KV / head_dim 128) is
NOT reduced: it is what the GQA layout under test is made of.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.mistral_3_5_d_p.reference import golden_cache, model as reference
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.demos.mistral_3_5_d_p.reference.mistral_config import reduced_text_config

# fp32 on both sides, so the two oracles must agree to near machine precision, not merely to PCC.
EXACT = 2e-5


@pytest.fixture(scope="module")
def cfg():
    """Reduced depth/width, real head geometry. Small enough for a host forward in seconds."""
    return reduced_text_config(num_hidden_layers=2, hidden_size=1024, intermediate_size=2048, vocab_size=2048)


@pytest.fixture(scope="module")
def rope(cfg):
    """One cos/sin table shared by both oracles (the tables themselves are pinned in
    test_reference_config.py, so here they must not be the thing under test)."""
    return reference.golden_cos_sin_hf(
        SEQ_LEN,
        C.HEAD_DIM,
        theta=C.ROPE_THETA,
        factor=C.YARN_FACTOR,
        orig_max_pos=C.YARN_ORIG_MAX_POS,
        beta_fast=C.YARN_BETA_FAST,
        beta_slow=C.YARN_BETA_SLOW,
        truncate=C.YARN_TRUNCATE,
    )


SEQ_LEN = 128


def _rel_max_diff(a, b):
    scale = max(a.abs().max().item(), 1e-12)
    return (a - b).abs().max().item() / scale


def test_rms_norm_oracles_agree(cfg, reset_seeds):
    x = torch.randn(1, SEQ_LEN, cfg.hidden_size)
    weight = torch.randn(cfg.hidden_size) * 0.1 + 1.0
    ref = reference.rms_norm_reference(x, weight, cfg.rms_norm_eps)
    gold = reference.golden_rms_norm(x, weight, cfg.rms_norm_eps)
    assert _rel_max_diff(ref, gold) < EXACT, f"rms_norm oracles differ by {_rel_max_diff(ref, gold)}"


def test_mlp_oracles_agree(cfg, reset_seeds):
    """Dense SwiGLU (silu), the exact variant: ``down(silu(gate(x)) * up(x))``, no clamp, no alpha."""
    h, i = cfg.hidden_size, cfg.intermediate_size
    x = torch.randn(1, SEQ_LEN, h) * 0.1
    gate_w, up_w, down_w = (torch.randn(i, h) * 0.02, torch.randn(i, h) * 0.02, torch.randn(h, i) * 0.02)
    ref = reference.mlp_reference(
        x, {"gate_proj.weight": gate_w, "up_proj.weight": up_w, "down_proj.weight": down_w}, cfg
    )
    gold = reference.golden_mlp(x, gate_w, up_w, down_w)
    assert _rel_max_diff(ref, gold) < EXACT, f"mlp oracles differ by {_rel_max_diff(ref, gold)}"


def test_attention_oracles_agree(cfg, rope, reset_seeds):
    """Whole attention block, plus the post-RoPE K / raw V pair the KV cache stores."""
    lw = reference.random_layer_weights(cfg, seed=11)
    x = torch.randn(1, SEQ_LEN, cfg.hidden_size) * 0.1
    attn_sd = {
        "q_proj.weight": lw["q"],
        "k_proj.weight": lw["k"],
        "v_proj.weight": lw["v"],
        "o_proj.weight": lw["o"],
    }
    cos, sin = rope
    ref = reference.attention_reference(x, attn_sd, cfg, cos_sin=(cos.unsqueeze(0), sin.unsqueeze(0)))
    gold_out, gold_k, gold_v = reference.golden_attention(
        x,
        lw,
        n_q=cfg.num_attention_heads,
        n_kv=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        cos=cos,
        sin=sin,
    )
    logger.info(f"attention oracle agreement: out={_rel_max_diff(ref.output, gold_out):.3e}")
    assert _rel_max_diff(ref.output, gold_out) < EXACT
    assert _rel_max_diff(ref.k_post_rope, gold_k) < EXACT, "post-RoPE K disagrees"
    assert _rel_max_diff(ref.v, gold_v) < EXACT, "raw V disagrees"


@pytest.mark.parametrize("offset", [0, 128], ids=["chunk0", "chunk1"])
def test_attention_oracles_agree_chunked(cfg, offset, reset_seeds):
    """Same block with a nonzero global offset and a prefix in the cache — the chunked shape the
    device's ring cache-read reproduces. Catches an off-by-``offset`` causal mask in either oracle."""
    lw = reference.random_layer_weights(cfg, seed=12)
    x = torch.randn(1, SEQ_LEN, cfg.hidden_size) * 0.1
    n_kv, hd = cfg.num_key_value_heads, cfg.head_dim
    past_k = torch.randn(1, n_kv, offset, hd) * 0.1 if offset else None
    past_v = torch.randn(1, n_kv, offset, hd) * 0.1 if offset else None
    cos, sin = reference.golden_cos_sin_hf(
        SEQ_LEN,
        hd,
        offset=offset,
        theta=C.ROPE_THETA,
        factor=C.YARN_FACTOR,
        orig_max_pos=C.YARN_ORIG_MAX_POS,
        beta_fast=C.YARN_BETA_FAST,
        beta_slow=C.YARN_BETA_SLOW,
        truncate=C.YARN_TRUNCATE,
    )
    attn_sd = {f"{n}_proj.weight": lw[n[0]] for n in ("q", "k", "v", "o")}
    ref = reference.attention_reference(
        x,
        attn_sd,
        cfg,
        offset=offset,
        past_k=past_k,
        past_v=past_v,
        cos_sin=(cos.unsqueeze(0), sin.unsqueeze(0)),
    )
    gold_out, _, _ = reference.golden_attention(
        x,
        lw,
        n_q=cfg.num_attention_heads,
        n_kv=n_kv,
        head_dim=hd,
        cos=cos,
        sin=sin,
        offset=offset,
        past_k=past_k,
        past_v=past_v,
    )
    assert _rel_max_diff(ref.output, gold_out) < EXACT, f"chunked attention oracles differ at offset={offset}"


def test_decoder_layer_oracles_agree(cfg, rope, reset_seeds):
    """The composition (both residual adds and both norms), after each piece agrees alone."""
    lw = reference.random_layer_weights(cfg, seed=13)
    x = torch.randn(1, SEQ_LEN, cfg.hidden_size) * 0.1
    ref, _ = reference.decoder_layer_reference(x, reference.hf_layer_state_dict(lw), cfg)
    cos, sin = rope
    gold, _, _ = reference.golden_decoder_layer(
        x,
        lw,
        n_q=cfg.num_attention_heads,
        n_kv=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        eps=cfg.rms_norm_eps,
        cos=cos,
        sin=sin,
    )
    logger.info(f"decoder layer oracle agreement: {_rel_max_diff(ref, gold):.3e}")
    assert _rel_max_diff(ref, gold) < EXACT


def test_whole_model_oracles_agree(cfg, reset_seeds):
    """M1's first row: the whole-model CPU reference forward against the inline golden, all layers,
    random weights — embedding, every decoder layer, the final norm and the LM head."""
    model = reference.build_reference_model(cfg, seed=21)
    token_ids = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN))
    ref = reference.model_reference_forward(model, token_ids)
    w = golden_cache.inline_golden_weights(model.state_dict(), cfg.num_hidden_layers)
    gold_logits, gold_hidden, gold_kv = reference.golden_model(token_ids, w, cfg, C)

    ok, pcc = comp_pcc(ref.logits, gold_logits, 0.9999)
    logger.info(
        f"whole-model oracle agreement: logits pcc={pcc} " f"hidden={_rel_max_diff(ref.hidden_states, gold_hidden):.3e}"
    )
    assert ok, f"whole-model logits disagree: {pcc}"
    assert _rel_max_diff(ref.hidden_states, gold_hidden) < EXACT
    for i, ((ref_k, ref_v), (gold_k, gold_v)) in enumerate(zip(ref.kv, gold_kv)):
        assert _rel_max_diff(ref_k, gold_k) < EXACT, f"layer {i} K disagrees"
        assert _rel_max_diff(ref_v, gold_v) < EXACT, f"layer {i} V disagrees"
