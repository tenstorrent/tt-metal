# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HOST-ONLY: pin the torch reference to the real ``transformers`` Ministral3 modules.
**No TT hardware needed.**

The device tests PCC the TT modules against ``reference/torch_reference.py``. That is only
meaningful if the reference itself is right, so this test drives the actual
``Ministral3DecoderLayer`` / ``MistralRMSNorm`` / ``MistralMLP`` with the same weights and requires
an exact match. It also fixes the one mechanism that has no config field: with
``llama_4_scaling_beta = 0`` the Ministral3 Q temperature must be an exact no-op, which is why the
reference can omit it.

Uses a small (hidden 512) config for speed — the mechanisms are shape-independent; the real shapes
are checked in test_checkpoint_ingest.py.

Run:  pytest models/demos/mistral_medium_d_p/tests/unit/test_reference_model.py
"""

import pytest
import torch

from models.demos.mistral_medium_d_p.reference.torch_reference import (
    decoder_layer,
    gqa_attention,
    random_layer_weights,
    rms_norm,
    swiglu_mlp,
)
from models.demos.mistral_medium_d_p.tt.rope_tables import build_hf_cos_sin

HIDDEN, N_Q, N_KV, HEAD_DIM, FFN, EPS = 512, 8, 2, 64, 1024, 1e-5
SEQ = 96
YARN = dict(rope_theta=1000000.0, yarn_factor=64.0, yarn_orig_max_pos=4096, yarn_beta_fast=4.0, yarn_beta_slow=1.0)


def _cfg(**overrides):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Ministral3Config(
        hidden_size=HIDDEN,
        intermediate_size=FFN,
        num_hidden_layers=1,
        num_attention_heads=N_Q,
        num_key_value_heads=N_KV,
        head_dim=HEAD_DIM,
        hidden_act="silu",
        rms_norm_eps=EPS,
        vocab_size=256,
        max_position_embeddings=262144,
        sliding_window=None,
        rope_parameters={
            "rope_type": "yarn",
            "type": "yarn",
            "rope_theta": 1000000.0,
            "factor": 64.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 4.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
            "llama_4_scaling_beta": 0,
        },
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_rms_norm_matches_hf():
    from transformers.models.mistral.modeling_mistral import MistralRMSNorm

    torch.manual_seed(0)
    x = torch.randn(1, SEQ, HIDDEN)
    gain = 1.0 + torch.randn(HIDDEN) * 0.02
    hf = MistralRMSNorm(HIDDEN, eps=EPS)
    hf.weight.data = gain.clone()
    torch.testing.assert_close(rms_norm(x, gain, EPS), hf(x), rtol=1e-6, atol=1e-6)


def test_swiglu_mlp_matches_hf():
    from transformers.models.mistral.modeling_mistral import MistralMLP

    torch.manual_seed(0)
    x = torch.randn(1, SEQ, HIDDEN)
    hf = MistralMLP(_cfg())
    got = swiglu_mlp(x, hf.gate_proj.weight.data, hf.up_proj.weight.data, hf.down_proj.weight.data)
    torch.testing.assert_close(got, hf(x), rtol=1e-5, atol=1e-5)
    assert (
        hf.gate_proj.bias is None and hf.up_proj.bias is None and hf.down_proj.bias is None
    ), "MistralMLP grew a bias — the TT MLP asserts there is none"


def _hf_layer(cfg, w):
    from transformers.models.ministral3.modeling_ministral3 import Ministral3DecoderLayer

    layer = Ministral3DecoderLayer(cfg, layer_idx=0)
    layer.self_attn.q_proj.weight.data = w["q"].clone()
    layer.self_attn.k_proj.weight.data = w["k"].clone()
    layer.self_attn.v_proj.weight.data = w["v"].clone()
    layer.self_attn.o_proj.weight.data = w["o"].clone()
    layer.mlp.gate_proj.weight.data = w["gate"].clone()
    layer.mlp.up_proj.weight.data = w["up"].clone()
    layer.mlp.down_proj.weight.data = w["down"].clone()
    layer.input_layernorm.weight.data = w["input_layernorm"].clone()
    layer.post_attention_layernorm.weight.data = w["post_attention_layernorm"].clone()
    return layer.eval()


def _run_hf_layer(cfg, layer, x):
    """Drive one HF decoder layer with an explicit causal mask and YaRN position embeddings."""
    from transformers.models.ministral3.modeling_ministral3 import Ministral3RotaryEmbedding

    seq = x.shape[1]
    pos_ids = torch.arange(seq)[None]
    cos, sin = Ministral3RotaryEmbedding(cfg)(x, pos_ids)
    mask = torch.triu(torch.full((seq, seq), torch.finfo(x.dtype).min), diagonal=1)[None, None]
    with torch.no_grad():
        out = layer(
            x,
            position_embeddings=(cos, sin),
            attention_mask=mask,
            position_ids=pos_ids,
            past_key_values=None,
        )
    return out[0] if isinstance(out, tuple) else out


def test_attention_matches_hf():
    cfg = _cfg()
    w = random_layer_weights(HIDDEN, N_Q, N_KV, HEAD_DIM, FFN, seed=1)
    layer = _hf_layer(cfg, w)
    torch.manual_seed(0)
    x = torch.randn(1, SEQ, HIDDEN) * 0.1

    cos, sin = build_hf_cos_sin(SEQ, HEAD_DIM, **YARN)
    ours = gqa_attention(x, w, cos, sin, n_q=N_Q, n_kv=N_KV, head_dim=HEAD_DIM)

    from transformers.models.ministral3.modeling_ministral3 import Ministral3RotaryEmbedding

    hf_cos, hf_sin = Ministral3RotaryEmbedding(cfg)(x, torch.arange(SEQ)[None])
    mask = torch.triu(torch.full((SEQ, SEQ), torch.finfo(x.dtype).min), diagonal=1)[None, None]
    with torch.no_grad():
        hf_out, _ = layer.self_attn(
            x, position_embeddings=(hf_cos, hf_sin), attention_mask=mask, position_ids=torch.arange(SEQ)[None]
        )
    torch.testing.assert_close(ours, hf_out, rtol=2e-4, atol=2e-4)


def test_decoder_layer_matches_hf():
    cfg = _cfg()
    w = random_layer_weights(HIDDEN, N_Q, N_KV, HEAD_DIM, FFN, seed=2)
    layer = _hf_layer(cfg, w)
    torch.manual_seed(0)
    x = torch.randn(1, SEQ, HIDDEN) * 0.1

    cos, sin = build_hf_cos_sin(SEQ, HEAD_DIM, **YARN)
    ours = decoder_layer(x, w, cos, sin, n_q=N_Q, n_kv=N_KV, head_dim=HEAD_DIM, eps=EPS)
    torch.testing.assert_close(ours, _run_hf_layer(cfg, layer, x), rtol=2e-4, atol=2e-4)


def test_llama4_q_temperature_is_a_no_op_at_beta_zero():
    """The mechanism with no config field: at beta=0 the reference may omit it — prove it.

    If a future checkpoint sets beta != 0 this test still passes (it only asserts the beta=0 case),
    but ``checkpoint.assert_supported`` refuses to build, which is the intended guard.
    """
    from transformers.models.ministral3.modeling_ministral3 import get_llama_4_attn_scale

    pos = torch.arange(0, 262144, 4096)[None]
    scale = get_llama_4_attn_scale(pos, 0, 4096)
    assert torch.equal(scale, torch.ones_like(scale)), "beta=0 must make the Q temperature exactly 1.0"
    # And that it would NOT be a no-op at the Ministral3Config class default of 0.1.
    assert not torch.equal(get_llama_4_attn_scale(pos, 0.1, 4096), torch.ones_like(scale))
