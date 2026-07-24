# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 timbre encoder core (4 layers) with REAL checkpoint weights.

The timbre encoder is the third conditioning stream (reference-audio timbre). Its core is
reused via AceStepLyricEncoder (Linear 64->2048 + 4 layers + norm). This validates it against
genuine `encoder.timbre_encoder.*` trained tensors, completing real-weight coverage of all
three conditioning encoders. Requires model.safetensors; skips if absent.
"""

import pytest
import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.reference.weight_utils import checkpoint_path, load_module_weights
from models.experimental.acestep.tt.encoder_layer import AceStepEncoderLayerConfig
from models.experimental.acestep.tt.lyric_encoder import AceStepLyricEncoder, AceStepLyricEncoderConfig
from models.experimental.acestep.tests.test_utils import (
    HEAD_DIM,
    NUM_ATTENTION_HEADS,
    NUM_KEY_VALUE_HEADS,
    RMS_NORM_EPS,
    SLIDING_WINDOW,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)

TIMBRE_HIDDEN_DIM = 64
SEQ_LENS = [256, 512]


def _have_checkpoint():
    try:
        checkpoint_path()
        return True
    except AssertionError:
        return False


def _wT(w, d):
    return make_lazy_weight(
        w.detach().clone().transpose(-1, -2).contiguous(), d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )


def _norm(w, d):
    return make_lazy_weight(w.detach().clone(), d, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


def _bias(b, d):
    return make_lazy_weight(
        b.detach().clone().reshape(1, -1).contiguous(), d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )


def _layer_cfg(rl, at, d):
    a = rl.self_attn
    return AceStepEncoderLayerConfig(
        input_layernorm_weight=_norm(rl.input_layernorm.weight, d),
        post_attention_layernorm_weight=_norm(rl.post_attention_layernorm.weight, d),
        wq=_wT(a.q_proj.weight, d),
        wk=_wT(a.k_proj.weight, d),
        wv=_wT(a.v_proj.weight, d),
        wo=_wT(a.o_proj.weight, d),
        q_norm_weight=_norm(a.q_norm.weight, d),
        k_norm_weight=_norm(a.k_norm.weight, d),
        w1=_wT(rl.mlp.gate_proj.weight, d),
        w2=_wT(rl.mlp.down_proj.weight, d),
        w3=_wT(rl.mlp.up_proj.weight, d),
        n_heads=NUM_ATTENTION_HEADS,
        n_kv_heads=NUM_KEY_VALUE_HEADS,
        head_dim=HEAD_DIM,
        eps=RMS_NORM_EPS,
        sliding_window=SLIDING_WINDOW if at == "sliding_attention" else None,
    )


@pytest.mark.skipif(not _have_checkpoint(), reason="model.safetensors not downloaded")
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=[f"S{s}" for s in SEQ_LENS])
def test_timbre_encoder_real_weights_vs_hf(device, seq_len):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    cfg._attn_implementation = "eager"
    te = m.AceStepTimbreEncoder(cfg).eval()
    load_module_weights(te, "encoder.timbre_encoder.")

    x = torch.randn(1, seq_len, TIMBRE_HIDDEN_DIM, dtype=torch.float32)

    # Reference core: embed -> layers -> norm (AceStepTimbreEncoder.forward pre-slice).
    with torch.no_grad():
        e = te.embed_tokens(x)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        cos, sin = te.rotary_emb(e, position_ids)
        h = e
        for rl in te.layers:
            mask = (
                None
                if rl.attention_type == "full_attention"
                else m.create_4d_mask(
                    seq_len=seq_len,
                    dtype=torch.float32,
                    device=h.device,
                    attention_mask=None,
                    sliding_window=SLIDING_WINDOW,
                    is_sliding_window=True,
                    is_causal=False,
                )
            )
            (h,) = rl(hidden_states=h, position_embeddings=(cos, sin), attention_mask=mask)
        ref_core = te.norm(h)

    rope = Qwen3RotaryEmbedding(cfg)
    cos_h, sin_h = rope(torch.zeros(1, seq_len, HEAD_DIM), position_ids)
    sliding_mask = m.create_4d_mask(
        seq_len=seq_len,
        dtype=torch.float32,
        device=x.device,
        attention_mask=None,
        sliding_window=SLIDING_WINDOW,
        is_sliding_window=True,
        is_causal=False,
    )

    layer_types = [rl.attention_type for rl in te.layers]
    tt = AceStepLyricEncoder(
        AceStepLyricEncoderConfig(
            embed_weight=_wT(te.embed_tokens.weight, device),
            embed_bias=_bias(te.embed_tokens.bias, device),
            norm_weight=_norm(te.norm.weight, device),
            layer_configs=[_layer_cfg(rl, at, device) for rl, at in zip(te.layers, layer_types)],
            layer_types=layer_types,
            eps=RMS_NORM_EPS,
        )
    )

    emb_tt = to_ttnn_tensor(x.reshape(1, 1, seq_len, TIMBRE_HIDDEN_DIM), device)
    cos_tt = ttnn.from_torch(cos_h.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin_h.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sliding_tt = ttnn.from_torch(sliding_mask, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    out_tt = tt.forward(emb_tt, cos_tt, sin_tt, sliding_mask=sliding_tt)
    out = to_torch(out_tt, expected_shape=(1, 1, seq_len, cfg.hidden_size)).reshape(1, seq_len, cfg.hidden_size)

    assert_pcc(ref_core, out, 0.97)
