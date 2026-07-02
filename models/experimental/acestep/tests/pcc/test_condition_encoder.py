# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 ConditionEncoder assembly vs genuine HF sub-computations.

Builds the DiT cross-attention context: text_projector(text) + lyric_encoder(lyric) +
timbre_encoder(timbre), packed together. With all-valid masks, pack_sequences == concat
(verified separately), so the reference context = concat(lyric_core, timbre_core, text_proj).

We compare against the REAL HF pieces:
  - text: te.text_projector applied to text_hidden_states
  - lyric: genuine AceStepLyricEncoder.last_hidden_state
  - timbre: genuine AceStepTimbreEncoder core (embed -> layers -> norm), which is what our
    reused AceStepLyricEncoder reproduces.

Deep multi-encoder composition -> threshold 0.96.
"""

import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.tt.condition_encoder import (
    AceStepConditionEncoder,
    AceStepConditionEncoderConfig,
)
from models.experimental.acestep.tt.encoder_layer import AceStepEncoderLayerConfig
from models.experimental.acestep.tt.lyric_encoder import AceStepLyricEncoderConfig
from models.experimental.acestep.tests.test_utils import (
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_ATTENTION_HEADS,
    NUM_KEY_VALUE_HEADS,
    RMS_NORM_EPS,
    SLIDING_WINDOW,
    TEXT_HIDDEN_DIM,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)

TIMBRE_HIDDEN_DIM = 64
LYRIC_LEN = 256
TIMBRE_LEN = 256
TEXT_LEN = 64


def _wT(w, device):
    return make_lazy_weight(
        w.detach().clone().transpose(-1, -2).contiguous(), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )


def _norm(w, device):
    return make_lazy_weight(w.detach().clone(), device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


def _bias(b, device):
    return make_lazy_weight(
        b.detach().clone().reshape(1, -1).contiguous(), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )


def _layer_cfg(rl, at, device):
    a = rl.self_attn
    return AceStepEncoderLayerConfig(
        input_layernorm_weight=_norm(rl.input_layernorm.weight, device),
        post_attention_layernorm_weight=_norm(rl.post_attention_layernorm.weight, device),
        wq=_wT(a.q_proj.weight, device),
        wk=_wT(a.k_proj.weight, device),
        wv=_wT(a.v_proj.weight, device),
        wo=_wT(a.o_proj.weight, device),
        q_norm_weight=_norm(a.q_norm.weight, device),
        k_norm_weight=_norm(a.k_norm.weight, device),
        w1=_wT(rl.mlp.gate_proj.weight, device),
        w2=_wT(rl.mlp.down_proj.weight, device),
        w3=_wT(rl.mlp.up_proj.weight, device),
        n_heads=NUM_ATTENTION_HEADS,
        n_kv_heads=NUM_KEY_VALUE_HEADS,
        head_dim=HEAD_DIM,
        eps=RMS_NORM_EPS,
        sliding_window=SLIDING_WINDOW if at == "sliding_attention" else None,
    )


def _init_encoder_layers(layers):
    for rl in layers:
        a = rl.self_attn
        for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
            lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
        a.q_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.q_norm.weight))
        a.k_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.k_norm.weight))
        for lin in (rl.mlp.gate_proj, rl.mlp.up_proj, rl.mlp.down_proj):
            lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
        rl.input_layernorm.weight.copy_(1.0 + 0.02 * torch.randn_like(rl.input_layernorm.weight))
        rl.post_attention_layernorm.weight.copy_(1.0 + 0.02 * torch.randn_like(rl.post_attention_layernorm.weight))


def _lyric_cfg_from(ref_enc, device):
    layer_types = [rl.attention_type for rl in ref_enc.layers]
    return AceStepLyricEncoderConfig(
        embed_weight=_wT(ref_enc.embed_tokens.weight, device),
        embed_bias=_bias(ref_enc.embed_tokens.bias, device),
        norm_weight=_norm(ref_enc.norm.weight, device),
        layer_configs=[_layer_cfg(rl, at, device) for rl, at in zip(ref_enc.layers, layer_types)],
        layer_types=layer_types,
        eps=RMS_NORM_EPS,
    )


def _encode_core(ref_enc, m, x, seq_len):
    """Genuine reference: embed -> layers -> norm (matches encoder core)."""
    e = ref_enc.embed_tokens(x)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = ref_enc.rotary_emb(e, position_ids)
    h = e
    for rl in ref_enc.layers:
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
    return ref_enc.norm(h)


def test_condition_encoder_vs_hf(device):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    cfg._attn_implementation = "eager"
    ce = m.AceStepConditionEncoder(cfg).eval()
    with torch.no_grad():
        ce.text_projector.weight.copy_(0.02 * torch.randn_like(ce.text_projector.weight))
        for enc in (ce.lyric_encoder, ce.timbre_encoder):
            enc.embed_tokens.weight.copy_(0.02 * torch.randn_like(enc.embed_tokens.weight))
            enc.embed_tokens.bias.copy_(0.01 * torch.randn_like(enc.embed_tokens.bias))
            enc.norm.weight.copy_(1.0 + 0.02 * torch.randn_like(enc.norm.weight))
            _init_encoder_layers(enc.layers)

    text = torch.randn(1, TEXT_LEN, TEXT_HIDDEN_DIM, dtype=torch.float32)
    lyric = torch.randn(1, LYRIC_LEN, TEXT_HIDDEN_DIM, dtype=torch.float32)
    timbre = torch.randn(1, TIMBRE_LEN, TIMBRE_HIDDEN_DIM, dtype=torch.float32)

    # Reference pieces.
    with torch.no_grad():
        text_ref = text @ ce.text_projector.weight.t()  # no bias
        lyric_ref = _encode_core(ce.lyric_encoder, m, lyric, LYRIC_LEN)
        timbre_ref = _encode_core(ce.timbre_encoder, m, timbre, TIMBRE_LEN)
        ctx_ref = torch.cat([lyric_ref, timbre_ref, text_ref], dim=1)  # all-valid pack == concat

    # TT module.
    tt = AceStepConditionEncoder(
        AceStepConditionEncoderConfig(
            text_projector_weight=_wT(ce.text_projector.weight, device),
            lyric_encoder=_lyric_cfg_from(ce.lyric_encoder, device),
            timbre_encoder=_lyric_cfg_from(ce.timbre_encoder, device),
        )
    )

    rope = Qwen3RotaryEmbedding(cfg)
    lyric_pos = torch.arange(LYRIC_LEN).unsqueeze(0)
    timbre_pos = torch.arange(TIMBRE_LEN).unsqueeze(0)
    lcos, lsin = rope(torch.zeros(1, LYRIC_LEN, HEAD_DIM), lyric_pos)
    tcos, tsin = rope(torch.zeros(1, TIMBRE_LEN, HEAD_DIM), timbre_pos)

    def mask_tt(seq):
        mk = m.create_4d_mask(
            seq_len=seq,
            dtype=torch.float32,
            device=text.device,
            attention_mask=None,
            sliding_window=SLIDING_WINDOW,
            is_sliding_window=True,
            is_causal=False,
        )
        return ttnn.from_torch(mk, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    text_tt = to_ttnn_tensor(text.reshape(1, 1, TEXT_LEN, TEXT_HIDDEN_DIM), device)
    lyric_tt = to_ttnn_tensor(lyric.reshape(1, 1, LYRIC_LEN, TEXT_HIDDEN_DIM), device)
    timbre_tt = to_ttnn_tensor(timbre.reshape(1, 1, TIMBRE_LEN, TIMBRE_HIDDEN_DIM), device)
    lcos_tt = ttnn.from_torch(lcos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    lsin_tt = ttnn.from_torch(lsin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tcos_tt = ttnn.from_torch(tcos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tsin_tt = ttnn.from_torch(tsin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    out_tt = tt.forward(
        text_tt,
        lyric_tt,
        timbre_tt,
        lcos_tt,
        lsin_tt,
        tcos_tt,
        tsin_tt,
        lyric_sliding=mask_tt(LYRIC_LEN),
        timbre_sliding=mask_tt(TIMBRE_LEN),
    )
    total = LYRIC_LEN + TIMBRE_LEN + TEXT_LEN
    out = to_torch(out_tt, expected_shape=(1, 1, total, HIDDEN_SIZE)).reshape(1, total, HIDDEN_SIZE)

    assert_pcc(ctx_ref, out, 0.96)
