# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ConditionEncoder -> AceStepDiTModel integration seam.

Validates that the two independently-validated subsystems compose correctly: the conditioning
context produced by AceStepConditionEncoder is a valid cross-attention input for AceStepDiTModel.

Reference chain (genuine HF pieces):
  ctx_ref = concat(lyric_core, timbre_core, text_proj)     # ConditionEncoder (all-valid pack)
  out_ref = AceStepDiTModel(..., encoder_hidden_states=ctx_ref)

TT chain (our modules):
  ctx_tt  = AceStepConditionEncoder(...)                    # our module
  out_tt  = AceStepDiTModel(..., encoder_hidden_states=ctx_tt)

Both feed the SAME reference DiT weights, so any divergence is the seam/composition. Small
config (2 DiT layers, short seqs) to bound runtime. Deep two-subsystem chain -> threshold 0.94.
"""

import pytest
import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.tt.condition_encoder import (
    AceStepConditionEncoder,
    AceStepConditionEncoderConfig,
)
from models.experimental.acestep.tt.dit_layer import AceStepDiTLayerConfig
from models.experimental.acestep.tt.dit_model import AceStepDiTModel, AceStepDiTModelConfig
from models.experimental.acestep.tt.dit_output import DiTOutputConfig
from models.experimental.acestep.tt.dit_stack import AceStepDiTStackConfig
from models.experimental.acestep.tt.encoder_layer import AceStepEncoderLayerConfig
from models.experimental.acestep.tt.lyric_encoder import AceStepLyricEncoderConfig
from models.experimental.acestep.tt.patch_embed import PatchEmbedConfig
from models.experimental.acestep.tt.timestep_embedding import TimestepEmbeddingConfig
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
IN_CHANNELS = 192
OUT_CHANNELS = 64
PATCH = 2
TIME_IN = 256
HIDDEN_CH = 64
N_DIT_LAYERS = 2
DIT_SEQ = 128  # audio latent seq (divisible by patch)
LYRIC_LEN = 256
TIMBRE_LEN = 256
TEXT_LEN = 64


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


def _enc_layer_cfg(rl, at, d):
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


def _lyric_cfg(enc, d):
    lt = [rl.attention_type for rl in enc.layers]
    return AceStepLyricEncoderConfig(
        embed_weight=_wT(enc.embed_tokens.weight, d),
        embed_bias=_bias(enc.embed_tokens.bias, d),
        norm_weight=_norm(enc.norm.weight, d),
        layer_configs=[_enc_layer_cfg(rl, at, d) for rl, at in zip(enc.layers, lt)],
        layer_types=lt,
        eps=RMS_NORM_EPS,
    )


def _te_cfg(te, d):
    return TimestepEmbeddingConfig(
        linear_1_weight=_wT(te.linear_1.weight, d),
        linear_1_bias=_bias(te.linear_1.bias, d),
        linear_2_weight=_wT(te.linear_2.weight, d),
        linear_2_bias=_bias(te.linear_2.bias, d),
        time_proj_weight=_wT(te.time_proj.weight, d),
        time_proj_bias=_bias(te.time_proj.bias, d),
        in_channels=TIME_IN,
        time_embed_dim=HIDDEN_SIZE,
    )


def _dit_layer_cfg(rl, at, d):
    a, c = rl.self_attn, rl.cross_attn
    return AceStepDiTLayerConfig(
        scale_shift_table=make_lazy_weight(
            rl.scale_shift_table.detach().clone(), d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        ),
        self_attn_norm_weight=_norm(rl.self_attn_norm.weight, d),
        mlp_norm_weight=_norm(rl.mlp_norm.weight, d),
        cross_attn_norm_weight=_norm(rl.cross_attn_norm.weight, d),
        wq=_wT(a.q_proj.weight, d),
        wk=_wT(a.k_proj.weight, d),
        wv=_wT(a.v_proj.weight, d),
        wo=_wT(a.o_proj.weight, d),
        q_norm_weight=_norm(a.q_norm.weight, d),
        k_norm_weight=_norm(a.k_norm.weight, d),
        c_wq=_wT(c.q_proj.weight, d),
        c_wk=_wT(c.k_proj.weight, d),
        c_wv=_wT(c.v_proj.weight, d),
        c_wo=_wT(c.o_proj.weight, d),
        c_q_norm_weight=_norm(c.q_norm.weight, d),
        c_k_norm_weight=_norm(c.k_norm.weight, d),
        w1=_wT(rl.mlp.gate_proj.weight, d),
        w2=_wT(rl.mlp.down_proj.weight, d),
        w3=_wT(rl.mlp.up_proj.weight, d),
        n_heads=NUM_ATTENTION_HEADS,
        n_kv_heads=NUM_KEY_VALUE_HEADS,
        head_dim=HEAD_DIM,
        dim=HIDDEN_SIZE,
        eps=RMS_NORM_EPS,
        sliding_window=SLIDING_WINDOW if at == "sliding_attention" else None,
        use_cross_attention=True,
    )


def _init_enc_layers(layers):
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


def _encode_core(enc, m, x, seq):
    e = enc.embed_tokens(x)
    pos = torch.arange(seq).unsqueeze(0)
    cos, sin = enc.rotary_emb(e, pos)
    h = e
    for rl in enc.layers:
        mask = (
            None
            if rl.attention_type == "full_attention"
            else m.create_4d_mask(
                seq_len=seq,
                dtype=torch.float32,
                device=h.device,
                attention_mask=None,
                sliding_window=SLIDING_WINDOW,
                is_sliding_window=True,
                is_causal=False,
            )
        )
        (h,) = rl(hidden_states=h, position_embeddings=(cos, sin), attention_mask=mask)
    return enc.norm(h)


def _mask_tt(m, seq, ref_dev, device):
    mk = m.create_4d_mask(
        seq_len=seq,
        dtype=torch.float32,
        device=ref_dev,
        attention_mask=None,
        sliding_window=SLIDING_WINDOW,
        is_sliding_window=True,
        is_causal=False,
    )
    return ttnn.from_torch(mk, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


@pytest.mark.slow
def test_condition_to_dit_seam(device):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    cfg._attn_implementation = "eager"
    cfg.num_hidden_layers = N_DIT_LAYERS

    ce = m.AceStepConditionEncoder(cfg).eval()
    dit = m.AceStepDiTModel(cfg).eval()
    with torch.no_grad():
        ce.text_projector.weight.copy_(0.02 * torch.randn_like(ce.text_projector.weight))
        for enc in (ce.lyric_encoder, ce.timbre_encoder):
            enc.embed_tokens.weight.copy_(0.02 * torch.randn_like(enc.embed_tokens.weight))
            enc.embed_tokens.bias.copy_(0.01 * torch.randn_like(enc.embed_tokens.bias))
            enc.norm.weight.copy_(1.0 + 0.02 * torch.randn_like(enc.norm.weight))
            _init_enc_layers(enc.layers)
        # DiT init
        for te in (dit.time_embed, dit.time_embed_r):
            for lin in (te.linear_1, te.linear_2, te.time_proj):
                lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
                lin.bias.copy_(0.01 * torch.randn_like(lin.bias))
        dit.proj_in[1].weight.copy_(0.02 * torch.randn_like(dit.proj_in[1].weight))
        dit.proj_in[1].bias.copy_(0.01 * torch.randn_like(dit.proj_in[1].bias))
        dit.condition_embedder.weight.copy_(0.02 * torch.randn_like(dit.condition_embedder.weight))
        dit.condition_embedder.bias.copy_(0.01 * torch.randn_like(dit.condition_embedder.bias))
        dit.norm_out.weight.copy_(1.0 + 0.02 * torch.randn_like(dit.norm_out.weight))
        dit.scale_shift_table.copy_(0.05 * torch.randn_like(dit.scale_shift_table))
        dit.proj_out[1].weight.copy_(0.02 * torch.randn_like(dit.proj_out[1].weight))
        dit.proj_out[1].bias.copy_(0.01 * torch.randn_like(dit.proj_out[1].bias))
        for rl in dit.layers:
            for a in (rl.self_attn, rl.cross_attn):
                for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
                    lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
                a.q_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.q_norm.weight))
                a.k_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.k_norm.weight))
            for lin in (rl.mlp.gate_proj, rl.mlp.up_proj, rl.mlp.down_proj):
                lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
            for nm in (rl.self_attn_norm, rl.cross_attn_norm, rl.mlp_norm):
                nm.weight.copy_(1.0 + 0.02 * torch.randn_like(nm.weight))
            rl.scale_shift_table.copy_(0.05 * torch.randn_like(rl.scale_shift_table))

    text = torch.randn(1, TEXT_LEN, TEXT_HIDDEN_DIM)
    lyric = torch.randn(1, LYRIC_LEN, TEXT_HIDDEN_DIM)
    timbre = torch.randn(1, TIMBRE_LEN, TIMBRE_HIDDEN_DIM)
    hidden = torch.randn(1, DIT_SEQ, HIDDEN_CH)
    context = torch.randn(1, DIT_SEQ, IN_CHANNELS - HIDDEN_CH)
    t = torch.rand(1)
    t_r = torch.rand(1)

    # Reference: build context then run reference DiT.
    with torch.no_grad():
        text_ref = text @ ce.text_projector.weight.t()
        lyric_ref = _encode_core(ce.lyric_encoder, m, lyric, LYRIC_LEN)
        timbre_ref = _encode_core(ce.timbre_encoder, m, timbre, TIMBRE_LEN)
        ctx_ref = torch.cat([lyric_ref, timbre_ref, text_ref], dim=1)
        (ref_out, *_) = dit(
            hidden_states=hidden,
            timestep=t,
            timestep_r=t_r,
            attention_mask=None,
            encoder_hidden_states=ctx_ref,
            encoder_attention_mask=None,
            context_latents=context,
        )

    # TT: ConditionEncoder -> DiT.
    tt_ce = AceStepConditionEncoder(
        AceStepConditionEncoderConfig(
            text_projector_weight=_wT(ce.text_projector.weight, device),
            lyric_encoder=_lyric_cfg(ce.lyric_encoder, device),
            timbre_encoder=_lyric_cfg(ce.timbre_encoder, device),
        )
    )
    ct = dit.proj_out[1]
    inp, outp, k = ct.weight.shape
    proj_out_w = ct.weight.permute(2, 1, 0).reshape(k * outp, inp).transpose(0, 1).contiguous()
    conv = dit.proj_in[1]
    oc, ic, kk = conv.weight.shape
    proj_in_w = conv.weight.reshape(oc, ic * kk).transpose(0, 1).contiguous()
    dit_layer_types = [rl.attention_type for rl in dit.layers]
    tt_dit = AceStepDiTModel(
        AceStepDiTModelConfig(
            time_embed=_te_cfg(dit.time_embed, device),
            time_embed_r=_te_cfg(dit.time_embed_r, device),
            patch_embed=PatchEmbedConfig(
                weight=make_lazy_weight(proj_in_w, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                bias=_bias(conv.bias, device),
                in_channels=IN_CHANNELS,
                out_channels=HIDDEN_SIZE,
                patch_size=PATCH,
            ),
            condition_embedder_weight=_wT(dit.condition_embedder.weight, device),
            condition_embedder_bias=_bias(dit.condition_embedder.bias, device),
            stack=AceStepDiTStackConfig(
                layer_configs=[_dit_layer_cfg(rl, at, device) for rl, at in zip(dit.layers, dit_layer_types)],
                layer_types=dit_layer_types,
            ),
            output=DiTOutputConfig(
                scale_shift_table=make_lazy_weight(
                    dit.scale_shift_table.detach().clone(), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                ),
                norm_out_weight=_norm(dit.norm_out.weight, device),
                proj_out_weight=make_lazy_weight(proj_out_w, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                proj_out_bias=_bias(ct.bias, device),
                dim=HIDDEN_SIZE,
                out_channels=OUT_CHANNELS,
                patch_size=PATCH,
                eps=RMS_NORM_EPS,
            ),
            dim=HIDDEN_SIZE,
        )
    )

    rope = Qwen3RotaryEmbedding(cfg)
    lcos, lsin = rope(torch.zeros(1, LYRIC_LEN, HEAD_DIM), torch.arange(LYRIC_LEN).unsqueeze(0))
    tcos, tsin = rope(torch.zeros(1, TIMBRE_LEN, HEAD_DIM), torch.arange(TIMBRE_LEN).unsqueeze(0))
    tprime = DIT_SEQ // PATCH
    dcos, dsin = rope(torch.zeros(1, tprime, HEAD_DIM), torch.arange(tprime).unsqueeze(0))
    dit_sliding = _mask_tt(m, tprime, hidden.device, device) if tprime > SLIDING_WINDOW else None

    def tf(x, s, c):
        return to_ttnn_tensor(x.reshape(1, 1, s, c), device)

    ctx_tt = tt_ce.forward(
        tf(text, TEXT_LEN, TEXT_HIDDEN_DIM),
        tf(lyric, LYRIC_LEN, TEXT_HIDDEN_DIM),
        tf(timbre, TIMBRE_LEN, TIMBRE_HIDDEN_DIM),
        ttnn.from_torch(lcos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        ttnn.from_torch(lsin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        ttnn.from_torch(tcos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        ttnn.from_torch(tsin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        lyric_sliding=_mask_tt(m, LYRIC_LEN, lyric.device, device),
        timbre_sliding=_mask_tt(m, TIMBRE_LEN, timbre.device, device),
    )

    out_tt = tt_dit.forward(
        tf(hidden, DIT_SEQ, HIDDEN_CH),
        tf(context, DIT_SEQ, IN_CHANNELS - HIDDEN_CH),
        t,
        t_r,
        ttnn.from_torch(dcos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        ttnn.from_torch(dsin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        ctx_tt,
        sliding_mask=dit_sliding,
    )
    out = to_torch(out_tt, expected_shape=(1, 1, DIT_SEQ, OUT_CHANNELS)).reshape(1, DIT_SEQ, OUT_CHANNELS)

    assert_pcc(ref_out, out, 0.94)
