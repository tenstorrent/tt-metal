# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 AudioTokenDetokenizer vs custom TT composition.

Reference = genuine HF AudioTokenDetokenizer: embed Linear + expand token to pool_window_size
patches + learned special_tokens + 2 encoder layers (sliding/full) + norm + proj_out Linear.
Single (B=1, T=1) token group. Composes AceStepEncoderLayer + RMSNorm1D. Threshold 0.98.

Correctness: P=pool_window_size (5) < sliding_window (128) -> sliding mask all-visible ->
pass None (all-zero additive mask != None in ttnn SDPA).
"""

import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.tt.detokenizer import AudioTokenDetokenizer, AudioTokenDetokenizerConfig
from models.experimental.acestep.tt.encoder_layer import AceStepEncoderLayerConfig
from models.experimental.acestep.tests.test_utils import (
    HEAD_DIM,
    HIDDEN_SIZE,
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

ACOUSTIC_DIM = 64


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


def test_detokenizer_vs_hf(device):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    cfg._attn_implementation = "eager"
    dt = m.AudioTokenDetokenizer(cfg).eval()
    P = cfg.pool_window_size
    with torch.no_grad():
        dt.embed_tokens.weight.copy_(0.02 * torch.randn_like(dt.embed_tokens.weight))
        dt.embed_tokens.bias.copy_(0.01 * torch.randn_like(dt.embed_tokens.bias))
        dt.special_tokens.copy_(0.02 * torch.randn_like(dt.special_tokens))
        dt.norm.weight.copy_(1.0 + 0.02 * torch.randn_like(dt.norm.weight))
        dt.proj_out.weight.copy_(0.02 * torch.randn_like(dt.proj_out.weight))
        dt.proj_out.bias.copy_(0.01 * torch.randn_like(dt.proj_out.bias))
        for rl in dt.layers:
            a = rl.self_attn
            for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
                lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
            a.q_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.q_norm.weight))
            a.k_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.k_norm.weight))
            for lin in (rl.mlp.gate_proj, rl.mlp.up_proj, rl.mlp.down_proj):
                lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
            rl.input_layernorm.weight.copy_(1.0 + 0.02 * torch.randn_like(rl.input_layernorm.weight))
            rl.post_attention_layernorm.weight.copy_(1.0 + 0.02 * torch.randn_like(rl.post_attention_layernorm.weight))

    x = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.float32)  # [B,T=1,D]
    with torch.no_grad():
        ref_out = dt(x)  # [1, P, acoustic]

    seq = P
    rope = Qwen3RotaryEmbedding(cfg)
    position_ids = torch.arange(seq).unsqueeze(0)
    cos, sin = rope(torch.zeros(1, seq, HEAD_DIM), position_ids)

    # P < sliding_window -> mask all-visible -> None.
    sliding_mask = (
        None
        if seq <= SLIDING_WINDOW
        else m.create_4d_mask(
            seq_len=seq,
            dtype=torch.float32,
            device=x.device,
            attention_mask=None,
            sliding_window=SLIDING_WINDOW,
            is_sliding_window=True,
            is_causal=False,
        )
    )

    layer_types = [rl.attention_type for rl in dt.layers]
    tt = AudioTokenDetokenizer(
        AudioTokenDetokenizerConfig(
            embed_weight=_wT(dt.embed_tokens.weight, device),
            embed_bias=_bias(dt.embed_tokens.bias, device),
            special_tokens=make_lazy_weight(
                dt.special_tokens.detach().clone(), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
            norm_weight=_norm(dt.norm.weight, device),
            proj_out_weight=_wT(dt.proj_out.weight, device),
            proj_out_bias=_bias(dt.proj_out.bias, device),
            layer_configs=[_layer_cfg(rl, at, device) for rl, at in zip(dt.layers, layer_types)],
            layer_types=layer_types,
            hidden_size=HIDDEN_SIZE,
            acoustic_dim=ACOUSTIC_DIM,
            pool_window_size=P,
            eps=RMS_NORM_EPS,
        )
    )

    x_tt = to_ttnn_tensor(x.reshape(1, 1, 1, HIDDEN_SIZE), device)
    cos_tt = ttnn.from_torch(cos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sliding_tt = (
        None
        if sliding_mask is None
        else ttnn.from_torch(sliding_mask, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    )

    out_tt = tt.forward(x_tt, cos_tt, sin_tt, sliding_mask=sliding_tt)
    out = to_torch(out_tt, expected_shape=(1, 1, P, ACOUSTIC_DIM)).reshape(1, P, ACOUSTIC_DIM)

    assert_pcc(ref_out, out, 0.98)
