# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 AceStepLyricEncoder (8 layers) vs custom TT composition.

Reference = genuine HF AceStepLyricEncoder: Linear(1024->2048)+bias, 8 alternating
full/sliding encoder layers, final RMSNorm, own RoPE. Deep stack -> threshold 0.97.
"""

import pytest
import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.tt.encoder_layer import AceStepEncoderLayerConfig
from models.experimental.acestep.tt.lyric_encoder import AceStepLyricEncoder, AceStepLyricEncoderConfig
from models.experimental.acestep.tests.test_utils import (
    HEAD_DIM,
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

SEQ_LENS = [256, 512]


def _wT(w, device):
    return make_lazy_weight(
        w.detach().clone().transpose(-1, -2).contiguous(),
        device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def _norm(w, device):
    return make_lazy_weight(w.detach().clone(), device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


def _bias(b, device):
    return make_lazy_weight(
        b.detach().clone().reshape(1, -1).contiguous(), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )


def _layer_cfg(ref_layer, attn_type, device):
    a = ref_layer.self_attn
    return AceStepEncoderLayerConfig(
        input_layernorm_weight=_norm(ref_layer.input_layernorm.weight, device),
        post_attention_layernorm_weight=_norm(ref_layer.post_attention_layernorm.weight, device),
        wq=_wT(a.q_proj.weight, device),
        wk=_wT(a.k_proj.weight, device),
        wv=_wT(a.v_proj.weight, device),
        wo=_wT(a.o_proj.weight, device),
        q_norm_weight=_norm(a.q_norm.weight, device),
        k_norm_weight=_norm(a.k_norm.weight, device),
        w1=_wT(ref_layer.mlp.gate_proj.weight, device),
        w2=_wT(ref_layer.mlp.down_proj.weight, device),
        w3=_wT(ref_layer.mlp.up_proj.weight, device),
        n_heads=NUM_ATTENTION_HEADS,
        n_kv_heads=NUM_KEY_VALUE_HEADS,
        head_dim=HEAD_DIM,
        eps=RMS_NORM_EPS,
        sliding_window=SLIDING_WINDOW if attn_type == "sliding_attention" else None,
    )


@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=[f"S{s}" for s in SEQ_LENS])
def test_lyric_encoder_vs_hf(device, seq_len):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    cfg._attn_implementation = "eager"
    ref = m.AceStepLyricEncoder(cfg).eval()
    with torch.no_grad():
        ref.embed_tokens.weight.copy_(0.02 * torch.randn_like(ref.embed_tokens.weight))
        ref.embed_tokens.bias.copy_(0.01 * torch.randn_like(ref.embed_tokens.bias))
        ref.norm.weight.copy_(1.0 + 0.02 * torch.randn_like(ref.norm.weight))
        for layer in ref.layers:
            a = layer.self_attn
            for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
                lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
            a.q_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.q_norm.weight))
            a.k_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.k_norm.weight))
            for lin in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
            layer.input_layernorm.weight.copy_(1.0 + 0.02 * torch.randn_like(layer.input_layernorm.weight))
            layer.post_attention_layernorm.weight.copy_(
                1.0 + 0.02 * torch.randn_like(layer.post_attention_layernorm.weight)
            )

    inputs_embeds = torch.randn(1, seq_len, TEXT_HIDDEN_DIM, dtype=torch.float32)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)

    with torch.no_grad():
        ref_out = ref(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        ref_last = ref_out.last_hidden_state  # [1, seq, hidden]

    # Reference RoPE (embed_tokens output space).
    rope = Qwen3RotaryEmbedding(cfg)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rope(torch.zeros(1, seq_len, HEAD_DIM), position_ids)

    # Sliding mask (all-valid padding -> full mask is None in reference).
    sliding_mask_torch = m.create_4d_mask(
        seq_len=seq_len,
        dtype=torch.float32,
        device=inputs_embeds.device,
        attention_mask=None,
        sliding_window=SLIDING_WINDOW,
        is_sliding_window=True,
        is_causal=False,
    )

    layer_types = [layer.attention_type for layer in ref.layers]
    layer_cfgs = [_layer_cfg(layer, at, device) for layer, at in zip(ref.layers, layer_types)]

    tt = AceStepLyricEncoder(
        AceStepLyricEncoderConfig(
            embed_weight=_wT(ref.embed_tokens.weight, device),
            embed_bias=_bias(ref.embed_tokens.bias, device),
            norm_weight=_norm(ref.norm.weight, device),
            layer_configs=layer_cfgs,
            layer_types=layer_types,
            eps=RMS_NORM_EPS,
        )
    )

    emb_tt = to_ttnn_tensor(inputs_embeds.reshape(1, 1, seq_len, TEXT_HIDDEN_DIM), device)
    cos_tt = ttnn.from_torch(cos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sliding_tt = ttnn.from_torch(sliding_mask_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    out_tt = tt.forward(emb_tt, cos_tt, sin_tt, sliding_mask=sliding_tt)
    out = to_torch(out_tt, expected_shape=(1, 1, seq_len, ref.config.hidden_size)).reshape(
        1, seq_len, ref.config.hidden_size
    )

    assert_pcc(ref_last, out, 0.97)
