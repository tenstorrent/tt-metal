# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 AceStepTimbreEncoder core vs reused AceStepLyricEncoder.

The timbre encoder core is structurally identical to the lyric encoder: an input Linear
(timbre_hidden_dim=64 -> hidden) + N encoder layers (alternating sliding/full) + final norm.
It then slices position 0 as the timbre embedding (the special_token prepend is commented out
in this checkpoint, and unpack_timbre_embeddings is data-dependent batching orchestration).

So we REUSE AceStepLyricEncoder directly (it derives dims from weights, so 64->2048 and 4
layers work unchanged) and slice the CLS position in the test. Zero new implementation code.
Threshold 0.97 (4-layer stack).
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
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)

TIMBRE_HIDDEN_DIM = 64
SEQ_LENS = [256, 512]  # > sliding_window so masks are real


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


@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=[f"S{s}" for s in SEQ_LENS])
def test_timbre_encoder_core_vs_hf(device, seq_len):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    cfg._attn_implementation = "eager"
    te = m.AceStepTimbreEncoder(cfg).eval()
    with torch.no_grad():
        te.embed_tokens.weight.copy_(0.02 * torch.randn_like(te.embed_tokens.weight))
        te.embed_tokens.bias.copy_(0.01 * torch.randn_like(te.embed_tokens.bias))
        te.norm.weight.copy_(1.0 + 0.02 * torch.randn_like(te.norm.weight))
        for rl in te.layers:
            a = rl.self_attn
            for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
                lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
            a.q_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.q_norm.weight))
            a.k_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.k_norm.weight))
            for lin in (rl.mlp.gate_proj, rl.mlp.up_proj, rl.mlp.down_proj):
                lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
            rl.input_layernorm.weight.copy_(1.0 + 0.02 * torch.randn_like(rl.input_layernorm.weight))
            rl.post_attention_layernorm.weight.copy_(1.0 + 0.02 * torch.randn_like(rl.post_attention_layernorm.weight))

    x = torch.randn(1, seq_len, TIMBRE_HIDDEN_DIM, dtype=torch.float32)

    # Reference core: embed -> layers -> norm (matches AceStepTimbreEncoder.forward pre-slice).
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
        ref_core = te.norm(h)  # [1, seq, hidden]

    # RoPE for TT path (head_dim space).
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
