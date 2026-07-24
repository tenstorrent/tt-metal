# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 lyric encoder (8 layers) with REAL checkpoint weights.

Validates the conditioning path's lyric encoder against genuine trained tensors from
`encoder.lyric_encoder.*` in model.safetensors. The trained norms deviate notably from random
init (q_norm mean ~0.85, final norm mean ~0.26), so this exercises the real distribution the
DiT cross-attention actually consumes. Requires model.safetensors; skips if absent.
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
    TEXT_HIDDEN_DIM,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)

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
def test_lyric_encoder_real_weights_vs_hf(device, seq_len):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    cfg._attn_implementation = "eager"
    ref = m.AceStepLyricEncoder(cfg).eval()
    load_module_weights(ref, "encoder.lyric_encoder.")

    inputs_embeds = torch.randn(1, seq_len, TEXT_HIDDEN_DIM, dtype=torch.float32)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)

    with torch.no_grad():
        ref_out = ref(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        ref_last = ref_out.last_hidden_state

    rope = Qwen3RotaryEmbedding(cfg)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rope(torch.zeros(1, seq_len, HEAD_DIM), position_ids)

    sliding_mask = m.create_4d_mask(
        seq_len=seq_len,
        dtype=torch.float32,
        device=inputs_embeds.device,
        attention_mask=None,
        sliding_window=SLIDING_WINDOW,
        is_sliding_window=True,
        is_causal=False,
    )

    layer_types = [rl.attention_type for rl in ref.layers]
    tt = AceStepLyricEncoder(
        AceStepLyricEncoderConfig(
            embed_weight=_wT(ref.embed_tokens.weight, device),
            embed_bias=_bias(ref.embed_tokens.bias, device),
            norm_weight=_norm(ref.norm.weight, device),
            layer_configs=[_layer_cfg(rl, at, device) for rl, at in zip(ref.layers, layer_types)],
            layer_types=layer_types,
            eps=RMS_NORM_EPS,
        )
    )

    emb_tt = to_ttnn_tensor(inputs_embeds.reshape(1, 1, seq_len, TEXT_HIDDEN_DIM), device)
    cos_tt = ttnn.from_torch(cos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sliding_tt = ttnn.from_torch(sliding_mask, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    out_tt = tt.forward(emb_tt, cos_tt, sin_tt, sliding_mask=sliding_tt)
    out = to_torch(out_tt, expected_shape=(1, 1, seq_len, cfg.hidden_size)).reshape(1, seq_len, cfg.hidden_size)

    assert_pcc(ref_last, out, 0.97)
