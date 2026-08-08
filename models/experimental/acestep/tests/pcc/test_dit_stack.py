# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 stacked DiT layers vs custom AceStepDiTStack.

Validates that stacking AceStepDiTLayer (the real generative depth, alternating full/sliding
with shared cross-attention context + shared timestep modulation) accumulates correctly.
Uses N=4 layers (idx 0..3 -> sliding,full,sliding,full) to keep runtime bounded while
exercising both attention types and a shared encoder context. Threshold 0.96 (4-layer depth).
"""

import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.tt.dit_layer import AceStepDiTLayerConfig
from models.experimental.acestep.tt.dit_stack import AceStepDiTStack, AceStepDiTStackConfig
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

N_LAYERS = 4
SEQ_LEN = 256
ENC_LEN = 128


def _wT(w, device):
    return make_lazy_weight(
        w.detach().clone().transpose(-1, -2).contiguous(),
        device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def _norm(w, device):
    return make_lazy_weight(w.detach().clone(), device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


def _init_layer(ref_layer):
    with torch.no_grad():
        for a in (ref_layer.self_attn, ref_layer.cross_attn):
            for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
                lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
            a.q_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.q_norm.weight))
            a.k_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.k_norm.weight))
        for lin in (ref_layer.mlp.gate_proj, ref_layer.mlp.up_proj, ref_layer.mlp.down_proj):
            lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
        for nm in (ref_layer.self_attn_norm, ref_layer.cross_attn_norm, ref_layer.mlp_norm):
            nm.weight.copy_(1.0 + 0.02 * torch.randn_like(nm.weight))
        ref_layer.scale_shift_table.copy_(0.05 * torch.randn_like(ref_layer.scale_shift_table))


def _layer_cfg(ref_layer, attn_type, device):
    a, c = ref_layer.self_attn, ref_layer.cross_attn
    return AceStepDiTLayerConfig(
        scale_shift_table=make_lazy_weight(
            ref_layer.scale_shift_table.detach().clone(), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        ),
        self_attn_norm_weight=_norm(ref_layer.self_attn_norm.weight, device),
        mlp_norm_weight=_norm(ref_layer.mlp_norm.weight, device),
        cross_attn_norm_weight=_norm(ref_layer.cross_attn_norm.weight, device),
        wq=_wT(a.q_proj.weight, device),
        wk=_wT(a.k_proj.weight, device),
        wv=_wT(a.v_proj.weight, device),
        wo=_wT(a.o_proj.weight, device),
        q_norm_weight=_norm(a.q_norm.weight, device),
        k_norm_weight=_norm(a.k_norm.weight, device),
        c_wq=_wT(c.q_proj.weight, device),
        c_wk=_wT(c.k_proj.weight, device),
        c_wv=_wT(c.v_proj.weight, device),
        c_wo=_wT(c.o_proj.weight, device),
        c_q_norm_weight=_norm(c.q_norm.weight, device),
        c_k_norm_weight=_norm(c.k_norm.weight, device),
        w1=_wT(ref_layer.mlp.gate_proj.weight, device),
        w2=_wT(ref_layer.mlp.down_proj.weight, device),
        w3=_wT(ref_layer.mlp.up_proj.weight, device),
        n_heads=NUM_ATTENTION_HEADS,
        n_kv_heads=NUM_KEY_VALUE_HEADS,
        head_dim=HEAD_DIM,
        dim=HIDDEN_SIZE,
        eps=RMS_NORM_EPS,
        sliding_window=SLIDING_WINDOW if attn_type == "sliding_attention" else None,
        use_cross_attention=True,
    )


def test_dit_stack_vs_hf(device):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    cfg._attn_implementation = "eager"

    ref_layers = [m.AceStepDiTLayer(cfg, layer_idx=i, use_cross_attention=True).eval() for i in range(N_LAYERS)]
    for rl in ref_layers:
        _init_layer(rl)
    layer_types = [rl.attention_type for rl in ref_layers]

    hidden = torch.randn(1, SEQ_LEN, HIDDEN_SIZE, dtype=torch.float32)
    encoder = torch.randn(1, ENC_LEN, HIDDEN_SIZE, dtype=torch.float32)
    temb = torch.randn(1, 6, HIDDEN_SIZE, dtype=torch.float32)

    rope = Qwen3RotaryEmbedding(cfg)
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
    cos, sin = rope(hidden, position_ids)

    sliding_mask = m.create_4d_mask(
        seq_len=SEQ_LEN,
        dtype=torch.float32,
        device=hidden.device,
        attention_mask=None,
        sliding_window=SLIDING_WINDOW,
        is_sliding_window=True,
        is_causal=False,
    )

    # Reference: run layers sequentially.
    x = hidden
    with torch.no_grad():
        for rl, at in zip(ref_layers, layer_types):
            mask = sliding_mask if at == "sliding_attention" else None
            (x,) = rl(
                hidden_states=x,
                position_embeddings=(cos, sin),
                temb=temb,
                encoder_hidden_states=encoder,
                attention_mask=mask,
                encoder_attention_mask=None,
            )
    ref_out = x

    tt = AceStepDiTStack(
        AceStepDiTStackConfig(
            layer_configs=[_layer_cfg(rl, at, device) for rl, at in zip(ref_layers, layer_types)],
            layer_types=layer_types,
        )
    )

    hidden_tt = to_ttnn_tensor(hidden.reshape(1, 1, SEQ_LEN, HIDDEN_SIZE), device)
    encoder_tt = to_ttnn_tensor(encoder.reshape(1, 1, ENC_LEN, HIDDEN_SIZE), device)
    cos_tt = ttnn.from_torch(cos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    temb_tt = ttnn.from_torch(
        temb.reshape(1, 6, 1, HIDDEN_SIZE), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    sliding_tt = ttnn.from_torch(sliding_mask, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    out_tt = tt.forward(hidden_tt, cos_tt, sin_tt, temb_tt, encoder_tt, sliding_mask=sliding_tt)
    out = to_torch(out_tt, expected_shape=(1, 1, SEQ_LEN, HIDDEN_SIZE)).reshape(1, SEQ_LEN, HIDDEN_SIZE)

    assert_pcc(ref_out, out, 0.96)
