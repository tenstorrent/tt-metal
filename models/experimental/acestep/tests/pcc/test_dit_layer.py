# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 AceStepDiTLayer (AdaLN + self + cross + MLP) vs custom TT module.

Reference = genuine HF AceStepDiTLayer. The core generative block with AdaLN modulation
from the timestep embedding. Composes RMSNorm1D + AceStepAttention (self + cross) + MLP1D
plus explicit ttnn modulation. Batch=1 for bring-up; threshold 0.98 (deep composition).
"""

import pytest
import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.tt.dit_layer import AceStepDiTLayer, AceStepDiTLayerConfig
from models.experimental.acestep.tests.test_utils import (
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_ATTENTION_HEADS,
    NUM_KEY_VALUE_HEADS,
    RMS_NORM_EPS,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)

SHAPES = [(256, 128), (512, 256)]  # (self seq, encoder seq)
LAYER_IDX = 1  # full_attention


def _wT(w, device):
    return make_lazy_weight(
        w.detach().clone().transpose(-1, -2).contiguous(),
        device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def _norm(w, device):
    return make_lazy_weight(w.detach().clone(), device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.parametrize("seq_len,enc_len", SHAPES, ids=[f"S{s}_E{e}" for s, e in SHAPES])
def test_dit_layer_vs_hf(device, seq_len, enc_len):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    cfg._attn_implementation = "eager"
    ref = m.AceStepDiTLayer(cfg, layer_idx=LAYER_IDX, use_cross_attention=True).eval()
    with torch.no_grad():
        for a in (ref.self_attn, ref.cross_attn):
            for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
                lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
            a.q_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.q_norm.weight))
            a.k_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.k_norm.weight))
        for lin in (ref.mlp.gate_proj, ref.mlp.up_proj, ref.mlp.down_proj):
            lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
        for nm in (ref.self_attn_norm, ref.cross_attn_norm, ref.mlp_norm):
            nm.weight.copy_(1.0 + 0.02 * torch.randn_like(nm.weight))
        ref.scale_shift_table.copy_(0.05 * torch.randn_like(ref.scale_shift_table))

    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.float32)
    encoder = torch.randn(1, enc_len, HIDDEN_SIZE, dtype=torch.float32)
    temb = torch.randn(1, 6, HIDDEN_SIZE, dtype=torch.float32)  # timestep_proj

    rope = Qwen3RotaryEmbedding(cfg)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rope(hidden, position_ids)

    with torch.no_grad():
        (ref_out,) = ref(
            hidden_states=hidden,
            position_embeddings=(cos, sin),
            temb=temb,
            encoder_hidden_states=encoder,
            attention_mask=None,
            encoder_attention_mask=None,
        )

    a, c = ref.self_attn, ref.cross_attn
    tt = AceStepDiTLayer(
        AceStepDiTLayerConfig(
            scale_shift_table=make_lazy_weight(
                ref.scale_shift_table.detach().clone(), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
            self_attn_norm_weight=_norm(ref.self_attn_norm.weight, device),
            mlp_norm_weight=_norm(ref.mlp_norm.weight, device),
            cross_attn_norm_weight=_norm(ref.cross_attn_norm.weight, device),
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
            w1=_wT(ref.mlp.gate_proj.weight, device),
            w2=_wT(ref.mlp.down_proj.weight, device),
            w3=_wT(ref.mlp.up_proj.weight, device),
            n_heads=NUM_ATTENTION_HEADS,
            n_kv_heads=NUM_KEY_VALUE_HEADS,
            head_dim=HEAD_DIM,
            dim=HIDDEN_SIZE,
            eps=RMS_NORM_EPS,
            sliding_window=None,
            use_cross_attention=True,
        )
    )

    hidden_tt = to_ttnn_tensor(hidden.reshape(1, 1, seq_len, HIDDEN_SIZE), device)
    encoder_tt = to_ttnn_tensor(encoder.reshape(1, 1, enc_len, HIDDEN_SIZE), device)
    cos_tt = ttnn.from_torch(cos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    # temb [1,6,dim] -> [1,6,B=1,dim] for device broadcast over seq.
    temb_tt = ttnn.from_torch(
        temb.reshape(1, 6, 1, HIDDEN_SIZE), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    out_tt = tt.forward(hidden_tt, cos_tt, sin_tt, temb_tt, encoder_hidden_states=encoder_tt)
    out = to_torch(out_tt, expected_shape=(1, 1, seq_len, HIDDEN_SIZE)).reshape(1, seq_len, HIDDEN_SIZE)

    assert_pcc(ref_out, out, 0.98)
