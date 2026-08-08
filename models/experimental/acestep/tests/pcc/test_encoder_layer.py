# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 AceStepEncoderLayer vs custom TT composition.

Reference = genuine HF AceStepEncoderLayer (pre-norm self-attn + MLP, bidirectional).
Composes RMSNorm1D + AceStepAttention + MLP1D. Tests both a full and a sliding layer.
Threshold 0.98 (accumulated bf16 error across 2 norms, attention, MLP, 2 residuals).
"""

import pytest
import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.tt.encoder_layer import AceStepEncoderLayer, AceStepEncoderLayerConfig
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

SEQ_LENS = [256, 512]
# layer_idx 1 = full_attention, 0 = sliding_attention.
LAYER_KINDS = [(1, "full"), (0, "sliding")]


def _wT(weight, device):
    return make_lazy_weight(
        weight.detach().clone().transpose(-1, -2).contiguous(),
        device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def _norm(weight, device):
    return make_lazy_weight(weight.detach().clone(), device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=[f"S{s}" for s in SEQ_LENS])
@pytest.mark.parametrize("layer_idx,kind", LAYER_KINDS, ids=[k for _, k in LAYER_KINDS])
def test_encoder_layer_vs_hf(device, seq_len, layer_idx, kind):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    cfg._attn_implementation = "eager"
    ref = m.AceStepEncoderLayer(cfg, layer_idx=layer_idx).eval()
    with torch.no_grad():
        a = ref.self_attn
        for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
            lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
        a.q_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.q_norm.weight))
        a.k_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(a.k_norm.weight))
        for lin in (ref.mlp.gate_proj, ref.mlp.up_proj, ref.mlp.down_proj):
            lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
        ref.input_layernorm.weight.copy_(1.0 + 0.02 * torch.randn_like(ref.input_layernorm.weight))
        ref.post_attention_layernorm.weight.copy_(1.0 + 0.02 * torch.randn_like(ref.post_attention_layernorm.weight))

    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.float32)

    rope = Qwen3RotaryEmbedding(cfg)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rope(hidden, position_ids)

    attn_mask_torch = None
    if kind == "sliding":
        attn_mask_torch = m.create_4d_mask(
            seq_len=seq_len,
            dtype=torch.float32,
            device=hidden.device,
            attention_mask=None,
            sliding_window=SLIDING_WINDOW,
            is_sliding_window=True,
            is_causal=False,
        )

    with torch.no_grad():
        (ref_out,) = ref(
            hidden_states=hidden,
            position_embeddings=(cos, sin),
            attention_mask=attn_mask_torch,
        )

    a = ref.self_attn
    tt = AceStepEncoderLayer(
        AceStepEncoderLayerConfig(
            input_layernorm_weight=_norm(ref.input_layernorm.weight, device),
            post_attention_layernorm_weight=_norm(ref.post_attention_layernorm.weight, device),
            wq=_wT(a.q_proj.weight, device),
            wk=_wT(a.k_proj.weight, device),
            wv=_wT(a.v_proj.weight, device),
            wo=_wT(a.o_proj.weight, device),
            q_norm_weight=_norm(a.q_norm.weight, device),
            k_norm_weight=_norm(a.k_norm.weight, device),
            w1=_wT(ref.mlp.gate_proj.weight, device),
            w2=_wT(ref.mlp.down_proj.weight, device),
            w3=_wT(ref.mlp.up_proj.weight, device),
            n_heads=NUM_ATTENTION_HEADS,
            n_kv_heads=NUM_KEY_VALUE_HEADS,
            head_dim=HEAD_DIM,
            eps=RMS_NORM_EPS,
            sliding_window=SLIDING_WINDOW if kind == "sliding" else None,
        )
    )

    hidden_tt = to_ttnn_tensor(hidden.reshape(1, 1, seq_len, HIDDEN_SIZE), device)
    cos_tt = ttnn.from_torch(cos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    mask_tt = None
    if attn_mask_torch is not None:
        mask_tt = ttnn.from_torch(attn_mask_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    out_tt = tt.forward(hidden_tt, cos_tt, sin_tt, attn_mask=mask_tt)
    out = to_torch(out_tt, expected_shape=(1, 1, seq_len, HIDDEN_SIZE)).reshape(1, seq_len, HIDDEN_SIZE)

    assert_pcc(ref_out, out, 0.98)
