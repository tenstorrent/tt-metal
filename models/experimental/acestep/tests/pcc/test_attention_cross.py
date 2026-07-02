# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 cross-attention vs custom AceStepAttention (is_cross_attention=True).

Cross-attention: query from hidden_states, key/value from encoder_hidden_states (different
seq len), per-head qk-norm, NO RoPE, non-causal. Reuses the same AceStepAttention class via
the is_cross_attention flag — no new implementation code.
"""

import pytest
import torch

import ttnn

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.tt.attention import AceStepAttention, AceStepAttentionConfig
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

LAYER_IDX = 1
# (query_seq, encoder_seq) pairs — cross-attn kv seq differs from query seq.
SHAPES = [(128, 96), (512, 256), (1024, 512)]


def _lazy_linear_T(weight, device):
    return make_lazy_weight(
        weight.detach().clone().transpose(-1, -2).contiguous(),
        device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


@pytest.mark.parametrize("q_seq,kv_seq", SHAPES, ids=[f"Q{q}_KV{k}" for q, k in SHAPES])
def test_cross_attention_vs_hf(device, q_seq, kv_seq):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    cfg._attn_implementation = "eager"
    ref = m.AceStepAttention(cfg, layer_idx=LAYER_IDX, is_cross_attention=True).eval()
    with torch.no_grad():
        for lin in (ref.q_proj, ref.k_proj, ref.v_proj, ref.o_proj):
            lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
        ref.q_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(ref.q_norm.weight))
        ref.k_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(ref.k_norm.weight))

    hidden = torch.randn(1, q_seq, HIDDEN_SIZE, dtype=torch.float32)
    encoder = torch.randn(1, kv_seq, HIDDEN_SIZE, dtype=torch.float32)

    with torch.no_grad():
        ref_out, _ = ref(
            hidden_states=hidden,
            attention_mask=None,
            encoder_hidden_states=encoder,
            position_embeddings=None,
        )

    tt = AceStepAttention(
        AceStepAttentionConfig(
            wq=_lazy_linear_T(ref.q_proj.weight, device),
            wk=_lazy_linear_T(ref.k_proj.weight, device),
            wv=_lazy_linear_T(ref.v_proj.weight, device),
            wo=_lazy_linear_T(ref.o_proj.weight, device),
            q_norm_weight=make_lazy_weight(
                ref.q_norm.weight.detach().clone(), device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            k_norm_weight=make_lazy_weight(
                ref.k_norm.weight.detach().clone(), device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            n_heads=NUM_ATTENTION_HEADS,
            n_kv_heads=NUM_KEY_VALUE_HEADS,
            head_dim=HEAD_DIM,
            eps=RMS_NORM_EPS,
            is_cross_attention=True,
        )
    )

    hidden_tt = to_ttnn_tensor(hidden.reshape(1, 1, q_seq, HIDDEN_SIZE), device)
    encoder_tt = to_ttnn_tensor(encoder.reshape(1, 1, kv_seq, HIDDEN_SIZE), device)

    out_tt = tt.forward(hidden_tt, encoder_hidden_states=encoder_tt)
    out = to_torch(out_tt, expected_shape=(1, 1, q_seq, HIDDEN_SIZE)).reshape(1, q_seq, HIDDEN_SIZE)

    assert_pcc(ref_out, out, 0.99)
