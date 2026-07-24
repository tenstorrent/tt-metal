# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 self-attention (full, bidirectional) vs custom AceStepAttention.

Reference = genuine HF AceStepAttention (full_attention layer, self, GQA 16/8, per-head
qk-norm, RoPE theta=1e6). This is bidirectional (diffusion) so is_causal=False, no KV cache.

Custom TT module: models/experimental/acestep/tt/attention.py — TTTv2-pattern class that
reuses ttnn SDPA + RMSNorm1D (qk-norm) + rotary_embedding_hf.
"""

import pytest
import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

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

# Full-attention layer index (layer_types[1] == "full_attention").
FULL_LAYER_IDX = 1
SEQ_LENS = [128, 512, 1024]


def _lazy_linear_T(weight, device):
    # HF Linear [out,in] -> ttnn.linear wants [in,out].
    return make_lazy_weight(
        weight.detach().clone().transpose(-1, -2).contiguous(),
        device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=[f"S{s}" for s in SEQ_LENS])
def test_self_attention_full_vs_hf(device, seq_len):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    cfg._attn_implementation = "eager"  # deterministic reference path
    ref = m.AceStepAttention(cfg, layer_idx=FULL_LAYER_IDX, is_cross_attention=False, is_causal=False).eval()
    with torch.no_grad():
        for lin in (ref.q_proj, ref.k_proj, ref.v_proj, ref.o_proj):
            lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
        ref.q_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(ref.q_norm.weight))
        ref.k_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(ref.k_norm.weight))

    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.float32)

    # Reference RoPE (theta=1e6).
    rope = Qwen3RotaryEmbedding(cfg)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rope(hidden, position_ids)  # [1, seq, head_dim]

    with torch.no_grad():
        ref_out, _ = ref(
            hidden_states=hidden,
            attention_mask=None,  # full bidirectional, no padding
            position_embeddings=(cos, sin),
        )  # [1, seq, hidden]

    # TT module.
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
            is_cross_attention=False,
            sliding_window=None,
        )
    )

    hidden_tt = to_ttnn_tensor(hidden.reshape(1, 1, seq_len, HIDDEN_SIZE), device)
    cos_tt = ttnn.from_torch(cos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    out_tt = tt.forward(hidden_tt, cos=cos_tt, sin=sin_tt)
    out = to_torch(out_tt, expected_shape=(1, 1, seq_len, HIDDEN_SIZE)).reshape(1, seq_len, HIDDEN_SIZE)

    assert_pcc(ref_out, out, 0.99)
