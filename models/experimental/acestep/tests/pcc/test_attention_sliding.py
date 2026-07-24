# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 sliding-window self-attention (window=128) vs custom AceStepAttention.

Sliding layers (layer_types[i] == "sliding_attention") attend only to |i-j| <= 128
(bidirectional local window). We reuse the genuine reference `create_4d_mask` to build the
additive mask and feed it to our custom module's attn_mask path — same AceStepAttention class,
no new code, just exercises the mask branch.
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
    SLIDING_WINDOW,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)

SLIDING_LAYER_IDX = 0  # layer_types[0] == "sliding_attention"
SEQ_LENS = [256, 512, 1024]  # > window so the mask actually constrains


def _lazy_linear_T(weight, device):
    return make_lazy_weight(
        weight.detach().clone().transpose(-1, -2).contiguous(),
        device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=[f"S{s}" for s in SEQ_LENS])
def test_self_attention_sliding_vs_hf(device, seq_len):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    cfg._attn_implementation = "eager"
    ref = m.AceStepAttention(cfg, layer_idx=SLIDING_LAYER_IDX, is_cross_attention=False, is_causal=False).eval()
    assert ref.sliding_window == SLIDING_WINDOW, f"expected sliding_window={SLIDING_WINDOW}, got {ref.sliding_window}"
    with torch.no_grad():
        for lin in (ref.q_proj, ref.k_proj, ref.v_proj, ref.o_proj):
            lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
        ref.q_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(ref.q_norm.weight))
        ref.k_norm.weight.copy_(1.0 + 0.02 * torch.randn_like(ref.k_norm.weight))

    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.float32)

    rope = Qwen3RotaryEmbedding(cfg)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rope(hidden, position_ids)

    # Genuine reference sliding mask: bidirectional local window |i-j|<=128.
    mask = m.create_4d_mask(
        seq_len=seq_len,
        dtype=torch.float32,
        device=hidden.device,
        attention_mask=None,
        sliding_window=SLIDING_WINDOW,
        is_sliding_window=True,
        is_causal=False,
    )  # [1, 1, seq, seq] additive (0 keep / -inf mask)

    with torch.no_grad():
        ref_out, _ = ref(hidden_states=hidden, attention_mask=mask, position_embeddings=(cos, sin))

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
            sliding_window=SLIDING_WINDOW,
        )
    )

    hidden_tt = to_ttnn_tensor(hidden.reshape(1, 1, seq_len, HIDDEN_SIZE), device)
    cos_tt = ttnn.from_torch(cos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    # Additive mask on device, TILE layout, broadcast over heads [1,1,seq,seq].
    mask_tt = ttnn.from_torch(mask, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    out_tt = tt.forward(hidden_tt, cos=cos_tt, sin=sin_tt, attn_mask=mask_tt)
    out = to_torch(out_tt, expected_shape=(1, 1, seq_len, HIDDEN_SIZE)).reshape(1, seq_len, HIDDEN_SIZE)

    assert_pcc(ref_out, out, 0.99)
