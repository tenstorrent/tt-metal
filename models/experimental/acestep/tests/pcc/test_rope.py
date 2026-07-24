# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 RoPE vs ttnn.experimental.rotary_embedding_hf (reuse ttnn op).

ACE-Step self-attention uses Qwen3RotaryEmbedding (theta=1e6) + apply_rotary_pos_emb,
which is HF rotate_half convention. TTNN provides `rotary_embedding_hf` that matches
this convention exactly, so we reuse it directly (no custom RoPE class).

We build cos/sin from the REAL Qwen3RotaryEmbedding to stay faithful to ACE-Step, then
apply on-device and compare against HF apply_rotary_pos_emb.

Prefill contract (from the op docstring):
  input [batch=1, num_heads, seq_len, head_dim], cos/sin [1, 1, seq_len, head_dim], TILE layout.
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from models.experimental.acestep.tests.test_utils import (
    HEAD_DIM,
    NUM_ATTENTION_HEADS,
    NUM_KEY_VALUE_HEADS,
    ROPE_THETA,
    SEQUENCE_LENGTHS,
    assert_pcc,
    require_single_device,
    to_torch,
)


def _hf_rotary(seq_len):
    cfg = SimpleNamespace(
        head_dim=HEAD_DIM,
        hidden_size=HEAD_DIM * NUM_ATTENTION_HEADS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        max_position_embeddings=32768,
        rope_parameters={"rope_type": "default", "rope_theta": ROPE_THETA},
        rope_scaling=None,
    )
    rope = Qwen3RotaryEmbedding(cfg)
    position_ids = torch.arange(seq_len).unsqueeze(0)  # [1, seq_len]
    dummy = torch.zeros(1, seq_len, HEAD_DIM)
    cos, sin = rope(dummy, position_ids)  # [1, seq_len, head_dim]
    return cos, sin


@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS, ids=[f"S{seq_len}" for seq_len in SEQUENCE_LENGTHS])
@pytest.mark.parametrize("n_heads", [NUM_ATTENTION_HEADS, NUM_KEY_VALUE_HEADS], ids=["q16", "kv8"])
def test_rope_vs_qwen3(device, seq_len, n_heads):
    require_single_device(device)
    torch.manual_seed(42)

    cos, sin = _hf_rotary(seq_len)  # [1, seq_len, head_dim], float32

    x = torch.randn(1, n_heads, seq_len, HEAD_DIM, dtype=torch.float32)

    # HF reference: apply_rotary_pos_emb unsqueezes cos/sin at dim=1 to broadcast over heads.
    with torch.no_grad():
        ref, _ = apply_rotary_pos_emb(x, x, cos, sin)  # rotate q (use x for both)

    # Device: cos/sin must be [1, 1, seq_len, head_dim] TILE layout.
    cos_tt = ttnn.from_torch(cos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    x_tt = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    out_tt = ttnn.experimental.rotary_embedding_hf(x_tt, cos_tt, sin_tt, is_decode_mode=False)
    out = to_torch(out_tt, expected_shape=(1, n_heads, seq_len, HEAD_DIM))

    assert_pcc(ref, out, 0.99)
