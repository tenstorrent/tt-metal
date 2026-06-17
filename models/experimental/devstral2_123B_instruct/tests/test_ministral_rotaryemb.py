# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side numerical parity for the Devstral-2-123B rotary embedding.

These tests do not need a device — they exercise the pure-python table builders
(``precompute_cos_sin``, ``compute_llama4_scale``) against HF's
``Ministral3RotaryEmbedding`` + ``apply_rotary_pos_emb`` using the real Devstral-2 RoPE
configuration (YaRN with ``factor=16``, ``original_max_position_embeddings=16384``, ...).

The TT module precomputes ``cos`` / ``sin`` in **pairwise-interleaved** layout (the form
expected by ``ttnn.experimental.rotary_embedding_llama``); HF works in split-half. The
equivalence test below verifies the two paths agree once Q is permuted between layouts.
"""

from __future__ import annotations

import pytest
import torch
from transformers.models.ministral3.modeling_ministral3 import (
    Ministral3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from models.experimental.devstral2_123B_instruct.tests._devstral_weights import (
    devstral2_test_model_args,
    require_text_config,
)
from models.experimental.devstral2_123B_instruct.tt.model_args import Devstral2Args
from models.experimental.devstral2_123B_instruct.tt.tt_ministral_rotary_emb import (
    compute_llama4_scale,
    precompute_cos_sin,
)


@pytest.fixture(scope="module")
def text_cfg():
    return require_text_config()


@pytest.fixture(scope="module")
def args(text_cfg):
    return devstral2_test_model_args(text_cfg)


@pytest.mark.parametrize("max_pos", [4096])
def test_yarn_inv_freq_matches_hf(args, max_pos):
    """YaRN-scaled cos/sin tables match HF ``Ministral3RotaryEmbedding``."""
    hf = Ministral3RotaryEmbedding(_minimal_hf_cfg_like(args)).eval()
    pos = torch.arange(max_pos).unsqueeze(0)
    cos_hf, sin_hf = hf(torch.zeros((1, max_pos, args.head_dim), dtype=torch.float32), pos)
    cos_hf = cos_hf[0]  # (max_pos, head_dim) in HF split-half layout
    sin_hf = sin_hf[0]

    cos_tt, sin_tt = precompute_cos_sin(args.head_dim, max_pos, args.rope)
    # Convert TT (interleaved) → split-half layout for comparison with HF: cos_tt[2k] == cos_hf[k],
    # cos_tt[2k+1] == cos_hf[k + D/2].
    half = args.head_dim // 2
    cos_tt_as_split = torch.empty_like(cos_hf)
    sin_tt_as_split = torch.empty_like(sin_hf)
    cos_tt_as_split[:, :half] = cos_tt[:, 0::2]
    cos_tt_as_split[:, half:] = cos_tt[:, 1::2]
    sin_tt_as_split[:, :half] = sin_tt[:, 0::2]
    sin_tt_as_split[:, half:] = sin_tt[:, 1::2]

    torch.testing.assert_close(cos_tt_as_split, cos_hf.to(torch.float32), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(sin_tt_as_split, sin_hf.to(torch.float32), atol=1e-3, rtol=1e-3)


def test_llama4_scale_matches_hf_formula(args):
    pos = torch.arange(args.max_seq_len).to(torch.float32)
    expected = 1.0 + args.rope.llama_4_scaling_beta * torch.log1p(
        torch.floor(pos / args.rope.original_max_position_embeddings)
    )
    actual = compute_llama4_scale(
        args.max_seq_len,
        args.rope.original_max_position_embeddings,
        args.rope.llama_4_scaling_beta,
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_rope_layout_equivalence(args):
    """Permute(split→interleaved) ∘ pairwise-RoPE ∘ unpermute == HF split-half RoPE (up to dtype)."""
    head_dim = args.head_dim
    half = head_dim // 2
    seq = 64

    cos_tt, sin_tt = precompute_cos_sin(head_dim, args.max_seq_len, args.rope)
    pos = torch.arange(seq)

    hf = Ministral3RotaryEmbedding(_minimal_hf_cfg_like(args)).eval()
    cos_hf, sin_hf = hf(torch.zeros((1, seq, head_dim), dtype=torch.float32), pos.unsqueeze(0))

    q = torch.randn(1, args.num_attention_heads, seq, head_dim, dtype=torch.float32)
    k = torch.randn(1, args.num_key_value_heads, seq, head_dim, dtype=torch.float32)

    # HF path
    q_hf, _ = apply_rotary_pos_emb(q, k, cos_hf, sin_hf)

    # TT path: split-half → interleaved, pairwise rotate, interleaved → split-half.
    idx = torch.empty(head_dim, dtype=torch.long)
    idx[0::2] = torch.arange(half)
    idx[1::2] = torch.arange(half, head_dim)
    q_perm = q[..., idx]  # interleaved layout

    cos_tt_b = cos_tt[pos].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
    sin_tt_b = sin_tt[pos].unsqueeze(0).unsqueeze(0)

    def _rotate_pairwise(x):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = -x_odd
        out[..., 1::2] = x_even
        return out

    q_rot_interleaved = q_perm * cos_tt_b + _rotate_pairwise(q_perm) * sin_tt_b

    # Inverse permutation: interleaved[i] = split_half[idx[i]] ⇒ split_half[idx[i]] = interleaved[i]
    inv_idx = torch.empty(head_dim, dtype=torch.long)
    inv_idx[idx] = torch.arange(head_dim)
    q_tt = q_rot_interleaved[..., inv_idx]

    torch.testing.assert_close(q_tt, q_hf, atol=1e-3, rtol=1e-3)


def _minimal_hf_cfg_like(args: Devstral2Args):
    """Build a thin HF config sufficient for ``Ministral3RotaryEmbedding.__init__``."""
    from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

    return Ministral3Config(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        max_position_embeddings=args.max_seq_len,
        rope_parameters={
            "type": args.rope.rope_type,
            "rope_theta": args.rope.rope_theta,
            "factor": args.rope.factor,
            "original_max_position_embeddings": args.rope.original_max_position_embeddings,
            "beta_fast": args.rope.beta_fast,
            "beta_slow": args.rope.beta_slow,
            "mscale": args.rope.mscale,
            "mscale_all_dim": args.rope.mscale_all_dim,
            "llama_4_scaling_beta": args.rope.llama_4_scaling_beta,
            "max_position_embeddings": args.max_seq_len,
        },
    )
