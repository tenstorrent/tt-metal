# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Backbone (Block 1) building blocks, shared by four stubs.

`attention`, `m_l_p`, `decoder_layer` and `tts_backbone` are the same arithmetic at four
granularities -- the decomposition split one layer into its parts -- so the math lives here once
and each stub is a thin adapter over it.  That also means the 26-layer backbone builds its RoPE
and causal-mask tables ONCE and shares them across every layer.

Everything in `__call__` is pure ttnn: `models/common/native_probe.py` graduates a stub only at
`torch_ops == 0`, and `ttnn.from_torch` itself counts.  Weight prep (including the RoPE row
permutation) happens in `from_module`, which runs at build time and is not probed.
"""

from __future__ import annotations

import ttnn

from models.demos.voxtral_tts_full.tt_common import (
    causal_bias_table,
    interleaved_to_halves,
    rope_tables,
    stage,
    stage_weight,
    tt_apply_rope,
    tt_gqa_attention,
    tt_linear,
    tt_merge_heads,
    tt_rms_norm,
    tt_split_heads,
    tt_swiglu,
    verify_rope_permutation,
)

MAX_SEQ = 2048  # config.max_position_embeddings
ROPE_THETA = 1_000_000.0  # config.rope_theta
NORM_EPS = 1e-5  # config.rms_norm_eps
HEAD_DIM = 128


class TtBackboneTables:
    """RoPE cos/sin and the causal mask, precomputed for `max_seq` and sliced per call.

    A probed forward cannot build these (that is torch), and every entry of the mask depends
    only on `rel = j - i`, so the length-S versions are the top-left corners of these."""

    def __init__(self, device, max_seq=MAX_SEQ, head_dim=HEAD_DIM, theta=ROPE_THETA):
        cos, sin = rope_tables(max_seq, head_dim, theta)
        self.cos = stage(cos, device)
        self.sin = stage(sin, device)
        self.bias = stage(causal_bias_table(max_seq), device)
        self.head_dim = head_dim

    def rope(self, seq):
        d = self.head_dim
        return (ttnn.slice(self.cos, [0, 0, 0, 0], [1, 1, seq, d]),
                ttnn.slice(self.sin, [0, 0, 0, 0], [1, 1, seq, d]))

    def mask(self, seq):
        return ttnn.slice(self.bias, [0, 0, 0, 0], [1, 1, seq, seq])


class TtBackboneAttention:
    """GQA (32 query heads over 8 KV heads) with Mistral-native RoPE.

    The checkpoint rotates INTERLEAVED pairs (`view_as_complex`).  Rather than pay for an
    interleave on device, `interleaved_to_halves` folds that pairing into the wq/wk rows at build
    time so the cheap half-split rotation is exact.  The permutation is identical on both sides
    of q.k^T, so it cancels in the scores and leaves v / wo untouched."""

    def __init__(self, tables, weights, dims):
        self.tables = tables
        self.wq, self.wk, self.wv, self.wo = weights
        self.n_heads, self.n_kv_heads, self.head_dim = dims

    @classmethod
    def from_module(cls, device, module, tables, verify=True):
        n_heads = int(getattr(module, "n_heads", 32))
        n_kv_heads = int(getattr(module, "n_kv_heads", 8))
        head_dim = int(getattr(module, "head_dim", HEAD_DIM))
        wq = module.q_proj.detach().float()
        wk = module.k_proj.detach().float()
        if verify:
            verify_rope_permutation(wq, n_heads, head_dim, ROPE_THETA)
            verify_rope_permutation(wk, n_kv_heads, head_dim, ROPE_THETA)
        weights = (
            stage_weight(interleaved_to_halves(wq, n_heads, head_dim), device),
            stage_weight(interleaved_to_halves(wk, n_kv_heads, head_dim), device),
            stage_weight(module.v_proj, device),
            stage_weight(module.o_proj, device),
        )
        return cls(tables, weights, (n_heads, n_kv_heads, head_dim))

    def __call__(self, h, causal=True):
        _, seq, _ = h.shape
        d = self.head_dim
        cos, sin = self.tables.rope(seq)
        q = tt_apply_rope(tt_split_heads(tt_linear(h, self.wq), self.n_heads, d), cos, sin)
        k = tt_apply_rope(tt_split_heads(tt_linear(h, self.wk), self.n_kv_heads, d), cos, sin)
        v = tt_split_heads(tt_linear(h, self.wv), self.n_kv_heads, d)
        mask = self.tables.mask(seq) if causal else None
        attn = tt_gqa_attention(q, k, v, mask, self.n_heads, self.n_kv_heads, d, seq)
        return tt_linear(tt_merge_heads(attn), self.wo)


class TtBackboneMLP:
    """`w2(silu(w1 x) * w3 x)` -- `gate_proj` / `down_proj` / `up_proj` are the reference's
    `w1` / `w2` / `w3`."""

    def __init__(self, weights):
        self.w1, self.w2, self.w3 = weights

    @classmethod
    def from_module(cls, device, module):
        return cls((
            stage_weight(module.gate_proj, device),
            stage_weight(module.down_proj, device),
            stage_weight(module.up_proj, device),
        ))

    def __call__(self, h):
        return tt_swiglu(h, self.w1, self.w2, self.w3)


class TtRMSNorm:
    def __init__(self, weight, eps=NORM_EPS):
        self.weight = weight
        self.eps = eps

    @classmethod
    def from_module(cls, device, module):
        return cls(stage(module.weight.detach().float().view(1, 1, -1), device),
                   float(getattr(module, "eps", NORM_EPS)))

    def __call__(self, x):
        return tt_rms_norm(x, self.weight, self.eps)


class TtBackboneLayer:
    """One pre-norm GQA + SwiGLU block: x -> norm -> attn -> +x -> norm -> mlp -> +x."""

    def __init__(self, input_layernorm, attn, post_attention_layernorm, mlp):
        self.input_layernorm = input_layernorm
        self.self_attn = attn
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp

    @classmethod
    def from_module(cls, device, module, tables, verify=True):
        return cls(
            TtRMSNorm.from_module(device, module.input_layernorm),
            TtBackboneAttention.from_module(device, module.self_attn, tables, verify=verify),
            TtRMSNorm.from_module(device, module.post_attention_layernorm),
            TtBackboneMLP.from_module(device, module.mlp),
        )

    def __call__(self, x, causal=True):
        x = ttnn.add(x, self.self_attn(self.input_layernorm(x), causal=causal))
        return ttnn.add(x, self.mlp(self.post_attention_layernorm(x)))
