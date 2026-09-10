# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""The Conformer layer `UpsampleConformerEncoder` is built from: relative-position
multi-head self-attention (`RelPositionMultiHeadedAttention`, Transformer-XL /
Dai et al. 2019 style -- NOT RoPE, and no analog elsewhere in this codebase) plus
a plain feed-forward, confirmed against real upstream source directly
(`cosyvoice/transformer/{attention,encoder_layer,positionwise_feed_forward,
embedding}.py`) -- not assumed from CosyVoice1's plain (non-relative) Conformer.

**The Conformer layer is smaller than its name suggests, verified from the real
checkpoint's own config** (`cosyvoice2.yaml`'s `flow.encoder`): `macaron_style=False`
and `use_cnn_module=False`. Tracing `ConformerEncoderLayer.forward` at those flags,
the macaron-FFN branch and the `ConvolutionModule` branch are both skipped entirely
(`feed_forward_macaron is None`, `conv_module is None`), so the layer collapses to a
**plain pre-norm Transformer layer**: `x = x + self_attn(norm(x))`, then
`x = x + FFN(norm(x))`. No `ConvolutionModule` is ported here at all -- it is never
instantiated for this checkpoint.

**The genuinely novel mechanism is the relative-position attention itself.** Two
learnable per-head bias vectors (`pos_bias_u`, `pos_bias_v`, shape `[heads, d_k]`)
are added to `q` before two separate score matmuls: `matrix_ac = (q+pos_bias_u) @ k^T`
(ordinary content-content) and `matrix_bd = (q+pos_bias_v) @ p^T` (content-position),
where `p = linear_pos(pos_emb)` and `pos_emb` is a single `[1, 2T-1, d_model]`
sinusoidal table spanning every relative offset from `+(T-1)` to `-(T-1)` in one
shot (not a per-pair lookup). `matrix_bd` comes out shaped `(T, 2T-1)` and `rel_shift`
-- a zero-pad + reshape + slice sequence -- turns it into `(T, T)`. This was verified
numerically (not just read and trusted) before porting: after `rel_shift`,
`matrix_bd[i, j]` exactly equals `(q_i + pos_bias_v) . pe[offset = i - j]`, matching
the mechanism's closed-form definition, not just its own re-derivation --
`tests/pcc/test_conformer_encoder.py::test_rel_shift_recovers_relative_offset_alignment`
keeps this claim checked permanently, not just verified once in scratch.

**`rel_shift` runs via a host round-trip, not natively on device.** Torch's `rel_shift`
`.view()`-reinterprets a flat, contiguous row-major buffer as if it had a different
last-two-dim split (`(T, 2T)` -> `(2T, T)`) -- valid only because torch tensors are
flat row-major. `ttnn` `TILE_LAYOUT` tensors are swizzled into 32x32 tiles, not flat
row-major, and a *much* simpler reshape (inserting a size-1 dim, in the CFM estimator
phase) already hit `TT_FATAL: Invalid arguments to reshape` there; a reshape that
reinterprets which axis is which is a strictly harder case. Bringing `matrix_bd` to
torch, applying the literal real `rel_shift` (transcribed verbatim below, the same
function the closed-form check verifies), and re-uploading the `(T, T)` result is
the same category of tradeoff this bring-up has made repeatedly (Qwen2LM's per-token
decode round-trip, the CFM's per-Euler-step round-trip): correctness first, a
flagged and deliberate deferral, not a hidden shortcut. A native on-device version
(`ROW_MAJOR_LAYOUT`, which may not carry the same tile restriction) is a plausible
follow-up if perf work is ever in scope -- not attempted here without evidence it is
needed.

`EspnetRelPositionalEncoding`'s table depends only on `d_model`/`T`, not on any
learned weight, so -- like RoPE tables elsewhere in this codebase -- it is computed
on the host and uploaded once per length, not rebuilt on device.

Scope of this module: the Conformer layer in isolation (attention + FFN), not yet
the outer `UpsampleConformerEncoder` (`PreLookaheadLayer`, the encoder's own
`Upsample1D`, the two 6-block/4-block stacks) -- a separate, later piece, matching
this bring-up's "validate the novel mechanism in isolation first" discipline (the
same split the CFM estimator phase used).

`matcha-tts`/`diffusers`/`conformer`/`wenet`/`espnet` are not installed in this
environment, so -- same situation as the CFM estimator, HiFT and SineGen2 --
`*Ref` classes are a line-by-line transcription of the real source, built from
genuine `torch.nn`/`torch.nn.functional` primitives throughout, not an importable
real class to call directly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn

# The real checkpoint's verified encoder config (cosyvoice2.yaml's flow.encoder).
D_MODEL = 512
ATTENTION_HEADS = 8
LINEAR_UNITS = 2048
LAYER_NORM_EPS = 1e-12  # ConformerEncoderLayer/TransformerEncoderLayer's own eps -- NOT the CFM decoder's 1e-5.


# ---------------------------------------------------------------------------
# torch reference
# ---------------------------------------------------------------------------


def sinusoidal_rel_pos_table_torch(t_len: int, d_model: int) -> torch.Tensor:
    """`cosyvoice.transformer.embedding.EspnetRelPositionalEncoding`'s `pe` table
    at `offset=0`, verbatim (`extend_pe` + `position_encoding` collapsed together
    for the only case this port needs). Returns `[1, 2*t_len-1, d_model]`: index
    `0` is relative offset `+(t_len-1)`, index `t_len-1` is offset `0`, index
    `2*t_len-2` is offset `-(t_len-1)`.
    """
    pe_positive = torch.zeros(t_len, d_model)
    pe_negative = torch.zeros(t_len, d_model)
    position = torch.arange(0, t_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
    pe_positive[:, 0::2] = torch.sin(position * div_term)
    pe_positive[:, 1::2] = torch.cos(position * div_term)
    pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
    pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
    pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
    pe_negative = pe_negative[1:].unsqueeze(0)
    return torch.cat([pe_positive, pe_negative], dim=1)


def rel_shift_torch(x: torch.Tensor) -> torch.Tensor:
    """`RelPositionMultiHeadedAttention.rel_shift`, verbatim. x: [B, H, T, 2T-1] ->
    [B, H, T, T], `out[..., i, j] == x[..., i, (T-1) - (i - j)]` (verified
    numerically, not just read -- see module docstring)."""
    zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1), device=x.device, dtype=x.dtype)
    x_padded = torch.cat([zero_pad, x], dim=-1)
    x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
    return x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]


class RelPositionMultiHeadedAttentionRef(nn.Module):
    """`cosyvoice.transformer.attention.RelPositionMultiHeadedAttention`, verbatim
    (dropout omitted -- inference only, matching this package's other `*Ref`
    classes)."""

    def __init__(self, n_head: int, n_feat: int):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def forward_qkv(self, query, key, value):
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, pos_emb: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """mask: [B, 1, T] or [B, T, T], True = valid (upstream's `~make_pad_mask` convention)."""
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = rel_shift_torch(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        invalid = (~mask).unsqueeze(1)  # (batch, 1, *, time2)
        scores = scores.masked_fill(invalid, -float("inf"))
        attn = torch.softmax(scores, dim=-1).masked_fill(invalid, 0.0)
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(query.size(0), -1, self.h * self.d_k)
        return self.linear_out(x)


class PositionwiseFeedForwardRef(nn.Module):
    """`cosyvoice.transformer.positionwise_feed_forward.PositionwiseFeedForward`
    at the real config's activation (`swish`, i.e. `nn.SiLU`)."""

    def __init__(self, idim: int, hidden_units: int):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(F.silu(self.w_1(x)))


class ConformerEncoderLayerRef(nn.Module):
    """`cosyvoice.transformer.encoder_layer.ConformerEncoderLayer` reduced to the
    real config's active path (`normalize_before=True`, no macaron FFN, no conv
    module -- see module docstring)."""

    def __init__(self, size: int = D_MODEL, num_heads: int = ATTENTION_HEADS, linear_units: int = LINEAR_UNITS):
        super().__init__()
        self.self_attn = RelPositionMultiHeadedAttentionRef(num_heads, size)
        self.feed_forward = PositionwiseFeedForwardRef(size, linear_units)
        self.norm_mha = nn.LayerNorm(size, eps=LAYER_NORM_EPS)
        self.norm_ff = nn.LayerNorm(size, eps=LAYER_NORM_EPS)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm_mha(x)
        x = residual + self.self_attn(h, h, h, pos_emb, mask)

        residual = x
        h = self.norm_ff(x)
        x = residual + self.feed_forward(h)
        return x


# ---------------------------------------------------------------------------
# TTNN port. [N, L, C] throughout, per this package's convention.
# ---------------------------------------------------------------------------


def _linear_weight(device, weight: torch.Tensor, dtype):
    return ttnn.from_torch(
        weight.detach().float().t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )


def _bias(device, bias: torch.Tensor, dtype):
    return ttnn.from_torch(bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


class TtRelPositionMultiHeadedAttention:
    def __init__(self, device, module: RelPositionMultiHeadedAttentionRef, dtype=ttnn.bfloat16):
        self.device = device
        self.h, self.d_k = module.h, module.d_k
        self.scale = module.d_k**-0.5
        self.wq = _linear_weight(device, module.linear_q.weight, dtype)
        self.bq = _bias(device, module.linear_q.bias, dtype)
        self.wk = _linear_weight(device, module.linear_k.weight, dtype)
        self.bk = _bias(device, module.linear_k.bias, dtype)
        self.wv = _linear_weight(device, module.linear_v.weight, dtype)
        self.bv = _bias(device, module.linear_v.bias, dtype)
        self.wo = _linear_weight(device, module.linear_out.weight, dtype)
        self.bo = _bias(device, module.linear_out.bias, dtype)
        self.w_pos = _linear_weight(device, module.linear_pos.weight, dtype)
        # [1, H, 1, d_k] -- broadcasts directly against a (B, H, T, d_k) tensor, a
        # simpler equivalent of the reference's (B, T, H, d_k)-broadcast + double
        # transpose (the transpose dance is a no-op for the actual arithmetic;
        # PCC-tested against the reference below, not assumed).
        self.pos_bias_u = ttnn.from_torch(
            module.pos_bias_u.detach().float().reshape(1, self.h, 1, self.d_k),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.pos_bias_v = ttnn.from_torch(
            module.pos_bias_v.detach().float().reshape(1, self.h, 1, self.d_k),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.dtype = dtype

    def _heads(self, x, b, t):
        return ttnn.transpose(ttnn.reshape(x, (b, t, self.h, self.d_k)), 1, 2)  # [B, H, T, d_k]

    def _rel_shift(self, matrix_bd, b, t):
        """Host round-trip -- see module docstring for why this does not run
        natively on device."""
        host = ttnn.to_torch(matrix_bd).float()
        shifted = rel_shift_torch(host)
        return ttnn.from_torch(shifted, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)

    def __call__(self, x, pos_emb, attn_bias):
        """x: [B, T, D]. pos_emb: [1, 2T-1, D] (host-built, uploaded by the
        caller). attn_bias: additive, broadcastable to [B, 1, T, T] (0 = valid,
        large-negative = masked)."""
        b, t, _ = x.shape
        q = self._heads(ttnn.linear(x, self.wq, bias=self.bq), b, t)
        k = self._heads(ttnn.linear(x, self.wk, bias=self.bk), b, t)
        v = self._heads(ttnn.linear(x, self.wv, bias=self.bv), b, t)

        p = ttnn.linear(pos_emb, self.w_pos)
        p = self._heads(p, 1, p.shape[1])  # [1, H, 2T-1, d_k]

        q_u = ttnn.add(q, self.pos_bias_u)
        q_v = ttnn.add(q, self.pos_bias_v)

        matrix_ac = ttnn.matmul(q_u, ttnn.transpose(k, -2, -1))
        matrix_bd_raw = ttnn.matmul(q_v, ttnn.transpose(p, -2, -1))  # [B, H, T, 2T-1]
        matrix_bd = self._rel_shift(matrix_bd_raw, b, t)

        scores = ttnn.multiply(ttnn.add(matrix_ac, matrix_bd), self.scale)
        scores = ttnn.add(scores, attn_bias)
        attn = ttnn.softmax(scores, dim=-1)

        out = ttnn.matmul(attn, v)  # [B, H, T, d_k]
        out = ttnn.reshape(ttnn.transpose(out, 1, 2), (b, t, self.h * self.d_k))
        return ttnn.linear(out, self.wo, bias=self.bo)


class TtConformerEncoderLayer:
    def __init__(self, device, module: ConformerEncoderLayerRef, dtype=ttnn.bfloat16):
        self.self_attn = TtRelPositionMultiHeadedAttention(device, module.self_attn, dtype=dtype)
        self.w1 = _linear_weight(device, module.feed_forward.w_1.weight, dtype)
        self.b1 = _bias(device, module.feed_forward.w_1.bias, dtype)
        self.w2 = _linear_weight(device, module.feed_forward.w_2.weight, dtype)
        self.b2 = _bias(device, module.feed_forward.w_2.bias, dtype)
        self.norm_mha_w = ttnn.from_torch(
            module.norm_mha.weight.detach().float().reshape(1, 1, -1),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.norm_mha_b = ttnn.from_torch(
            module.norm_mha.bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.norm_ff_w = ttnn.from_torch(
            module.norm_ff.weight.detach().float().reshape(1, 1, -1),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.norm_ff_b = ttnn.from_torch(
            module.norm_ff.bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, x, pos_emb, attn_bias):
        h = ttnn.layer_norm(x, weight=self.norm_mha_w, bias=self.norm_mha_b, epsilon=LAYER_NORM_EPS)
        x = ttnn.add(x, self.self_attn(h, pos_emb, attn_bias))

        h = ttnn.layer_norm(x, weight=self.norm_ff_w, bias=self.norm_ff_b, epsilon=LAYER_NORM_EPS)
        h = ttnn.linear(h, self.w1, bias=self.b1)
        h = ttnn.silu(h)
        h = ttnn.linear(h, self.w2, bias=self.b2)
        return ttnn.add(x, h)
