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
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

import ttnn

from ..hifigan.conv import TtConv1d

# The real checkpoint's verified encoder config (cosyvoice2.yaml's flow.encoder).
D_MODEL = 512
ATTENTION_HEADS = 8
LINEAR_UNITS = 2048
LAYER_NORM_EPS = 1e-12  # ConformerEncoderLayer/TransformerEncoderLayer's own eps -- NOT the CFM decoder's 1e-5.
EMBED_LAYER_NORM_EPS = 1e-5  # LinearNoSubsampling.out's / after_norm's own eps -- different again.
NUM_BLOCKS = 6  # token-rate Conformer stack (cosyvoice2.yaml's num_blocks)
NUM_UP_BLOCKS = 4  # mel-rate Conformer stack -- hardcoded in real source, not a yaml param
PRE_LOOKAHEAD_LEN = 3
UPSAMPLE_STRIDE = 2


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


class LinearNoSubsamplingRef(nn.Module):
    """`cosyvoice.transformer.subsampling.LinearNoSubsampling` +
    `EspnetRelPositionalEncoding.forward`'s `x*xscale` step folded in (the two are
    always used together in this encoder, and `position_encoding`'s own table is
    `sinusoidal_rel_pos_table_torch`, already built above) -- `right_context=0`,
    `subsampling_rate=1`: despite the name, this changes width (input_size ->
    output_size) but never length."""

    def __init__(self, idim: int = D_MODEL, odim: int = D_MODEL):
        super().__init__()
        self.linear = nn.Linear(idim, odim)
        self.norm = nn.LayerNorm(odim, eps=EMBED_LAYER_NORM_EPS)
        self.xscale = math.sqrt(odim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, idim] -> [B, T, odim], already scaled by `xscale` -- the
        caller builds `pos_emb` separately via `sinusoidal_rel_pos_table_torch`."""
        x = self.norm(self.linear(x))
        return x * self.xscale


class PreLookaheadLayerRef(nn.Module):
    """`cosyvoice.transformer.upsample_encoder.PreLookaheadLayer`, `finalize=True`
    path only (`context` always empty -- see module docstring: `streaming=False`
    is this phase's scope). `conv1` is padded on the RIGHT by `pre_lookahead_len`
    (a genuine look-ahead, the mirror image of the CFM estimator's causal convs,
    not causal itself); `conv2` is causal (left-pad by `kernel_size-1`)."""

    def __init__(self, channels: int = D_MODEL, pre_lookahead_len: int = PRE_LOOKAHEAD_LEN):
        super().__init__()
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=pre_lookahead_len + 1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] -> [B, T, C]."""
        residual = x
        h = x.transpose(1, 2)  # [B, C, T]
        h = F.pad(h, (0, self.pre_lookahead_len), value=0.0)
        h = F.leaky_relu(self.conv1(h))
        h = F.pad(h, (self.conv2.kernel_size[0] - 1, 0), value=0.0)
        h = self.conv2(h)
        return h.transpose(1, 2) + residual


class Upsample1DRef(nn.Module):
    """`cosyvoice.transformer.upsample_encoder.Upsample1D` -- the encoder's OWN
    upsample class (distinct from the CFM decoder's unused `matcha.Upsample1D`):
    nearest-interpolate by `stride`, then a causal (left-pad by `stride*2`) conv
    of kernel `stride*2+1` -- shape-preserving after the interpolation, so the net
    effect is exactly length x `stride`. This is what replaces CosyVoice1's
    separate `length_regulator` for the token-rate -> mel-rate expansion."""

    def __init__(self, channels: int = D_MODEL, stride: int = UPSAMPLE_STRIDE):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(channels, channels, stride * 2 + 1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] -> [B, T*stride, C]."""
        h = x.transpose(1, 2)  # [B, C, T]
        h = F.interpolate(h, scale_factor=float(self.stride), mode="nearest")
        h = F.pad(h, (self.stride * 2, 0), value=0.0)
        h = self.conv(h)
        return h.transpose(1, 2)


class UpsampleConformerEncoderRef(nn.Module):
    """`cosyvoice.transformer.upsample_encoder.UpsampleConformerEncoder`,
    `streaming=False` (`finalize=True`, `context` always empty) only -- see module
    docstring. Two independent `LinearNoSubsamplingRef` instances (`embed`/
    `up_embed`, real source builds them as two separate weight sets, not a shared
    one) and two independent Conformer stacks (6 blocks at token rate, 4 more at
    mel rate -- `NUM_UP_BLOCKS=4` is hardcoded in real source, not a yaml
    parameter)."""

    def __init__(
        self,
        d_model: int = D_MODEL,
        num_heads: int = ATTENTION_HEADS,
        linear_units: int = LINEAR_UNITS,
        num_blocks: int = NUM_BLOCKS,
        num_up_blocks: int = NUM_UP_BLOCKS,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = LinearNoSubsamplingRef(d_model, d_model)
        self.pre_lookahead_layer = PreLookaheadLayerRef(d_model)
        self.encoders = nn.ModuleList(
            [ConformerEncoderLayerRef(d_model, num_heads, linear_units) for _ in range(num_blocks)]
        )
        self.up_layer = Upsample1DRef(d_model, UPSAMPLE_STRIDE)
        self.up_embed = LinearNoSubsamplingRef(d_model, d_model)
        self.up_encoders = nn.ModuleList(
            [ConformerEncoderLayerRef(d_model, num_heads, linear_units) for _ in range(num_up_blocks)]
        )
        self.after_norm = nn.LayerNorm(d_model, eps=EMBED_LAYER_NORM_EPS)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """xs: [B, T, d_model] -> [B, 2T, d_model]. No padding (single full-length
        utterance) -- masks are all-valid throughout, matching this package's
        existing batch=1/no-padding testing scope."""
        b, t_len, _ = xs.shape
        mask = torch.ones(b, 1, t_len, dtype=torch.bool)

        xs = self.embed(xs)
        pos_emb = sinusoidal_rel_pos_table_torch(t_len, self.d_model)
        xs = self.pre_lookahead_layer(xs)
        for layer in self.encoders:
            xs = layer(xs, mask, pos_emb)

        xs = self.up_layer(xs)
        t_len2 = xs.shape[1]
        mask2 = torch.ones(b, 1, t_len2, dtype=torch.bool)
        xs = self.up_embed(xs)
        pos_emb2 = sinusoidal_rel_pos_table_torch(t_len2, self.d_model)
        for layer in self.up_encoders:
            xs = layer(xs, mask2, pos_emb2)

        return self.after_norm(xs)

    @classmethod
    def from_checkpoint(cls, encoder_state_dict: dict, **kwargs) -> "UpsampleConformerEncoderRef":
        """Real weights from `flow.pt`'s `encoder.*` keys (strip that prefix
        first -- see `tt/checkpoint.py`'s `sub_state_dict`). Confirmed
        empirically against the real checkpoint: every key matches this
        class's own `state_dict()` 1:1 EXCEPT `embed`/`up_embed`, where real
        upstream `LinearNoSubsampling` wraps its Linear+LayerNorm in
        `self.out = nn.Sequential(...)` (`embed.out.0.*`/`embed.out.1.*`)
        while this class keeps them as separate `linear`/`norm` attributes --
        remapped here (`out.0.* -> linear.*`, `out.1.* -> norm.*`), nothing
        else needs renaming (`pre_lookahead_layer`, all 6+4 `encoders`/
        `up_encoders` transformer-layer internals, `up_layer`, `after_norm`
        all match real upstream's own attribute names directly)."""
        import re

        remapped = {}
        for k, v in encoder_state_dict.items():
            nk = re.sub(r"^embed\.out\.0\.", "embed.linear.", k)
            nk = re.sub(r"^embed\.out\.1\.", "embed.norm.", nk)
            nk = re.sub(r"^up_embed\.out\.0\.", "up_embed.linear.", nk)
            nk = re.sub(r"^up_embed\.out\.1\.", "up_embed.norm.", nk)
            remapped[nk] = v
        ref = cls(**kwargs)
        ref.load_state_dict(remapped, strict=True)
        return ref


# ---------------------------------------------------------------------------
# TTNN port. [N, L, C] throughout, per this package's convention.
# ---------------------------------------------------------------------------


def _linear_weight(device, weight: torch.Tensor, dtype):
    return ttnn.from_torch(
        weight.detach().float().t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )


def _bias(device, bias: torch.Tensor, dtype):
    return ttnn.from_torch(bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def flow_encoder_trace() -> bool:
    """Cached/traced whole-encoder forward in `TtUpsampleConformerEncoder.__call__`
    (embed -> pre-lookahead -> Conformer stack -> upsample -> Conformer stack), added
    2026-09-22. Unlike the CFM solver (10 replays of one captured step per call), the
    encoder runs ONCE per utterance -- the trace here buys nothing within a single call;
    it pays off across calls at the SAME (token_len, batch_size), e.g. the warm-repeat
    measurement in `rtf_warm.py` or any real session that reuses a prompt length. Off by
    default, matching `TtQwen2LM`'s `use_decode_trace` / the CFM's `COSYVOICE2_FLOW_CFM_TRACE`
    -- opt-in, needs a nonzero `trace_region_size`. `COSYVOICE2_FLOW_ENCODER_TRACE=1` turns it
    on; read at construction time."""
    return os.environ.get("COSYVOICE2_FLOW_ENCODER_TRACE", "0") == "1"


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


class TtPaddedConv1d(TtConv1d):
    """A `ttnn.conv1d` with an explicit, possibly-asymmetric `(pad_left, pad_right)`
    -- confirmed from `ttnn.conv1d`'s own docstring to accept a `[pad_left,
    pad_right]` tuple directly. This module needs a RIGHT-padded conv too
    (`PreLookaheadLayer.conv1`, a genuine look-ahead, not causal), which is why the
    `pad` tuple is a constructor argument rather than always `(k-1, 0)` the way
    `tt/flow/decoder.py`'s `TtCausalConv1d` fixes it.

    A thin subclass of `hifigan.conv.TtConv1d` (ported 2026-09-22, same change and
    for the same reason as `TtCausalConv1d`): `TtConv1d` already generalizes to
    asymmetric `padding` tuples, so this class only carries the `pad` argument
    through and restores the single-tensor return every call site here expects.
    Inherits prepared, per-geometry-cached weights and the relative-L2,
    float64-arbitrated resolver in place of the old absolute `max|out|`-within-2%
    check.
    """

    def __init__(
        self,
        device,
        weight: torch.Tensor,
        bias: torch.Tensor,
        pad: tuple[int, int],
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
    ):
        super().__init__(
            device,
            weight,
            bias,
            stride=1,
            padding=tuple(pad),
            dilation=1,
            groups=1,
            dtype=dtype,
            weights_dtype=weights_dtype,
            high_fidelity=True,
        )

    @classmethod
    def from_module(cls, device, module: nn.Conv1d, pad: tuple[int, int], dtype=ttnn.bfloat16):
        return cls(device, module.weight, module.bias, pad, dtype=dtype)

    def __call__(self, x, input_length: int, batch_size: int = 1):
        out, _ = super().__call__(x, input_length, batch_size)
        return out


class TtLinearNoSubsampling:
    def __init__(self, device, module: LinearNoSubsamplingRef, dtype=ttnn.bfloat16):
        self.weight = _linear_weight(device, module.linear.weight, dtype)
        self.bias = _bias(device, module.linear.bias, dtype)
        self.norm_w = ttnn.from_torch(
            module.norm.weight.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.norm_b = ttnn.from_torch(
            module.norm.bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.xscale = module.xscale

    def __call__(self, x):
        h = ttnn.linear(x, self.weight, bias=self.bias)
        h = ttnn.layer_norm(h, weight=self.norm_w, bias=self.norm_b, epsilon=EMBED_LAYER_NORM_EPS)
        return ttnn.multiply(h, self.xscale)


class TtPreLookaheadLayer:
    def __init__(self, device, module: PreLookaheadLayerRef, dtype=ttnn.bfloat16):
        self.pre_lookahead_len = module.pre_lookahead_len
        self.conv1 = TtPaddedConv1d.from_module(device, module.conv1, pad=(0, module.pre_lookahead_len), dtype=dtype)
        self.conv2 = TtPaddedConv1d.from_module(
            device, module.conv2, pad=(module.conv2.kernel_size[0] - 1, 0), dtype=dtype
        )

    def __call__(self, x, length: int, batch_size: int = 1):
        # conv1 is right-padded by pre_lookahead_len: TtPaddedConv1d's `pad` already
        # encodes that, and ttnn.conv1d's `input_length` is the UNPADDED length
        # (padding is applied internally, same convention TtCausalConv1d already
        # relies on in decoder.py) -- so this is just `self.conv1(x, length, ...)`.
        h = self.conv1(x, length, batch_size)
        h = ttnn.leaky_relu(h, negative_slope=0.01)
        h = self.conv2(h, length, batch_size)
        return ttnn.add(h, x)


class TtUpsample1D:
    """The encoder's OWN upsample (distinct from the CFM decoder's unused
    `matcha.Upsample1D`): `ttnn.repeat_interleave` along the sequence axis is
    exactly nearest-neighbor upsampling by an integer factor (each element
    repeated `stride` times, matching `F.interpolate(..., mode="nearest")` at an
    integer scale) -- simpler than reshaping to `[N,1,L,C]` for `ttnn.upsample`,
    and exact rather than approximate for this integer-scale case."""

    def __init__(self, device, module: Upsample1DRef, dtype=ttnn.bfloat16):
        self.stride = module.stride
        self.conv = TtPaddedConv1d.from_module(device, module.conv, pad=(module.stride * 2, 0), dtype=dtype)

    def __call__(self, x, length: int, batch_size: int = 1):
        h = ttnn.repeat_interleave(x, self.stride, dim=1)
        return self.conv(h, length * self.stride, batch_size)


class TtUpsampleConformerEncoder:
    """`UpsampleConformerEncoderRef` on device -- `streaming=False` only (see
    module docstring).

    **Cached/traced forward (`COSYVOICE2_FLOW_ENCODER_TRACE=1` / `use_trace=True`), added
    2026-09-22.** Unlike the CFM solver, this module runs exactly ONCE per utterance -- so
    a trace does not amortize a loop the way the CFM's ten-Euler-step trace does. What it
    amortizes instead is repeated calls at the SAME `(token_len, batch_size)`: the fixed
    all-zero attention bias and the sinusoidal relative-position table are both pure
    functions of `length`, so at a fixed length the entire captured graph -- embed,
    pre-lookahead, the causal Conformer stack, the upsample, the second Conformer stack,
    and the final LayerNorm -- is byte-for-byte the same computation every time, and only
    the token embeddings themselves (`xs`) differ between calls. Sized here for whatever
    `length` the caller actually passes at capture time -- Stage 1 (whole-utterance,
    non-streaming) passes the real `prompt_token_len + token_len` for a given utterance,
    NOT a fixed streaming-chunk length (see `tt/flow/flow.py`'s `inference`, which calls
    this with `full_token.shape[1]`); a streaming caller passing a 100-frame chunk instead
    would get a trace captured/cached for THAT geometry, which is a different, equally
    valid use of this same mechanism, not a special case of it.

    **Measured real result, real checkpoint, real Stage 1 lengths (2026-09-22): capture
    always fails.** `TtRelPositionMultiHeadedAttention._rel_shift` does a deliberate host
    round-trip once per Conformer layer (see that method's own docstring) -- all 10 layers
    this encoder calls hit it, so `begin_trace_capture`/`body()`/`end_trace_capture` always
    raises `TT_FATAL: Reads are not supported during trace capture`, at every geometry, not
    just some. The fallback in `_call_traced` below is correct (every existing "traced" PCC
    test actually exercises this exact fallback, hence passing with PCC identical to eager
    -- not a false positive, just not evidence tracing works) and now remembers the failure
    (`_trace_unavailable`) rather than re-paying a doomed capture attempt's cost (measured:
    ~2.5x plain eager) on every subsequent call. Making this module genuinely traceable
    needs `_rel_shift` ported to run natively on device -- a real, separate piece of work,
    flagged here rather than solved.
    """

    def __init__(self, device, module: UpsampleConformerEncoderRef, dtype=ttnn.bfloat16, use_trace: bool | None = None):
        self.device = device
        self.d_model = module.d_model
        self.embed = TtLinearNoSubsampling(device, module.embed, dtype=dtype)
        self.pre_lookahead_layer = TtPreLookaheadLayer(device, module.pre_lookahead_layer, dtype=dtype)
        self.encoders = [TtConformerEncoderLayer(device, layer, dtype=dtype) for layer in module.encoders]
        self.up_layer = TtUpsample1D(device, module.up_layer, dtype=dtype)
        self.up_embed = TtLinearNoSubsampling(device, module.up_embed, dtype=dtype)
        self.up_encoders = [TtConformerEncoderLayer(device, layer, dtype=dtype) for layer in module.up_encoders]
        self.after_norm_w = ttnn.from_torch(
            module.after_norm.weight.detach().float().reshape(1, 1, -1),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.after_norm_b = ttnn.from_torch(
            module.after_norm.bias.detach().float().reshape(1, 1, -1),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.use_trace = flow_encoder_trace() if use_trace is None else use_trace
        # Keep the captured trace across calls at the same (length, batch_size) -- same
        # naming convention as the CFM's COSYVOICE2_CFM_TRACE_CACHE.
        self._cache_trace = os.environ.get("COSYVOICE2_ENCODER_TRACE_CACHE", "1") != "0"
        self._trace_id = None
        self._trace_key = None
        self._next_h = None
        self._xs_buf = self._bias1_buf = self._pos_emb_buf = self._bias2_buf = self._pos_emb2_buf = None
        # Measured 2026-09-22, real checkpoint, real Stage 1 lengths: capture always fails
        # with `TT_FATAL: Reads are not supported during trace capture`, from
        # `TtRelPositionMultiHeadedAttention._rel_shift`'s deliberate host round-trip (see
        # that method's own docstring) -- every one of the 10 Conformer layers this encoder
        # calls hits it, so this module cannot be traced AT ALL as currently built, not
        # merely at some geometries. `_call_traced` still tries once per new geometry (the
        # graceful per-call fallback below is what every existing PCC test's "traced" case
        # actually exercises, and it is correct -- eager, exactly, just not traced) and
        # remembers a proven failure here so it does not keep re-paying the failed attempt's
        # cost (measured: ~2.5x plain eager, from the 2 warm-up passes plus the aborted
        # capture, every call) once it is known to be futile. Making this module genuinely
        # traceable needs `_rel_shift`'s relative-position shift ported to run natively on
        # device -- out of scope for this round, flagged here rather than silently eaten by
        # the fallback.
        self._trace_unavailable = False

    def _pos_emb(self, t_len: int):
        emb = sinusoidal_rel_pos_table_torch(t_len, self.d_model)
        return ttnn.from_torch(emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

    def __call__(self, xs, length: int, batch_size: int = 1, use_trace: bool | None = None):
        """xs: ttnn [B, T, d_model] -> ttnn [B, T*stride, d_model]. No padding
        (single full-length utterance, matching this package's batch=1/
        no-padding testing scope) -- `attn_bias` is all-zero (all-valid) at both
        stages."""
        if use_trace is None:
            use_trace = self.use_trace
        if not use_trace:
            return self._call_eager(xs, length, batch_size)
        return self._call_traced(xs, length, batch_size)

    def _call_eager(self, xs, length: int, batch_size: int = 1):
        bias1 = ttnn.from_torch(
            torch.zeros(batch_size, 1, 1, length), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        h = self.embed(xs)
        pos_emb = self._pos_emb(length)
        h = self.pre_lookahead_layer(h, length, batch_size)
        for layer in self.encoders:
            h = layer(h, pos_emb, bias1)

        h = self.up_layer(h, length, batch_size)
        length2 = length * self.up_layer.stride
        bias2 = ttnn.from_torch(
            torch.zeros(batch_size, 1, 1, length2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        h = self.up_embed(h)
        pos_emb2 = self._pos_emb(length2)
        for layer in self.up_encoders:
            h = layer(h, pos_emb2, bias2)

        return ttnn.layer_norm(h, weight=self.after_norm_w, bias=self.after_norm_b, epsilon=EMBED_LAYER_NORM_EPS)

    # ------------------------------------------------------------------
    # Traced path. One capture per distinct (length, batch_size); replayed once per call
    # at that geometry. See the class docstring.
    # ------------------------------------------------------------------

    def _trace_key_for(self, length: int, batch_size: int):
        return (length, batch_size)

    def _release_trace(self) -> None:
        if self._trace_id is not None:
            ttnn.release_trace(self.device, self._trace_id)
            self._trace_id = None
        self._trace_key = None
        self._next_h = None  # allocated inside the capture; release_trace reclaims it
        for name in ("_xs_buf", "_bias1_buf", "_pos_emb_buf", "_bias2_buf", "_pos_emb2_buf"):
            t = getattr(self, name, None)
            if t is not None:
                ttnn.deallocate(t)
                setattr(self, name, None)

    def release_encoder_trace(self) -> None:
        """Public wrapper, matching `TtQwen2LM.release_decode_trace` / `TtCausalConditionalCFM.release_cfm_trace`."""
        self._release_trace()

    def _capture(self, xs, length: int, batch_size: int):
        self._xs_buf = ttnn.from_torch(
            torch.zeros(batch_size, length, self.d_model),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.copy(xs, self._xs_buf)
        # Bias and position tables are pure functions of `length` (no padding, this
        # package's whole testing scope) -- built once here, at capture time, and never
        # refreshed on replay or reuse, unlike the CFM's per-step buffers.
        self._bias1_buf = ttnn.from_torch(
            torch.zeros(batch_size, 1, 1, length),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._pos_emb_buf = self._pos_emb(length)
        length2 = length * self.up_layer.stride
        self._bias2_buf = ttnn.from_torch(
            torch.zeros(batch_size, 1, 1, length2),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._pos_emb2_buf = self._pos_emb(length2)

        def body():
            h = self.embed(self._xs_buf)
            h = self.pre_lookahead_layer(h, length, batch_size)
            for layer in self.encoders:
                h = layer(h, self._pos_emb_buf, self._bias1_buf)
            h = self.up_layer(h, length, batch_size)
            h = self.up_embed(h)
            for layer in self.up_encoders:
                h = layer(h, self._pos_emb2_buf, self._bias2_buf)
            return ttnn.layer_norm(h, weight=self.after_norm_w, bias=self.after_norm_b, epsilon=EMBED_LAYER_NORM_EPS)

        # Warm the program cache and every conv's prepared-weight / verified-config cache
        # (TtPaddedConv1d in this module, TtConvTranspose1d inside TtUpsample1D) before
        # capture -- both are host work a trace cannot contain, and both are already keyed
        # by (input_length, batch_size), so two full eager passes at this exact geometry
        # populate every cache capture will need.
        for _ in range(2):
            ttnn.deallocate(body())
        ttnn.synchronize_device(self.device)

        self._trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        try:
            self._next_h = body()  # allocated inside the capture -- see TtCausalConditionalCFM._capture's note
        finally:
            ttnn.end_trace_capture(self.device, self._trace_id, cq_id=0)
        self._trace_key = self._trace_key_for(length, batch_size)

    def _reuse_trace(self, xs, length: int, batch_size: int) -> bool:
        if not self._cache_trace or self._trace_key != self._trace_key_for(length, batch_size):
            return False
        if self._trace_id is None:
            return False
        ttnn.copy(xs, self._xs_buf)
        # See TtCausalConditionalCFM._reuse_trace's note: `_capture` syncs before its first
        # `execute_trace` (after its own warm-up writes); a reuse has no equivalent sync on
        # its path to the replay otherwise.
        ttnn.synchronize_device(self.device)
        return True

    def _call_traced(self, xs, length: int, batch_size: int = 1):
        if self._trace_unavailable:
            return self._call_eager(xs, length, batch_size)

        traced = False
        try:
            if not self._reuse_trace(xs, length, batch_size):
                self._release_trace()  # a stale trace of a different geometry must go first
                self._capture(xs, length, batch_size)
            traced = True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"encoder trace capture unavailable, falling back to eager permanently: {e}")
            self._release_trace()
            self._trace_unavailable = True

        if not traced:
            return self._call_eager(xs, length, batch_size)

        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=True)
        # Always clone out: `_next_h` is allocated INSIDE the capture, so its address is
        # reused by every future replay -- handing it to the caller uncloned would let the
        # NEXT call silently overwrite output the caller may still be holding.
        return ttnn.clone(self._next_h)
