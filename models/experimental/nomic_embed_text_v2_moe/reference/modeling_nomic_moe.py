# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vendored PyTorch reference for nomic-ai/nomic-embed-text-v2-moe.

This is the golden model the TTNN port is validated against. It covers only the inference
path of one pinned checkpoint; see `configuration_nomic_moe.py` for the assumptions that
are asserted rather than branched on.

Two properties are deliberate:

*   **Name isomorphism.** Every parameter sits at exactly the upstream path (`attn.Wqkv`,
    `emb_ln`, `mlp.experts.mlp.w1`, ...), so `load_state_dict(strict=True)` against the
    real checkpoint *is* the structural proof -- no remapping layer to get wrong.
*   **No `einops`.** Upstream leans on `rearrange`/`repeat`; every such call is written
    out as an explicit `view`/`cat` here, because those layout choices are precisely what
    the TTNN port has to reproduce and they should be readable, not inferred.

Excluded from upstream's 2556 lines: the vision tower, all task heads, the pooler, gated
MLP variants, DynamicNTK and xPos rotary, the megablocks bridge, the custom
`from_pretrained`, KV-cache, gradient checkpointing, the pre-norm branch, and every
`use_flash_attn` / `fused_*` path.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental.nomic_embed_text_v2_moe.reference.configuration_nomic_moe import NomicMoEConfig

# ---------------------------------------------------------------------------------------
# Rotary embeddings (GPT-NeoX halves)
# ---------------------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """GPT-NeoX convention: split the last axis in half, `(x1, x2) -> (-x2, x1)`.

    NOT the GPT-J / interleaved convention, which pairs even and odd lanes. The two are
    different maps and produce different -- both finite, both plausible -- outputs. The
    checkpoint's `rotary_emb_interleaved` is False.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to `x` of shape (batch, seqlen, nheads, headdim).

    `cos`/`sin` are (seqlen, rotary_dim // 2) -- i.e. HALF width. Upstream widens them with
    ``repeat(cos, "... d -> ... 1 (2 d)")``, where `(2 d)` makes 2 the *outer* factor, so
    the result is `concat([c, c])` and not `repeat_interleave`. Getting that backwards
    yields the GPT-J lane pairing, which composed with NeoX `rotate_half` is not a rotation
    at all.
    """
    rotary_dim = cos.shape[-1] * 2
    assert rotary_dim <= x.shape[-1]
    seqlen = x.shape[1]
    cos = cos[:seqlen]
    sin = sin[:seqlen]

    # (S, rotary_dim // 2) -> (S, 1, rotary_dim); the singleton broadcasts over heads.
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)

    rotated = x[..., :rotary_dim] * cos + rotate_half(x[..., :rotary_dim]) * sin
    if rotary_dim == x.shape[-1]:
        return rotated
    return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)


class NomicBertRotaryEmbedding(nn.Module):
    """Caches `cos`/`sin` at HALF the rotary width; `apply_rotary_emb` widens at use.

    `inv_freq` is non-persistent upstream, so it is absent from the checkpoint.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    def _update_cos_sin_cache(self, seqlen: int, device, dtype) -> None:
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            self._seq_len_cached = seqlen
            # Positions in fp32 regardless of model dtype: `t * inv_freq` grows large and
            # bf16 would collapse distinct late positions onto the same angle.
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        """`qkv`: (batch, seqlen, 3, nheads, headdim). Rotates q and k, passes v through."""
        seqlen = qkv.shape[1]
        self._update_cos_sin_cache(seqlen, device=qkv.device, dtype=qkv.dtype)
        q_rot = apply_rotary_emb(qkv[:, :, 0], self._cos_cached, self._sin_cached)
        k_rot = apply_rotary_emb(qkv[:, :, 1], self._cos_cached, self._sin_cached)
        return torch.stack((q_rot, k_rot, qkv[:, :, 2]), dim=2)


# ---------------------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------------------


class NomicBertEmbeddings(nn.Module):
    """Word + token-type embeddings. No position embeddings -- position is rotary-only.

    `padding_idx` is passed to mirror upstream's construction, but it only zeroes the row
    at *init*; loading the checkpoint overwrites it. The trained `<pad>` row is NOT zero
    (absmax ~1.5e-2), so a TTNN port must not reintroduce zeroing.
    """

    def __init__(self, config: NomicMoEConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.type_vocab_size = config.type_vocab_size
        if self.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        if self.type_vocab_size > 0:
            if token_type_ids is None:
                seqlen = embeddings.shape[1]
                token_type_ids = torch.zeros(seqlen, dtype=torch.long, device=embeddings.device)
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        return embeddings


# ---------------------------------------------------------------------------------------
# Dense MLP (even-numbered layers)
# ---------------------------------------------------------------------------------------


class NomicBertMLP(nn.Module):
    def __init__(self, config: NomicMoEConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        # `activation_function == "gelu"` maps to nn.GELU(approximate="none") upstream, i.e.
        # the exact erf form. The tanh approximation differs by ~5e-4 and would later read
        # as a hardware precision problem.
        self.activation = nn.GELU(approximate="none")
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


# ---------------------------------------------------------------------------------------
# MoE (odd-numbered layers)
# ---------------------------------------------------------------------------------------


class NomicRouter(nn.Module):
    """Softmax over ALL experts in fp32, then top-k. The top-k weights are NOT renormalised.

    That is the single most-copied bug in this architecture: Mixtral, Switch and most MoE
    reference code divide by the top-k sum. Nomic does not (`moe_normalize_expert_weights`
    is False), so the routed weights sum to ~0.67 on real data, and the residual branch is
    correspondingly attenuated. Renormalising still scores PCC 0.993 against the correct
    output -- above any 0.99 gate.
    """

    def __init__(self, hidden_size: int, moe_num_experts: int, moe_top_k: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.layer = nn.Linear(hidden_size, moe_num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        weights = self.layer(x.view(-1, x.shape[-1])).softmax(dim=-1, dtype=torch.float32)
        top_weights, top_experts = torch.topk(weights, self.moe_top_k, dim=-1)
        # NO renormalisation here. See the class docstring.
        return weights.to(x.dtype), top_weights.to(x.dtype), top_experts


class NomicExpertMLP(nn.Module):
    """Eight experts packed into two `[E * ffn_hidden, hidden]` parameter blocks.

    The expert axis is the OUTER one: expert `e` owns rows `e * ffn_hidden : (e+1) * ...`.
    Both `w1` and `w2` are stored `[ffn_hidden, hidden]` per expert, so `w1` is applied
    transposed and `w2` is applied as-is. Every other combination either raises a shape
    error or -- for `w2` viewed as `(E, hidden, ffn_hidden)`, which typechecks because
    `24576 * 768 == 8 * 768 * 3072` -- silently produces garbage (measured PCC -0.013).
    """

    def __init__(self, hidden_size: int, ffn_hidden_size: int, moe_num_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts
        self.w1 = nn.Parameter(torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))
        self.activation_fn = nn.GELU(approximate="none")

    def expert_weights(self, expert_idx: int):
        shape = (self.moe_num_experts, self.ffn_hidden_size, self.hidden_size)
        return self.w1.view(*shape)[expert_idx], self.w2.view(*shape)[expert_idx]

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        expert_w1, expert_w2 = self.expert_weights(expert_idx)
        return self.activation_fn(x.matmul(expert_w1.t())).matmul(expert_w2)


class NomicExperts(nn.Module):
    """Weighted sum of the top-k expert outputs, plus ONE shared bias added at the end.

    The bias is a single `[hidden]` vector for all eight experts and is added *after* the
    weighted sum -- not inside the per-expert loop. Adding it per-expert instead scales it
    by the routed-weight sum, an almost-constant offset of `(sum(w) - 1) * bias`. PCC
    mean-centres, so it scores 0.9999998 and no PCC threshold can catch it; the tests gate
    this one on max-abs.
    """

    def __init__(self, config: NomicMoEConfig):
        super().__init__()
        self.moe_num_experts = config.num_experts
        self.mlp = NomicExpertMLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            moe_num_experts=config.num_experts,
        )
        self.bias = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(self, x: torch.Tensor, top_weights: torch.Tensor, top_experts: torch.Tensor) -> torch.Tensor:
        bsz, q_len, hidden_size = x.shape
        x = x.view(-1, hidden_size)
        out = torch.zeros_like(x)

        expert_mask = F.one_hot(top_experts, num_classes=self.moe_num_experts).permute(2, 1, 0)
        for expert_idx in range(self.moe_num_experts):
            topk_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.shape[0] == 0:
                continue
            expert_tokens = x[token_idx]
            expert_out = self.mlp(expert_tokens, expert_idx) * top_weights[token_idx, topk_idx, None]
            out.index_add_(0, token_idx, expert_out)

        out = out.reshape(bsz, q_len, hidden_size)
        return out + self.bias  # ONE bias, ONCE, after the sum.

    def dense_forward(self, x: torch.Tensor, dense_weights: torch.Tensor) -> torch.Tensor:
        """Arithmetically equivalent all-experts formulation -- the shape the TTNN port uses.

        Instead of gathering each expert's tokens, run every token through every expert and
        weight by a dense `[tokens, num_experts]` routing tensor whose non-top-k entries are
        zero. Same result, no ragged gather/scatter, and it maps onto two broadcast-batch
        matmuls on device.

        `dense_weights`: (tokens, num_experts), zero off the top-k.
        """
        bsz, q_len, hidden_size = x.shape
        flat = x.reshape(1, -1, hidden_size)  # (1, T, H)
        shape = (self.moe_num_experts, self.mlp.ffn_hidden_size, hidden_size)

        w1 = self.mlp.w1.view(*shape).transpose(1, 2)  # (E, H, F)
        w2 = self.mlp.w2.view(*shape)  # (E, F, H)

        act = self.mlp.activation_fn(torch.matmul(flat, w1))  # (E, T, F)
        per_expert = torch.matmul(act, w2)  # (E, T, H)

        gate = dense_weights.t().unsqueeze(-1).to(per_expert.dtype)  # (E, T, 1)
        out = (per_expert * gate).sum(dim=0)  # (T, H)
        return out.reshape(bsz, q_len, hidden_size) + self.bias


class NomicMoELayer(nn.Module):
    def __init__(self, config: NomicMoEConfig):
        super().__init__()
        self.router = NomicRouter(config.hidden_size, config.num_experts, config.moe_top_k)
        self.experts = NomicExperts(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upstream's block passes an inverted pad mask here and NomicMoELayer.forward
        # ignores it entirely. We do not thread it through at all: applying it would zero
        # the real tokens, since in that mask 1 means "pad".
        _weights, top_weights, top_experts = self.router(x)
        return self.experts(x, top_weights, top_experts)


# ---------------------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------------------


class NomicBertAttention(nn.Module):
    """Bidirectional MHA with a fused three-major QKV projection and full-head rotary.

    `Wqkv` produces `[q(768) | k(768) | v(768)]`; within each block, heads are contiguous.
    `norm_factor` is a non-persistent buffer upstream and unused on the SDPA path (SDPA's
    default scale, `1/sqrt(head_dim)`, already matches), so it is omitted here.
    """

    def __init__(self, config: NomicMoEConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.Wqkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.rotary_emb = NomicBertRotaryEmbedding(dim=config.rotary_dim, base=config.rotary_emb_base)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape

        qkv = self.Wqkv(hidden_states)
        # Upstream: rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=head_dim).
        # `three` is the OUTER factor -- q, k and v are contiguous blocks of 768, and heads
        # are contiguous inside each. Splitting head-major instead scores PCC 0.08.
        qkv = qkv.view(bsz, seqlen, 3, self.num_heads, self.head_dim)
        qkv = self.rotary_emb(qkv)

        # (B, S, H, D) -> (B, H, S, D)
        query = qkv[:, :, 0].permute(0, 2, 1, 3)
        key = qkv[:, :, 1].permute(0, 2, 1, 3)
        value = qkv[:, :, 2].permute(0, 2, 1, 3)

        # is_causal=False is not the SDPA default; this is an encoder.
        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=False)

        # (B, H, S, D) -> (B, S, H * D)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(bsz, seqlen, self.embed_dim)
        return self.out_proj(attn_output)


# ---------------------------------------------------------------------------------------
# Block / encoder / model
# ---------------------------------------------------------------------------------------


class NomicBertBlock(nn.Module):
    """Post-norm block: `norm1(attn(x) + x)` then `norm2(mlp(h) + h)`.

    Post-norm, not pre-norm -- the residual is added *before* the norm, so every sub-block
    output is re-centred. That is why numerical error does not compound across the 12
    layers the way it does in a pre-norm decoder.
    """

    def __init__(self, config: NomicMoEConfig, moe: bool):
        super().__init__()
        self.moe = moe
        self.attn = NomicBertAttention(config)
        self.mlp = NomicMoELayer(config) if moe else NomicBertMLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(hidden_states, attention_mask=attention_mask)
        hidden_states = self.norm1(attn_out + hidden_states)
        mlp_out = self.mlp(hidden_states)
        return self.norm2(mlp_out + hidden_states)


class NomicBertEncoder(nn.Module):
    def __init__(self, config: NomicMoEConfig):
        super().__init__()
        # `i % every_n == 1`, so layer 0 is dense and the MoE layers are the odd ones.
        self.layers = nn.ModuleList(
            [NomicBertBlock(config, moe=config.is_moe_layer(i)) for i in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states


def build_extended_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(B, S) keep-mask of 1/0 -> (B, 1, 1, S) additive mask of 0 / dtype-min.

    This is `PreTrainedModel.get_extended_attention_mask` for a non-decoder, written out.
    """
    extended = attention_mask[:, None, None, :].to(dtype=dtype)
    return (1.0 - extended) * torch.finfo(dtype).min


class NomicBertModel(nn.Module):
    """Encoder-only backbone. Returns `last_hidden_state` -- no pooler, no task head.

    Two upstream behaviours are deliberately NOT reproduced:

    *   Upstream *requires* `attention_mask` and raises `AttributeError` without it. Here
        it defaults to all-ones.
    *   Upstream's `matryoshka_dim` slices `sequence_output[:, :matryoshka_dim]` -- the
        SEQUENCE axis, not the feature axis. That is a bug; truncation belongs after
        pooling and lives in `pipeline.py`.
    """

    def __init__(self, config: NomicMoEConfig):
        super().__init__()
        self.config = config
        self.embeddings = NomicBertEmbeddings(config)
        self.emb_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.encoder = NomicBertEncoder(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids, token_type_ids=token_type_ids)
        hidden_states = self.emb_ln(hidden_states)

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        extended_mask = build_extended_attention_mask(attention_mask, hidden_states.dtype)

        return self.encoder(hidden_states, attention_mask=extended_mask)
