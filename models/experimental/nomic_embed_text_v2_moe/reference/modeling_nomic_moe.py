# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference for nomic-ai/nomic-embed-text-v2-moe. Golden model for the TTNN port.

Inference path only. Parameter names mirror upstream exactly, so load_state_dict(strict=True)
against the real checkpoint validates the structure without a remapping layer.

Config fields that upstream branches on at runtime are asserted in configuration_nomic_moe.py
rather than implemented here. Not implemented: vision tower, task heads, pooler, gated MLP,
DynamicNTK and xPos rotary, megablocks, KV cache, pre-norm.

einops is not used. The layout choices are exactly what the TTNN port must reproduce, so they
are written out as explicit view/cat calls.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental.nomic_embed_text_v2_moe.reference.configuration_nomic_moe import NomicMoEConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """GPT-NeoX convention: split the last axis in half, (x1, x2) -> (-x2, x1).

    Not the GPT-J interleaved convention, which pairs even and odd lanes. Both produce finite
    output, so picking the wrong one fails silently. config.rotary_emb_interleaved is False.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to x of shape (batch, seqlen, nheads, headdim).

    cos and sin are (seqlen, rotary_dim // 2), i.e. half width. Widening them with
    repeat_interleave instead of cat gives the GPT-J lane pairing, which combined with NeoX
    rotate_half is not a rotation and does not preserve the per-plane norm.
    """
    rotary_dim = cos.shape[-1] * 2
    assert rotary_dim <= x.shape[-1]
    seqlen = x.shape[1]

    # Trailing singleton broadcasts over heads.
    cos = torch.cat((cos[:seqlen], cos[:seqlen]), dim=-1).unsqueeze(-2)
    sin = torch.cat((sin[:seqlen], sin[:seqlen]), dim=-1).unsqueeze(-2)

    rotated = x[..., :rotary_dim] * cos + rotate_half(x[..., :rotary_dim]) * sin
    if rotary_dim == x.shape[-1]:
        return rotated
    return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)


class NomicBertRotaryEmbedding(nn.Module):
    """Caches cos/sin at half the rotary width. inv_freq is non-persistent, so it is not in
    the checkpoint."""

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
            # Positions stay fp32 regardless of model dtype: t * inv_freq grows large, and in
            # bf16 distinct late positions collapse onto the same angle.
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        """qkv: (batch, seqlen, 3, nheads, headdim). Rotates q and k, passes v through."""
        self._update_cos_sin_cache(qkv.shape[1], device=qkv.device, dtype=qkv.dtype)
        q_rot = apply_rotary_emb(qkv[:, :, 0], self._cos_cached, self._sin_cached)
        k_rot = apply_rotary_emb(qkv[:, :, 1], self._cos_cached, self._sin_cached)
        return torch.stack((q_rot, k_rot, qkv[:, :, 2]), dim=2)


class NomicBertEmbeddings(nn.Module):
    """Word and token-type embeddings. Position is rotary-only, so there is no learned table."""

    def __init__(self, config: NomicMoEConfig):
        super().__init__()
        # padding_idx mirrors upstream construction. It zeroes the row only at init; loading
        # the checkpoint restores a trained, non-zero <pad> row. Do not reintroduce zeroing.
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.type_vocab_size = config.type_vocab_size
        if self.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        if self.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = torch.zeros(embeddings.shape[1], dtype=torch.long, device=embeddings.device)
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        return embeddings


class NomicBertMLP(nn.Module):
    """Dense FFN, used on even-numbered layers."""

    def __init__(self, config: NomicMoEConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        # Exact erf, not the tanh approximation. They differ by ~5e-4, which is small enough
        # to pass a loose PCC gate and large enough to look like a device precision problem.
        self.activation = nn.GELU(approximate="none")
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class NomicRouter(nn.Module):
    """Softmax over all experts in fp32, then top-k.

    The top-k weights are NOT renormalized (config.moe_normalize_expert_weights is False), so
    they sum to less than 1 and the MoE branch is attenuated relative to the residual. Most
    MoE implementations divide by the top-k sum; doing that here scores PCC ~0.99, which sits
    right on a typical gate.
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
        return weights.to(x.dtype), top_weights.to(x.dtype), top_experts

    def dense_weights(self, top_weights: torch.Tensor, top_experts: torch.Tensor) -> torch.Tensor:
        """Scatter the top-k weights back to a (tokens, num_experts) tensor, zero off the top-k.

        This is the routing form NomicExperts.dense_forward and the TTNN port consume.
        """
        dense = torch.zeros(
            top_weights.shape[0],
            self.moe_num_experts,
            dtype=top_weights.dtype,
            device=top_weights.device,
        )
        return dense.scatter_(1, top_experts, top_weights)


class NomicExpertMLP(nn.Module):
    """Experts packed into two [num_experts * ffn_hidden, hidden] blocks, expert axis outer.

    Both w1 and w2 are stored [ffn_hidden, hidden] per expert, so w1 is applied transposed and
    w2 is not. Viewing w2 as (num_experts, hidden, ffn_hidden) instead also succeeds, because
    the element count is symmetric in those two dims, and every downstream matmul typechecks.
    Nothing raises; the output is uncorrelated noise.
    """

    def __init__(self, hidden_size: int, ffn_hidden_size: int, moe_num_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts
        self.w1 = nn.Parameter(torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))
        self.activation_fn = nn.GELU(approximate="none")

    @property
    def expert_shape(self) -> tuple[int, int, int]:
        return (self.moe_num_experts, self.ffn_hidden_size, self.hidden_size)

    def expert_weights(self, expert_idx: int):
        return self.w1.view(*self.expert_shape)[expert_idx], self.w2.view(*self.expert_shape)[expert_idx]

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        expert_w1, expert_w2 = self.expert_weights(expert_idx)
        return self.activation_fn(x.matmul(expert_w1.t())).matmul(expert_w2)


class NomicExperts(nn.Module):
    """Weighted sum of the top-k expert outputs, plus one shared bias.

    The bias is a single [hidden] vector for all experts, added after the sum. Adding it inside
    the per-expert loop scales it by the routed-weight sum, giving an offset of
    (sum(w) - 1) * bias. That offset is nearly constant, and PCC mean-centres, so it scores
    0.9999998 against real weights. Gate this on max-abs, not PCC.
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
            expert_out = self.mlp(x[token_idx], expert_idx) * top_weights[token_idx, topk_idx, None]
            out.index_add_(0, token_idx, expert_out)

        return out.reshape(bsz, q_len, hidden_size) + self.bias

    def dense_forward(self, x: torch.Tensor, dense_weights: torch.Tensor) -> torch.Tensor:
        """All-experts formulation, arithmetically equivalent to forward.

        Runs every token through every expert and zeroes the non-top-k contributions via
        dense_weights, instead of gathering each expert's tokens. This is the shape the TTNN
        port uses: two broadcast-batch matmuls and a reduce, no ragged gather or scatter.

        dense_weights: (tokens, num_experts), zero off the top-k.
        """
        bsz, q_len, hidden_size = x.shape
        flat = x.reshape(1, -1, hidden_size)

        w1 = self.mlp.w1.view(*self.mlp.expert_shape).transpose(1, 2)
        w2 = self.mlp.w2.view(*self.mlp.expert_shape)

        per_expert = torch.matmul(self.mlp.activation_fn(torch.matmul(flat, w1)), w2)
        gate = dense_weights.t().unsqueeze(-1).to(per_expert.dtype)
        out = (per_expert * gate).sum(dim=0)
        return out.reshape(bsz, q_len, hidden_size) + self.bias


class NomicMoELayer(nn.Module):
    def __init__(self, config: NomicMoEConfig):
        super().__init__()
        self.router = NomicRouter(config.hidden_size, config.num_experts, config.moe_top_k)
        self.experts = NomicExperts(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upstream passes an inverted pad mask here (1 means pad) and then ignores it. Not
        # threaded through: applying it would zero the real tokens.
        _weights, top_weights, top_experts = self.router(x)
        return self.experts(x, top_weights, top_experts)


class NomicBertAttention(nn.Module):
    """Bidirectional MHA with a fused three-major QKV projection and full-head rotary.

    norm_factor is a non-persistent buffer upstream and unused on the SDPA path, since SDPA's
    default scale already equals 1/sqrt(head_dim). It is not defined here.
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

        # Three-major: q, k and v are contiguous blocks of embed_dim, heads contiguous inside
        # each. A head-major view strides across the q/k/v boundaries and scores PCC 0.08.
        qkv = self.Wqkv(hidden_states).view(bsz, seqlen, 3, self.num_heads, self.head_dim)
        qkv = self.rotary_emb(qkv)

        query = qkv[:, :, 0].permute(0, 2, 1, 3)
        key = qkv[:, :, 1].permute(0, 2, 1, 3)
        value = qkv[:, :, 2].permute(0, 2, 1, 3)

        # is_causal is not the SDPA default here; this is an encoder.
        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=False)

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(bsz, seqlen, self.embed_dim)
        return self.out_proj(attn_output)


class NomicBertBlock(nn.Module):
    """Post-norm block: norm1(attn(x) + x), then norm2(mlp(h) + h).

    Residual is added before the norm, so every sub-block output is re-centred. This is why
    numerical error does not compound across layers the way it does in a pre-norm decoder.
    """

    def __init__(self, config: NomicMoEConfig, moe: bool):
        super().__init__()
        self.moe = moe
        self.attn = NomicBertAttention(config)
        self.mlp = NomicMoELayer(config) if moe else NomicBertMLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.norm1(self.attn(hidden_states, attention_mask=attention_mask) + hidden_states)
        return self.norm2(self.mlp(hidden_states) + hidden_states)


class NomicBertEncoder(nn.Module):
    def __init__(self, config: NomicMoEConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [NomicBertBlock(config, moe=config.is_moe_layer(i)) for i in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states


def build_extended_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(B, S) keep-mask of 1/0 to a (B, 1, 1, S) additive mask of 0 / dtype-min.

    PreTrainedModel.get_extended_attention_mask for a non-decoder, written out. Upstream's own
    call site is deprecated in transformers 5.12.
    """
    extended = attention_mask[:, None, None, :].to(dtype=dtype)
    return (1.0 - extended) * torch.finfo(dtype).min


class NomicBertModel(nn.Module):
    """Encoder-only backbone returning last_hidden_state. No pooler, no task head.

    Two upstream behaviours are deliberately not reproduced:
      - Upstream requires attention_mask and raises AttributeError without it; here it
        defaults to all-ones.
      - Upstream's matryoshka_dim slices the sequence axis, not the feature axis. Truncation
        belongs after pooling and lives in pipeline.py.
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
        hidden_states = self.emb_ln(self.embeddings(input_ids, token_type_ids=token_type_ids))

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

        return self.encoder(
            hidden_states, attention_mask=build_extended_attention_mask(attention_mask, hidden_states.dtype)
        )
