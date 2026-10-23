# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference for nomic-ai/nomic-embed-text-v2-moe. Golden model for the TTNN port.

Stage 2 of 3: input_ids (B, S) -> last_hidden_state (B, S, 768). Pooling and normalization are
postprocessing.py; this file stops at one vector per token.

Shape legend, used throughout:

    B  batch                     S  sequence, padded to the batch max
    H  hidden, 768               F  FFN hidden, 3072
    A  attention heads, 12       D  head dim, 64  (A * D == H)
    E  experts, 8                K  experts routed per token, 2
    T  tokens, B * S             the MoE path flattens batch and sequence together

The shape through one block, all of it (B, S, H) at the boundaries:

    x (B, S, H)
      Wqkv          -> (B, S, 3H)   -> view (B, S, 3, A, D)   three-major: [q | k | v]
      rotary        -> (B, S, 3, A, D)
      SDPA          -> (B, A, S, D) -> reshape (B, S, H)
      out_proj      -> (B, S, H)
      norm1(+x)     -> (B, S, H)
      dense FFN     -> (B, S, F) -> (B, S, H)      even layers
      MoE FFN       -> flatten (T, H), route, weighted sum -> (B, S, H)   odd layers
      norm2(+h)     -> (B, S, H)

Inference path only. Parameter names mirror upstream exactly, so load_state_dict(strict=True)
against the real checkpoint validates the structure without a remapping layer.

Config fields that upstream branches on at runtime are hard-coded to this checkpoint's values
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
    """Rotate the halves of the last axis, GPT-NeoX convention.

    Splits the last axis in half and maps (x1, x2) to (-x2, x1). Not the GPT-J interleaved
    convention, which pairs even and odd lanes. Both produce finite output, so picking the
    wrong one fails silently. config.rotary_emb_interleaved is False.

    Args:
        x: Any tensor whose last axis is even, typically (B, S, A, D).

    Returns:
        torch.Tensor: Same shape as x.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate one of q or k by position.

    cos and sin are widened from D//2 to D by concatenation, then unsqueezed to (S, 1, D) so
    the trailing singleton broadcasts over heads. Widening with repeat_interleave instead gives
    the GPT-J lane pairing, which combined with NeoX rotate_half is not a rotation and does not
    preserve the per-plane norm.

    Args:
        x: Query or key, (B, S, A, D).
        cos: Cosine table, (S, D // 2), half width as cached.
        sin: Sine table, (S, D // 2).

    Returns:
        torch.Tensor: (B, S, A, D), same shape as x.
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
    """Position tables for rotary embeddings.

    Caches cos/sin at (S, D // 2), half the rotary width; apply_rotary_emb widens them to D.
    inv_freq is (D // 2,) and non-persistent, so it is not in the checkpoint.
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
        """Grow the cos/sin cache to cover seqlen positions, if it does not already.

        Args:
            seqlen: Number of positions needed, S.
            device: Device to build the tables on.
            dtype: Model dtype the tables are cast to after being computed in fp32.
        """
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
        """Rotate q and k by position, passing v through unchanged.

        Args:
            qkv: Packed projection output, (B, S, 3, A, D), index 0/1/2 being q/k/v.

        Returns:
            torch.Tensor: (B, S, 3, A, D), same shape, with q and k rotated.
        """
        self._update_cos_sin_cache(qkv.shape[1], device=qkv.device, dtype=qkv.dtype)
        q_rot = apply_rotary_emb(qkv[:, :, 0], self._cos_cached, self._sin_cached)
        k_rot = apply_rotary_emb(qkv[:, :, 1], self._cos_cached, self._sin_cached)
        return torch.stack((q_rot, k_rot, qkv[:, :, 2]), dim=2)


class NomicBertEmbeddings(nn.Module):
    """Word and token-type embeddings, mapping token ids to vectors.

    Position is rotary-only, applied later inside attention, so there is no learned position
    table here.
    """

    def __init__(self, config: NomicMoEConfig):
        super().__init__()
        # padding_idx mirrors upstream construction. It zeroes the row only at init; loading
        # the checkpoint restores a trained, non-zero <pad> row. Do not reintroduce zeroing.
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.type_vocab_size = config.type_vocab_size
        if self.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Look up and sum the word and token-type embeddings.

        Args:
            input_ids: (B, S) int64 token ids.
            token_type_ids: (S,) int64, or None to default to zeros. type_vocab_size is 1, so
                this term folds to a constant.

        Returns:
            torch.Tensor: (B, S, H) fp32.
        """
        embeddings = self.word_embeddings(input_ids)
        if self.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = torch.zeros(embeddings.shape[1], dtype=torch.long, device=embeddings.device)
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        return embeddings


class NomicBertMLP(nn.Module):
    """Dense FFN, used on even-numbered layers.

    Widens H to F, applies exact-erf GELU, and projects back to H.
    """

    def __init__(self, config: NomicMoEConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        # Exact erf, not the tanh approximation. They differ by ~5e-4, which is small enough
        # to pass a loose PCC gate and large enough to look like a device precision problem.
        self.activation = nn.GELU(approximate="none")
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the dense feed-forward network.

        Args:
            x: (B, S, H) fp32.

        Returns:
            torch.Tensor: (B, S, H) fp32, via (B, S, F) at the activation.
        """
        return self.fc2(self.activation(self.fc1(x)))


class NomicRouter(nn.Module):
    """Choose which experts each token goes to.

    Softmax over all experts in fp32, then top-k. This is where (B, S, H) flattens to (T, ...),
    with T = B * S.

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
        """Score every expert and select the top-k per token.

        Args:
            x: (B, S, H) fp32 block input.

        Returns:
            tuple: weights (T, E) fp32, the full softmax over all E experts; top_weights
            (T, K) fp32, the K largest and deliberately not renormalized; top_experts (T, K)
            int64 expert indices. T = B * S.
        """
        weights = self.layer(x.view(-1, x.shape[-1])).softmax(dim=-1, dtype=torch.float32)
        top_weights, top_experts = torch.topk(weights, self.moe_top_k, dim=-1)
        return weights.to(x.dtype), top_weights.to(x.dtype), top_experts

    def dense_weights(self, top_weights: torch.Tensor, top_experts: torch.Tensor) -> torch.Tensor:
        """Scatter the top-k weights back to a dense per-expert tensor.

        This is the routing form NomicExperts.dense_forward and the TTNN port consume.

        Args:
            top_weights: (T, K) fp32 routed weights.
            top_experts: (T, K) int64 expert indices.

        Returns:
            torch.Tensor: (T, E) fp32, holding the routed weight at the top-k positions and
            zero everywhere else.
        """
        dense = torch.zeros(
            top_weights.shape[0],
            self.moe_num_experts,
            dtype=top_weights.dtype,
            device=top_weights.device,
        )
        return dense.scatter_(1, top_experts, top_weights)


class NomicExpertMLP(nn.Module):
    """All experts' weights, packed into two (E*F, H) blocks with the expert axis outer.

    w1 and w2 are stored (E*F, H); view either as (E, F, H) to index one expert, where expert e
    owns rows e*F .. (e+1)*F.

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
        """Shape to view w1 or w2 as when indexing a single expert.

        Returns:
            tuple: (E, F, H). Viewing as (E, H, F) instead also succeeds, because the element
            count is symmetric in F and H, and produces uncorrelated noise.
        """
        return (self.moe_num_experts, self.ffn_hidden_size, self.hidden_size)

    def expert_weights(self, expert_idx: int):
        """Slice out one expert's two weight matrices.

        Args:
            expert_idx: Which expert, in [0, E).

        Returns:
            tuple: (w1, w2), each (F, H). w1 is applied transposed, w2 is not.
        """
        return self.w1.view(*self.expert_shape)[expert_idx], self.w2.view(*self.expert_shape)[expert_idx]

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Run one expert over the tokens routed to it.

        Args:
            x: (N, H) fp32, the N tokens this expert received.
            expert_idx: Which expert to apply, in [0, E).

        Returns:
            torch.Tensor: (N, H) fp32, via (N, F) at the activation.
        """
        expert_w1, expert_w2 = self.expert_weights(expert_idx)
        return self.activation_fn(x.matmul(expert_w1.t())).matmul(expert_w2)


class NomicExperts(nn.Module):
    """Weighted sum of the top-k expert outputs, plus one shared bias.

    The bias is a single (H,) vector for all experts, added after the sum. Adding it inside
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
        """Combine expert outputs using a ragged per-expert loop.

        Gathers each expert's tokens by value, runs them, and scatter-adds the results back.
        Flattens to (T, H) internally. The gather and scatter are data-dependent, which is why
        this path is not torch.fx-traceable; dense_forward is the device-friendly equivalent.

        Args:
            x: (B, S, H) fp32 block input.
            top_weights: (T, K) fp32 routed weights.
            top_experts: (T, K) int64 expert indices.

        Returns:
            torch.Tensor: (B, S, H) fp32, the weighted expert sum plus the shared bias.
        """
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
        """Combine expert outputs by running every token through every expert.

        Arithmetically equivalent to forward, but with no ragged gather or scatter: two
        broadcast-batch matmuls and a reduce. This is the formulation the TTNN port uses.

        The internal chain, which is the memory cost the port must size for::

            x             -> flat       (1, T, H)
            w1 as (E,H,F) -> matmul     (E, T, F)   every token through every expert
            GELU, w2      -> per_expert (E, T, H)
            gate (E,T,1), multiply, sum over E -> (T, H) -> reshape (B, S, H)

        The (E, T, F) intermediate scales with batch times sequence length, not sequence length
        alone.

        Args:
            x: (B, S, H) fp32 block input.
            dense_weights: (T, E) fp32 from NomicRouter.dense_weights, zero off the top-k.

        Returns:
            torch.Tensor: (B, S, H) fp32, identical to what forward produces.
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
    """Router plus experts: the FFN used on odd-numbered layers."""

    def __init__(self, config: NomicMoEConfig):
        super().__init__()
        self.router = NomicRouter(config.hidden_size, config.num_experts, config.moe_top_k)
        self.experts = NomicExperts(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route each token and combine the selected experts' outputs.

        Args:
            x: (B, S, H) fp32 block input.

        Returns:
            torch.Tensor: (B, S, H) fp32.
        """
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
        """Project, rotate, attend, and project back.

        The shape chain is (B, S, H) -> Wqkv (B, S, 3H) -> view (B, S, 3, A, D) -> rotary ->
        SDPA (B, A, S, D) -> reshape (B, S, H) -> out_proj.

        Args:
            hidden_states: (B, S, H) fp32 block input.
            attention_mask: (B, 1, 1, S) additive mask, 0.0 for real tokens and dtype-min for
                padding, or None for no masking.

        Returns:
            torch.Tensor: (B, S, H) fp32.
        """
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
        """Apply attention then the FFN, each with a residual and a following norm.

        Args:
            hidden_states: (B, S, H) fp32.
            attention_mask: (B, 1, 1, S) additive mask, or None.

        Returns:
            torch.Tensor: (B, S, H) fp32, shape unchanged.
        """
        hidden_states = self.norm1(self.attn(hidden_states, attention_mask=attention_mask) + hidden_states)
        return self.norm2(self.mlp(hidden_states) + hidden_states)


class NomicBertEncoder(nn.Module):
    """The 12 blocks in sequence, alternating dense and MoE FFNs.

    Every block preserves its input shape, so the stack does too.
    """

    def __init__(self, config: NomicMoEConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [NomicBertBlock(config, moe=config.is_moe_layer(i)) for i in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run every block in order.

        Args:
            hidden_states: (B, S, H) fp32 post-embedding input.
            attention_mask: (B, 1, 1, S) additive mask, or None.

        Returns:
            torch.Tensor: (B, S, H) fp32.
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states


def build_extended_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a keep-mask into an additive mask ready to add onto attention scores.

    The two singleton axes broadcast over heads and query positions when added to the
    (B, A, S, S) scores inside SDPA. This is PreTrainedModel.get_extended_attention_mask for a
    non-decoder, written out; upstream's own call site is deprecated in transformers 5.12.

    Args:
        attention_mask: (B, S) int64, 1 for real tokens and 0 for padding.
        dtype: Model dtype, used for the large-negative fill value.

    Returns:
        torch.Tensor: (B, 1, 1, S) of dtype, 0.0 at real tokens and dtype-min at padding.
    """
    extended = attention_mask[:, None, None, :].to(dtype=dtype)
    return (1.0 - extended) * torch.finfo(dtype).min


class NomicBertModel(nn.Module):
    """Encoder-only backbone returning last_hidden_state. No pooler, no task head.

    Pooling to (B, 768) is postprocessing.mean_pool, not part of this model.

    Two upstream behaviours are deliberately not reproduced:
      - Upstream requires attention_mask and raises AttributeError without it; here it
        defaults to all-ones.
      - Upstream's matryoshka_dim slices the sequence axis, not the feature axis. Truncation
        belongs after pooling and lives in postprocessing.py.
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
        """Embed the tokens and run the encoder stack.

        Args:
            input_ids: (B, S) int64 token ids, from preprocessing.tokenize.
            attention_mask: (B, S) int64, 1 for real tokens and 0 for padding, or None to
                default to all-ones.
            token_type_ids: (S,) int64, or None. type_vocab_size is 1, so this folds to a
                constant.

        Returns:
            torch.Tensor: (B, S, 768) fp32, one contextual vector per token.
        """
        hidden_states = self.emb_ln(self.embeddings(input_ids, token_type_ids=token_type_ids))

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

        return self.encoder(
            hidden_states, attention_mask=build_extended_attention_mask(attention_mask, hidden_states.dtype)
        )
