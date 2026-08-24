# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone DeepSeek DSpark drafter (PyTorch reference).

DSpark is the speculative-decoding module fused into DeepSeek-V4-Flash-0731 as
``mtp.*``. It is not a Qwen/Gemma model and is not the 43-layer Flash target.
One draft round:

1. Fuse hidden states from target layers ``dspark_target_layer_ids``
   (Flash: 40, 41, 42) with ``main_proj`` / ``main_norm``.
2. Inject that context as extra K/V. The draft block (anchor + noise tokens)
   supplies the queries; attention inside the block is bidirectional.
3. Sample left-to-right with a rank-``r`` Markov logit bias, and score each
   position with a confidence head used to truncate the verified prefix.

This file implements that algorithm with a dense sliding-window backbone so it
can be unit-tested alone (no 256-expert MoE, no CSA/HCA indexer). Module names
under ``mtp.{0,1,2}`` follow the 0731 checkpoint for the DSpark-specific pieces
(``main_proj``, ``markov_head``, ``confidence_head``). The embedding table and
LM head are owned here for standalone use, or aliased from the target via
:meth:`DSparkModel.share_from_target`.

Paper: Cheng et al., "DSpark: Confidence-Scheduled Speculative Decoding with
Semi-Autoregressive Generation", arXiv:2607.05147.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DSparkConfig:
    """Drafter geometry.

    :meth:`tiny` is for unit tests. :meth:`flash_0731` copies the DSpark knobs
    from DeepSeek-V4-Flash-0731 ``config.json`` (not the target's MLA/MoE widths).
    """

    hidden_size: int = 64
    num_target_layers: int = 3
    num_stages: int = 3
    num_attention_heads: int = 4
    head_dim: int = 16
    intermediate_size: int = 128
    vocab_size: int = 256
    rms_norm_eps: float = 1.0e-6
    sliding_window: int = 32
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0

    dspark_block_size: int = 5
    dspark_markov_rank: int = 16
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: tuple[int, ...] = (40, 41, 42)

    def __post_init__(self) -> None:
        if self.num_target_layers < 1:
            raise ValueError("num_target_layers must be >= 1")
        if self.num_stages < 1:
            raise ValueError("num_stages must be >= 1")
        if self.dspark_block_size < 1:
            raise ValueError("dspark_block_size must be >= 1")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        if not (0 <= self.dspark_noise_token_id < self.vocab_size):
            raise ValueError(
                f"dspark_noise_token_id={self.dspark_noise_token_id} is outside vocab_size={self.vocab_size}"
            )

    @property
    def qkv_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @classmethod
    def tiny(cls, **overrides) -> DSparkConfig:
        """Small CPU-friendly config used by ``test_dspark.py``."""
        kwargs = dict(
            hidden_size=32,
            num_target_layers=3,
            num_stages=3,
            num_attention_heads=4,
            head_dim=8,
            intermediate_size=64,
            vocab_size=64,
            sliding_window=8,
            dspark_block_size=5,
            dspark_markov_rank=8,
            dspark_noise_token_id=63,
            dspark_target_layer_ids=(0, 1, 2),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def ttnn_tiny(cls, **overrides) -> DSparkConfig:
        """Tile-aligned config for the ttnn port (``matmul_decode`` + prefetcher).

        Hidden / QKV / vocab are 1024 so every projection is a multiple of 64 in N
        (32 B-cores of 32-wide shards) and of 32 in K (32 A-cores of 32-wide shards).
        ``dspark_block_size`` and ``sliding_window`` are one tile so attention never
        needs padding. Markov rank matches hidden so the sequential head shares the
        same GCB geometry as the backbone.
        """
        kwargs = dict(
            hidden_size=1024,
            num_target_layers=3,
            num_stages=1,
            num_attention_heads=32,
            head_dim=32,
            intermediate_size=1024,
            vocab_size=1024,
            sliding_window=32,
            dspark_block_size=32,
            dspark_markov_rank=1024,
            dspark_noise_token_id=1023,
            dspark_target_layer_ids=(0, 1, 2),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def flash_0731(cls, **overrides) -> DSparkConfig:
        """DSpark knobs as shipped in DeepSeek-V4-Flash-0731 ``config.json``."""
        kwargs = dict(
            hidden_size=4096,
            num_target_layers=3,
            num_stages=3,
            num_attention_heads=64,
            head_dim=64,
            intermediate_size=2048,
            vocab_size=129280,
            sliding_window=128,
            dspark_block_size=5,
            dspark_markov_rank=256,
            dspark_noise_token_id=128799,
            dspark_target_layer_ids=(40, 41, 42),
        )
        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass
class DSparkOutput:
    """One draft round, before the target verifies the prefix."""

    draft_ids: torch.Tensor  # [B, gamma]
    logits: torch.Tensor  # [B, gamma, V] after the Markov bias
    base_logits: torch.Tensor  # [B, gamma, V] parallel backbone only
    confidence: torch.Tensor  # [B, gamma] in (0, 1)
    prefix_survival: torch.Tensor  # [B, gamma] cumulative product of confidence
    hidden_states: torch.Tensor  # [B, gamma, D]
    context: torch.Tensor  # [B, S, D] fused target context
    block_input_ids: torch.Tensor  # [B, gamma] anchor + noise tokens


def prefix_survival(confidence: torch.Tensor) -> torch.Tensor:
    """Per-position prefix survival ``a_j = prod_{i<=j} c_i``."""
    return torch.cumprod(confidence, dim=-1)


def truncate_prefix(confidence: torch.Tensor, min_survival: float) -> torch.Tensor:
    """Longest prefix whose survival stays ``>= min_survival``.

    Returns integer lengths ``[B]`` in ``1..gamma``. Position 0 is always kept.
    Truncation is causal: everything after the first drop is discarded.
    """
    survival = prefix_survival(confidence)
    keep = survival >= min_survival
    keep[..., 0] = True
    after_first_drop = (~keep).int().cumsum(dim=-1) > 0
    keep = keep & ~after_first_drop
    return keep.sum(dim=-1)


class DSparkRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1.0e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class DSparkRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:seq_len].to(device=device, dtype=dtype),
            self.sin_cached[:seq_len].to(device=device, dtype=dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., ::2] = -x2
    out[..., 1::2] = x1
    return out


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``x`` ``[B, H, S, Dh]`` at ``position_ids`` ``[B, S]``."""
    cos = cos[position_ids]
    sin = sin[position_ids]
    cos = torch.stack((cos, cos), dim=-1).flatten(-2).unsqueeze(1)
    sin = torch.stack((sin, sin), dim=-1).flatten(-2).unsqueeze(1)
    return x * cos + _rotate_half(x) * sin


def dspark_block_mask(
    ctx_len: int,
    block_size: int,
    sliding_window: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive mask ``[1, 1, G, S+G]``.

    Every block query sees the last ``sliding_window`` context tokens and every
    token in the draft block (non-causal inside the block).
    """
    total = ctx_len + block_size
    mask = torch.full((block_size, total), float("-inf"), device=device, dtype=dtype)
    ctx_start = max(0, ctx_len - sliding_window)
    mask[:, ctx_start:ctx_len] = 0
    mask[:, ctx_len:] = 0
    return mask.view(1, 1, block_size, total)


class DSparkAttention(nn.Module):
    """Block queries attend to ``[context KV ; block KV]`` (KV injection)."""

    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        qkv = config.qkv_dim
        self.q_proj = nn.Linear(config.hidden_size, qkv, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, qkv, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, qkv, bias=False)
        self.o_proj = nn.Linear(qkv, config.hidden_size, bias=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        attn_mask: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        block_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, gamma, _ = hidden_states.shape
        cos, sin = rope

        query = apply_rope(self._split(self.q_proj(hidden_states)), cos, sin, block_position_ids)
        key_ctx = apply_rope(self._split(self.k_proj(context)), cos, sin, context_position_ids)
        value_ctx = self._split(self.v_proj(context))
        key_blk = apply_rope(self._split(self.k_proj(hidden_states)), cos, sin, block_position_ids)
        value_blk = self._split(self.v_proj(hidden_states))

        key = torch.cat([key_ctx, key_blk], dim=2)
        value = torch.cat([value_ctx, value_blk], dim=2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale + attn_mask.to(dtype=query.dtype)
        attn = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        out = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch, gamma, -1)
        return self.o_proj(out)


class DSparkMLP(nn.Module):
    """SwiGLU MLP (stand-in for each checkpoint stage's 256-expert MoE)."""

    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DSparkStage(nn.Module):
    """One uncompressed draft decoder stage with KV-injected attention."""

    def __init__(self, config: DSparkConfig, stage_idx: int):
        super().__init__()
        self.stage_idx = stage_idx
        self.attn_norm = DSparkRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = DSparkAttention(config)
        self.ffn_norm = DSparkRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = DSparkMLP(config)

    def forward(self, hidden_states: torch.Tensor, context: torch.Tensor, **attn_kwargs) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states), context, **attn_kwargs)
        return hidden_states + self.mlp(self.ffn_norm(hidden_states))


class DSparkMarkovHead(nn.Module):
    """Low-rank first-order transition ``B(x_{k-1}, ·) = W1[x_{k-1}] @ W2``."""

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.rank = rank
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(self, prev_ids: torch.Tensor) -> torch.Tensor:
        """``[B, V]`` logit bias given previous-token ids ``[B]``."""
        return self.markov_w2(self.markov_w1(prev_ids))


class DSparkConfidenceHead(nn.Module):
    """``c_k = sigmoid(w^T [h_k ; W1[x_{k-1}]])``."""

    def __init__(self, hidden_size: int, markov_rank: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size + markov_rank, 1, bias=False)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        logits = self.proj(torch.cat([hidden, markov_embed], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits.float()).to(hidden.dtype)


class DSparkModel(nn.Module):
    """DSpark drafter: fuse target hiddens → parallel block → Markov sample."""

    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.config = config
        fused = config.hidden_size * config.num_target_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.mtp = nn.ModuleList()
        for i in range(config.num_stages):
            stage = DSparkStage(config, i)
            if i == 0:
                stage.main_proj = nn.Linear(fused, config.hidden_size, bias=False)
                stage.main_norm = DSparkRMSNorm(config.hidden_size, config.rms_norm_eps)
            if i == config.num_stages - 1:
                stage.norm = DSparkRMSNorm(config.hidden_size, config.rms_norm_eps)
                stage.markov_head = DSparkMarkovHead(config.vocab_size, config.dspark_markov_rank)
                stage.confidence_head = DSparkConfidenceHead(config.hidden_size, config.dspark_markov_rank)
            self.mtp.append(stage)

        self.rotary_emb = DSparkRotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)

    @property
    def main_proj(self) -> nn.Linear:
        return self.mtp[0].main_proj

    @property
    def main_norm(self) -> DSparkRMSNorm:
        return self.mtp[0].main_norm

    @property
    def markov_head(self) -> DSparkMarkovHead:
        return self.mtp[-1].markov_head

    @property
    def confidence_head(self) -> DSparkConfidenceHead:
        return self.mtp[-1].confidence_head

    def share_from_target(self, embed_tokens: nn.Embedding, lm_head: nn.Linear) -> None:
        """Alias the target's frozen embedding / LM head."""
        if embed_tokens.weight.shape != self.embed_tokens.weight.shape:
            raise ValueError("target embed_tokens shape does not match DSparkConfig")
        if lm_head.weight.shape != self.lm_head.weight.shape:
            raise ValueError("target lm_head shape does not match DSparkConfig")
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.embed_tokens.weight.requires_grad_(False)
        self.lm_head.weight.requires_grad_(False)

    def fuse_target_hiddens(self, target_hiddens: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """``H_ctx = RMSNorm(W_c [H^{l1}; ...; H^{lm}])``.

        ``target_hiddens`` is a tuple of ``num_target_layers`` tensors ``[B, S, D]``
        or a stacked tensor ``[B, S, L, D]``.
        """
        if isinstance(target_hiddens, torch.Tensor):
            if target_hiddens.dim() != 4:
                raise ValueError(f"stacked target_hiddens must be [B, S, L, D], got {tuple(target_hiddens.shape)}")
            stacked = target_hiddens
        else:
            if len(target_hiddens) != self.config.num_target_layers:
                raise ValueError(f"expected {self.config.num_target_layers} target layers, got {len(target_hiddens)}")
            stacked = torch.stack(tuple(target_hiddens), dim=2)
        return self.main_norm(self.main_proj(stacked.flatten(-2)))

    def build_block_input_ids(self, anchor_ids: torch.Tensor) -> torch.Tensor:
        """``[anchor, noise, ..., noise]`` of length ``dspark_block_size``."""
        gamma = self.config.dspark_block_size
        noise = torch.full(
            (anchor_ids.shape[0], gamma - 1),
            self.config.dspark_noise_token_id,
            dtype=anchor_ids.dtype,
            device=anchor_ids.device,
        )
        return torch.cat([anchor_ids.view(-1, 1), noise], dim=1)

    def forward(
        self,
        target_hiddens: torch.Tensor | tuple[torch.Tensor, ...],
        anchor_ids: torch.Tensor,
        *,
        greedy: bool = True,
        temperature: float = 1.0,
        min_survival: float | None = None,
    ) -> DSparkOutput:
        """Draft one block conditioned on captured target layer states.

        ``anchor_ids`` is the last token the target committed (the bonus token).
        """
        if anchor_ids.dim() == 2:
            if anchor_ids.shape[-1] != 1:
                raise ValueError(f"anchor_ids must be [B] or [B, 1], got {tuple(anchor_ids.shape)}")
            anchor_ids = anchor_ids.squeeze(-1)

        context = self.fuse_target_hiddens(target_hiddens)
        batch, ctx_len, _ = context.shape
        gamma = self.config.dspark_block_size
        device, dtype = context.device, context.dtype

        block_ids = self.build_block_input_ids(anchor_ids)
        hidden = self.embed_tokens(block_ids)

        ctx_pos = torch.arange(ctx_len, device=device).unsqueeze(0).expand(batch, -1)
        blk_pos = torch.arange(ctx_len, ctx_len + gamma, device=device).unsqueeze(0).expand(batch, -1)
        rope = self.rotary_emb(ctx_len + gamma, device, dtype)
        attn_mask = dspark_block_mask(ctx_len, gamma, self.config.sliding_window, device, dtype)
        attn_kwargs = dict(
            attn_mask=attn_mask,
            rope=rope,
            block_position_ids=blk_pos,
            context_position_ids=ctx_pos,
        )
        for stage in self.mtp:
            hidden = stage(hidden, context, **attn_kwargs)
        hidden = self.mtp[-1].norm(hidden)

        base_logits = self.lm_head(hidden)
        draft_ids, logits, confidence = self._markov_sample(
            hidden, base_logits, anchor_ids, greedy=greedy, temperature=temperature
        )
        survival = prefix_survival(confidence)
        if min_survival is not None:
            lengths = truncate_prefix(confidence, min_survival)
            keep = torch.arange(gamma, device=device).view(1, -1) < lengths.unsqueeze(-1)
            draft_ids = torch.where(keep, draft_ids, torch.zeros_like(draft_ids))
        return DSparkOutput(
            draft_ids=draft_ids,
            logits=logits,
            base_logits=base_logits,
            confidence=confidence,
            prefix_survival=survival,
            hidden_states=hidden,
            context=context,
            block_input_ids=block_ids,
        )

    def _markov_sample(
        self,
        hidden: torch.Tensor,
        base_logits: torch.Tensor,
        anchor_ids: torch.Tensor,
        *,
        greedy: bool,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, gamma, _vocab = base_logits.shape
        draft = torch.empty(batch, gamma, dtype=torch.long, device=hidden.device)
        step_logits = torch.empty_like(base_logits)
        conf = torch.empty(batch, gamma, dtype=hidden.dtype, device=hidden.device)
        prev = anchor_ids
        markov = self.markov_head
        conf_head = self.confidence_head
        for k in range(gamma):
            logits_k = base_logits[:, k] + markov.bias(prev)
            step_logits[:, k] = logits_k
            conf[:, k] = conf_head(hidden[:, k], markov.embed(prev))
            if greedy:
                next_id = logits_k.argmax(dim=-1)
            else:
                scaled = (logits_k.float() / max(temperature, 1e-5)).softmax(dim=-1)
                next_id = torch.multinomial(scaled, num_samples=1).squeeze(-1)
            draft[:, k] = next_id
            prev = next_id
        return draft, step_logits, conf
