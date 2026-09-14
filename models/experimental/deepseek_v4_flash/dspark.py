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


def ngram_continuation(tokens: list[int], gamma: int, *, min_n: int = 2, max_n: int = 8) -> list[int]:
    """Prompt-lookup draft: tokens that followed the current suffix earlier in ``tokens``."""
    n = len(tokens)
    if gamma < 1 or n < min_n + 1:
        return []
    max_n = min(max_n, n - 1)
    for ngram in range(max_n, min_n - 1, -1):
        needle = tokens[-ngram:]
        limit = n - ngram
        for i in range(limit - 1, -1, -1):
            if tokens[i : i + ngram] == needle:
                cont = tokens[i + ngram : i + ngram + gamma]
                if cont:
                    return list(cont)
    return []


def speculative_accept_lengths(draft_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Longest matching prefix of ``draft_ids`` against ``target_ids``, both ``[B, L]``.

    Speculative decoding accepts ``d_0..d_{k-1}`` iff they equal the target greedy
    tokens at those positions, and rejects from the first mismatch. Returns lengths
    ``[B]`` in ``0..L``.
    """
    if draft_ids.shape != target_ids.shape:
        raise ValueError(f"draft/target shape mismatch: {tuple(draft_ids.shape)} vs {tuple(target_ids.shape)}")
    match = draft_ids == target_ids
    after_first_miss = (~match).int().cumsum(dim=-1) > 0
    return (match & ~after_first_miss).sum(dim=-1)


def speculative_accept_rate(draft_ids: torch.Tensor, target_ids: torch.Tensor) -> dict[str, float]:
    """Token- and first-position accept rates for one drafted block vs the target."""
    lengths = speculative_accept_lengths(draft_ids, target_ids)
    gamma = draft_ids.shape[-1]
    n = draft_ids.numel()
    first = (draft_ids[:, 0] == target_ids[:, 0]).float().mean().item()
    prefix = lengths.float().mean().item() / max(gamma, 1)
    token = (draft_ids == target_ids).float().mean().item()
    return {
        "first_token": float(first),
        "mean_prefix_frac": float(prefix),
        "mean_accept_len": float(lengths.float().mean().item()),
        "token": float(token),
        "gamma": float(gamma),
        "n_blocks": float(draft_ids.shape[0]),
        "n_tokens": float(n),
    }


def fuse_flash_mtp_pack(
    pack: torch.Tensor,
    main_proj_weight: torch.Tensor,
    main_norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """``[B, 3, hc, D]`` MTP pack → fused context ``[B, 1, D]`` via ``mtp.0.main_proj``.

    One hidden per tapped layer (``main_proj`` is ``[D, 3D]``). Stream 0 is the
    primary HC residual used as that hidden.
    """
    stacked = pack.float()[:, :, 0, :].unsqueeze(1)
    x = stacked.flatten(-2)
    y = F.linear(x, main_proj_weight.float())
    rms = torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return y * rms * main_norm_weight.float()


def markov_bias(prev_ids: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """First-order Markov logit rows ``[B, V]`` from ``mtp.2.markov_head`` tables."""
    ids = prev_ids.view(-1).long()
    return F.embedding(ids, w1.float()) @ w2.float().T


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
        skip_attn = attn_kwargs.pop("skip_attn", False)
        if not skip_attn:
            hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states), context, **attn_kwargs)
        return hidden_states + self.mlp(self.ffn_norm(hidden_states))


class _FlashRMSNorm(nn.Module):
    """Reference RMSNorm used by the native Flash MTP stages."""

    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = nn.Parameter(weight.float(), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        return (x_f * torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class _FlashHyperConnection(nn.Module):
    """PyTorch reference of the checkpoint's mHC connection."""

    def __init__(self, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, config: DSparkConfig):
        super().__init__()
        self.fn = nn.Parameter(fn.float(), requires_grad=False)
        self.base = nn.Parameter(base.float(), requires_grad=False)
        self.scale = nn.Parameter(scale.float(), requires_grad=False)
        self.hc = 4
        self.hidden = config.hidden_size
        self.eps = config.rms_norm_eps
        self.hc_eps = 1.0e-6
        self.sinkhorn_iters = 20

    def forward(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [B,S,H,D] -> [B,S,(2+H)*H]
        flat = streams.float()
        flat = flat * torch.rsqrt(flat.square().mean(dim=(-1, -2), keepdim=True) + self.eps)
        mixed = F.linear(flat.flatten(-2), self.fn)
        hc = self.hc
        pre_w, post_w, comb_w = mixed.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2.0 * torch.sigmoid(post_w * post_scale + post_b)
        comb = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        collapsed = (pre.unsqueeze(-1) * streams).sum(dim=-2)
        return post.to(streams.dtype), comb.to(streams.dtype), collapsed.to(streams.dtype)


class _FlashMLA(nn.Module):
    """Native MTP MLA, including sinks, partial RoPE, and grouped output."""

    def __init__(self, weights: dict[str, torch.Tensor], config: DSparkConfig):
        super().__init__()
        self.hidden = config.hidden_size
        self.heads = 64
        self.head_dim = 512
        self.rope_dim = 64
        self.groups = 8
        self.group_rank = 1024
        self.eps = config.rms_norm_eps
        self.q_a = nn.Parameter(weights["wq_a"].float(), requires_grad=False)
        self.q_b = nn.Parameter(weights["wq_b"].float(), requires_grad=False)
        self.q_norm = _FlashRMSNorm(weights["q_norm"], self.eps)
        self.kv = nn.Parameter(weights["wkv"].float(), requires_grad=False)
        self.kv_norm = _FlashRMSNorm(weights["kv_norm"], self.eps)
        self.o_a = nn.Parameter(weights["wo_a"].float(), requires_grad=False)
        self.o_b = nn.Parameter(weights["wo_b"].float(), requires_grad=False)
        self.sinks = nn.Parameter(weights["attn_sink"].float(), requires_grad=False)

    @staticmethod
    def _rope(
        x: torch.Tensor, positions: torch.Tensor, rope_dim: int, theta: float, sin_sign: float = 1.0
    ) -> torch.Tensor:
        # x [B,H,S,D], positions [B,S]
        if rope_dim == 0:
            return x
        inv = theta ** (-torch.arange(0, rope_dim, 2, device=x.device, dtype=torch.float32) / rope_dim)
        angles = positions.float().unsqueeze(-1) * inv
        cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).unsqueeze(1)
        sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).unsqueeze(1)
        # V4 lays heads out as [non-RoPE channels | trailing RoPE channels].
        xr = x[..., -rope_dim:]
        x1, x2 = xr[..., ::2], xr[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return torch.cat([x[..., :-rope_dim], xr * cos + rotated * (sin * sin_sign)], dim=-1)

    def forward(
        self,
        query_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        query_positions: torch.Tensor,
        context_positions: torch.Tensor,
        mask: torch.Tensor,
        theta: float,
    ) -> torch.Tensor:
        b, q_len, _ = query_hidden.shape
        kv_input = torch.cat([context_hidden, query_hidden], dim=1)
        q = F.linear(self.q_norm(F.linear(query_hidden, self.q_a)), self.q_b)
        q = q.view(b, q_len, self.heads, self.head_dim).transpose(1, 2)
        q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + self.eps)
        kv = self.kv_norm(F.linear(kv_input, self.kv))
        k = kv.view(b, kv.shape[1], 1, self.head_dim).transpose(1, 2)
        q = self._rope(q, query_positions, self.rope_dim, theta)
        k = self._rope(k, torch.cat([context_positions, query_positions], dim=1), self.rope_dim, theta)
        v = k.transpose(1, 2).contiguous().view(b, kv.shape[1], self.head_dim)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (self.head_dim**-0.5)
        scores = scores + mask.to(scores.dtype)
        sink = self.sinks.view(1, self.heads, 1, 1).expand(b, -1, q_len, -1)
        probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
        out = torch.matmul(probs.to(v.dtype), v.view(b, 1, -1, self.head_dim))
        out = out.transpose(1, 2).contiguous().view(b, q_len, self.heads, self.head_dim)
        # K=V in Flash: undo the value-side rotation at the query position.
        out = self._rope(out.transpose(1, 2), query_positions, self.rope_dim, theta, sin_sign=-1.0).transpose(1, 2)
        out = out.reshape(b, q_len, self.groups, -1)
        out = torch.einsum("bsgk,gok->bsgo", out, self.o_a.view(self.groups, self.group_rank, -1))
        out = out.reshape(b, q_len, -1)
        return F.linear(out, self.o_b)

    def forward_full(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        theta: float,
    ) -> torch.Tensor:
        """Run native MLA over a context-prefix plus draft block sequence."""
        b, seq, _ = hidden.shape
        q = F.linear(self.q_norm(F.linear(hidden, self.q_a)), self.q_b)
        q = q.view(b, seq, self.heads, self.head_dim).transpose(1, 2)
        q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + self.eps)
        kv = self.kv_norm(F.linear(hidden, self.kv))
        k = kv.view(b, seq, 1, self.head_dim).transpose(1, 2)
        q = self._rope(q, positions, self.rope_dim, theta)
        k = self._rope(k, positions, self.rope_dim, theta)
        v = k.transpose(1, 2).contiguous().view(b, seq, self.head_dim)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (self.head_dim**-0.5)
        scores = scores + mask.to(scores.dtype)
        sink = self.sinks.view(1, self.heads, 1, 1).expand(b, -1, seq, -1)
        probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
        out = torch.matmul(probs.to(v.dtype), v.view(b, 1, seq, self.head_dim))
        out = out.transpose(1, 2).contiguous().view(b, seq, self.heads, self.head_dim)
        out = self._rope(out.transpose(1, 2), positions, self.rope_dim, theta, sin_sign=-1.0).transpose(1, 2)
        out = out.reshape(b, seq, self.groups, -1)
        out = torch.einsum("bsgk,gok->bsgo", out, self.o_a.view(self.groups, self.group_rank, -1))
        return F.linear(out.reshape(b, seq, -1), self.o_b)


class _FlashMoE(nn.Module):
    """Native 256-route MTP MoE with lazy expert selection at inference."""

    def __init__(self, weights: dict[str, object], config: DSparkConfig):
        super().__init__()
        self.hidden = config.hidden_size
        self.intermediate = 2048
        self.experts = 256
        self.top_k = 6
        self.scaling = 1.5
        self.limit = 10.0
        self.gate = nn.Parameter(weights["gate"].float(), requires_grad=False)
        self.bias = nn.Parameter(weights["bias"].float(), requires_grad=False)
        self.shared_w1 = nn.Parameter(weights["shared_w1"].float(), requires_grad=False)
        self.shared_w2 = nn.Parameter(weights["shared_w2"].float(), requires_grad=False)
        self.shared_w3 = nn.Parameter(weights["shared_w3"].float(), requires_grad=False)
        self.expert_w1 = nn.ParameterList([nn.Parameter(w.float(), requires_grad=False) for w in weights["expert_w1"]])
        self.expert_w2 = nn.ParameterList([nn.Parameter(w.float(), requires_grad=False) for w in weights["expert_w2"]])
        self.expert_w3 = nn.ParameterList([nn.Parameter(w.float(), requires_grad=False) for w in weights["expert_w3"]])

    def _shared(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.linear(x, self.shared_w1)
        up = F.linear(x, self.shared_w3)
        return F.linear(F.silu(gate) * up, self.shared_w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, self.hidden)
        scores = torch.sqrt(F.softplus(F.linear(flat, self.gate)))
        indices = torch.topk(scores + self.bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1.0e-20) * self.scaling
        routed = torch.zeros_like(flat)
        for expert in torch.unique(indices).tolist():
            rows, slots = torch.where(indices == expert)
            if rows.numel() == 0:
                continue
            selected = flat[rows]
            gate = F.linear(selected, self.expert_w1[expert]).clamp(max=self.limit)
            up = F.linear(selected, self.expert_w3[expert]).clamp(min=-self.limit, max=self.limit)
            value = F.linear(F.silu(gate) * up, self.expert_w2[expert])
            routed.index_add_(0, rows, value * weights[rows, slots].unsqueeze(-1))
        return (routed + self._shared(flat)).view_as(x)


class _FlashMTPStage(nn.Module):
    def __init__(self, weights: dict[str, object], config: DSparkConfig):
        super().__init__()
        self.attn_norm = _FlashRMSNorm(weights["attn_norm"], config.rms_norm_eps)
        self.ffn_norm = _FlashRMSNorm(weights["ffn_norm"], config.rms_norm_eps)
        self.attn = _FlashMLA(weights["attn"], config)
        self.moe = _FlashMoE(weights["moe"], config)
        self.attn_hc = _FlashHyperConnection(
            weights["hc_attn_fn"], weights["hc_attn_base"], weights["hc_attn_scale"], config
        )
        self.ffn_hc = _FlashHyperConnection(
            weights["hc_ffn_fn"], weights["hc_ffn_base"], weights["hc_ffn_scale"], config
        )

    @staticmethod
    def _mix(post: torch.Tensor, comb: torch.Tensor, output: torch.Tensor, streams: torch.Tensor) -> torch.Tensor:
        return post.unsqueeze(-1) * output.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), streams)

    def forward(
        self,
        streams: torch.Tensor,
        context_streams: torch.Tensor,
        query_positions: torch.Tensor,
        context_positions: torch.Tensor,
        mask: torch.Tensor,
        theta: float,
    ) -> torch.Tensor:
        post, comb, collapsed = self.attn_hc(streams)
        _, _, context_collapsed = self.attn_hc(context_streams)
        attn = self.attn(
            self.attn_norm(collapsed),
            self.attn_norm(context_collapsed),
            query_positions,
            context_positions,
            mask,
            theta,
        )
        streams = self._mix(post, comb, attn, streams)
        post, comb, collapsed = self.ffn_hc(streams)
        mlp = self.moe(self.ffn_norm(collapsed))
        return self._mix(post, comb, mlp, streams)

    def forward_full(
        self,
        streams: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        theta: float,
    ) -> torch.Tensor:
        """Apply one checkpoint decoder stage to the whole prefix+block."""
        post, comb, collapsed = self.attn_hc(streams)
        attn = self.attn.forward_full(self.attn_norm(collapsed), positions, mask, theta)
        streams = self._mix(post, comb, attn, streams)
        post, comb, collapsed = self.ffn_hc(streams)
        mlp = self.moe(self.ffn_norm(collapsed))
        return self._mix(post, comb, mlp, streams)


class FlashDSparkModel(nn.Module):
    """Checkpoint-faithful PyTorch DSpark reference.

    Unlike :class:`DSparkModel`, this class loads the native ``mtp.*`` MLA,
    mHC, router, and all routed experts. It is intentionally a reference path;
    the ttnn implementation is wired separately after this output is validated.
    """

    def __init__(
        self,
        config: DSparkConfig,
        embed: torch.Tensor,
        lm_head: torch.Tensor,
        main_proj: torch.Tensor,
        main_norm: torch.Tensor,
        stages: list[dict[str, object]],
        norm: torch.Tensor,
        hc_head_fn: torch.Tensor,
        hc_head_base: torch.Tensor,
        hc_head_scale: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
        confidence: torch.Tensor,
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Parameter(embed.float(), requires_grad=False)
        self.lm_head = nn.Parameter(lm_head.float(), requires_grad=False)
        self.main_proj = nn.Parameter(main_proj.float(), requires_grad=False)
        self.main_norm = _FlashRMSNorm(main_norm, config.rms_norm_eps)
        self.stages = nn.ModuleList([_FlashMTPStage(w, config) for w in stages])
        self.norm = _FlashRMSNorm(norm, config.rms_norm_eps)
        self.hc_head_fn = nn.Parameter(hc_head_fn.float(), requires_grad=False)
        self.hc_head_base = nn.Parameter(hc_head_base.float(), requires_grad=False)
        self.hc_head_scale = nn.Parameter(hc_head_scale.float(), requires_grad=False)
        self.markov_w1 = nn.Parameter(markov_w1.float(), requires_grad=False)
        self.markov_w2 = nn.Parameter(markov_w2.float(), requires_grad=False)
        self.confidence = nn.Parameter(confidence.float(), requires_grad=False)

    @staticmethod
    def _dq(loader, name: str, dequantize_weight: callable) -> torch.Tensor:
        return dequantize_weight(
            loader.get_tensor(name, translate=False),
            loader.get_scale(name, translate=False),
            dtype=torch.float32,
        ).float()

    @classmethod
    def from_checkpoint(cls, loader, dequantize_weight, config: DSparkConfig | None = None):
        config = config or DSparkConfig.flash_0731()

        def raw(name: str) -> torch.Tensor:
            return loader.get_tensor(name, translate=False).float()

        stages = []
        for stage in range(3):
            prefix = f"mtp.{stage}"
            expert_w1, expert_w2, expert_w3 = [], [], []
            for expert in range(256):
                expert_w1.append(cls._dq(loader, f"{prefix}.ffn.experts.{expert}.w1.weight", dequantize_weight))
                expert_w2.append(cls._dq(loader, f"{prefix}.ffn.experts.{expert}.w2.weight", dequantize_weight))
                expert_w3.append(cls._dq(loader, f"{prefix}.ffn.experts.{expert}.w3.weight", dequantize_weight))
            stages.append(
                {
                    "attn_norm": raw(f"{prefix}.attn_norm.weight"),
                    "ffn_norm": raw(f"{prefix}.ffn_norm.weight"),
                    "attn": {
                        "wq_a": cls._dq(loader, f"{prefix}.attn.wq_a.weight", dequantize_weight),
                        "q_norm": raw(f"{prefix}.attn.q_norm.weight"),
                        "wq_b": cls._dq(loader, f"{prefix}.attn.wq_b.weight", dequantize_weight),
                        "wkv": cls._dq(loader, f"{prefix}.attn.wkv.weight", dequantize_weight),
                        "kv_norm": raw(f"{prefix}.attn.kv_norm.weight"),
                        "wo_a": cls._dq(loader, f"{prefix}.attn.wo_a.weight", dequantize_weight),
                        "wo_b": cls._dq(loader, f"{prefix}.attn.wo_b.weight", dequantize_weight),
                        "attn_sink": raw(f"{prefix}.attn.attn_sink"),
                    },
                    "moe": {
                        "gate": raw(f"{prefix}.ffn.gate.weight"),
                        "bias": raw(f"{prefix}.ffn.gate.bias"),
                        "shared_w1": cls._dq(loader, f"{prefix}.ffn.shared_experts.w1.weight", dequantize_weight),
                        "shared_w2": cls._dq(loader, f"{prefix}.ffn.shared_experts.w2.weight", dequantize_weight),
                        "shared_w3": cls._dq(loader, f"{prefix}.ffn.shared_experts.w3.weight", dequantize_weight),
                        "expert_w1": expert_w1,
                        "expert_w2": expert_w2,
                        "expert_w3": expert_w3,
                    },
                    "hc_attn_fn": raw(f"{prefix}.hc_attn_fn"),
                    "hc_attn_base": raw(f"{prefix}.hc_attn_base"),
                    "hc_attn_scale": raw(f"{prefix}.hc_attn_scale"),
                    "hc_ffn_fn": raw(f"{prefix}.hc_ffn_fn"),
                    "hc_ffn_base": raw(f"{prefix}.hc_ffn_base"),
                    "hc_ffn_scale": raw(f"{prefix}.hc_ffn_scale"),
                }
            )
        return cls(
            config,
            raw("embed.weight"),
            raw("head.weight"),
            cls._dq(loader, "mtp.0.main_proj.weight", dequantize_weight),
            raw("mtp.0.main_norm.weight"),
            stages,
            raw("mtp.2.norm.weight"),
            raw("mtp.2.hc_head_fn"),
            raw("mtp.2.hc_head_base"),
            raw("mtp.2.hc_head_scale"),
            raw("mtp.2.markov_head.markov_w1.weight"),
            raw("mtp.2.markov_head.markov_w2.weight"),
            raw("mtp.2.confidence_head.proj.weight"),
        )

    def _head(self, streams: torch.Tensor) -> torch.Tensor:
        flat = streams.float()
        flat = flat * torch.rsqrt(flat.square().mean(dim=(-1, -2), keepdim=True) + self.config.rms_norm_eps)
        mixes = F.linear(flat.flatten(-2), self.hc_head_fn)
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + 1.0e-6
        return (pre.unsqueeze(-1) * streams).sum(dim=-2)

    def forward(
        self,
        target_hiddens: torch.Tensor,
        anchor_ids: torch.Tensor,
        *,
        absolute_position: int = 0,
        greedy: bool = True,
        temperature: float = 1.0,
    ) -> DSparkOutput:
        if target_hiddens.ndim != 4 or target_hiddens.shape[2] != 3:
            raise ValueError("target_hiddens must have shape [B, S, 3, D]")
        if anchor_ids.ndim == 2:
            anchor_ids = anchor_ids.squeeze(-1)
        b, context_len, _, _ = target_hiddens.shape
        gamma = self.config.dspark_block_size
        fused = F.linear(target_hiddens.float().reshape(b, context_len, -1), self.main_proj.float())
        context = self.main_norm(fused)
        ids = self.build_block_input_ids(anchor_ids)
        block = F.embedding(ids, self.embed_tokens).to(context.dtype)
        hc = 4
        context_streams = context.unsqueeze(-2).expand(-1, -1, hc, -1)
        block_streams = block.unsqueeze(-2).expand(-1, -1, hc, -1)
        context_pos = torch.arange(absolute_position, absolute_position + context_len, device=context.device)
        query_pos = torch.arange(
            absolute_position + context_len, absolute_position + context_len + gamma, device=context.device
        )
        context_pos = context_pos.view(1, -1).expand(b, -1)
        query_pos = query_pos.view(1, -1).expand(b, -1)
        total = context_len + gamma
        mask = torch.zeros(b, 1, gamma, total, device=context.device, dtype=torch.float32)
        context_start = max(0, context_len - self.config.sliding_window)
        if context_start:
            mask[:, :, :, :context_start] = float("-inf")
        for stage in self.stages:
            block_streams = stage(
                block_streams,
                context_streams,
                query_pos,
                context_pos,
                mask,
                self.config.rope_theta,
            )
        hidden = self.norm(self._head(block_streams))
        base_logits = F.linear(hidden.float(), self.lm_head.float())
        draft, logits, confidence = self._markov_sample(base_logits, hidden, anchor_ids, greedy, temperature)
        return DSparkOutput(
            draft_ids=draft,
            logits=logits,
            base_logits=base_logits,
            confidence=confidence,
            prefix_survival=prefix_survival(confidence),
            hidden_states=hidden,
            context=context,
            block_input_ids=ids,
        )

    def build_block_input_ids(self, anchor_ids: torch.Tensor) -> torch.Tensor:
        noise = torch.full(
            (anchor_ids.shape[0], self.config.dspark_block_size - 1),
            self.config.dspark_noise_token_id,
            dtype=anchor_ids.dtype,
            device=anchor_ids.device,
        )
        return torch.cat([anchor_ids.view(-1, 1), noise], dim=1)

    def _markov_sample(self, base_logits, hidden, anchor_ids, greedy, temperature):
        draft, logits, confidence = [], [], []
        prev = anchor_ids
        for k in range(self.config.dspark_block_size):
            markov_embed = F.embedding(prev, self.markov_w1)
            row = base_logits[:, k] + F.linear(markov_embed, self.markov_w2)
            probs = torch.softmax(row.float() / max(temperature, 1.0e-5), dim=-1)
            next_id = row.argmax(dim=-1) if greedy else torch.multinomial(probs, 1).squeeze(-1)
            conf_in = torch.cat([hidden[:, k].float(), markov_embed.float()], dim=-1)
            confidence.append(torch.sigmoid(F.linear(conf_in, self.confidence)).squeeze(-1))
            draft.append(next_id)
            logits.append(row)
            prev = next_id
        return torch.stack(draft, 1), torch.stack(logits, 1), torch.stack(confidence, 1)


def load_flash_dspark(loader, dequantize_weight, config: DSparkConfig | None = None) -> FlashDSparkModel:
    """Load the checkpoint-faithful Flash DSpark reference model.

    This is deliberately separate from :meth:`DSparkModel.load_flash_mtp_heads`,
    which remains the lightweight dense stand-in used by the original unit tests.
    """

    return FlashDSparkModel.from_checkpoint(loader, dequantize_weight, config=config)


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

    def load_flash_mtp_heads(self, loader, dequantize_weight) -> None:
        """Load embed, LM head, fusion, Markov, and confidence from a 0731 checkpoint.

        The dense SwiGLU stages load each MTP block's **shared experts** (the 256-route
        MoE is skipped). MLA is not mapped; ``o_proj`` is zeroed so attention does not
        inject random noise when :meth:`forward` is run with ``skip_attn=True``.
        """
        if self.config.hidden_size != 4096 or self.config.dspark_markov_rank != 256:
            raise ValueError("load_flash_mtp_heads expects DSparkConfig.flash_0731 geometry")

        def dq(name: str) -> torch.Tensor:
            weight = loader.get_tensor(name, translate=False)
            scale = loader.get_scale(name, translate=False)
            return dequantize_weight(weight, scale, dtype=torch.bfloat16)

        def raw(name: str) -> torch.Tensor:
            return loader.get_tensor(name, translate=False).to(torch.bfloat16)

        with torch.no_grad():
            self.embed_tokens.weight.copy_(raw("embed.weight"))
            self.lm_head.weight.copy_(raw("head.weight"))
            self.main_proj.weight.copy_(dq("mtp.0.main_proj.weight"))
            self.main_norm.weight.copy_(raw("mtp.0.main_norm.weight"))
            self.mtp[-1].norm.weight.copy_(raw("mtp.2.norm.weight"))
            self.markov_head.markov_w1.weight.copy_(raw("mtp.2.markov_head.markov_w1.weight"))
            self.markov_head.markov_w2.weight.copy_(raw("mtp.2.markov_head.markov_w2.weight"))
            self.confidence_head.proj.weight.copy_(raw("mtp.2.confidence_head.proj.weight"))
            for i, stage in enumerate(self.mtp):
                prefix = f"mtp.{i}"
                stage.attn_norm.weight.copy_(raw(f"{prefix}.attn_norm.weight"))
                stage.ffn_norm.weight.copy_(raw(f"{prefix}.ffn_norm.weight"))
                stage.mlp.gate_proj.weight.copy_(dq(f"{prefix}.ffn.shared_experts.w1.weight"))
                stage.mlp.down_proj.weight.copy_(dq(f"{prefix}.ffn.shared_experts.w2.weight"))
                stage.mlp.up_proj.weight.copy_(dq(f"{prefix}.ffn.shared_experts.w3.weight"))
                stage.attn.o_proj.weight.zero_()
        self.to(torch.bfloat16)

    def draft_from_anchor_embed(self, anchor_ids: torch.Tensor) -> torch.Tensor:
        """Draft a block using ``embed(anchor)`` as a stand-in for layers 40–42.

        Traced decode does not export those hiddens; repeating the embedding through
        ``main_proj`` and the shared-expert MLPs (attention skipped) is the host-side
        approximation used by the 32-chip demo.
        """
        if anchor_ids.dim() == 0:
            anchor_ids = anchor_ids.view(1)
        if anchor_ids.dim() == 2:
            anchor_ids = anchor_ids.squeeze(-1)
        emb = self.embed_tokens(anchor_ids).unsqueeze(1)  # [B, 1, D]
        hiddens = emb.unsqueeze(2).expand(-1, -1, self.config.num_target_layers, -1).contiguous()
        out = self.forward(hiddens, anchor_ids, greedy=True, skip_attn=True)
        return out.draft_ids

    def draft_markov_ids(self, anchor_ids: torch.Tensor, gamma: int | None = None) -> torch.Tensor:
        """Greedy first-order Markov drafts ``[B, gamma]`` from the rank-``r`` head.

        This is the sequential DSpark sampler with a zero backbone (the checkpoint
        Markov tables are trained as a residual on MTP logits; used alone they are a
        cheap bigram drafter that does not need target hiddens).
        """
        if anchor_ids.dim() == 2:
            anchor_ids = anchor_ids.squeeze(-1)
        gamma = self.config.dspark_block_size if gamma is None else gamma
        prev = anchor_ids
        cols = []
        for _ in range(gamma):
            nxt = self.markov_head.bias(prev).argmax(dim=-1)
            cols.append(nxt)
            prev = nxt
        return torch.stack(cols, dim=1)

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
        skip_attn: bool = False,
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
            hidden = stage(hidden, context, skip_attn=skip_attn, **attn_kwargs)
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
