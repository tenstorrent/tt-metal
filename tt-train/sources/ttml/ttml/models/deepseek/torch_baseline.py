# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference ("GPU baseline") implementation of nano DeepSeek-V3.

This is a verbatim copy of the model section from
``tt-train_nanoGPT-gpu-baseline/train_deepseek_torch.py``. The tokenizer,
dataset, and training loop are intentionally omitted — this module exposes
only the model so it can be imported by tests that compare the ttml
implementation in :mod:`ttml.models.deepseek` against this reference.

Nothing in this file is imported from the regular deepseek package
``__init__``; torch is only pulled in when a test (or user) explicitly
imports ``ttml.models.deepseek.gpu_baseline``.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# DeepSeek-V3 model — adapted from DeepSeek-V3/inference/model.py for training
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ModelArgs:
    vocab_size: int = 256
    dim: int = 512
    inter_dim: int = 1536
    moe_inter_dim: int = 256
    n_layers: int = 8
    n_dense_layers: int = 2
    n_heads: int = 8
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_expert_groups: int = 2
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_scale: float = 2.5
    q_lora_rank: int = 256
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    max_seq_len: int = 256
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    base = args.rope_theta
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    """Multi-head Latent Attention — naive mode, no KV cache."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank

        self.wq_a = nn.Linear(args.dim, args.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(args.q_lora_rank)
        self.wq_b = nn.Linear(args.q_lora_rank, args.n_heads * self.qk_head_dim, bias=False)
        self.wkv_a = nn.Linear(args.dim, args.kv_lora_rank + args.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(args.kv_lora_rank)
        self.wkv_b = nn.Linear(args.kv_lora_rank, args.n_heads * (args.qk_nope_head_dim + args.v_head_dim), bias=False)
        self.wo = nn.Linear(args.n_heads * args.v_head_dim, args.dim, bias=False)
        self.softmax_scale = self.qk_head_dim**-0.5

    def forward(self, x, freqs_cis, mask):
        bsz, seqlen, _ = x.size()

        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        q = torch.cat([q_nope, q_pe], dim=-1)
        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

        scores = torch.einsum("bshd,bthd->bsht", q, k) * self.softmax_scale
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(2)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        x = torch.einsum("bsht,bthd->bshd", scores, v)
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.n_routed_experts = args.n_routed_experts
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_buffer("expert_bias", torch.zeros(args.n_routed_experts))

    def forward(self, x):
        scores = F.linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores

        # Add bias for load-balanced selection (bias only affects which experts are chosen)
        biased = scores + self.expert_bias

        if self.n_groups > 1:
            biased = biased.view(x.size(0), self.n_groups, -1)
            group_scores = biased.topk(2, dim=-1)[0].sum(dim=-1)
            top_group_idx = group_scores.topk(self.topk_groups, dim=-1)[1]
            group_mask = biased.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, top_group_idx, False)
            biased = biased.masked_fill(group_mask.unsqueeze(-1), float("-inf")).flatten(1)

        indices = torch.topk(biased, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) for _ in range(args.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)
        self._tokens_per_expert = torch.zeros(args.n_routed_experts)

    def forward(self, x):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
        self._tokens_per_expert += counts.float().cpu()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        return (y + z).view(shape)

    @torch.no_grad()
    def update_expert_bias(self, coeff: float = 0.001):
        """Auxiliary-loss-free load balancing matching ttml implementation."""
        mean_count = self._tokens_per_expert.mean()
        delta = coeff * torch.sign(mean_count - self._tokens_per_expert)
        delta -= delta.mean()
        self.gate.expert_bias += delta.to(self.gate.expert_bias.device)
        self._tokens_per_expert.zero_()


class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x, freqs_cis, mask):
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([Block(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    def get_moe_layers(self):
        return [layer.ffn for layer in self.layers if isinstance(layer.ffn, MoE)]

    def forward(self, tokens):
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[:seqlen]
        mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        h = self.norm(h)
        return self.head(h)
