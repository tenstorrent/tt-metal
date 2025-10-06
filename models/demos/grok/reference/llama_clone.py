# SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE-FILE
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# this folder
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
# Small modifications by Tenstorrent for CPU compatibility


import math
from typing import Optional, Tuple

# import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F

# from fairscale.nn.model_parallel.layers import (
#     ColumnParallelLinear,
#     RowParallelLinear,
#     VocabParallelEmbedding,
# )
from torch import nn

# from .args import ModelArgs


class FakeParallelLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        input_is_parallel: bool = False,
        init_method=lambda x: x,
    ):
        super().__init__(in_features, out_features, bias=bias)


RowParallelLinear = ColumnParallelLinear = FakeParallelLinear


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_scaling(freqs: torch.Tensor, scale_factor: float = 8):
    # Values obtained from grid search
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False, scale_factor: float = 8):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs, scale_factor)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.head_dim = 128
        self.n_local_heads = 64
        self.n_local_kv_heads = 8
        self.n_rep = 8

        # self.wq = ColumnParallelLinear(
        #     8192,
        #     8192,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wk = ColumnParallelLinear(
        #     8192,
        #     1024,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wv = ColumnParallelLinear(
        #     8192,
        #     1024,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        self.wqkv = ColumnParallelLinear(
            10240,
            8192,
            bias=False,
            gather_output=False,
            init_method=lambda x: x.squeeze(dim=(0, 1)),
        )
        self.wo = RowParallelLinear(
            8192,
            8192,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x.squeeze(dim=(0, 1)),
        )

        self.cache_k = torch.zeros(
            (
                32,
                128 * 1024,
                8,
                128,
            )
        )
        self.cache_v = torch.zeros(
            (
                32,
                128 * 1024,
                8,
                128,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        # xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # wq, wk, wv = self.wqkv.weight[:, :8192], self.wqkv.weight[:, 8192:9216], self.wqkv.weight[:, 9216:]
        # xq, xk, xv = x @ wq, x @ wk, x @ wv
        x_qkv = x @ self.wqkv.weight
        x_qkv = x_qkv.squeeze(1)
        B, N = x_qkv.shape
        blk_big, blk_mid, blk_tail = 8 * 128, 128, 128
        block = blk_big + blk_mid + blk_tail  # 1280
        repeats = N // block  # 8

        xr = x_qkv.view(B, repeats, block)  # [32, 8, 1280]

        xq = xr[:, :, :blk_big].reshape(B, repeats * blk_big)  # [32, 8192] = [32, 8*1024]
        xk = xr[:, :, blk_big : blk_big + blk_mid].reshape(B, repeats * blk_mid)  # [32, 1024] = [32, 128*8]
        xv = xr[:, :, blk_big + blk_mid :].reshape(B, repeats * blk_tail)  # [32, 1024] = [32, 128*8]

        xq = xq.unsqueeze(1)
        xk = xk.unsqueeze(1)
        xv = xv.unsqueeze(1)

        xq, xk, xv = xq.contiguous(), xk.contiguous(), xv.contiguous()

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, hidden_dim=32768, dim=8192):
        super().__init__()
        hidden_dim = hidden_dim
        dim = dim

        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)

    def forward(self, x):
        # Note that the SGLang implementation uses the tanh approximation on the first element to gelu
        return self.w2(F.gelu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    def __init__(self, num_experts=8):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(8192, num_experts, bias=False)
        self.top_k = 2

    def forward(self, x):
        gate_logits = self.gate(x)
        topk_logits, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1, dtype=torch.float32).to(x.dtype)
        weights = torch.zeros(B, S, T, E, dtype=x.dtype, device=x.device)
        weights.scatter_(dim=-1, index=topk_idx, src=topk_weights)
        return weights


class MoE(nn.Module):
    def __init__(self, num_experts=8, gate=False):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([FeedForward(hidden_dim=16384, dim=8192) for _ in range(num_experts)])
        self.gate = nn.Linear(8192, num_experts, bias=False) if gate else None
        self.top_k = 2

    def forward(self, x):
        """
        x: (B, S, T, H)  e.g. (1, 1, 32, 8192)
        gate(x): (B, S, T, E)  where E = num_experts
        each expert(x): (B, S, T, H)
        """
        B, S, T, H = x.shape
        E = self.num_experts

        if self.gate is not None:
            gate_logits = self.gate(x)  # (B, S, T, E)

            # If your gate returns (B, T, E), add back the singleton stream dim:
            if gate_logits.dim() == 3:
                gate_logits = gate_logits.unsqueeze(1)  # -> (B, 1, T, E)

            topk_weights = F.softmax(gate_logits, dim=-1, dtype=torch.float32).to(x.dtype)  # (B, S, T, K)
            topk_logits, topk_idx = torch.topk(topk_weights, self.top_k, dim=-1)  # (B, S, T, K), (B, S, T, K)

            # Scatter top-k weights back into full (B, S, T, E)
            weights = torch.zeros(B, S, T, E, dtype=x.dtype, device=x.device)  # (B, S, T, E)
            weights.scatter_(dim=-1, index=topk_idx, src=topk_logits)  # fill only selected experts
        else:
            # Uniform mix over all experts
            weights = torch.full((B, S, T, E), 1.0 / E, dtype=x.dtype, device=x.device)

        print(weights)
        print(topk_idx)

        # return weights, topk_weights, topk_idx
        # Weighted sum across experts
        y = torch.zeros_like(x)  # (B, S, T, H)
        for e, expert in enumerate(self.experts):
            y = y + expert(x) * weights[..., e].unsqueeze(-1)  # (B, S, T, H) * (B, S, T, 1)

        return y


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int):
        super().__init__()
        self.n_heads = 64
        self.dim = 8192
        self.head_dim = 128
        self.attention = Attention()
        self.feed_forward = FeedForward()
        self.moe = MoE(num_experts=8, gate=True)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(8192, eps=1e-5)
        self.ffn_norm = RMSNorm(8192, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x
        an = self.attention_norm(x)
        at = self.attention(an, start_pos, freqs_cis, mask)
        h = h + at
        res = h
        fin = self.ffn_norm(h)
        out = res + self.feed_forward(fin) + self.moe(fin)
        return out


class Transformer(nn.Module):
    def __init__(self, seq_len=32):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size,
            params.dim,
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params, llama3))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(params.dim, params.vocab_size, bias=False, init_method=lambda x: x)

        if llama3:
            self.freqs_cis = precompute_freqs_cis(
                params.dim // params.n_heads,
                params.max_seq_len * 2,
                params.rope_theta,
                params.use_scaled_rope,
                params.rope_scaling_factor,
            )
        else:
            self.freqs_cis = precompute_freqs_cis(
                8192 // 64,
                params.max_seq_len * 2,
                params.rope_theta,
                params.use_scaled_rope,
                params.rope_scaling_factor,
            )

    @torch.inference_mode()
    def forward(self, embeddings: torch.Tensor, start_pos: int, mode: str = "decode"):
        _bsz, seqlen, _dim = embeddings.shape
        h = embeddings  # self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=embeddings.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=embeddings.device), mask]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        if mode == "decode":
            h = self.norm(h)
            h = self.output(h)
        else:
            assert mode == "prefill", "Invalid mode, only decode and prefill are supported"
        return h.float()


def reverse_permute(tensor, n_heads, dim1, dim2):
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def permute(tensor, n_heads, dim1, dim2):
    return tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def convert_rope_style_hf_to_meta(cos_hf: torch.Tensor, sin_hf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts RoPE cos/sin tensors from Hugging Face style (half-dim duplicated)
    to Meta style (pairwise duplicated / odd-even interleaved).

    Args:
        cos_hf: Cosine tensor in HF format [..., seq_len, head_dim]
                (e.g., [c0, c1, ..., c_{d/2-1}, c0, c1, ..., c_{d/2-1}])
        sin_hf: Sine tensor in HF format [..., seq_len, head_dim]
                (e.g., [s0, s1, ..., s_{d/2-1}, s0, s1, ..., s_{d/2-1}])

    Returns:
        A tuple containing (cos_meta, sin_meta) in Meta format [..., seq_len, head_dim]
        (e.g., [c0, c0, c1, c1, ..., c_{d/2-1}, c_{d/2-1}],
         [s0, s0, s1, s1, ..., s_{d/2-1}, s_{d/2-1}])
    """
    # Input validation (optional but good practice)
    if cos_hf.shape != sin_hf.shape:
        raise ValueError("cos_hf and sin_hf must have the same shape.")
    if len(cos_hf.shape) < 2:
        raise ValueError("Input tensors must have at least 2 dimensions (seq_len, head_dim).")

    head_dim = cos_hf.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"Head dimension ({head_dim}) must be even.")

    half_head_dim = head_dim // 2

    # Select the first half (contains the unique frequencies)
    cos_unique = cos_hf[..., :half_head_dim]
    sin_unique = sin_hf[..., :half_head_dim]

    # Repeat each unique frequency pairwise
    cos_meta = torch.repeat_interleave(cos_unique, repeats=2, dim=-1)
    sin_meta = torch.repeat_interleave(sin_unique, repeats=2, dim=-1)

    return cos_meta, sin_meta
