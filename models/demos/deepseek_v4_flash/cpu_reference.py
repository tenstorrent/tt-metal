# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dtype = x.dtype
    x_float = x.float()
    x_norm = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    return (x_norm * weight.float()).to(dtype)


def v4_router(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    *,
    topk: int,
    route_scale: float,
    scoring_func: str = "sqrtsoftplus",
    bias: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = x.float() @ gate_weight.float().T
    if scoring_func == "softmax":
        scores = scores.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        scores = scores.sigmoid()
    elif scoring_func == "sqrtsoftplus":
        scores = F.softplus(scores).sqrt()
    else:
        raise ValueError(f"Unsupported DeepSeek V4 Flash scoring_func {scoring_func!r}")

    original_scores = scores
    selection_scores = scores if bias is None else scores + bias.float()
    if tid2eid is not None:
        if input_ids is None:
            raise ValueError("input_ids is required for hash-routed layers")
        indices = tid2eid[input_ids.reshape(-1)].to(torch.long)
    else:
        indices = selection_scores.topk(topk, dim=-1).indices

    weights = original_scores.gather(-1, indices)
    if scoring_func != "softmax":
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * route_scale, indices


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mix_hc = (2 + hc_mult) * hc_mult
    if mixes.shape[-1] != mix_hc:
        raise ValueError(f"Expected mixes last dim {mix_hc}, got {mixes.shape[-1]}")
    pre_logits = mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    post_logits = mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    comb_logits = mixes[..., 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]

    pre = torch.sigmoid(pre_logits) + eps
    post = 2 * torch.sigmoid(post_logits)
    comb = comb_logits.reshape(*mixes.shape[:-1], hc_mult, hc_mult).softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hyperconnection_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    norm_eps: float = 1e-6,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = x.shape
    x_flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(dim=-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, hc_fn.float()) * rsqrt
    pre, post, comb = hc_split_sinkhorn(
        mixes, hc_scale.float(), hc_base.float(), hc_mult=hc_mult, sinkhorn_iters=sinkhorn_iters, eps=hc_eps
    )
    y = torch.sum(pre.unsqueeze(-1) * x_float_view(x_flat, shape), dim=2)
    return y.to(x.dtype), post, comb


def hyperconnection_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.type_as(x)


def x_float_view(x_flat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return x_flat.view(*shape).float()


def window_topk_indices(window_size: int, batch_size: int, seq_len: int, start_pos: int) -> torch.Tensor:
    if start_pos >= window_size - 1:
        start = start_pos % window_size
        matrix = torch.cat([torch.arange(start + 1, window_size), torch.arange(0, start + 1)])
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seq_len).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seq_len, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


def compress_topk_indices(ratio: int, batch_size: int, seq_len: int, start_pos: int, offset: int) -> torch.Tensor:
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seq_len // ratio).repeat(seq_len, 1)
        mask = matrix >= torch.arange(1, seq_len + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


def compressor_prefill(
    x: torch.Tensor,
    wkv_weight: torch.Tensor,
    wgate_weight: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    compress_ratio: int,
    head_dim: int,
    norm_eps: float = 1e-6,
    overlap: bool | None = None,
) -> torch.Tensor:
    if overlap is None:
        overlap = compress_ratio == 4
    batch_size, seq_len, _ = x.shape
    if seq_len < compress_ratio:
        return x.new_empty(batch_size, 0, head_dim)
    cutoff = seq_len - (seq_len % compress_ratio)
    coff = 2 if overlap else 1
    kv = F.linear(x.float(), wkv_weight.float())[:, :cutoff]
    score = F.linear(x.float(), wgate_weight.float())[:, :cutoff]
    kv = kv.unflatten(1, (-1, compress_ratio))
    score = score.unflatten(1, (-1, compress_ratio)) + ape.float()
    if overlap:
        kv = _overlap_transform(kv, head_dim, value=0)
        score = _overlap_transform(score, head_dim, value=float("-inf"))
    pooled = (kv * score.softmax(dim=2)).sum(dim=2)
    expected_dim = coff * head_dim if not overlap else head_dim
    if pooled.shape[-1] != expected_dim:
        raise ValueError(f"Unexpected pooled dim {pooled.shape[-1]} for head_dim={head_dim}, overlap={overlap}")
    return rms_norm(pooled, norm_weight, norm_eps)


def _overlap_transform(tensor: torch.Tensor, head_dim: int, *, value: float) -> torch.Tensor:
    batch_size, seq_blocks, ratio, _ = tensor.shape
    out = tensor.new_full((batch_size, seq_blocks, 2 * ratio, head_dim), value)
    out[:, :, ratio:] = tensor[:, :, :, head_dim:]
    out[:, 1:, :ratio] = tensor[:, :-1, :, :head_dim]
    return out


def indexer_topk(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    *,
    index_topk: int,
    compress_ratio: int,
    start_pos: int,
    offset: int,
) -> torch.Tensor:
    batch_size, seq_len, _, _ = q.shape
    end_pos = start_pos + seq_len
    kv = kv_cache[:, : end_pos // compress_ratio]
    index_score = torch.einsum("bshd,btd->bsht", q.float(), kv.float())
    index_score = (index_score.relu() * weights.float().unsqueeze(-1)).sum(dim=2)
    if start_pos == 0:
        mask = torch.arange(seq_len // compress_ratio).repeat(seq_len, 1) >= (
            torch.arange(1, seq_len + 1).unsqueeze(1) // compress_ratio
        )
        index_score = index_score + torch.where(mask, float("-inf"), 0.0)
    topk = min(index_topk, end_pos // compress_ratio)
    topk_idxs = index_score.topk(topk, dim=-1).indices
    if start_pos == 0:
        mask = topk_idxs >= torch.arange(1, seq_len + 1).unsqueeze(1) // compress_ratio
        topk_idxs = torch.where(mask, -1, topk_idxs + offset)
    else:
        topk_idxs = topk_idxs + offset
    return topk_idxs


def sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    safe_indices = topk_idxs.clamp_min(0)
    batch_index = torch.arange(kv.shape[0], device=kv.device).view(-1, 1, 1).expand_as(safe_indices)
    gathered = kv[batch_index, safe_indices]
    valid = topk_idxs >= 0
    scores = torch.einsum("bshd,bskd->bshk", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink_scores = attn_sink.float().view(1, 1, -1, 1).expand(q.shape[0], q.shape[1], q.shape[2], 1)
    all_scores = torch.cat([scores, sink_scores], dim=-1)
    probs = all_scores.softmax(dim=-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", probs, gathered.float()).to(q.dtype)


def swiglu_expert(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    *,
    route_weight: torch.Tensor | None = None,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    dtype = x.dtype
    gate = F.linear(x.float(), w1.float())
    up = F.linear(x.float(), w3.float())
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    hidden = F.silu(gate) * up
    if route_weight is not None:
        hidden = route_weight.float() * hidden
    return F.linear(hidden.to(dtype), w2.to(dtype)).to(dtype)


def combine_routed_experts(
    x: torch.Tensor,
    route_weights: torch.Tensor,
    route_indices: torch.Tensor,
    experts: Mapping[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    shared_expert: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    flat_x = x.reshape(-1, x.shape[-1])
    flat_weights = route_weights.reshape(flat_x.shape[0], -1)
    flat_indices = route_indices.reshape(flat_x.shape[0], -1)
    output = torch.zeros_like(flat_x, dtype=torch.float32)
    for expert_id, weights in experts.items():
        token_idx, top_idx = torch.where(flat_indices == expert_id)
        if token_idx.numel() == 0:
            continue
        w1, w2, w3 = weights
        routed = swiglu_expert(
            flat_x[token_idx],
            w1,
            w2,
            w3,
            route_weight=flat_weights[token_idx, top_idx, None],
            swiglu_limit=swiglu_limit,
        )
        output[token_idx] += routed.float()
    if shared_expert is not None:
        output += swiglu_expert(flat_x, *shared_expert).float()
    return output.to(x.dtype).view_as(x)
