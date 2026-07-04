# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Pure-torch reference for the DeepSeek-V4-Flash **DSpark** speculative module.

DSpark is the same V4-Flash checkpoint plus a bolted-on speculative-decoding
head (DeepSpec). It adds, under the ``mtp.*`` checkpoint namespace, a small stack
of ``n_mtp`` transformer stages that draft ``block_size`` tokens per step:

* **main tap** — the main model exposes ``main_hidden``: at every target layer in
  ``dspark_target_layer_ids`` it takes the residual-stream stack's mean over the
  ``hc_mult`` axis and concatenates them along the feature dim
  (``[B, S, len(target)*dim]``). Only ``mtp.0`` consumes it, via ``main_proj``
  (``len(target)*dim -> dim``) + ``main_norm``.
* **DSparkBlock** — a full V4 block (mHC + attention + 256-expert MoE) but with a
  **sliding-window-only** attention (``compress_ratio == 0``) that runs over the
  ``block_size`` draft query tokens against a sliding KV cache seeded from the
  main token.
* **heads** (last stage only) — ``hc_head`` collapse + ``norm`` + the shared
  ``lm_head``, plus a **Markov head** (embed ``vocab->markov_rank`` then
  ``markov_rank->vocab``) that adds a per-step logit bias inside a length
  ``block_size`` greedy sampling loop, plus a **confidence head**
  (``dim+markov_rank -> 1``).

This module is a faithful, dependency-light (torch + safetensors, no tilelang /
CUDA) port of the DSpark-relevant classes in the checkpoint's
``inference/model.py``: the ``sparse_attn`` tilelang kernel is re-expressed as a
dense gathered-softmax, and ``hc_split_sinkhorn`` as the plain torch Sinkhorn from
``modular_deepseek_v4.py`` (both are bit-equivalent to the fused kernels up to
fp arithmetic order). It runs on CPU under the *system* interpreter (it never
imports ttnn) so it can serve as the gold reference the ttnn port is PCC-gated
against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class DSparkArgs:
    """DSpark hyperparameters, read from the top-level HF ``config.json``."""

    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 2048
    n_layers: int = 43
    n_mtp_layers: int = 3
    n_heads: int = 64
    head_dim: int = 512
    rope_head_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    q_lora_rank: int = 1024
    window_size: int = 128
    n_routed_experts: int = 256
    n_activated_experts: int = 6
    n_shared_experts: int = 1
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    temperature: float = 0.0  # 0 -> greedy argmax (deterministic reference)
    # dspark
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: tuple = (40, 41, 42)
    dspark_markov_rank: int = 256

    @staticmethod
    def from_config_json(path: str | Path, n_mtp_layers: int | None = None) -> "DSparkArgs":
        with open(path) as fh:
            c = json.load(fh)
        # NOTE: the top-level HF ``config.json`` advertises ``num_nextn_predict_layers: 1``,
        # but the DSpark checkpoint actually ships 3 MTP stages (``mtp.0/1/2``) with the
        # heads on the *last* stage. Trust the checkpoint (count the stages), not that
        # field. ``n_mtp_layers`` overrides the default (see :func:`count_mtp_stages`).
        return DSparkArgs(
            vocab_size=c["vocab_size"],
            dim=c["hidden_size"],
            moe_inter_dim=c["moe_intermediate_size"],
            n_layers=c["num_hidden_layers"],
            n_mtp_layers=n_mtp_layers if n_mtp_layers is not None else 3,
            n_heads=c["num_attention_heads"],
            head_dim=c["head_dim"],
            rope_head_dim=c["qk_rope_head_dim"],
            o_groups=c["o_groups"],
            o_lora_rank=c["o_lora_rank"],
            q_lora_rank=c["q_lora_rank"],
            window_size=c["sliding_window"],
            n_routed_experts=c["n_routed_experts"],
            n_activated_experts=c["num_experts_per_tok"],
            n_shared_experts=c["n_shared_experts"],
            route_scale=c["routed_scaling_factor"],
            swiglu_limit=c["swiglu_limit"],
            norm_eps=c["rms_norm_eps"],
            rope_theta=c["rope_theta"],
            hc_mult=c["hc_mult"],
            hc_sinkhorn_iters=c["hc_sinkhorn_iters"],
            hc_eps=c["hc_eps"],
            dspark_block_size=c["dspark_block_size"],
            dspark_noise_token_id=c["dspark_noise_token_id"],
            dspark_target_layer_ids=tuple(c["dspark_target_layer_ids"]),
            dspark_markov_rank=c["dspark_markov_rank"],
        )


# --------------------------------------------------------------------------- #
# RoPE (base theta, no YaRN — DSpark attention has compress_ratio == 0)
# --------------------------------------------------------------------------- #
def precompute_freqs_cis(dim: int, seqlen: int, base: float) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """In-place interleaved rotary embedding on the last ``x.size(-1)`` dims."""
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        freqs_cis = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    xc = torch.view_as_real(xc * freqs_cis).flatten(-2)
    y.copy_(xc)
    return y


def dense_sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """Dense re-expression of the ``sparse_attn`` tilelang kernel.

    ``q`` ``[b, m, h, d]``, ``kv`` ``[b, n, d]`` (a single shared MQA head),
    ``attn_sink`` ``[h]``, ``topk_idxs`` ``[b, m, k]`` (``-1`` = masked). Gathers
    the ``k`` KV rows per query, scores ``scale * q·kv``, softmaxes with an extra
    ``exp(sink - max)`` term folded into the denominator (contributing 0 to the
    numerator), and returns ``[b, m, h, d]``.
    """
    b, m, h, d = q.shape
    k = topk_idxs.shape[-1]
    idx = topk_idxs.clamp(min=0).long()  # [b, m, k]
    valid = (topk_idxs != -1).view(b, m, 1, k)  # [b,m,1,k]
    # gather kv rows -> [b, m, k, d]
    gathered = torch.gather(
        kv.unsqueeze(1).expand(b, m, kv.shape[1], d),
        2,
        idx.unsqueeze(-1).expand(b, m, k, d),
    )
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(~valid, float("-inf"))
    row_max = scores.max(dim=-1, keepdim=True).values  # [b,m,h,1]
    row_max = torch.nan_to_num(row_max, neginf=0.0)
    exp = torch.exp(scores - row_max)
    exp = exp.masked_fill(~valid, 0.0)
    denom = exp.sum(dim=-1, keepdim=True) + torch.exp(attn_sink.view(1, 1, h, 1).float() - row_max)
    out = torch.einsum("bmhk,bmkd->bmhd", exp, gathered.float()) / denom
    return out.to(q.dtype)


# --------------------------------------------------------------------------- #
# Sub-modules
# --------------------------------------------------------------------------- #
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class DSparkAttention(torch.nn.Module):
    """Sliding-window MLA over the ``block_size`` draft query tokens.

    Mirrors ``DSparkAttention`` in ``inference/model.py`` (``compress_ratio == 0``).
    ``main_x`` is the (single) main-token hidden used to seed the sliding KV cache;
    the draft tokens ``x`` (``block_size`` of them) attend to that cache + their own
    block via :func:`dense_sparse_attn`.
    """

    def __init__(self, args: DSparkArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.window_size = args.window_size
        self.eps = args.norm_eps
        self.softmax_scale = self.head_dim**-0.5

        self.attn_sink = torch.nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        self.wq_a = torch.nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = torch.nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = torch.nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = torch.nn.Linear(
            self.n_heads * self.head_dim // self.n_groups, self.n_groups * self.o_lora_rank, bias=False
        )
        self.wo_b = torch.nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False)

        self.register_buffer("freqs_cis", torch.empty(0), persistent=False)
        self.kv_cache: Optional[torch.Tensor] = None  # [b, window, head_dim]

    def _dspark_topk_idxs(self, bsz: int, block_size: int, start_pos: int) -> torch.Tensor:
        win = self.window_size
        matrix = torch.cat([torch.arange(min(win, start_pos + 1)), win + torch.arange(block_size)])
        return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()

    def seed_cache(self, main_x: torch.Tensor, start_pos: int) -> None:
        """Prefill / per-step seeding: write the main token's KV into the sliding cache."""
        bsz, seqlen, _ = main_x.shape
        win, rd = self.window_size, self.rope_head_dim
        main_freqs = self.freqs_cis[start_pos : start_pos + seqlen]
        main_kv = self.kv_norm(self.wkv(main_x))
        apply_rotary_emb(main_kv[..., -rd:], main_freqs)
        if self.kv_cache is None:
            self.kv_cache = main_kv.new_zeros(bsz, win, self.head_dim)
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = main_kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = main_kv[:, -win:].split(
                    [win - cutoff, cutoff], dim=1
                )
        else:
            self.kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1)

    def forward(self, x: torch.Tensor, start_pos: int, main_x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = main_x.shape
        win, rd = self.window_size, self.rope_head_dim
        if start_pos == 0:
            self.seed_cache(main_x, start_pos)
            return x  # prefill only fills the cache

        block_size = x.shape[1]
        freqs = self.freqs_cis[start_pos + seqlen : start_pos + seqlen + block_size]

        q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs)

        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -rd:], freqs)

        topk_idxs = self._dspark_topk_idxs(bsz, block_size, start_pos)
        self.seed_cache(main_x, start_pos)  # writes main_kv at start_pos % win
        kv_full = torch.cat([self.kv_cache[:bsz], kv], dim=1)  # [b, win + block, head_dim]
        o = dense_sparse_attn(q, kv_full, self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs, inverse=True)

        o = o.view(bsz, block_size, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a.float())
        return self.wo_b(o.flatten(2).to(x.dtype))


class Gate(torch.nn.Module):
    """MoE router (score-based, sqrtsoftplus + additive selection bias)."""

    def __init__(self, args: DSparkArgs):
        super().__init__()
        self.topk = args.n_activated_experts
        self.route_scale = args.route_scale
        self.weight = torch.nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = torch.nn.Parameter(torch.zeros(args.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        scores = F.linear(x.float(), self.weight.float())
        scores = F.softplus(scores).sqrt()
        original = scores
        scores = scores + self.bias
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original.gather(1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


def _swiglu_expert(x: torch.Tensor, w1, w2, w3, swiglu_limit: float, weights: Optional[torch.Tensor] = None):
    dtype = x.dtype
    gate = F.linear(x, w1).float()
    up = F.linear(x, w3).float()
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    y = F.silu(gate) * up
    if weights is not None:
        y = weights * y
    return F.linear(y.to(dtype), w2)


class MoE(torch.nn.Module):
    """Score-routed 256-expert MoE with a lazy expert ``provider``.

    To keep memory bounded (256 experts × 3 stages would be ~tens of GB if all
    dequantized), routed experts are loaded on demand: only the experts some token
    in the batch actually selects are pulled from ``provider(e) -> (w1, w2, w3)``.
    The shared expert (always active) is materialised eagerly.
    """

    def __init__(self, args: DSparkArgs):
        super().__init__()
        self.dim = args.dim
        self.n_routed = args.n_routed_experts
        self.swiglu_limit = args.swiglu_limit
        self.gate = Gate(args)
        self.shared_w1 = torch.nn.Parameter(torch.empty(args.moe_inter_dim, args.dim))
        self.shared_w2 = torch.nn.Parameter(torch.empty(args.dim, args.moe_inter_dim))
        self.shared_w3 = torch.nn.Parameter(torch.empty(args.moe_inter_dim, args.dim))
        self.expert_provider = None  # set by load_dspark_weights: e -> (w1, w2, w3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x, dtype=torch.float32)
        for i in torch.unique(indices).tolist():
            idx, top = torch.where(indices == i)
            w1, w2, w3 = self.expert_provider(i)
            y[idx] += _swiglu_expert(x[idx], w1, w2, w3, self.swiglu_limit, weights[idx, top, None])
        y = y + _swiglu_expert(x, self.shared_w1, self.shared_w2, self.shared_w3, self.swiglu_limit)
        return y.type_as(x).view(shape)


def _sinkhorn_comb(comb_logits: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    comb = torch.softmax(comb_logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


class DSparkBlock(torch.nn.Module):
    """One DSpark MTP stage: mHC + :class:`DSparkAttention` + :class:`MoE`.

    ``stage_id == 0`` owns ``main_proj`` / ``main_norm`` (consuming ``main_hidden``);
    ``stage_id == n_mtp - 1`` owns the ``norm`` + ``hc_head`` + markov / confidence
    heads used by :meth:`DSparkModel.forward_head`.
    """

    def __init__(self, args: DSparkArgs, stage_id: int):
        super().__init__()
        self.args = args
        self.stage_id = stage_id
        self.hc_mult = args.hc_mult
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.hc_iters = args.hc_sinkhorn_iters
        hc_dim = args.hc_mult * args.dim
        mix_hc = (2 + args.hc_mult) * args.hc_mult

        self.attn = DSparkAttention(args)
        self.ffn = MoE(args)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.hc_attn_fn = torch.nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_ffn_fn = torch.nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_attn_base = torch.nn.Parameter(torch.empty(mix_hc))
        self.hc_ffn_base = torch.nn.Parameter(torch.empty(mix_hc))
        self.hc_attn_scale = torch.nn.Parameter(torch.empty(3))
        self.hc_ffn_scale = torch.nn.Parameter(torch.empty(3))

        if stage_id == 0:
            n_tgt = len(args.dspark_target_layer_ids)
            self.main_proj = torch.nn.Linear(args.dim * n_tgt, args.dim, bias=False)
            self.main_norm = RMSNorm(args.dim, args.norm_eps)
        if stage_id == args.n_mtp_layers - 1:
            self.norm = RMSNorm(args.dim, args.norm_eps)
            self.markov_w1 = torch.nn.Parameter(torch.empty(args.vocab_size, args.dspark_markov_rank))
            self.markov_w2 = torch.nn.Parameter(torch.empty(args.vocab_size, args.dspark_markov_rank))
            self.confidence_proj = torch.nn.Parameter(torch.empty(1, args.dim + args.dspark_markov_rank))
            self.hc_head_fn = torch.nn.Parameter(torch.empty(args.hc_mult, hc_dim))
            self.hc_head_base = torch.nn.Parameter(torch.empty(args.hc_mult))
            self.hc_head_scale = torch.nn.Parameter(torch.empty(1))

    # -- mHC helpers ------------------------------------------------------- #
    def hc_pre(self, x: torch.Tensor, fn, scale, base):
        shape, dtype = x.shape, x.dtype
        hc = self.hc_mult
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(xf * rsqrt, fn.float())
        pre_w, post_w, comb_w = mixes.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = base.split([hc, hc, hc * hc])
        pre = torch.sigmoid(pre_w * scale[0] + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * scale[1] + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * scale[2] + comb_b.view(hc, hc)
        comb = _sinkhorn_comb(comb_logits, self.hc_iters, self.hc_eps)
        collapsed = (pre.unsqueeze(-1) * x.view(shape)).sum(dim=2)
        return collapsed.to(dtype), post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def hc_head(self, x: torch.Tensor, fn, scale, base):
        shape, dtype = x.shape, x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(xf * rsqrt, fn.float())
        pre = torch.sigmoid(mixes * scale + base) + self.hc_eps
        return (pre.unsqueeze(-1) * x.view(shape)).sum(dim=2).to(dtype)

    def forward(self, x: torch.Tensor, start_pos: int, main_x: torch.Tensor) -> torch.Tensor:
        residual = x
        collapsed, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        attn_out = self.attn(self.attn_norm(collapsed), start_pos, main_x)
        if start_pos == 0:
            return x  # prefill: attention only seeded the cache
        x = self.hc_post(attn_out, residual, post, comb)

        residual = x
        collapsed, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        ffn_out = self.ffn(self.ffn_norm(collapsed))
        return self.hc_post(ffn_out, residual, post, comb)


def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


class DSparkModel(torch.nn.Module):
    """Standalone DSpark speculative stack (embed + lm_head are the shared main
    ones). Consumes ``main_hidden`` + the accepted token and drafts ``block_size``
    tokens with logits + a per-token confidence."""

    def __init__(self, args: DSparkArgs):
        super().__init__()
        self.args = args
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        self.temperature = args.temperature
        self.embed = torch.nn.Parameter(torch.empty(args.vocab_size, args.dim))  # shared main embed
        self.head = torch.nn.Parameter(torch.empty(args.vocab_size, args.dim))  # shared main lm_head
        self.mtp = torch.nn.ModuleList([DSparkBlock(args, s) for s in range(args.n_mtp_layers)])

    # -- stage-0 embed of the draft block --------------------------------- #
    def forward_embed(self, main_hidden: torch.Tensor, input_ids: torch.Tensor):
        stage0 = self.mtp[0]
        main_x = stage0.main_norm(stage0.main_proj(main_hidden))
        draft_ids = input_ids.new_full((input_ids.size(0), self.block_size), self.noise_token_id)
        draft_ids[:, 0] = input_ids
        x = F.embedding(draft_ids, self.embed)
        x = x.unsqueeze(2).repeat(1, 1, self.args.hc_mult, 1)
        return x, main_x

    def markov_head(self, token_ids: torch.Tensor):
        embed = F.embedding(token_ids, self.mtp[-1].markov_w1)
        logits = F.linear(embed.float(), self.mtp[-1].markov_w2.float())
        return logits, embed

    def forward_head(self, x: torch.Tensor, input_ids: torch.Tensor):
        last = self.mtp[-1]
        x = last.hc_head(x, last.hc_head_fn, last.hc_head_scale, last.hc_head_base)
        logits = F.linear(last.norm(x).float(), self.head.float())
        output_ids = input_ids.new_empty(input_ids.size(0), self.block_size + 1)
        output_ids[:, 0] = input_ids
        markov_embeds = []
        for i in range(self.block_size):
            bias, m_embed = self.markov_head(output_ids[:, i])
            logits[:, i] = logits[:, i] + bias
            markov_embeds.append(m_embed)
            output_ids[:, i + 1] = _sample(logits[:, i], self.temperature)
        markov_embed = torch.stack(markov_embeds, dim=1)
        conf_in = torch.cat([x, markov_embed], dim=-1)
        confidence = F.linear(conf_in.float(), last.confidence_proj.float()).squeeze(-1)
        return output_ids, logits, confidence

    def forward_spec(self, input_ids: torch.Tensor, main_hidden: torch.Tensor, start_pos: int):
        x, main_x = self.forward_embed(main_hidden, input_ids)
        for layer in self.mtp:
            x = layer(x, start_pos, main_x)
        if start_pos == 0:
            return None
        return self.forward_head(x, input_ids)


# --------------------------------------------------------------------------- #
# Weight loading (native ``mtp.*`` checkpoint names, dequantized)
# --------------------------------------------------------------------------- #
def _make_expert_provider(wfn, prefix):
    """Lazy, memoized routed-expert loader: ``e -> (w1, w2, w3)`` (dequantized)."""
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def provider(e: int):
        return (
            wfn(f"{prefix}.ffn.experts.{e}.w1.weight"),
            wfn(f"{prefix}.ffn.experts.{e}.w2.weight"),
            wfn(f"{prefix}.ffn.experts.{e}.w3.weight"),
        )

    return provider


def count_mtp_stages(loader) -> int:
    """Number of ``mtp.<s>.*`` stages present in the checkpoint (the config's
    ``num_nextn_predict_layers`` is unreliable for the DSpark checkpoint)."""
    import re

    stages = set()
    for k in loader.keys():
        m = re.match(r"mtp\.(\d+)\.", k)
        if m:
            stages.add(int(m.group(1)))
    return (max(stages) + 1) if stages else 0


def load_dspark_weights(model: DSparkModel, loader, quant) -> None:
    """Populate ``model`` from the checkpoint via the lazy ``loader`` + ``quant``
    dequantizers, using the native (untranslated) ``mtp.*`` names."""

    def w(name: str) -> torch.Tensor:
        t = loader.get_tensor(name, translate=False)
        s = loader.get_scale(name, translate=False)
        return quant.dequantize_weight(t, s).float()

    args = model.args
    model.embed.data.copy_(w("embed.weight"))
    model.head.data.copy_(w("head.weight"))
    for s, block in enumerate(model.mtp):
        p = f"mtp.{s}"
        block.attn.attn_sink.data.copy_(w(f"{p}.attn.attn_sink"))
        block.attn.wq_a.weight.data.copy_(w(f"{p}.attn.wq_a.weight"))
        block.attn.q_norm.weight.data.copy_(w(f"{p}.attn.q_norm.weight"))
        block.attn.wq_b.weight.data.copy_(w(f"{p}.attn.wq_b.weight"))
        block.attn.wkv.weight.data.copy_(w(f"{p}.attn.wkv.weight"))
        block.attn.kv_norm.weight.data.copy_(w(f"{p}.attn.kv_norm.weight"))
        block.attn.wo_a.weight.data.copy_(w(f"{p}.attn.wo_a.weight"))
        block.attn.wo_b.weight.data.copy_(w(f"{p}.attn.wo_b.weight"))
        block.attn.freqs_cis = precompute_freqs_cis(args.rope_head_dim, 8192, args.rope_theta)

        block.ffn.gate.weight.data.copy_(w(f"{p}.ffn.gate.weight"))
        block.ffn.gate.bias.data.copy_(w(f"{p}.ffn.gate.bias"))
        block.ffn.shared_w1.data.copy_(w(f"{p}.ffn.shared_experts.w1.weight"))
        block.ffn.shared_w2.data.copy_(w(f"{p}.ffn.shared_experts.w2.weight"))
        block.ffn.shared_w3.data.copy_(w(f"{p}.ffn.shared_experts.w3.weight"))
        # Routed experts are loaded lazily (only the selected ones), dequantized on
        # demand from the checkpoint's MXFP4 packing.
        block.ffn.expert_provider = _make_expert_provider(w, p)

        block.attn_norm.weight.data.copy_(w(f"{p}.attn_norm.weight"))
        block.ffn_norm.weight.data.copy_(w(f"{p}.ffn_norm.weight"))
        block.hc_attn_fn.data.copy_(w(f"{p}.hc_attn_fn"))
        block.hc_ffn_fn.data.copy_(w(f"{p}.hc_ffn_fn"))
        block.hc_attn_base.data.copy_(w(f"{p}.hc_attn_base"))
        block.hc_ffn_base.data.copy_(w(f"{p}.hc_ffn_base"))
        block.hc_attn_scale.data.copy_(w(f"{p}.hc_attn_scale"))
        block.hc_ffn_scale.data.copy_(w(f"{p}.hc_ffn_scale"))

        if s == 0:
            block.main_proj.weight.data.copy_(w(f"{p}.main_proj.weight"))
            block.main_norm.weight.data.copy_(w(f"{p}.main_norm.weight"))
        if s == args.n_mtp_layers - 1:
            block.norm.weight.data.copy_(w(f"{p}.norm.weight"))
            block.markov_w1.data.copy_(w(f"{p}.markov_head.markov_w1.weight"))
            block.markov_w2.data.copy_(w(f"{p}.markov_head.markov_w2.weight"))
            block.confidence_proj.data.copy_(w(f"{p}.confidence_head.proj.weight"))
            block.hc_head_fn.data.copy_(w(f"{p}.hc_head_fn"))
            block.hc_head_base.data.copy_(w(f"{p}.hc_head_base"))
            block.hc_head_scale.data.copy_(w(f"{p}.hc_head_scale"))


__all__ = [
    "DSparkArgs",
    "DSparkModel",
    "DSparkBlock",
    "DSparkAttention",
    "MoE",
    "load_dspark_weights",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "dense_sparse_attn",
]
