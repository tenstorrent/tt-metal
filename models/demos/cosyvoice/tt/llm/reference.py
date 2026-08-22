# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TransformerLM in plain torch, driven only by the flat weight export.

The same device-free proof `tt/flow/reference.py` provides for the CFM estimator,
and for a sharper reason: this stage has four places where the config *implies*
one thing and the code does another, and each of them produces a network that runs
and is silently wrong.

1. `static_chunk_size: 1` on the text encoder. `subsequent_chunk_mask` turns that
   into a plain causal mask. The flow encoder, same class, leaves it at 0 and
   attends fully.
2. `input_layer: 'linear_legacy'` on the AR decoder appends a **ReLU** the plain
   `'linear'` variant does not have.
3. `ConformerEncoder` defaults its FFN activation to swish, `TransformerEncoder`
   to relu. Neither is set in the yaml, so the two stacks differ.
4. Two LayerNorm epsilons: 1e-5 on the embedding norm, 1e-12 on the block norms.

If this file reproduces the captured goldens, all four are right.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..flow.encoder import espnet_rel_positional_encoding

EPS_BLOCK = 1e-12
EPS_EMBED = 1e-5


def rel_shift(x: torch.Tensor) -> torch.Tensor:
    """`RelPositionMultiHeadedAttention.rel_shift` verbatim."""
    zero_pad = torch.zeros((*x.size()[:3], 1), dtype=x.dtype, device=x.device)
    x_padded = torch.cat([zero_pad, x], dim=-1)
    x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
    x = x_padded[:, :, 1:].view_as(x)
    return x[:, :, :, : x.size(-1) // 2 + 1]


def rel_pos_attention(x, pos_emb, w, p, n_head, d_k, mask=None, cache=None):
    """One rel-pos attention layer. Returns `(out, (k, v))`."""
    b, t, _ = x.shape

    def proj(src, name, src_b=None):
        y = F.linear(src, w[f"{p}.{name}.weight"], w.get(f"{p}.{name}.bias"))
        return y.view(src_b or b, -1, n_head, d_k).transpose(1, 2)

    q = proj(x, "linear_q")
    k = proj(x, "linear_k")
    v = proj(x, "linear_v")
    if cache is not None:
        k = torch.cat([cache[0], k], dim=2)
        v = torch.cat([cache[1], v], dim=2)
    new_cache = (k, v)

    pe = F.linear(pos_emb, w[f"{p}.linear_pos.weight"])
    pe = pe.view(pos_emb.shape[0], -1, n_head, d_k).transpose(1, 2)

    qt = q.transpose(1, 2)  # (b, t, h, d_k) so the [h, d_k] biases broadcast over time
    ac = torch.matmul((qt + w[f"{p}.pos_bias_u"]).transpose(1, 2), k.transpose(-2, -1))
    bd = torch.matmul((qt + w[f"{p}.pos_bias_v"]).transpose(1, 2), pe.transpose(-2, -1))
    if ac.shape != bd.shape:
        bd = rel_shift(bd)
    scores = (ac + bd) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    attn = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, n_head * d_k)
    return F.linear(ctx, w[f"{p}.linear_out.weight"], w[f"{p}.linear_out.bias"]), new_cache


def _layer(x, pos_emb, w, p, meta, norm_a, norm_b, act, mask=None, cache=None):
    """One pre-norm block. `norm_a`/`norm_b` name the two LayerNorms, which differ
    between the Conformer (`norm_mha`/`norm_ff`) and Transformer (`norm1`/`norm2`)
    variants but sit in the same places."""
    c = x.shape[-1]
    h = F.layer_norm(x, (c,), w[f"{p}.{norm_a}.weight"], w[f"{p}.{norm_a}.bias"], EPS_BLOCK)
    a, new_cache = rel_pos_attention(h, pos_emb, w, f"{p}.self_attn", meta["n_head"], meta["d_k"], mask, cache)
    x = x + a
    h = F.layer_norm(x, (c,), w[f"{p}.{norm_b}.weight"], w[f"{p}.{norm_b}.bias"], EPS_BLOCK)
    f = F.linear(h, w[f"{p}.feed_forward.w_1.weight"], w[f"{p}.feed_forward.w_1.bias"])
    f = act(f)
    f = F.linear(f, w[f"{p}.feed_forward.w_2.weight"], w[f"{p}.feed_forward.w_2.bias"])
    return x + f, new_cache


def _embed(x, w, p, meta):
    """`embed.out`: Linear -> LayerNorm(1e-5) -> [ReLU] -> scale by sqrt(d_model)."""
    h = F.linear(x, w[f"{p}.embed.out.0.weight"], w[f"{p}.embed.out.0.bias"])
    h = F.layer_norm(h, (h.shape[-1],), w[f"{p}.embed.out.1.weight"], w[f"{p}.embed.out.1.bias"], EPS_EMBED)
    if meta.get("embed_has_relu"):
        h = F.relu(h)
    return h * math.sqrt(meta["d_model"])


def _activation(meta):
    return F.relu if meta.get("ffn_activation", "relu") == "relu" else F.silu


def causal_additive(size: int) -> torch.Tensor:
    keep = torch.tril(torch.ones(size, size, dtype=torch.bool))
    return torch.where(keep, 0.0, -float("inf")).reshape(1, 1, size, size)


def text_encoder(w, xs, meta, *, prefix="text_encoder", causal=True):
    """`[1, T, 512]` -> `[1, T, 1024]`. Causal because `static_chunk_size: 1`."""
    h = _embed(xs, w, prefix, meta)
    pos = espnet_rel_positional_encoding(xs.shape[1], meta["d_model"])
    mask = causal_additive(xs.shape[1]) if causal else None
    act = _activation(meta)
    for i in range(meta["n_layers"]):
        h, _ = _layer(h, pos, w, f"{prefix}.encoders.{i}", meta, "norm_mha", "norm_ff", act, mask)
    return F.layer_norm(h, (h.shape[-1],), w[f"{prefix}.after_norm.weight"], w[f"{prefix}.after_norm.bias"], EPS_BLOCK)


def ar_forward_chunk(w, xs, meta, caches=None, *, prefix="llm", causal=True):
    """One `forward_chunk` of the 14-block decoder. Returns `(ys, caches)`.

    `caches` is a list of `(k, v)` pairs, unpacked rather than in the reference's
    `[layers, head, T, 2*d_k]` packing.
    """
    chunk = xs.shape[1]
    cache_t1 = 0 if not caches else caches[0][0].shape[2]
    pos = espnet_rel_positional_encoding(cache_t1 + chunk, meta["d_model"])
    mask = causal_additive(chunk) if (causal and chunk > 1) else None

    h = _embed(xs, w, prefix, meta)
    act = _activation(meta)
    new_caches = []
    for i in range(meta["n_layers"]):
        cache = caches[i] if caches else None
        h, nc = _layer(h, pos, w, f"{prefix}.encoders.{i}", meta, "norm1", "norm2", act, mask, cache)
        new_caches.append(nc)
    ys = F.layer_norm(h, (h.shape[-1],), w[f"{prefix}.after_norm.weight"], w[f"{prefix}.after_norm.bias"], EPS_BLOCK)
    return ys, new_caches


def pack_cache(caches) -> torch.Tensor:
    """Back to the reference's `[layers, head, T, 2*d_k]` shape, for comparing
    against a captured `out_att_cache`."""
    return torch.cat([torch.cat([k, v], dim=-1) for k, v in caches], dim=0)
