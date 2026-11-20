# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch ground-truth reference for one decode step of Hy4-preview's sparse attention
(tenstorrent/tt-metal#65889, with the decode side of the indexer, #65888).

Pure torch, runs on CPU. It adds no arithmetic of its own: a decode step is the attention
reference (attention.py) and the indexer reference (indexer.py) applied to one new token and a
cache, in the order HF's HYV4Attention.forward runs them when it is given a cache:

    new token     q_resid, (k_pass | k_rope) = latent(x)          the 576-wide cache entry
                  index key = indexer.keys(x)                     128 wide, full layers only
    append        both go into the cache at the token's position
    select        full layer:   top-min(2048, P + 1) of the indexer score over every cached key,
                                the new token's included
                  shared layer: the picks the nearest full layer below made in this same step
    attend        softmax over the SELECTED keys and the per-head sink, values summed, the sink's
                  share dropped; then the sigmoid output gate and o_proj

HF attends over every cached key with an additive mask that removes the keys that were not
picked. Here the picked rows are gathered and attention runs on those rows alone, which is what a
device has to do at a context of a million tokens; the two are the same sum with the zero terms
left out, and tests/test_attention_decode_host.py pins the step to the last row of HF's
full-sequence attention and to HF's own cached decode.

The same step is written three ways, all equal up to the order of the float sums:

* ``stages`` (expanded): kv_b_proj turns every selected latent into per-head keys and values,
  then the dense core of attention.py runs on them. This is HF's arithmetic.
* ``absorbed_stages``: kv_b_proj is folded into the query and into the output instead,
      q_abs[h]   = q_nope[h] W_k[h]                    [H, kv_lora]
      logit[h,t] = (q_abs[h] . k_pass[t] + q_rope[h] . k_rope[t]) * scaling
      ctx[h]     = sum_t prob[h,t] k_pass[t]           [H, kv_lora]
      attn[h]    = W_v[h] ctx[h]                       [H, v]
  so the keys and values of the 2048 selected rows are never expanded: the selected rows are only
  ever read as the 576-wide latent. tt/attention_decode.py runs this form by default.
* ``lse_merge``: the absorbed form with the selected rows left on the chips that hold them. Every
  chip attends over its own rows and returns (ctx, log-sum-exp) per head; the partial results are
  merged with softmax weights, the sink entering the total once. This is the design #65889 names,
  written in torch so that the claim "it gives the same answer" has a test; the device module
  does not run it (see tt/attention_decode.py for why).

A pick list can carry entries that are not real picks (``valid`` False): the device's selection has
a fixed width, and a query that sees fewer positions than that gets masked slots in the tail. They
are removed with the same additive mask HF uses.

Everything is per user and per layer: x is [1, hidden], the cache holds [P, width] rows.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.experimental.hy4_preview.reference.attention import Hy4Attention, rope_cos_sin
from models.experimental.hy4_preview.reference.indexer import Hy4Indexer


@dataclass
class DecodeCache:
    """One user's cache of one layer, in position order: row p is position p.

    ``latent`` [P, kv_lora + rope] is (k_pass | k_rope); ``index_keys`` [P, index head dim] exists on
    full layers only.
    """

    latent: torch.Tensor
    index_keys: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.latent.shape[0]

    def append(self, latent: torch.Tensor, index_key: torch.Tensor | None = None):
        """Add one token's rows ([1, width] each) at the next position."""
        self.latent = torch.cat([self.latent, latent], dim=0)
        if self.index_keys is not None:
            assert index_key is not None, "a full layer caches an index key with every token"
            self.index_keys = torch.cat([self.index_keys, index_key], dim=0)


@dataclass
class NewToken:
    """What the new token contributes before anything is read from the cache."""

    position: int
    cos: torch.Tensor  # [1, rope]
    sin: torch.Tensor  # [1, rope]
    q_resid: torch.Tensor  # [1, q_lora]
    latent: torch.Tensor  # [1, kv_lora + rope], the cache entry
    index_key: torch.Tensor | None = None  # [1, index head dim], the cache entry of a full layer
    index_q: torch.Tensor | None = None  # [1, index heads, index head dim]
    index_w: torch.Tensor | None = None  # [1, index heads]


@dataclass
class DecodeOutput:
    out: torch.Tensor  # [1, hidden], the sublayer output
    picks: torch.Tensor  # int64 [K], the positions attended over (best first on a full layer)
    valid: torch.Tensor  # bool [K]
    token: NewToken


def additive_mask(valid: torch.Tensor) -> torch.Tensor:
    """bool [K] -> additive fp32 [1, K]: 0 on a real pick, the most negative value elsewhere."""
    return torch.zeros(1, valid.shape[0]).masked_fill(~valid[None, :], torch.finfo(torch.float32).min)


class Hy4AttentionDecode:
    """A decode step of one layer. ``indexer`` is the layer's own on a full layer, None on a shared one."""

    def __init__(self, attention: Hy4Attention, indexer: Hy4Indexer | None = None):
        self.attention = attention
        self.indexer = indexer
        self.cfg = attention.cfg

    @property
    def is_full(self) -> bool:
        return self.indexer is not None

    def as_bf16(self) -> "Hy4AttentionDecode":
        """This layer run the way the checkpoint is released and HF runs it: every parameter and every
        activation in bf16, the norms and the indexer's score sums in fp32 (HYV4RMSNorm,
        HYV4Indexer.forward). The methods of this class follow the dtype of what they are given, so
        the copy's ``new_token`` on a bf16 ``x`` returns q_resid, the index key and the indexer's
        query heads rounded to bf16 at the points where HF rounds them.

        A device runs the model at this precision, not in fp32. The selection is a discrete choice,
        so the two precisions do not pick the same 2048 positions: the scores next to the cut change
        sides. How many do is the floor for any bf16 implementation compared with the fp32 reference,
        and tests/test_attention_decode.py measures a device's selection against that floor.
        """
        attention = copy.deepcopy(self.attention).to(torch.bfloat16)
        indexer = None if self.indexer is None else copy.deepcopy(self.indexer).to(torch.bfloat16)
        return Hy4AttentionDecode(attention, indexer)

    # ------------------------------------------------------------------------------------------
    # the cache
    # ------------------------------------------------------------------------------------------
    def prefill_cache(self, x: torch.Tensor, start: int = 0) -> DecodeCache:
        """The cache a prompt leaves behind. x: [P, hidden], the layer's pre-normed inputs at
        positions ``start ..``; P may be 0."""
        positions = torch.arange(start, start + x.shape[0])
        cos, sin = (t.to(x.dtype) for t in rope_cos_sin(positions, self.cfg))
        _, k_pass, k_rope = self.attention.latent(x, cos, sin)
        index_keys = self.indexer.keys(x, cos, sin) if self.is_full else None
        return DecodeCache(torch.cat([k_pass, k_rope], dim=-1), index_keys)

    def new_token(self, x: torch.Tensor, position: int) -> NewToken:
        """x: [1, hidden] at ``position`` -> its cache entries and the queries it will ask."""
        assert x.shape == (1, self.cfg.hidden_size), tuple(x.shape)
        cos, sin = (t.to(x.dtype) for t in rope_cos_sin(torch.tensor([position]), self.cfg))
        q_resid, k_pass, k_rope = self.attention.latent(x, cos, sin)
        token = NewToken(position, cos, sin, q_resid, torch.cat([k_pass, k_rope], dim=-1))
        if self.is_full:
            token.index_key = self.indexer.keys(x, cos, sin)
            token.index_q = self.indexer.queries(q_resid, cos, sin)
            token.index_w = self.indexer.head_weights(x)
        return token

    # ------------------------------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------------------------------
    def scores(self, token: NewToken, cache: DecodeCache) -> torch.Tensor:
        """The indexer score of every cached position for the new token: fp32 [P]. The cache must
        already hold the token's own key."""
        assert self.is_full, "a shared layer has no indexer of its own"
        return Hy4Indexer.scores(token.index_q, token.index_w, cache.index_keys)[0]

    def select(self, token: NewToken, cache: DecodeCache) -> torch.Tensor:
        """The top-min(topk, P) cached positions for the new token, best first: int64 [K]. At decode
        the query is the last position, so nothing is masked and every pick is real."""
        scores = self.scores(token, cache)
        return scores.topk(min(self.indexer.cfg.topk, scores.shape[0])).indices

    @staticmethod
    def gather(cache: DecodeCache, picks: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """The latents of the picked positions, [K, kv_lora + rope], zero where the entry is not valid
        (such an entry may name a position the cache does not hold yet)."""
        if valid is None:
            return cache.latent[picks.long()]
        return cache.latent[torch.where(valid, picks.long(), 0)] * valid[:, None]

    # ------------------------------------------------------------------------------------------
    # attention over the selected rows
    # ------------------------------------------------------------------------------------------
    def _split(self, selected: torch.Tensor):
        c = self.cfg
        return selected.split([c.kv_lora_rank, c.qk_rope_head_dim], dim=-1)

    def stages(self, x: torch.Tensor, token: NewToken, selected: torch.Tensor, valid: torch.Tensor | None = None):
        """HF's arithmetic on the selected rows. selected: [K, kv_lora + rope]; valid: bool [K] or None.

        Returns every intermediate a device stage can be checked against:
            q [H, 1, qk]   gate [1, H * v] (logits)   keys [H, K, qk]   values [H, K, v]
            attn [H, 1, v]   out [1, hidden]
        """
        a = self.attention
        valid = torch.ones(selected.shape[0], dtype=torch.bool) if valid is None else valid
        k_pass, k_rope = self._split(selected)
        q = a.queries(token.q_resid, token.cos, token.sin)
        keys, values = a.keys_values(k_pass, k_rope)
        gate = F.linear(x, a.linear_gate)
        attn = a.core(q, keys, values, additive_mask(valid))
        out = F.linear(attn.transpose(0, 1).reshape(1, -1) * torch.sigmoid(gate), a.o_proj)
        return {"q": q, "gate": gate, "keys": keys, "values": values, "attn": attn, "out": out}

    def absorb_weights(self):
        """kv_b_proj per head: (W_k [H, nope, kv_lora], W_v [H, v, kv_lora])."""
        c = self.cfg
        w = self.attention.kv_b_proj.view(c.num_heads, c.kv_b_head_dim, c.kv_lora_rank)
        return w[:, : c.qk_nope_head_dim], w[:, c.qk_nope_head_dim :]

    def latent_queries(self, token: NewToken) -> torch.Tensor:
        """The queries moved into the latent's own space: [H, kv_lora + rope], (q_nope W_k | q_rope),
        so that a logit is one dot product with a cache row."""
        c = self.cfg
        q = self.attention.queries(token.q_resid, token.cos, token.sin)[:, 0]  # [H, qk]
        w_k, _ = self.absorb_weights()
        q_abs = torch.einsum("hn,hnk->hk", q[:, : c.qk_nope_head_dim], w_k)
        return torch.cat([q_abs, q[:, c.qk_nope_head_dim :]], dim=-1)

    def absorbed_stages(
        self, x: torch.Tensor, token: NewToken, selected: torch.Tensor, valid: torch.Tensor | None = None
    ):
        """The same step with kv_b_proj folded into the query and the output (see the module doc).

        Returns q_lat [H, kv_lora + rope], logits [H, K] (scaled and masked, without the sink),
        probs [H, K] (the sink's share dropped), ctx [H, kv_lora], attn [H, 1, v], gate, out.
        """
        a, c = self.attention, self.cfg
        valid = torch.ones(selected.shape[0], dtype=torch.bool) if valid is None else valid
        k_pass, _ = self._split(selected)
        q_lat = self.latent_queries(token)
        logits = torch.matmul(q_lat, selected.t()) * c.scaling + additive_mask(valid)
        combined = torch.cat([logits, a.learnable_sink_param.reshape(-1, 1)], dim=-1)
        probs = F.softmax(combined - combined.max(dim=-1, keepdim=True).values, dim=-1)[:, :-1]
        ctx = torch.matmul(probs, k_pass)
        _, w_v = self.absorb_weights()
        attn = torch.einsum("hvk,hk->hv", w_v, ctx)[:, None, :]
        gate = F.linear(x, a.linear_gate)
        out = F.linear(attn.transpose(0, 1).reshape(1, -1) * torch.sigmoid(gate), a.o_proj)
        return {"q_lat": q_lat, "logits": logits, "probs": probs, "ctx": ctx, "attn": attn, "gate": gate, "out": out}

    def lse_merge(self, x: torch.Tensor, token: NewToken, cache: DecodeCache, picks, valid, layout):
        """The step with the selected rows left where the context-parallel cache holds them.

        ``layout`` is tt/kv_layout.py BlockCyclicLayout. Chip c attends, with all heads, over the
        picks it owns and returns ctx_c [H, kv_lora] (softmax over its own rows only) and
        lse_c [H] = log sum exp of its logits; a chip that owns none returns -inf and zeros. The
        merge weighs each partial by exp(lse_c - total), total being the log-sum-exp over the chips
        and the sink. Returns (out [1, hidden], rows per chip [num_chips]).
        """
        a, c = self.attention, self.cfg
        picks = picks.long()
        valid = torch.ones_like(picks, dtype=torch.bool) if valid is None else valid
        q_lat = self.latent_queries(token)
        owner = layout.chip(picks)
        partial_ctx, partial_lse, rows = [], [], []
        for chip in range(layout.num_chips):
            mine = picks[(owner == chip) & valid]
            rows.append(mine.numel())
            local = cache.latent[mine]  # this chip's own rows, [n, kv_lora + rope]
            logits = torch.matmul(q_lat, local.t()) * c.scaling  # [H, n]
            lse = torch.logsumexp(logits, dim=-1)  # -inf when n == 0
            probs = F.softmax(logits, dim=-1) if mine.numel() else logits
            partial_ctx.append(torch.matmul(probs, local[:, : c.kv_lora_rank]))
            partial_lse.append(lse)
        lse = torch.stack(partial_lse + [a.learnable_sink_param.float()])  # [chips + 1, H]
        weight = F.softmax(lse, dim=0)[:-1]  # the sink takes its share and gives nothing
        ctx = (weight[:, :, None] * torch.stack(partial_ctx)).sum(0)
        _, w_v = self.absorb_weights()
        attn = torch.einsum("hvk,hk->hv", w_v, ctx)[:, None, :]
        gate = F.linear(x, a.linear_gate)
        out = F.linear(attn.transpose(0, 1).reshape(1, -1) * torch.sigmoid(gate), a.o_proj)
        return out, torch.tensor(rows)

    # ------------------------------------------------------------------------------------------
    # the step
    # ------------------------------------------------------------------------------------------
    def step(
        self,
        x: torch.Tensor,
        cache: DecodeCache,
        picks: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        position: int | None = None,
    ) -> DecodeOutput:
        """One new token. x: [1, hidden], the layer's pre-normed input; ``cache`` holds the earlier
        positions and gains this one (in place).

        ``picks`` (int [K] positions) is the selection to attend with: required on a shared layer
        (the full layer's picks of this step), optional on a full layer, where it replaces the
        layer's own selection (a device test hands in the device's picks to separate the attention
        arithmetic from near-ties in the top-k). ``valid`` (bool [K]) flags the real entries.
        ``position`` defaults to the cache length: the token's position in the context, which RoPE
        uses too.
        """
        position = len(cache) if position is None else position
        assert position == len(cache), f"the cache holds {len(cache)} positions, the token is at {position}"
        token = self.new_token(x, position)
        cache.append(token.latent, token.index_key)
        if picks is None:
            assert self.is_full, "a shared layer attends over the picks of the full layer below it"
            picks = self.select(token, cache)
        picks = picks.long()
        valid = torch.ones_like(picks, dtype=torch.bool) if valid is None else valid.bool()
        assert int(picks[valid].max()) <= position, "a pick is ahead of the query"
        out = self.stages(x, token, self.gather(cache, picks, valid), valid)["out"]
        return DecodeOutput(out, picks, valid, token)
