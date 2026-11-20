# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch ground-truth reference for Hy4-preview's DSA indexer (top-k token selection).

Pure torch, runs on CPU. Mirrors HF transformers ``models/hy_v4/modeling_hy_v4.py`` HYV4Indexer and
the way HYV4Attention / HYV4Model hand its picks from a "full" layer to the "shared" layers above
it (tenstorrent/tt-metal#65887, #65888).

    key     = LayerNorm(x W_k)                       [T, 128]; RoPE on the trailing 64 dims, by the key's position
    q       = q_resid W_qb  -> 32 heads x 128        RoPE on the trailing 64 dims of every head, by the query's position
    w       = x W_w * 32^-0.5 * 128^-0.5             [S, 32], one signed weight per indexer head
    score   = sum_h  w[s, h] * relu(q[s, h] . key[t])        [S, T], fp32
    picks   = top-2048 of score over the positions the query may see (t <= s)

What the cache stores per token is ``key``: 128 values, on the 21 full layers only. ``q_resid`` is
the attention sublayer's RMSNorm(x W_qa) (reference/attention.py ``latent``) and ``x`` is the same
pre-normed hidden state the attention projections take.

Notes on HF's exact behaviour, all mirrored here:
* ``k_norm`` is a LayerNorm with a bias, eps = the model's ``rms_norm_eps`` (1e-5), not an RMSNorm.
* RoPE sits on the trailing 64 dims (the attention keys have it there too), same cos / sin as attention.
* A head's weight can be negative, so a score can be; the weight cannot be folded into the relu.
* The top-k is ``min(2048, T)`` wide whatever the query's position. A query that may see fewer
  positions than that also gets masked positions back (their score is -inf, their order arbitrary);
  the causal mask removes them again in attention. The selected SET of a query is therefore
  ``picks & allowed`` (selection_mask), which is what a device implementation has to reproduce.
* Layer L is "full" (has its own indexer) for L == 0 or L % 4 == 1, 21 of the 78 layers; every
  other layer is "shared" and attends over the picks of the nearest full layer below it.

Checkpoint names under ``model.layers.{L}.self_attn.indexer.``:
    wq_b.weight  wk.weight  k_norm.weight  k_norm.bias  weights_proj.weight

The last part of this file, striped_select, is not HF's: it is the SC1 device algorithm
(tt/indexer.py) written in torch, chip by chip, so that the way the selection is distributed over
the context-parallel stripes and pushed through a bf16 top-k can be pinned to the plain top-k above
without a device, at any context length.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental.hy4_preview.reference.attention import apply_rope

# The score the distributed selection gives a slot the query may not see: far below any real score,
# still far below after a real score is subtracted from it, and a power of two so that bf16 and fp32
# both hold it exactly.
NEG = -(2.0**100)


@dataclass
class IndexerConfig:
    """Indexer hyperparameters. Defaults are the tencent/Hy4-preview config.json values."""

    hidden_size: int = 6144
    q_lora_rank: int = 2048
    num_heads: int = 32  # index_n_heads
    head_dim: int = 128  # index_head_dim, the width of a cached index key
    qk_rope_head_dim: int = 64  # the trailing dims of a key / query head that carry RoPE
    topk: int = 2048  # index_topk
    norm_eps: float = 1e-5  # k_norm (LayerNorm) takes the model's rms_norm_eps
    rope_theta: float = 1e7

    @property
    def pass_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    @property
    def weight_scale(self) -> float:
        # HF applies n_heads^-0.5 and the "softmax scale" head_dim^-0.5 to the head weights
        return self.num_heads**-0.5 * self.head_dim**-0.5


def full_layers(num_layers: int) -> list[str]:
    """HYV4Config's default ``indexer_types``: full on layer 0 and every layer with L % 4 == 1."""
    return ["full" if layer == 0 or (layer - 1) % 4 == 0 else "shared" for layer in range(num_layers)]


def picks_source(indexer_types) -> list[int]:
    """For every layer, the layer whose indexer picks it attends over: itself when full, else the
    nearest full layer below. HF threads this through the layer loop as ``prev_topk_indices``."""
    source, last = [], None
    for layer, kind in enumerate(indexer_types):
        assert kind in ("full", "shared"), (layer, kind)
        if kind == "full":
            last = layer
        if last is None:
            raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
        source.append(last)
    return source


def selection_mask(indices: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """The selected set as a mask. indices: [S, k] picks; allowed: bool [S, T], the positions each
    query may see. Returns bool [S, T], True where the query attends: picked AND allowed."""
    picked = torch.zeros_like(allowed).scatter(-1, indices.long(), True)
    return picked & allowed


def causal_allowed(query_positions: torch.Tensor, num_keys: int) -> torch.Tensor:
    """bool [S, T]: key position t is visible to the query at absolute position p when t <= p."""
    return torch.arange(num_keys)[None, :] <= query_positions[:, None]


def compare_selection(scores: torch.Tensor, selected: torch.Tensor, other: torch.Tensor, k: int, tol: float) -> dict:
    """Mismatches between the reference's selected set and another implementation's, split into
    near-ties and real errors.

    scores: [S, T] reference scores, -inf where a position is not visible. selected / other: bool
    [S, T]. A position on which the two disagree is a near-tie when its reference score lies within
    ``tol`` (absolute) of the reference's k-th largest score for that query: an implementation whose
    scores are off by up to tol / 2 may legitimately put it on either side of the cut. Anything
    else is a real error, as is any disagreement on a query that sees no more than k positions,
    where there is no cut at all.

    Returns counts over all queries: ``missing`` (reference has it, other does not), ``extra``
    (the reverse), ``near_tie`` and ``real`` (the two add up to missing + extra), ``queries`` with
    any mismatch, and ``size_diff`` (the largest |other's set size - reference's|).
    """
    visible = torch.isfinite(scores)
    has_cut = visible.sum(-1, keepdim=True) > k
    kth = scores.topk(min(k, scores.shape[-1]), dim=-1).values[:, -1:]
    differ = selected ^ other
    near = differ & has_cut & visible & ((scores - kth).abs() <= tol)
    return {
        "missing": int((selected & ~other).sum()),
        "extra": int((other & ~selected).sum()),
        "near_tie": int(near.sum()),
        "real": int((differ & ~near).sum()),
        "queries": int(differ.any(-1).sum()),
        "size_diff": int((other.sum(-1) - selected.sum(-1)).abs().max()),
    }


class Hy4Indexer(nn.Module):
    """One full layer's indexer. Everything is per prompt: x [S, hidden], no batch dim."""

    # attribute -> name under model.layers.{L}.self_attn.indexer.
    CHECKPOINT_NAMES = {
        "wq_b": "wq_b.weight",
        "wk": "wk.weight",
        "k_norm_weight": "k_norm.weight",
        "k_norm_bias": "k_norm.bias",
        "weights_proj": "weights_proj.weight",
    }

    def __init__(self, cfg: IndexerConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg
        self.wq_b = nn.Parameter(torch.empty(c.num_heads * c.head_dim, c.q_lora_rank))
        self.wk = nn.Parameter(torch.empty(c.head_dim, c.hidden_size))
        self.k_norm_weight = nn.Parameter(torch.ones(c.head_dim))
        self.k_norm_bias = nn.Parameter(torch.zeros(c.head_dim))
        self.weights_proj = nn.Parameter(torch.empty(c.num_heads, c.hidden_size))
        for name in ("wq_b", "wk", "weights_proj"):
            nn.init.normal_(getattr(self, name), std=0.006)

    def weights(self) -> dict:
        """The five parameters by checkpoint name (HF state_dict keys), as the device module takes them."""
        return {name: getattr(self, attr).detach() for attr, name in self.CHECKPOINT_NAMES.items()}

    def load_weights(self, weights: dict):
        with torch.no_grad():
            for attr, name in self.CHECKPOINT_NAMES.items():
                p = getattr(self, attr)
                p.copy_(weights[name].reshape(p.shape))
        return self

    # ------------------------------------------------------------------------------------------
    def keys(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x: [T, hidden]; cos / sin: [T, rope] at the tokens' positions -> index keys [T, head_dim].

        This is the per-token cache entry of a full layer.
        """
        c = self.cfg
        k = F.layer_norm(F.linear(x, self.wk), (c.head_dim,), self.k_norm_weight, self.k_norm_bias, c.norm_eps)
        k_pass, k_rope = k.split([c.pass_dim, c.qk_rope_head_dim], dim=-1)
        return torch.cat([k_pass, apply_rope(k_rope, cos, sin)], dim=-1)

    def queries(self, q_resid: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """q_resid: [S, q_lora]; cos / sin: [S, rope] at the queries' positions -> [S, heads, head_dim]."""
        c = self.cfg
        q = F.linear(q_resid, self.wq_b).view(-1, c.num_heads, c.head_dim)
        q_pass, q_rope = q.split([c.pass_dim, c.qk_rope_head_dim], dim=-1)
        return torch.cat([q_pass, apply_rope(q_rope, cos[:, None, :], sin[:, None, :])], dim=-1)

    def head_weights(self, x: torch.Tensor) -> torch.Tensor:
        """x: [S, hidden] -> [S, heads], the signed per-head weights with both scales applied."""
        return F.linear(x, self.weights_proj).float() * self.cfg.weight_scale

    @staticmethod
    def scores(q: torch.Tensor, w: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """q: [S, heads, head_dim]; w: [S, heads]; keys: [T, head_dim] -> [S, T], fp32, unmasked."""
        per_head = F.relu(torch.matmul(q.float(), keys.float().transpose(-1, -2)))  # [S, heads, T]
        return torch.matmul(w.unsqueeze(-2), per_head).squeeze(-2)

    def select(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """scores: [S, T]; mask: bool [S, T] (True = visible) or additive -> int32 picks [S, min(topk, T)],
        best first. Masked positions can appear when a query sees fewer than that many (see module doc)."""
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask
        return scores.topk(min(self.cfg.topk, scores.shape[-1]), dim=-1).indices.to(torch.int32)

    def forward(
        self,
        x: torch.Tensor,
        q_resid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
        cached_keys: torch.Tensor | None = None,
    ):
        """S new tokens against the P cached keys plus their own.

        x: [S, hidden]; q_resid: [S, q_lora]; cos / sin: [S, rope] at the new tokens' positions;
        mask: [S, P + S], bool or additive; cached_keys: [P, head_dim] or None.
        Returns (picks int32 [S, min(topk, P + S)], keys [P + S, head_dim]); keys is the cache after this call.
        """
        keys = self.keys(x, cos, sin)
        if cached_keys is not None:
            keys = torch.cat([cached_keys, keys], dim=0)
        scores = self.scores(self.queries(q_resid, cos, sin), self.head_weights(x), keys)
        return self.select(scores, mask), keys


# ---------------------------------------------------------------------------------------------
# The SC1 selection in torch: stripes, local candidates, one gather, the cut
# ---------------------------------------------------------------------------------------------
def _bf16(t: torch.Tensor) -> torch.Tensor:
    """What a bf16 device tensor holds of an fp32 one (round to nearest even), back in fp32."""
    return t.bfloat16().float()


def top_through_bf16(scores: torch.Tensor, k: int, passes: int):
    """Top-k of fp32 ``scores`` [..., n] with a top-k that only ranks bf16 values (the device's at k = 2048).

    bf16 keeps 8 significant bits, so ranking the scores as they are ties everything within about
    0.4% of the k-th. bf16 resolves a value relative to its own magnitude, though: pass 1 takes the
    k-th largest of the rounded scores as the centre, and every further pass ranks
    bf16(scores - centre) and moves the centre onto the new k-th, which adds 8 bits per pass next to
    the cut. Rounding is monotonic, so no pass can move an element across the cut it resolves.

    Returns (centre, cut, slots): ``slots`` [..., k] are the last pass's picks, best first, and an
    element is in the top-k exactly when bf16(score - centre) >= cut (centre is None after a single
    pass: the score itself is ranked). centre and cut are [..., 1].
    """
    centre = None
    for step in range(passes):
        ranked = scores if centre is None else scores - centre
        top, slots = _bf16(ranked).topk(k, dim=-1)
        cut = top[..., -1:]
        if step < passes - 1:
            centre = cut if centre is None else centre + cut
    return centre, cut, slots


def striped_select(scores: torch.Tensor, positions: torch.Tensor, layout, topk: int, slots=None, refine_passes=2):
    """The selection as SC1 computes it, from scores that are already known.

    scores: [S, T] fp32, unmasked, in position order; positions: [S] the queries' positions;
    ``layout``: tt/kv_layout.py BlockCyclicLayout (position <-> (chip, slot)); ``slots``: the
    stripes' size, at least layout.slots(T).

    Per chip: mask every slot whose position is after the query (that covers the unfilled ones)
    with NEG; keep the chip's top-``topk`` slots as candidates, or all of them when the stripe is
    no longer than that, as (score, position) pairs. Then gather the candidates of every chip and
    take the top-``topk`` of them.

    Returns (picked, valid): picked is int64 [S, K], the selected positions, best first, with
    K = min(topk, chips * min(slots, topk)); valid is bool [S, K], False where the entry is a masked
    slot because the query sees fewer than K positions.
    """
    S, T = scores.shape
    slots = layout.slots(T) if slots is None else slots
    stripes = layout.split(scores.float(), dim=1, slots=slots)  # [chips, S, slots]
    slot_position = layout.position(torch.arange(layout.num_chips)[:, None], torch.arange(slots)[None, :])
    slot_position = slot_position[:, None, :].expand(-1, S, -1)
    masked = torch.where(slot_position <= positions[None, :, None], stripes, torch.tensor(NEG))

    if slots <= topk:
        cand, cand_position = masked, slot_position
    else:
        _, _, cand_slots = top_through_bf16(masked, topk, refine_passes)
        cand, cand_position = masked.gather(-1, cand_slots), slot_position.gather(-1, cand_slots)

    # every chip's candidates, per query
    everything = cand.permute(1, 0, 2).reshape(S, -1)
    picked = cand_position.permute(1, 0, 2).reshape(S, -1)
    if everything.shape[-1] > topk:
        _, _, index = top_through_bf16(everything, topk, refine_passes)
        everything, picked = everything.gather(-1, index), picked.gather(-1, index)
    return picked, everything > 0.5 * NEG


def picked_mask(picked: torch.Tensor, valid: torch.Tensor, num_positions: int) -> torch.Tensor:
    """(picked, valid) as a bool [S, num_positions] mask, the form selection_mask returns. Raises if
    a valid pick is not one of the first ``num_positions`` positions or a position is picked twice."""
    S = picked.shape[0]
    mask = torch.zeros(S, num_positions, dtype=torch.bool)
    query = torch.arange(S)[:, None].expand_as(picked)
    if valid.any():
        assert int(picked[valid].max()) < num_positions, "an unfilled slot was selected"
    mask[query[valid], picked[valid]] = True
    assert int(mask.sum()) == int(valid.sum()), "a position was selected twice"
    return mask
