# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch ground-truth reference for Hy4-preview's prefill attention: the attention sublayer on a
whole prompt with the DSA selection in it (tenstorrent/tt-metal#65887).

Pure torch, runs on CPU. Mirrors HF transformers ``models/hy_v4/modeling_hy_v4.py``
HYV4Attention.forward with the eager core, on a prompt that starts at position 0:

    q_resid, latent      = the MLA projections of reference/attention.py (latent = k_pass | k_rope)
    picks                = a "full" layer: its indexer's top-2048 per query (reference/indexer.py)
                           a "shared" layer: the picks of the nearest full layer below, handed in
    mask[s, t]           = 0 where t <= s AND t is among query s's picks, the dtype's minimum elsewhere
    out                  = ((softmax over the allowed keys and the sink) @ v * sigmoid(gate)) W_o

So a query attends over at most 2048 positions, whatever the prompt length. Up to 2048 tokens every
query picks every position it may see and the layer is plain causal attention; above that the
selection decides the output. What the layer leaves in the cache for decode is, per token, the
576-wide latent and, on a full layer, the 128-wide index key.

HF builds the mask the same way: the causal mask, then ``masked_fill`` of every key that is not
among the picks. The picks of a query that sees fewer than 2048 positions also name positions after
it (reference/indexer.py); the causal part of the mask removes those again.

The last part of this file, threshold_select, is not HF's: it is the selection in the form the
device prefill computes it (tt/attention_prefill.py), written in torch chip by chip. Prefill needs
the selected set as a mask over the positions, not as a list, so it never gathers the picks: it
finds the score of the cut and compares every score with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental.hy4_preview.reference.attention import Hy4Attention, causal_mask
from models.experimental.hy4_preview.reference.indexer import NEG, Hy4Indexer, selection_mask, top_through_bf16


def dsa_mask(picks: torch.Tensor, num_keys: int, dtype=torch.float32) -> torch.Tensor:
    """The additive mask HYV4Attention attends with. picks: [S, k] positions per query (S == num_keys:
    the prompt starts at position 0). Returns [S, num_keys]: 0 where the key is picked and not after
    the query, the dtype's most negative value elsewhere."""
    seq_len = picks.shape[0]
    assert seq_len == num_keys, "the reference prefill takes a whole prompt: one query per key"
    not_picked = torch.ones(seq_len, num_keys, dtype=torch.bool).scatter(-1, picks.long(), False)
    return causal_mask(seq_len, dtype).masked_fill(not_picked, torch.finfo(dtype).min)


@dataclass
class PrefillOutput:
    """What one prefill attention sublayer produces for a prompt of S tokens.

    ``out``         [S, hidden]            the sublayer's output
    ``picks``       int32 [S, min(topk, S)]  the picks this layer attended with; a full layer's go to the
                                           shared layers above it (HF's ``prev_topk_indices``)
    ``selected``    bool [S, S]            the set each query attended over: picked and not after it
    ``latent``      [S, kv_lora + rope]    (k_pass | k_rope), the cache entry of every token
    ``index_keys``  [S, index_head_dim]    the index-key cache entry of every token; None on a shared layer
    """

    out: torch.Tensor
    picks: torch.Tensor
    selected: torch.Tensor
    latent: torch.Tensor
    index_keys: torch.Tensor | None


class Hy4AttentionPrefill(nn.Module):
    """One attention sublayer in prefill. ``indexer`` is the layer's own on a full layer and None on
    a shared one. x: [S, hidden], the layer's pre-normed input for positions 0 .. S-1."""

    def __init__(self, attention: Hy4Attention, indexer: Hy4Indexer | None = None):
        super().__init__()
        self.attention = attention
        self.indexer = indexer

    @property
    def is_full(self) -> bool:
        return self.indexer is not None

    def select(self, x: torch.Tensor, q_resid: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """A full layer's selection: (picks int32 [S, min(topk, S)], index keys [S, index_head_dim])."""
        assert self.indexer is not None, "a shared layer has no indexer: it takes the picks of the full layer below"
        return self.indexer(x, q_resid, cos, sin, causal_mask(x.shape[0]))

    def attend(self, x, q_resid, k_pass, k_rope, cos, sin, mask):
        """Everything after the latent: queries, keys and values, the core with its sink, gate, o_proj."""
        attn = self.attention
        q = attn.queries(q_resid, cos, sin)
        keys, values = attn.keys_values(k_pass, k_rope)
        heads = attn.core(q, keys, values, mask)  # [H, S, v]
        heads = heads.transpose(0, 1).reshape(x.shape[0], -1)  # [S, H*v], head-major
        return F.linear(heads * attn.gate(x), attn.o_proj)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, prev_picks: torch.Tensor | None = None):
        """x: [S, hidden]; cos / sin: [S, rope] at the tokens' positions; prev_picks: the picks of the
        full layer below, for a shared layer. Returns a PrefillOutput."""
        seq_len = x.shape[0]
        q_resid, k_pass, k_rope = self.attention.latent(x, cos, sin)
        if self.indexer is not None:
            picks, index_keys = self.select(x, q_resid, cos, sin)
        else:
            if prev_picks is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            picks, index_keys = prev_picks, None
        mask = dsa_mask(picks, seq_len, x.dtype)
        out = self.attend(x, q_resid, k_pass, k_rope, cos, sin, mask)
        allowed = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
        return PrefillOutput(
            out=out,
            picks=picks,
            selected=selection_mask(picks, allowed),
            latent=torch.cat([k_pass, k_rope], dim=-1),
            index_keys=index_keys,
        )


def additive_mask(selected: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """A selected set (bool [S, T]) as the additive mask the attention core takes."""
    return torch.zeros(selected.shape, dtype=dtype).masked_fill(~selected, torch.finfo(dtype).min)


# ---------------------------------------------------------------------------------------------
# The prefill selection as SC1 computes it: stripes, the cut, a compare
# ---------------------------------------------------------------------------------------------
def threshold_select(
    scores: torch.Tensor, positions: torch.Tensor, layout, topk: int, slots: int | None = None, refine_passes: int = 2
) -> torch.Tensor:
    """The selected set of a block of queries as tt/attention_prefill.py computes it, from scores
    that are already known.

    scores: [S, T] fp32, unmasked, in position order; positions: [S] the queries' positions;
    ``layout``: tt/kv_layout.py BlockCyclicLayout; ``slots``: the stripes' size, at least
    layout.slots(T).

    Per chip: mask every slot whose position is after the query (that covers the unfilled ones) with
    NEG. The chip's candidates are its top-``topk`` scores, or all of them when the stripe is no
    longer than that. The candidates of every chip are gathered and the cut is the ``topk``-th
    largest of them, found with the top-k that only ranks bf16 values
    (reference/indexer.py top_through_bf16): it returns a centre and a cut such that a score is in the
    top-k when bf16(score - centre) >= cut. Every chip then compares its own scores with that, and
    a slot is selected when it passes and the query may see it. With no more than ``topk`` slots in
    all there is no cut and a query selects everything it sees.

    Returns bool [S, T]. It is the top-``topk`` set, except that scores the refinement passes cannot
    tell from the ``topk``-th one are all kept: the set can be larger than ``topk`` by such ties, where
    a list of picks (striped_select) keeps an arbitrary ``topk`` of them.
    """
    S, T = scores.shape
    slots = layout.slots(T) if slots is None else slots
    stripes = layout.split(scores.float(), dim=1, slots=slots)  # [chips, S, slots]
    slot_position = layout.position(torch.arange(layout.num_chips)[:, None], torch.arange(slots)[None, :])
    visible = slot_position[:, None, :] <= positions[None, :, None]
    masked = torch.where(visible, stripes, torch.tensor(NEG))

    centre = cut = None
    cand = masked
    if slots > topk:
        centre, cut, cand_slots = top_through_bf16(masked, topk, refine_passes)  # each chip's own cut
        cand = masked.gather(-1, cand_slots)
    everything = cand.permute(1, 0, 2).reshape(S, -1)
    if everything.shape[-1] > topk:
        centre, cut, _ = top_through_bf16(everything, topk, refine_passes)  # [S, 1]: the same on every chip

    if cut is None:
        selected = visible
    else:
        ranked = masked if centre is None else masked - centre
        selected = (ranked.bfloat16().float() >= cut) & visible
    return layout.merge(selected, T, dim=1)
