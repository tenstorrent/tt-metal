# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview prefill attention on the SC1 layout (tenstorrent/tt-metal#65887): the attention
sublayer on a whole prompt, attending over the positions the DSA indexer selects, and leaving in the
cache what decode reads.

Mirrors the pure-torch ground truth in ../reference/attention_prefill.py. It composes three modules
and owns no weight: TtHy4Attention (projections, gate, sinks, o_proj, tt/attention.py), TtHy4Indexer
(index keys, queries and scores, tt/indexer.py) and TtHy4KvCache (tt/kv_cache.py).

Frame (tt/frames.py). TtHy4Attention and TtHy4Indexer work in the frame the prefill stack stores its
caches in, and so does everything here: ``cos`` / ``sin`` are the interleaved tables
(frames.rope_tables), the latent's k_rope is in interleaved pairs, and an index key is
[rope interleaved | pass] in the Hadamard basis. This module only moves those tensors (to the
stripes, to the cache), so it holds no frame arithmetic of its own; scores, the selection and the
sublayer's output are the reference's.

One layer, a prompt of T tokens at positions 0 .. T-1, hidden states [1, 1, T, hidden/4] split over
the hidden axis:

    latent     q_resid, k_pass, k_rope            TtHy4Attention.latent: whole on every chip
    full layer index keys, index queries, head weights      TtHy4Indexer.forward: whole on every chip
               selection                          below: the set every query attends over, as a mask
    shared     selection                          the one the full layer below stored (TtHy4IndexCache)
    cache      latent, and the index key on a full layer    TtHy4KvCache.write_prefill, striped block-cyclic
    attention  q, keys, values of this chip's 8 heads       TtHy4Attention.head_projections
               the dense core with its sink, in query chunks, under the selection mask
    output     gate, o_proj                       TtHy4Attention.output: split over the hidden axis again

The selection as a mask. Prefill attends densely and head-parallel: every chip holds the keys and
values of its own 8 heads for all T positions and masks what a query does not attend to. It needs
the selected set as a [queries, positions] mask, not as a list of picks, so nothing is gathered or
scattered. For a chunk of S queries:

    stripes    the index keys dealt block-cyclic to the chips (tt/kv_layout.py), bf16     per chip [slots, 128]
    masked     TtHy4Indexer.masked_scores: every chip scores its own stripe; NEG where the
               slot's position is after the query (unfilled slots included)               per chip [S, slots] fp32
    candidates the chip's scores, or its top-2048 when the stripe is longer than that
    all        gather(candidates) over both mesh axes                                     [S, 32 x K], on every chip
    cut        the 2048th largest of ``all`` through the bf16 top-k and its refinement passes
               (TtHy4Indexer's): a centre and a cut such that a score is in the top-2048
               exactly when bf16(score - centre) >= cut
    selected   bf16(masked - centre) >= cut AND the query may see the slot                0 / 1
    mask       (chip, slot) order -> position order, a reshape and a permute; additive: 0 where
               selected, MASKED elsewhere                                                 [S, T] bf16, on every chip

  With no more than 2048 slots in all the stripes together there is no cut and a query selects
  everything it may see. On the 32-chip mesh the first round of stripes already holds 4096 slots, so the
  cut is computed for every prompt; for a short one it comes out below every visible score.
* The set is HF's top-2048 except for scores the refinement passes cannot tell from the 2048th: all
  of those are selected, so a query can attend over a few positions more than 2048 where HF keeps an
  arbitrary 2048 of them. reference/attention_prefill.py threshold_select is this algorithm in torch.
* The scores go through the indexer's matmuls, so two scores closer than the matmul error can land on
  either side of the cut. A position that swaps there has a score next to the cut, by construction.
* The mask covers causality: a position after the query is never visible, so it is never selected.
* The selection is stored whole, [T, T + 32] bf16 per chip, for the shared layers above
  (TtHy4IndexCache holds it the way it holds a decode step's picks).

The core and the sink. HF appends one logit per head, the sink, to a query's logits and drops its
share after the softmax. Here the sink is a key: the keys and values get one more tile of 32 zero
rows, a zero key scores 0 with every query and a zero value adds nothing to the output. The first of
those rows is given the head's sink as its logit and the other 31 are masked, by one constant
[1, heads, 1, T + 32] tensor added to the logits. So nothing is concatenated to or sliced off a
[queries, T] tensor and, for a prompt that is a whole number of tiles, every tensor of the core stays
on whole tiles. (TtHy4Attention.core concatenates a sink column and slices it off again; it is not
used here.)

Query chunks. The selection and the core both run on ``query_chunk`` queries at a time, so the
intermediates are [chunk, T] (scores, mask, attention logits per head), not [T, T]. What does grow
with T squared is the stored mask, 2 T^2 bytes per chip, and the work: every query is scored against
every position and attends densely over all T keys with the unselected ones masked. This is the
functional form. It is exact, and it is quadratic: a prompt of 128K tokens would need a 32 GB mask on
every chip and 256x the work of 8K. The sparse form (attend over the 2048 selected rows only,
tenstorrent/tt-metal#65889 for decode) replaces the core without changing the selection.

A prompt continues nothing here: the keys and values come from this call's own latent, not from the
cache, so the call covers positions 0 .. T-1. Chunked prefill of a long prompt against the cache needs
the cache read path of sparse attention.

T is a multiple of 32, which keeps every tensor here on whole tiles (the query chunks, the mask, the
keys with the sink's tile); the cache's device write path asks the same of where a write starts. A
prompt of another length is padded by the caller: the pad tokens come after every real token, so no
real query sees them, their output rows are ignored, and decode overwrites their cache rows as the
context grows into them.

About ttnn.transformer.sparse_sdpa and ttnn.experimental.indexer_score_dsa, which the issue names.
Both are plain ttnn ops on this branch. sparse_sdpa attends in the "absorbed" form: q [1, H, S, 576]
against one row-major [1, 1, T, 576] latent table that must be whole on the chip (or block-cyclic
inside that one tensor), a per-query uint32 index list with sentinels, an optional sink, output
[1, H, S, 512] in latent space. It asserts H is a multiple of 32 and at least 32. On SC1 a chip has 8
heads and 1/4 of kv_b's 512 input columns for them, and 1/32 of the latent rows. Using the op means,
per chip: the absorbed query (q_nope @ W_k_nope, 8 x 192 x 512, which SC1 shards over the hidden
axis and would have to gather), 24 dummy heads to reach 32, a row-major copy of the whole latent on
every chip instead of a 1/32 stripe, and the value projection (512 -> 256 per head) after the op,
again sharded over the hidden axis. indexer_score_dsa computes exactly the indexer's score
(sum over heads of relu(q . k) * w) with any head count, but its causal cut-off assumes key row t
is position t (or its own block-cyclic chunking over a mesh axis), not kv_layout's stripes of 128
over all 32 chips, and it returns the scores in the input dtype; TtHy4Indexer keeps them in fp32
for the cut. Neither is used here.

Tensors: hidden states and latents are [1, 1, T, width] with the tokens down the tile rows; per-head
tensors are [1, heads, T, width]; a mask is [1, 1, queries, positions + 32].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.hy4_preview.reference.indexer import NEG
from models.experimental.hy4_preview.tt.attention import TtHy4Attention
from models.experimental.hy4_preview.tt.indexer import TtHy4IndexCache, TtHy4Indexer
from models.experimental.hy4_preview.tt.kv_layout import BlockCyclicLayout

# The additive mask value of a position a query does not attend to. A power of two, exact in bf16, and
# far enough below any logit that the softmax gives it nothing.
MASKED = -(2.0**30)
# Key rows appended for the sink: one whole tile. The first is the sink, the rest are masked.
SINK_ROWS = ttnn.TILE_SIZE


def _as(x, dtype):
    return x if x.dtype == dtype else ttnn.typecast(x, dtype)


@dataclass
class PrefillSelection:
    """The positions the queries of a prompt attend over, as the attention core takes it.

    ``mask`` [1, 1, num_queries, num_keys + SINK_ROWS] bf16, the same on every chip. Row s is the
    query at position s. Column t < num_keys: 0 where the query attends to position t, MASKED
    elsewhere; a position after the query is always MASKED. The last SINK_ROWS columns are 0: they
    belong to the sink's rows, which the core fills in.
    """

    mask: ttnn.Tensor
    num_queries: int
    num_keys: int

    def rows(self, start: int, end: int):
        """The mask of queries [start, end): [1, 1, end - start, num_keys + SINK_ROWS]."""
        if start == 0 and end == self.num_queries:
            return self.mask
        return ttnn.slice(self.mask, [0, 0, start, 0], [1, 1, end, self.num_keys + SINK_ROWS])

    def deallocate(self):
        self.mask.deallocate(True)


class TtHy4AttentionPrefill(LightweightModule):
    """One layer's attention sublayer in prefill.

    ``attention``   the layer's TtHy4Attention.
    ``layer``       the layer index; ``index_cache`` (a TtHy4IndexCache shared by the layers of the
                    stack) says whether it is a full or a shared layer and carries the selection up.
    ``indexer``     the layer's TtHy4Indexer on a full layer, None on a shared one.
    ``kv_cache``    a TtHy4KvCache holding this layer, or None to write nothing. ``user`` is the
                    cache user the prompt belongs to.
    ``query_chunk`` queries per pass of the selection and of the core, a multiple of 32.
    """

    def __init__(
        self,
        attention: TtHy4Attention,
        layer: int,
        index_cache: TtHy4IndexCache,
        indexer: TtHy4Indexer | None = None,
        kv_cache=None,
        user: int = 0,
        query_chunk: int = 256,
    ):
        assert index_cache.is_full(layer) == (indexer is not None), (
            f"layer {layer} is a {'full' if index_cache.is_full(layer) else 'shared'} layer: "
            f"{'it needs its indexer' if indexer is None else 'it has no indexer of its own'}"
        )
        assert query_chunk > 0 and query_chunk % ttnn.TILE_SIZE == 0, f"query_chunk = {query_chunk}: whole tile rows"
        self.attention = attention
        self.plan = attention.plan
        self.cfg = attention.cfg
        self.layer = layer
        self.index_cache = index_cache
        self.indexer = indexer
        self.kv_cache = kv_cache
        self.user = user
        self.query_chunk = query_chunk

        num_chips = self.plan.hidden_tp * self.plan.head_tp
        if indexer is not None:
            self.layout = indexer.layout
        elif kv_cache is not None:
            self.layout = kv_cache.layout
        else:
            self.layout = BlockCyclicLayout(num_chips=num_chips)
        assert self.layout.num_chips == num_chips, (self.layout.num_chips, num_chips)
        if kv_cache is not None:
            assert kv_cache.layout == self.layout, (kv_cache.layout, self.layout)
            assert layer in kv_cache.latent, f"the cache holds no latent for layer {layer}"
            assert indexer is None or layer in kv_cache.index_key, f"the cache holds no index key for layer {layer}"
            assert indexer is None or kv_cache.cfg.index_key_dim == indexer.cfg.head_dim
            assert kv_cache.cfg.latent_dim == self.cfg.latent_dim

        # The heads' sink logits as the device holds them, [1, num_heads, 1, 1] on the host: the
        # constant of ``_sink_columns`` is built from them once per prompt length.
        self._sinks = self.plan.to_torch_head_concat(attention.sinks, dim=1).float()
        self._sink_columns_cache = {}

    @property
    def is_full(self) -> bool:
        return self.indexer is not None

    # ------------------------------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------------------------------
    def stripes(self, index_keys):
        """Index keys of positions 0 .. T-1, [1,1,T,head_dim] whole on every chip -> the stripes the
        indexer scores: per chip [1,1,slots,head_dim], this chip's rows in slot order, zero where a
        slot has no position.

        The same cut TtHy4KvCache.write_prefill makes: pad to whole rounds, view a round as
        [chips x block] rows and let mesh_partition keep each chip its own block of every round.
        The selection scores these, not the cache's copy: the keys as the indexer computed them,
        whatever dtype the cache stores.
        """
        layout = self.layout
        T, width = index_keys.shape[-2], index_keys.shape[-1]
        slots = layout.slots(T)
        rounds = slots // layout.block
        round_len = layout.num_chips * layout.block
        tail = rounds * round_len - T
        x = _as(index_keys, ttnn.bfloat16)
        if tail:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, tail), (0, 0)], 0.0)
        if layout.num_chips == 1:
            return x
        x = ttnn.reshape(x, [1, rounds, round_len, width])
        x = ttnn.mesh_partition(x, 2)
        return ttnn.reshape(x, [1, 1, slots, width])

    def _gather_all(self, x):
        """Concatenate every chip's ``x`` on the last dim, on every chip: chip-major, chips row-major
        (along mesh axis 1 first, then along axis 0), the order kv_layout counts chips in, whichever
        mesh axis carries the hidden dimension."""
        plan = self.plan
        if plan.hidden_axis is None:
            return x
        return plan.all_gather(plan.all_gather(x, dim=3, axis=1), dim=3, axis=0)

    def _top(self, scores, k):
        """(centre, cut, slots) of the top-``k`` of fp32 ``scores`` [1,1,S,n]: the indexer's bf16 top-k
        with its refinement passes. A score is in the top-k exactly when bf16(score - centre) >= cut."""
        return self.indexer._top(scores, k, self.indexer.refine_passes)

    @staticmethod
    def _passes(scores, centre, cut):
        """0 / 1 fp32, shaped like ``scores``: the score is at or above the cut and is not a masked slot's."""
        visible = _as(ttnn.gt(scores, 0.5 * NEG), ttnn.float32)
        if cut is None:
            return visible
        ranked = scores if centre is None else ttnn.subtract(scores, centre)
        ranked = ttnn.typecast(ttnn.typecast(ranked, ttnn.bfloat16), ttnn.float32)
        return ttnn.multiply(_as(ttnn.ge(ranked, cut), ttnn.float32), visible)

    def selected(self, q, w, positions, stripes, num_keys: int):
        """The selected set of S queries, 0 / 1 fp32 [1,1,S,num_keys] in position order, the same on
        every chip.

        q: [1,S,heads,head_dim] and w: [1,1,S,heads] from TtHy4Indexer.forward; positions: the
        queries' sequence positions (ints); stripes: from ``stripes``; num_keys: the prompt length.
        """
        indexer, layout = self.indexer, self.layout
        topk, chips, block = indexer.cfg.topk, layout.num_chips, layout.block
        S, slots = q.shape[1], stripes.shape[-2]
        assert slots % block == 0 and chips * slots >= num_keys, (slots, block, chips, num_keys)

        masked = indexer.masked_scores(q, w, indexer.query_positions(positions), stripes)  # [1,1,S,slots]
        centre = cut = None
        cand = masked
        if slots > topk:
            centre, cut, cand_slots = self._top(masked, topk)  # this chip's own cut
            cand = ttnn.gather(masked, dim=-1, index=cand_slots)
        everything = self._gather_all(cand)  # [1,1,S,chips*K]
        if everything.shape[-1] > topk:
            centre, cut, _ = self._top(everything, topk)  # the same on every chip

        if slots <= topk:
            # every slot was a candidate: ``everything`` is every chip's scores, compared once
            chosen = self._passes(everything, centre, cut)
        else:
            chosen = self._gather_all(self._passes(masked, centre, cut))

        # [chips, slots] order -> position order. Inside one round the two are the same.
        rounds = slots // block
        if chips > 1 and rounds > 1:
            chosen = ttnn.reshape(chosen, [S, chips, rounds, block])
            chosen = ttnn.permute(chosen, (0, 2, 1, 3))
            chosen = ttnn.reshape(chosen, [1, 1, S, chips * slots])
        if chips * slots > num_keys:
            chosen = ttnn.slice(chosen, [0, 0, 0, 0], [1, 1, S, num_keys])
        return chosen

    @staticmethod
    def additive(chosen):
        """0 / 1 [1,1,S,T] -> the additive bf16 mask [1,1,S,T + SINK_ROWS]: 0 where chosen, MASKED
        elsewhere, and 0 in the sink's columns."""
        mask = ttnn.typecast(ttnn.multiply(ttnn.subtract(chosen, 1.0), -MASKED), ttnn.bfloat16)
        return ttnn.pad(mask, [(0, 0), (0, 0), (0, 0), (0, SINK_ROWS)], 0.0)

    def select(self, q, w, stripes, num_keys: int) -> PrefillSelection:
        """The selection of a whole prompt: the queries at positions 0 .. S-1 over the ``num_keys``
        positions whose keys are in ``stripes``. q: [1,S,heads,head_dim]; w: [1,1,S,heads]."""
        assert self.indexer is not None, "a shared layer has no indexer: it takes the selection of the full layer below"
        S = q.shape[1]
        H, D = self.indexer.cfg.num_heads, self.indexer.cfg.head_dim
        parts = []
        for start in range(0, S, self.query_chunk):
            end = min(start + self.query_chunk, S)
            whole = start == 0 and end == S
            q_c = q if whole else ttnn.slice(q, [0, start, 0, 0], [1, end, H, D])
            w_c = w if whole else ttnn.slice(w, [0, 0, start, 0], [1, 1, end, H])
            parts.append(self.additive(self.selected(q_c, w_c, list(range(start, end)), stripes, num_keys)))
        mask = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)
        if len(parts) > 1:
            for part in parts:
                part.deallocate(True)
        return PrefillSelection(mask=mask, num_queries=S, num_keys=num_keys)

    def selection_from_host(self, selected: torch.Tensor) -> PrefillSelection:
        """A selected set from the host (bool [S, T]) as a PrefillSelection, the same on every chip."""
        S, T = selected.shape
        mask = torch.zeros(1, 1, S, T + SINK_ROWS)
        mask[0, 0, :, :T].masked_fill_(~selected, MASKED)
        mask = ttnn.from_torch(mask, layout=ttnn.TILE_LAYOUT, device=self.plan.mesh_device, dtype=ttnn.bfloat16)
        return PrefillSelection(mask=mask, num_queries=S, num_keys=T)

    @staticmethod
    def selection_to_host(selection: PrefillSelection, chips: int | None = None) -> list[torch.Tensor]:
        """The selected set the chips hold: one bool [S, T] per chip, chips row-major (the first
        ``chips`` of them, default all)."""
        T = selection.num_keys
        shards = ttnn.get_device_tensors(selection.mask)[:chips]
        return [ttnn.to_torch(t).float()[0, 0, :, :T] > 0.5 * MASKED for t in shards]

    # ------------------------------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------------------------------
    def keys_values(self, kv, k_rope):
        """kv [1,1,T,heads*(nope+v)] and k_rope [1,1,T,rope], as TtHy4Attention.head_projections and
        latent return them -> (keys [1,heads,T+SINK_ROWS,qk], values [1,heads,T+SINK_ROWS,v]) of
        this chip's heads, with the sink's zero rows at the end."""
        rows = [(0, 0), (0, 0), (0, SINK_ROWS), (0, 0)]
        return self.attention.keys_values(ttnn.pad(kv, rows, 0.0), ttnn.pad(k_rope, rows, 0.0))

    def _sink_columns(self, num_keys: int):
        """[1, heads, 1, num_keys + SINK_ROWS] bf16, this chip's heads: 0 over the keys, the head's sink
        logit in column num_keys, MASKED in the columns after it."""
        if num_keys not in self._sink_columns_cache:
            plan = self.plan
            columns = torch.zeros(1, self.cfg.num_heads, 1, num_keys + SINK_ROWS)
            columns[..., num_keys] = self._sinks[..., 0]
            columns[..., num_keys + 1 :] = MASKED
            self._sink_columns_cache[num_keys] = ttnn.from_torch(
                columns,
                layout=ttnn.TILE_LAYOUT,
                device=plan.mesh_device,
                dtype=ttnn.bfloat16,
                mesh_mapper=plan.shard(head_dim=1),
            )
        return self._sink_columns_cache[num_keys]

    def core(self, q, keys_t, values, mask):
        """Dense attention with the sink as a key: q [1,heads,S,qk]; keys_t [1,heads,qk,T+SINK_ROWS]
        (the keys, transposed); values [1,heads,T+SINK_ROWS,v]; mask [1,1,S,T+SINK_ROWS], additive.
        Returns [1,heads,S,v]."""
        attn = self.attention
        num_keys = values.shape[-2] - SINK_ROWS
        logits = ttnn.matmul(q, keys_t, compute_kernel_config=attn.core_ckc)
        logits = ttnn.add(ttnn.multiply(logits, self.cfg.scaling), mask)
        logits = ttnn.add(logits, self._sink_columns(num_keys))
        probs = ttnn.softmax(logits, dim=-1)
        return ttnn.matmul(probs, values, compute_kernel_config=attn.core_ckc)

    def attend(self, q, keys, values, selection: PrefillSelection):
        """The core under the selection, ``query_chunk`` queries at a time.

        q: [1,heads,S,qk]; keys: [1,heads,T+SINK_ROWS,qk] and values: [1,heads,T+SINK_ROWS,v] from
        ``keys_values``, this chip's heads. Returns [1,heads,S,v].
        """
        S, T = q.shape[-2], keys.shape[-2] - SINK_ROWS
        assert (selection.num_queries, selection.num_keys) == (S, T), (selection.num_queries, selection.num_keys, S, T)
        heads, qk = q.shape[1], q.shape[-1]
        keys_t = ttnn.transpose(keys, -2, -1)
        parts = []
        for start in range(0, S, self.query_chunk):
            end = min(start + self.query_chunk, S)
            whole = start == 0 and end == S
            q_c = q if whole else ttnn.slice(q, [0, 0, start, 0], [1, heads, end, qk])
            parts.append(self.core(q_c, keys_t, values, selection.rows(start, end)))
        if len(parts) == 1:
            return parts[0]
        out = ttnn.concat(parts, dim=2)
        for part in parts:
            part.deallocate(True)
        return out

    def forward(self, x, cos, sin):
        """x: [1,1,T,hidden/htp], the layer's pre-normed input for positions 0 .. T-1, split over the
        hidden axis; cos / sin: [1,1,T,rope] at those positions, the interleaved tables
        (frames.rope_tables). T is a multiple of 32.

        Returns [1,1,T,hidden/htp], split over the hidden axis like x. On the way: a full layer stores
        its selection in the index cache, and the layer's latents (and index keys) go to the KV cache.
        """
        attn = self.attention
        T = x.shape[-2]
        assert T % ttnn.TILE_SIZE == 0, (
            f"prefill takes a prompt of whole tiles (a multiple of {ttnn.TILE_SIZE} tokens), got {T}: pad it, "
            "the pad tokens come last and nothing real attends to them"
        )
        q_resid, k_pass, k_rope = attn.latent(x, cos, sin)

        index_keys = None
        if self.indexer is not None:
            index_keys, index_q, index_w = self.indexer(x, q_resid, cos, sin)
            self.index_cache.store(self.layer, self.select(index_q, index_w, self.stripes(index_keys), T))
        selection = self.index_cache.picks(self.layer)
        assert (selection.num_queries, selection.num_keys) == (T, T), (
            f"layer {self.layer} got a selection for {selection.num_queries} queries over {selection.num_keys} "
            f"positions, the prompt has {T}"
        )

        if self.kv_cache is not None:
            self.kv_cache.write_prefill(
                self.layer, self.user, latent=self.kv_cache.pack(k_pass, k_rope), index_key=index_keys, start=0
            )

        q, gate, kv = attn.head_projections(x, q_resid, k_pass)
        q = attn.queries(q, cos, sin)
        keys, values = self.keys_values(kv, k_rope)
        return attn.output(self.attend(q, keys, values, selection), gate)
