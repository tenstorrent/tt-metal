# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One decode step of Hy4-preview's sparse attention on the SC1 layout (tenstorrent/tt-metal#65889,
with the decode side of the indexer, #65888).

Mirrors the pure-torch ground truth in ../reference/attention_decode.py. It adds no weights: it
composes the attention sublayer (tt/attention.py), the indexer (tt/indexer.py) and the
context-parallel cache (tt/kv_cache.py) for one new token of one user.

    new token   latent()                 q_resid and the 576-wide latent, whole on every chip
                indexer()                index key, 32 query heads, head weights (full layers)
                cache.write_decode       latent and index key land on the one chip that owns the position
    select      full layer:   indexer.select over the index-key stripes -> top-2048 positions,
                              the same list on every chip, kept in the TtHy4IndexCache
                shared layer: the list the full layer below made in this step
    locate      cache.locate             position -> this chip's storage row, and a 0/1 "mine" flag;
                                         done once per selection, reused by the shared layers
    gather      cache.gather             every chip reads 2048 rows of its own stripe; a row it does
                                         not own is some other row, zeroed with the flag
    collect     two all-reduces          each selected row is non-zero on exactly one chip, so the
                                         sum over the mesh IS the row: the 2048 selected latents,
                                         [1, 1, 2048, 576], whole on every chip, in pick order
    attend      head parallel            each chip runs its own 8 heads over the 2048 rows, with the
                                         sink; gate and o_proj as in prefill

The output is [1, 1, 1, hidden / 4], split over the hidden axis like the input.

Mesh and frame. SC1 runs on the Galaxy's own 8x4 mesh (tt/common.py MeshPlan: heads over mesh
axis 0, 8 chips; hidden over mesh axis 1, 4 chips), chips counted row-major, which is the chip
order of tt/kv_layout.py. The cache this step reads and appends to is the one prefill writes
(tt/frames.py): k_rope in interleaved RoPE pairs, index keys as [rope interleaved | pass] in the
Hadamard basis. Nothing here depends on that order: the attention and indexer modules produce the
new token's rows and queries in the cache's frame, cos / sin are uploaded as the interleaved
tables, and a logit or a score is a dot product of two vectors in the same frame.

Why head parallel, and not the log-sum-exp merge the issue describes
---------------------------------------------------------------------
#65889 has every chip attend with all 64 heads over the selected rows it holds and merges the
partial outputs by log-sum-exp. As written (keys and values expanded per chip) that needs kv_b for
all 64 heads on every chip, 32 copies of a weight that SC1 shards once. This module keeps the
1/32 layout of tt/attention.py and moves the rows instead of the weights:

* What moves: the 2048 selected latents, 2048 x 576 values (2.4 MB in bf16), once over each mesh
  axis, per layer and per token. The LSE design moves almost nothing (64 queries out, 64 partial
  results back).
* What it buys: every chip does the same work whatever the spread of the picks. They can all sit
  on one chip or on none of this chip's neighbours; the collect does not care, and the attention
  that follows always sees 2048 rows. With per-chip attention a chip may own anything from 0 to
  2048 of the picks, and a fixed-shape device program has to be sized for 2048 on every chip:
  64 heads x 2048 rows on each of 32 chips, 32 times the arithmetic, unless a native indexed op
  takes a row count at run time.
* kv_b is never applied to the selected rows (``core="absorb"``, the default): it is folded into
  the query (q_abs = q_nope W_k) and into the output (attn = W_v sum_t p_t k_pass_t), which is the
  absorbed form flash MLA decode uses. Both foldings contract over the 512 latent dims, which are
  the dims kv_b's input is split over the hidden axis by, so each chip folds its own 128-wide
  slice with its own 1/32 of the weight and the only things summed over the hidden axis are the
  8 x 2048 logits and the 8 x 256 outputs. ``core="expand"`` is the literal first version of the
  issue instead: kv_b on all 2048 rows, one all-reduce of 2048 x 3584 values (14.7 MB) over the
  hidden axis, then TtHy4Attention.keys_values and .core unchanged.
* The absorbed form is also what would let the LSE merge keep the 1/32 layout: with q in latent
  space a chip can attend over its own rows with all 64 heads and no weight at all, and return
  (sum_t p_t k_pass_t, lse) per head; W_v is applied after the merge, sharded as it is now.
  reference/attention_decode.py ``lse_merge`` is that design in torch, and the host test shows it
  equal to this one. It is the better end state (no 2.4 MB collect) once there is an op that
  attends over a per-chip row list of run-time length; until then it costs the 32x above.

So the trade is 2.4 MB of fabric traffic per layer per token against 32x the attention
arithmetic or a new op. Neither decides what this version costs: the number of ttnn calls does.

What this version costs
-----------------------
Measured on the 8x4 Galaxy at Hy4's widths, BFP8 weights and cache, untraced, one user per step
(tests/test_attention_decode.py test_decode_step_latency). "step" is ms per layer with the steps
issued back to back and one synchronize at the end; the pieces come from a second run in which
each is closed by a synchronize, which adds time of its own, so they add up to more:

    context  layer    step    new token  selection  locate  row gather  collect  attention
    4K       full     14.7    5.7-6.1    3.2        1.3     0.4         0.8      4.3
             shared    6.3    1.5        -          -       0.4         0.7      4.1
    128K     full     16.4    5.6-6.2    9.1        1.3     0.4         0.9      4.4
             shared    6.1    1.5        -          -       0.4         0.7      4.1
    1M       full     16.2    6.7        11.3       1.4     0.4         0.9      5.3
             shared    6.0    1.5        -          -       0.4         0.8      4.2

* The target is 75 tokens per second per user for 8 users at 1M context on one Galaxy: 13.3 ms for
  a step of the whole batch over 78 layers, 0.171 ms per layer. One user's step is 35 (shared) to
  95 (full) times that, and 8 users are 8 steps, one after the other. Over the model's 21 full and
  57 shared layers at 1M the attention sublayer alone is 0.68 s per token for one user.
* Nearly all of it is the host issuing ops, not the device working: a shared layer's "attention"
  is about 50 ttnn calls and takes 4.1 ms at every context, about 80 us per call on 32 chips, and a
  full layer's step at 1M is 1.5 ms longer than at 4K. Nothing here says what the kernels
  themselves take; that needs a trace. ``core="expand"`` is 0.3 to 1.0 ms per layer faster than
  "absorb" for the same reason (fewer calls), 14.7 MB all-reduce included. The indexer's
  ``head_sum`` makes no difference to the time.
* Row gather and collect together are 1.1 to 1.4 ms. cache.gather is ttnn.embedding over the
  stripe, which reads through the whole table (users x slots rows, tt/kv_cache.py): 0.4 ms from
  32,768 rows (one user at 1M) and 1.3 ms from 262,144 (a cache of 8 users at 1M). Everything else
  is the same with a cache of 8 users.
* The step's host uploads (below) take 1.4 to 1.9 ms per token.
* On a mesh opened with an L1_SMALL region the ttnn collectives keep their semaphores there: the
  steps of full and shared layers together hold 3328 bytes per core. The prefill stack opens the
  mesh with 1536, in which a step fails at its tenth collective (out of L1_SMALL). A mesh shared
  with prefill needs the larger region, or collectives on persistent semaphores as prefill has.

What is not here
----------------
* One user per call. The cache already holds ``max_users`` stripes and a step names its user. A
  batch of users in one call needs: the selection scored per user (indexer.select takes S queries
  but one set of stripes; per-user stripes are a batch dim on the score matmul), cache.locate with
  a row offset per user instead of one ``user`` argument, the collect on [1, users, 2048, 576],
  and the core with the users on the batch dim ([users, heads, 1, .] against [users, 1, 2048, .]).
  The projections, the gate and o_proj already take one row per user.
* What still comes from the host on every token, with what would replace it:
    cos / sin of the position       rope_cos_sin on the host, two uploads. On the device:
                                    position x inverse frequencies in fp32, then ttnn.cos / ttnn.sin,
                                    once their accuracy at arguments up to 1e6 is checked.
    the cache's append indices      a host table, one upload (cache.append_indices). On the device:
                                    kv_cache.locate_rows on the position tensor, the slot where the
                                    chip owns it and -1 elsewhere.
    the query position              one upload. On the device: a counter the step increments.
    the context length as an int    selection reads only the stripe rows a context of this length
                                    reaches (cache.stripe(length=)), so the shapes, and with them the
                                    programs, change every 4096 positions. A traced step has to
                                    score the whole stripe; the indexer masks by position anyway.
  Nothing is read back to the host inside a step.
* A row read that costs 2048 rows: cache.gather reads through the table (above). A tile-native
  row gather, or prefill's sparse_kv_push, is the op for it.
* ``core="absorb"`` keeps a second, head-major bf16 copy of this chip's 1/32 of kv_b (0.9 MB per
  layer per chip). An engine would store kv_b in that layout once.
* Rows a chip does not own are read from the user's slot 0 and multiplied by zero, and so is the
  row of a list entry that is not a real pick, so every row of the user's stripe must hold finite
  values (see "What a step needs from the cache").
* The cache this module reads is tt/kv_cache.py's container. A prefill by the other stack leaves
  its rows in another one (packed scaled-FP8 rows), so until decode attends over that storage in
  place (its sparse_kv_push + sparse_sdpa path, which also is one op where this module issues
  dozens), a bridge has to copy every row once, and the copy does not fit beside the original at
  8 users x 1M.

What a step needs from the cache
--------------------------------
For whoever fills the cache this step reads (host tests, or a bridge from the prefill stack's
storage). The step goes through TtHy4KvCache only, for one user, at ``step.position = N``, the
length of the user's context before this token.

* Container. ``cache.latent[layer]`` for every layer that runs, ``cache.index_key[layer]`` for
  every full layer: per chip ``[max_users, 1, slots, width]``, TILE, DRAM, bfloat8_b or bfloat16
  (one dtype for the whole cache), width 576 and 128; ``slots`` = 128 x the rounds of 4096
  positions the capacity holds. cache.gather (ttnn.embedding) and cache.write_decode
  (paged_update_cache) need exactly this: a row-major packed row (the prefill stack's scaled-FP8
  KVPE row) cannot be read or appended to by this module.
* Where a row is. Position p of user u: dim 0 = u, dim 2 = ``layout.slot(p)``, on chip
  ``layout.chip(p)``, chips row-major over the 8x4 mesh (tt/kv_layout.py). That is the prefill
  stripe: chip = (p // 128) % 32, row = (p // 4096) * 128 + p % 128.
* Frame (tt/frames.py). A latent row is ``[k_pass 512 | k_rope 64]``: the kv_a_layernorm output in
  HF's order, then k_rope after RoPE in interleaved pairs; this is the logical value of a prefill
  KVPE row (FP8 latent times its block scale | BF16 rope). An index-key row is the key in
  ``[rope 64 interleaved | pass 64]`` times frames.hadamard(128), which is a prefill index row as
  it is stored.
* Rows that must be valid: positions ``[0, N)`` of the user, the latent on every layer and the
  index key on every full layer. Row N is written by the step before anything is read, so a step
  may also re-run the last prompt token over a cache that already holds it.
* Rows that are read and not used. Each chip reads the user's slot 0 for every pick it does not
  own, and the owner chip reads the row of a list entry that is not a real pick (a context shorter
  than the top-k has such entries, and they may name any slot); both are multiplied by zero. So
  no row of the user's stripe, on any chip, may hold NaN or Inf: zero-initialised storage,
  whatever a prefill chunk's padding or an earlier prompt left behind, or BFP8 storage (which
  cannot hold either) all do. Their values never matter: the indexer masks by position (a slot
  whose position is above N cannot be a real pick), not by content.
* The cache keeps no lengths. The caller owns N and passes it as ``position``.
* ``begin_decode_step`` reads no device tensor. It takes ``cache.cfg.max_users`` and
  ``cache.cfg.capacity`` (asserts), the layout (through ``cache.append_indices``), and uploads
  four small host tensors: cos and sin of N (interleaved tables), the per-chip append indices and
  the query position.
* A step then calls, per layer: ``cache.pack``, ``cache.write_decode`` (rows N of both kinds),
  ``cache.stripe(layer, INDEX_KEY, user, length=N + 1)`` and ``cache.locate`` (full layers), and
  ``cache.gather(layer, rows, LATENT)``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.hy4_preview.reference.attention import rope_cos_sin
from models.experimental.hy4_preview.tt import frames
from models.experimental.hy4_preview.tt.attention import TtHy4Attention
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.indexer import Picks, TtHy4IndexCache, TtHy4Indexer
from models.experimental.hy4_preview.tt.kv_cache import INDEX_KEY, LATENT, TtHy4KvCache

# Added to the logit of a list entry that is not a real pick. Finite in bf16 and fp32, and far below
# any logit, so the entry's probability underflows to zero.
MASKED = -1.0e9

CORES = ("absorb", "expand")
# The pieces a step's time is split into, in the order they run.
STAGES = ("new token", "selection", "locate", "row gather", "collect", "attention")


def _cols(x, start, end):
    """Columns [start, end) of the last dim."""
    shape = list(x.shape)
    return ttnn.slice(x, [0, 0, 0, start], [shape[0], shape[1], shape[2], end])


def _as(x, dtype):
    return x if x.dtype == dtype else ttnn.typecast(x, dtype)


@dataclass
class Selection:
    """A full layer's picks in the form this step's attention reads them, per chip.

    ``rows``  uint32 row major [1, K]: the storage row of every pick on this chip (cache.locate).
    ``own``   bf16 [1, 1, K, 1]: 1 where this chip holds the pick and the pick is real, else 0.
    ``mask``  fp32 [1, 1, 1, K]: additive, 0 on a real pick and MASKED elsewhere; the same on every chip.
    """

    layer: int
    picks: Picks
    rows: ttnn.Tensor
    own: ttnn.Tensor
    mask: ttnn.Tensor

    @property
    def width(self) -> int:
        return self.picks.width

    def deallocate(self):
        for t in (self.rows, self.own, self.mask):
            t.deallocate(True)


@dataclass
class DecodeStep:
    """What the layers of one decode step share: the token's position and user, the position's
    cos / sin ([1, 1, 1, rope]), the cache's append indices, and the selection of the last full layer.

    ``seconds`` is filled when the step was begun with ``profile=True``: wall time per STAGES entry,
    each closed by a device synchronize, summed over the layers that ran.
    """

    position: int
    user: int
    cos: ttnn.Tensor
    sin: ttnn.Tensor
    indices: ttnn.Tensor
    query_position: ttnn.Tensor
    mesh_device: object = None
    profile: bool = False
    selection: Selection | None = None
    seconds: dict = field(default_factory=dict)
    _last: float = 0.0

    def start(self):
        if self.profile:
            ttnn.synchronize_device(self.mesh_device)
            self._last = time.perf_counter()

    def mark(self, stage: str):
        """Close ``stage``: everything queued since the last mark is charged to it."""
        if self.profile:
            ttnn.synchronize_device(self.mesh_device)
            now = time.perf_counter()
            self.seconds[stage] = self.seconds.get(stage, 0.0) + now - self._last
            self._last = now

    def end(self):
        """Free the step's device tensors (the picks themselves belong to the index cache)."""
        if self.selection is not None:
            self.selection.deallocate()
            self.selection = None
        for t in (self.cos, self.sin, self.indices, self.query_position):
            t.deallocate(True)


def begin_decode_step(
    plan: MeshPlan, cache: TtHy4KvCache, cfg, position: int, user: int = 0, profile: bool = False
) -> DecodeStep:
    """The per-step inputs every layer of the step takes. ``cfg`` is the attention config (RoPE);
    ``position`` is where the new token goes in ``user``'s context, which must already hold every
    position below it."""
    assert 0 <= user < cache.cfg.max_users, f"user {user} of {cache.cfg.max_users}"
    assert 0 <= position < cache.cfg.capacity, f"position {position} is outside the cache ({cache.cfg.capacity})"
    # TtHy4Attention takes the interleaved tables of the cache's frame (tt/frames.py)
    cos, sin = frames.rope_tables(*rope_cos_sin(torch.tensor([position]), cfg))

    def replicated(t, dtype):
        return ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=plan.mesh_device, dtype=dtype)

    return DecodeStep(
        position=position,
        user=user,
        cos=replicated(cos.reshape(1, 1, 1, -1), ttnn.bfloat16),
        sin=replicated(sin.reshape(1, 1, 1, -1), ttnn.bfloat16),
        indices=cache.append_indices([position if u == user else -1 for u in range(cache.cfg.max_users)]),
        query_position=replicated(torch.tensor([float(position)]).reshape(1, 1, 1, 1), ttnn.float32),
        mesh_device=plan.mesh_device,
        profile=profile,
    )


class TtHy4AttentionDecode(LightweightModule):
    """The decode step of one layer.

    ``attention`` is the layer's TtHy4Attention (its weights are used as they are), ``cache`` the
    engine's TtHy4KvCache, ``index_cache`` the engine's TtHy4IndexCache (one for all layers), and
    ``indexer`` the layer's TtHy4Indexer on a full layer, None on a shared one.

    ``core``: "absorb" (kv_b folded into the query and the output) or "expand" (kv_b on every
    selected row, then the dense core of tt/attention.py); see the module docstring.
    """

    def __init__(
        self,
        plan: MeshPlan,
        attention: TtHy4Attention,
        cache: TtHy4KvCache,
        layer: int,
        index_cache: TtHy4IndexCache,
        indexer: TtHy4Indexer | None = None,
        core: str = "absorb",
    ):
        assert core in CORES, core
        self.plan = plan
        self.attention = attention
        self.cache = cache
        self.layer = layer
        self.index_cache = index_cache
        self.indexer = indexer
        self.core = core
        self.cfg = c = attention.cfg
        self.heads = attention.heads

        assert index_cache.is_full(layer) == (indexer is not None), (
            f"layer {layer} is a {'full' if index_cache.is_full(layer) else 'shared'} layer: "
            f"{'it needs its indexer' if indexer is None else 'it has no indexer of its own'}"
        )
        assert cache.cfg.latent_dim == c.latent_dim, (cache.cfg.latent_dim, c.latent_dim)
        assert layer in cache.latent, f"the cache holds no latent stripe for layer {layer}"
        if indexer is not None:
            assert layer in cache.index_key, f"the cache holds no index-key stripe for layer {layer}"
            assert cache.cfg.index_key_dim == indexer.cfg.head_dim, (cache.cfg.index_key_dim, indexer.cfg.head_dim)
            assert (
                indexer.layout == cache.layout
            ), "the indexer and the cache must share one position <-> (chip, slot) layout"

        self.sinks = self.w_k = self.w_v = None
        if core == "absorb":
            # This chip's 1/32 of kv_b, [1, 1, in, heads * (nope + v)], once more as one matrix per
            # head: W_k^T [1, heads, nope, in] for the query side, W_v [1, heads, in, v] for the output.
            self.in_slice = c.kv_lora_rank // plan.hidden_tp
            w = _as(attention.w_kv_b, ttnn.bfloat16)
            w = ttnn.permute(ttnn.reshape(w, [1, self.in_slice, self.heads, c.kv_b_head_dim]), (0, 2, 1, 3))
            self.w_k = ttnn.transpose(_cols(w, 0, c.qk_nope_head_dim), -2, -1)
            self.w_v = _cols(w, c.qk_nope_head_dim, c.kv_b_head_dim)
            # one sink logit per head, down the rows like the logits: [1, 1, heads, 1]
            self.sinks = _as(ttnn.reshape(attention.sinks, [1, 1, self.heads, 1]), ttnn.float32)

    # ------------------------------------------------------------------------------------------
    # The new token
    # ------------------------------------------------------------------------------------------
    def _per_user(self, row, step: DecodeStep):
        """[1, 1, 1, width] -> the [1, 1, max_users, width] write_decode takes. Only ``step.user``'s
        row is written (the other users' append index is -1), so every row carries the same values."""
        users = self.cache.cfg.max_users
        return row if users == 1 else ttnn.repeat(row, [1, 1, users, 1])

    def new_token(self, x, step: DecodeStep):
        """x: [1, 1, 1, hidden / htp]. Appends the token's latent (and index key) to the cache.

        Returns (q_resid [1, 1, 1, q_lora], index_q, index_w): the last two are the indexer's query
        heads [1, 1, heads, head_dim] and head weights [1, 1, 1, heads] on a full layer, else None.
        """
        assert tuple(x.shape)[:3] == (1, 1, 1), f"a decode step takes one token, got {tuple(x.shape)}"
        q_resid, k_pass, k_rope = self.attention.latent(x, step.cos, step.sin)
        latent = self.cache.pack(k_pass, k_rope)
        index_key = index_q = index_w = None
        if self.indexer is not None:
            index_key, index_q, index_w = self.indexer(x, q_resid, step.cos, step.sin)
            index_key = self._per_user(index_key, step)
        self.cache.write_decode(self.layer, step.indices, latent=self._per_user(latent, step), index_key=index_key)
        return q_resid, index_q, index_w

    # ------------------------------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------------------------------
    def select(self, index_q, index_w, step: DecodeStep) -> Picks:
        """The top-k positions of the new token over this layer's index-key stripes, which already
        hold the token's own key. Only the rows a context of this length can reach are scored."""
        assert self.indexer is not None, f"layer {self.layer} is a shared layer"
        stripes = self.cache.stripe(self.layer, INDEX_KEY, user=step.user, length=step.position + 1)
        return self.indexer.select(index_q, index_w, step.query_position, stripes)

    def locate(self, picks: Picks, step: DecodeStep, layer: int | None = None) -> Selection:
        """Picks -> where every chip finds them in its stripe, and which entries count."""
        assert picks.num_queries == 1, f"a decode step has one query, got {picks.num_queries}"
        rows, mine = self.cache.locate(picks.positions, user=step.user)
        own = ttnn.multiply(mine, picks.valid)  # [1,1,1,K]: this chip holds it, and it is a real pick
        own = _as(ttnn.transpose(own, -2, -1), ttnn.bfloat16)  # [1,1,K,1], one flag per gathered row
        mask = ttnn.multiply(ttnn.subtract(picks.valid, 1.0), -MASKED)  # 0 where valid, MASKED where not
        return Selection(self.layer if layer is None else layer, picks, rows, own, mask)

    def selection(self, step: DecodeStep, index_q=None, index_w=None) -> Selection:
        """The selection this layer attends with: its own on a full layer (run now, stored for the
        shared layers above), else the stored one of the full layer below."""
        source = self.index_cache.source[self.layer]
        if self.indexer is not None:
            picks = self.select(index_q, index_w, step)
            step.mark("selection")
            self.index_cache.store(self.layer, picks)
            if step.selection is not None:
                step.selection.deallocate()
            step.selection = self.locate(picks, step)
            step.mark("locate")
        assert step.selection is not None and step.selection.layer == source, (
            f"layer {self.layer} attends over layer {source}'s picks, but this step holds "
            f"{'none' if step.selection is None else f'layer {step.selection.layer}s'}"
        )
        assert step.selection.picks is self.index_cache.picks(self.layer)
        return step.selection

    # ------------------------------------------------------------------------------------------
    # The selected rows
    # ------------------------------------------------------------------------------------------
    def gather(self, selection: Selection):
        """Every chip's read of its own stripe at the selection's rows: bf16 [1, 1, K, latent] per
        chip, zero on the rows the chip does not own and on entries that are not real picks."""
        rows = self.cache.gather(self.layer, selection.rows, LATENT)
        return ttnn.multiply(rows, selection.own)

    def collect(self, gathered):
        """Sum over the mesh. A row is non-zero on one chip only, so the sum is exact: the K selected
        latents, [1, 1, K, latent], whole and identical on every chip, in pick order."""
        plan = self.plan
        return plan.all_reduce(plan.all_reduce(gathered, plan.head_axis), plan.hidden_axis)

    def selected_rows(self, selection: Selection, step: DecodeStep | None = None):
        gathered = self.gather(selection)
        if step is not None:
            step.mark("row gather")
        selected = self.collect(gathered)
        if step is not None:
            step.mark("collect")
        return selected

    # ------------------------------------------------------------------------------------------
    # Attention over the selected rows, this chip's heads
    # ------------------------------------------------------------------------------------------
    def query_gate(self, x, q_resid):
        """q_b and the gate for the new token, this chip's heads: (q [1,1,1,heads*qk],
        gate logits [1,1,1,heads*v]), head-major. One all-reduce over the hidden axis carries both."""
        a, c, plan, h = self.attention, self.cfg, self.plan, self.heads
        q_p = a._matmul(plan.partition(q_resid, 3, plan.hidden_axis), a.w_q_b)
        g_p = a._matmul(x, a.w_gate)
        if plan.hidden_tp == 1:
            return q_p, g_p
        q_w, g_w = h * c.qk_head_dim, h * c.v_head_dim
        both = plan.all_reduce(ttnn.concat([q_p, g_p], dim=-1), plan.hidden_axis)
        return _cols(both, 0, q_w), _cols(both, q_w, q_w + g_w)

    def expanded_core(self, q, selected, mask):
        """kv_b on every selected row, then the dense core. q: [1,heads,1,qk]; selected:
        [1,1,K,latent]; mask: [1,1,1,K]. Returns [1,heads,1,v]."""
        a, c, plan = self.attention, self.cfg, self.plan
        k_pass = _cols(selected, 0, c.kv_lora_rank)
        k_rope = _cols(selected, c.kv_lora_rank, c.latent_dim)
        kv = plan.all_reduce(a._matmul(plan.partition(k_pass, 3, plan.hidden_axis), a.w_kv_b), plan.hidden_axis)
        keys, values = a.keys_values(kv, k_rope)
        return a.core(q, keys, values, _as(mask, a.dtype))

    def absorbed_logits(self, q, selected, mask):
        """q: [1,heads,1,qk] -> (logits fp32 [1,1,heads,K], scaled and masked, without the sink;
        this chip's hidden slice of the selected k_pass, [1,1,K,in])."""
        a, c, plan, h = self.attention, self.cfg, self.plan, self.heads
        K = selected.shape[-2]
        k_slice = plan.partition(_cols(selected, 0, c.kv_lora_rank), 3, plan.hidden_axis)
        k_rope = _cols(selected, c.kv_lora_rank, c.latent_dim)
        # q_nope W_k: this chip's slice of the query in latent space, complete without a collective
        q_abs = ttnn.matmul(_cols(q, 0, c.qk_nope_head_dim), self.w_k, compute_kernel_config=a.ckc, dtype=a.dtype)
        q_abs = ttnn.reshape(q_abs, [1, 1, h, self.in_slice])
        nope = ttnn.matmul(q_abs, ttnn.transpose(k_slice, -2, -1), compute_kernel_config=a.core_ckc, dtype=ttnn.float32)
        nope = plan.all_reduce(nope, plan.hidden_axis)  # [1,1,heads,K]: the sum over the 512 latent dims
        q_rope = ttnn.reshape(_cols(q, c.qk_nope_head_dim, c.qk_head_dim), [1, 1, h, c.qk_rope_head_dim])
        rope = ttnn.matmul(q_rope, ttnn.transpose(k_rope, -2, -1), compute_kernel_config=a.core_ckc, dtype=ttnn.float32)
        logits = ttnn.add(ttnn.multiply(ttnn.add(nope, rope), c.scaling), mask)
        assert tuple(logits.shape) == (1, 1, h, K), tuple(logits.shape)
        return logits, k_slice

    def absorbed_core(self, q, selected, mask):
        """The same attention without expanding a key or a value. Returns [1,heads,1,v]."""
        a, c, plan, h = self.attention, self.cfg, self.plan, self.heads
        K = selected.shape[-2]
        logits, k_slice = self.absorbed_logits(q, selected, mask)
        combined = ttnn.concat([logits, self.sinks], dim=-1)  # [1,1,heads,K+1]
        probs = ttnn.softmax(_as(combined, a.dtype), dim=-1)
        probs = ttnn.slice(probs, [0, 0, 0, 0], [1, 1, h, K])  # the sink's share is dropped
        ctx = ttnn.matmul(probs, k_slice, compute_kernel_config=a.core_ckc)  # [1,1,heads,in]
        ctx = ttnn.reshape(ctx, [1, h, 1, self.in_slice])
        out = ttnn.matmul(ctx, self.w_v, compute_kernel_config=a.ckc, dtype=ttnn.float32)  # [1,heads,1,v]
        out = plan.all_reduce(ttnn.reshape(out, [1, 1, h, c.v_head_dim]), plan.hidden_axis)
        return ttnn.reshape(_as(out, a.dtype), [1, h, 1, c.v_head_dim])

    def attend(self, x, q_resid, selected, selection: Selection, step: DecodeStep):
        """x: [1,1,1,hidden/htp]; selected: [1,1,K,latent] whole on every chip. Returns the sublayer
        output [1,1,1,hidden/htp], split over the hidden axis."""
        a = self.attention
        q, gate = self.query_gate(x, q_resid)
        q = a.queries(q, step.cos, step.sin)
        core = self.absorbed_core if self.core == "absorb" else self.expanded_core
        return a.output(core(q, selected, selection.mask), gate)

    # ------------------------------------------------------------------------------------------
    def forward(self, x, step: DecodeStep, picks: Picks | None = None):
        """One new token through this layer's attention sublayer.

        x: [1, 1, 1, hidden / htp], the layer's pre-normed input for the token at ``step.position``,
        split over the hidden axis. Returns [1, 1, 1, hidden / htp].

        The layers of a step run bottom-up with the same ``step``. ``picks`` overrides the selection
        (a test hands in a hand-made list; the caller keeps ownership of it): the layer then runs
        no selection of its own and stores nothing for the layers above.
        """
        step.start()
        q_resid, index_q, index_w = self.new_token(x, step)
        step.mark("new token")
        if picks is None:
            selection = self.selection(step, index_q, index_w)
        else:
            selection = self.locate(picks, step)
            step.mark("locate")
        selected = self.selected_rows(selection, step)
        out = self.attend(x, q_resid, selected, selection, step)
        step.mark("attention")
        if picks is not None:
            selection.deallocate()
        return out
