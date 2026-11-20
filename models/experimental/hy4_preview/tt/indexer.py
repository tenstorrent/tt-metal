# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview DSA indexer on the SC1 layout (tenstorrent/tt-metal#65887, #65888): the index key a
token adds to the cache, the top-2048 selection of a query over the context-parallel key stripes,
at decode and at prefill, and the hand-over of a full layer's picks to the shared layers above it.

Mirrors the pure-torch ground truth in ../reference/indexer.py.

Projections (tt/common.py MeshPlan; the 8x4 mesh: hidden axis = the 4-chip axis, head axis = the
8-chip axis). All three
weights are sharded once, 1/32 per chip:

    weight          in -> out            in split over    out split over                  then
    wq_b            2048 -> 32 x 128     hidden axis      head axis (4 indexer heads)     \\ one reduce(hidden),
    wk | w_proj     6144 -> 128 + 32     hidden axis      head axis (16 key dims, 4 w)    / one gather(head)

* q_resid is whole on every chip after the attention sublayer's qkv_a, so each chip keeps its own
  slice with mesh_partition, as attention's q_b does. x arrives split over the hidden axis.
* The two partials are concatenated, summed over the hidden axis in ONE all-reduce and gathered
  over the head axis in ONE all-gather; both run in fp32, and the result is rounded to bf16 once.
  After that the 32 query heads, the 128-wide key and the 32 head weights of every new token are
  whole on every chip: the key is complete on whichever chip owns the token's block, and every
  chip can score its own stripe with all 32 heads.
* The key's LayerNorm (with its bias), RoPE and the Hadamard rotation are chip-local.

Frame (tt/frames.py). Keys and queries are in the frame prefill stores index keys in
(tenstorrent/tt-metal#65935, #65886): ``[rope 64, interleaved pairs | pass 64]`` in the orthonormal
Hadamard basis, where HF and ../reference have ``[pass 64 | rope 64, half-split]`` and no rotation.
The key dims of wk, k_norm and every head of wq_b are reordered when the weights are loaded, and
both keys and queries are multiplied by the same orthonormal matrix, so every score is HF's and
the selection is unchanged. ``cos`` / ``sin`` are the interleaved tables (frames.rope_tables).

Selection (tt/kv_layout.py BlockCyclicLayout: chip c holds the keys of its blocks as [slots, 128]).
For S queries, on every chip, with no host round trip:

    per_head = relu(stripe @ q^T)                    [slots, S x 32], fp32, HiFi4
    score    = per_head @ W                          [slots, S] -> [S, slots]; W holds the signed head
                                                     weights, block-diagonal over the queries
                                                     (``head_sum="matmul"``; see below for "fp32")
    masked   = score where the slot's position <= the query's position, else NEG
    local    = top-2048 of masked                    the chip's candidates: (score, position) pairs
    all      = gather(local) over both mesh axes     [S, 32 x 2048] pairs, the same on every chip
    picks    = top-2048 of all                       [S, 2048] positions, best first

  The result is the global top-2048 as sequence positions, identical on every chip, with a 0/1
  ``valid`` flag per entry (a query that sees fewer than 2048 positions gets masked slots in the
  tail). A position is a (chip, slot) location by kv_layout's arithmetic; the cache's ``locate``
  turns the list into the rows each chip holds, on the device.

* Validity is a compare against a per-chip constant: ``slot_position`` holds the sequence position
  of every slot of that chip's stripe, so a slot whose position is after the query (which covers
  every unfilled slot) scores NEG and can never be a valid pick. The query position is the only per-step
  input besides the query itself.
* ttnn.topk at k = 2048: on Blackhole a bf16 input of up to 2^19 columns runs a multi-core large-k
  path; an fp32 input is rejected at every width tried (4096 to 65536). bf16 keeps 8 significant
  bits, which would turn every score within 0.4% of the cut into a tie, so every top-k here runs
  ``refine_passes`` times (reference/indexer.py top_through_bf16): the first pass ranks the scores
  rounded to bf16 and gives the k-th to 8 bits; the scores are then re-centred on it in fp32 and
  ranked again. bf16 resolves a value relative to its own magnitude, so after centring the scores
  next to the cut keep 8 more bits per pass. Two passes resolve the cut to 2^-16 of its value, finer
  than the matmul's own error. The candidates' fp32 scores and positions are read back with
  ttnn.gather, so only the ranking goes through bf16, never a value. Rounding is monotonic, so a
  pass can only misorder scores that are tied at its own resolution.
* The gather carries (score | position) as one fp32 tensor, so the merge costs one all-gather per
  mesh axis. Positions are exact in fp32 below 2^24.
* A stripe of at most 2048 slots (contexts up to 65,536) needs no local top-k: every slot is a
  candidate. When all the stripes together hold no more than 2048 slots there is no top-k at all.
* Decode is S = 1. Prefill runs the same code on a block of queries, in chunks that bound the
  [slots, S x 32] intermediate: functional, not tuned.
* ``head_sum`` picks how the 32 per-head terms are added up. "matmul" (the default) is the second
  matmul above. "fp32" multiplies by the head weights and sums over the heads with the elementwise
  fp32 ops instead (per_head as [S, slots, 32], one batched matmul over the queries), which leaves
  the first matmul as the only matmul in a score. Both give [S, slots] scores and feed the same
  selection. Measured on the Galaxy at Hy4's width on order-one scores, at every context from 4K
  to 1M (tests/test_indexer.py): the largest score error is up to 8e-3 with "matmul" and up to
  1.9e-3 with "fp32", 3 to 4.4 times smaller on the same scores, and a decode step costs the same
  either way. The default is still "matmul": the modules built on this one were verified with it,
  and "fp32" has not been timed on a prefill block.

Measured cost of a decode step for one full layer and one user on the 8x4 Galaxy (untraced, the
steps queued back to back; a synchronize after every step adds 1.5 to 3 ms), in ms:

    context   slots per chip   select()      forward(), one new token
    128K      4096             7.6 to 7.8    3.6 to 4.5
    200K      6272             7.7           3.6 to 5.5
    1M        32768            9.8 to 9.9    4.5 to 4.9

  Inside select(), at every context: scoring and masking 0.5 to 0.8 ms, each two-pass top-k 0.8
  to 1.3 ms, the all-gather of the candidates over both mesh axes 0.4 to 0.5 ms. The rest is
  ttnn.gather, which dominates: reading the picks' scores and positions out of the 65,536
  gathered candidates takes 2.75 ms per gather, and reading a chip's candidates out of its stripe
  0.5 ms per gather at 128K and 1.4 ms at 1M (the only part that grows with the context). BFP8
  stripes cost the same as bf16.

Tensors: hidden states and keys are [1, 1, T, width] with the tokens down the tile rows; query
heads are [1, S, heads, 128]; everything per query is [1, 1, S, width].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.hy4_preview.reference.indexer import NEG, IndexerConfig, picked_mask, picks_source
from models.experimental.hy4_preview.tt import frames
from models.experimental.hy4_preview.tt.common import MeshPlan, compute_config
from models.experimental.hy4_preview.tt.kv_layout import BlockCyclicLayout


def _cols(x, start, end):
    """Columns [start, end) of the last dim."""
    shape = list(x.shape)
    return ttnn.slice(x, [0, 0, 0, start], [shape[0], shape[1], shape[2], end])


def _rope(x, cos, sin, swap, ckc):
    """Interleaved RoPE on the last dim: x * cos + (x @ swap) * sin, (x @ swap) = (-x1, x0, -x3, x2, ...)
    (frames.rope_swap). cos / sin are interleaved tables; swap is [1, 1, d, d]."""
    rotated = ttnn.matmul(x, swap, compute_kernel_config=ckc, dtype=ttnn.float32)
    return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rotated, sin))


def _as(x, dtype):
    return x if x.dtype == dtype else ttnn.typecast(x, dtype)


@dataclass
class Picks:
    """A selection: the positions S queries attend over, the same on every chip.

    ``positions`` [1, 1, S, K] fp32: sequence positions. K = min(top-k, all stripe slots). Best first
                  whenever the merge ran a top-k (always on the 8x4 mesh); on a single chip the
                  local top-k's order, or slot order when the stripe holds no more than top-k slots.
    ``valid``     [1, 1, S, K] fp32: 1.0 where the entry is a real pick, 0.0 where it is a masked slot
                  (the query sees fewer than K positions); a masked entry still names a slot inside
                  the stripes, so a gather on it stays in range, and attention must mask it.
    Position p lives on chip BlockCyclicLayout.chip(p) at slot BlockCyclicLayout.slot(p).
    """

    positions: ttnn.Tensor
    valid: ttnn.Tensor
    num_queries: int

    @property
    def width(self) -> int:
        return self.positions.shape[-1]

    def deallocate(self):
        self.positions.deallocate(True)
        self.valid.deallocate(True)


class TtHy4Indexer(LightweightModule):
    """One full layer's indexer. ``weights`` holds the five tensors by checkpoint name, as
    reference.indexer.Hy4Indexer.weights() returns them (linear weights are [out, in]), in the
    checkpoint's frame; they are moved to the device frame here (module docstring).

    ``layout`` is the cache's position <-> (chip, slot) contract; it defaults to one chip per mesh
    device, blocks of 128. On a single device (``plan.hidden_axis is None``) it is one stripe.
    """

    def __init__(
        self,
        plan: MeshPlan,
        cfg: IndexerConfig,
        weights: dict,
        layout: BlockCyclicLayout | None = None,
        weight_dtype=ttnn.bfloat16,
        refine_passes: int = 2,
        max_per_head_elements: int = 1 << 26,
        head_sum: str = "matmul",
    ):
        assert head_sum in ("matmul", "fp32"), head_sum
        self.plan = plan
        self.cfg = cfg
        self.head_sum = head_sum
        c, htp, ctp = cfg, plan.hidden_tp, plan.head_tp
        self.num_chips = htp * ctp
        self.layout = layout or BlockCyclicLayout(num_chips=self.num_chips)
        assert self.layout.num_chips == self.num_chips, (self.layout.num_chips, self.num_chips)
        assert refine_passes >= 1
        self.refine_passes = refine_passes
        self.max_per_head_elements = max_per_head_elements

        for what, size, tp in (
            ("hidden_size", c.hidden_size, htp),
            ("q_lora_rank", c.q_lora_rank, htp),
            ("num_heads", c.num_heads, ctp),
            ("head_dim", c.head_dim, ctp),
        ):
            assert size % tp == 0, f"{what} = {size} does not split {tp} ways"
        assert htp == 1 or (c.q_lora_rank // htp) % ttnn.TILE_SIZE == 0, (
            f"q_lora_rank = {c.q_lora_rank} over {htp} hidden slices is {c.q_lora_rank // htp} wide; "
            f"mesh_partition needs a multiple of {ttnn.TILE_SIZE}"
        )
        # per chip: this chip's query heads, key dims and head weights
        self.q_width = c.num_heads // ctp * c.head_dim
        self.k_width = c.head_dim // ctp
        self.w_width = c.num_heads // ctp

        # The indexer decides a discrete choice, so everything runs HiFi4 with fp32 accumulation.
        self.ckc = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True)

        def linear(w):
            """An [out, in] weight as [1, 1, in, out], in split over the hidden axis, out over the head axis."""
            return ttnn.from_torch(
                w.float().t().contiguous().reshape(1, 1, w.shape[1], w.shape[0]),
                layout=ttnn.TILE_LAYOUT,
                device=plan.mesh_device,
                dtype=weight_dtype,
                mesh_mapper=plan.shard(hidden_dim=2, head_dim=3),
            )

        def replicated(t, shape):
            return ttnn.from_torch(
                t.float().contiguous().reshape(shape),
                layout=ttnn.TILE_LAYOUT,
                device=plan.mesh_device,
                dtype=ttnn.float32,
            )

        weights = frames.indexer_weights(weights, c)
        wq_b, wk, w_proj = weights["wq_b.weight"], weights["wk.weight"], weights["weights_proj.weight"]
        assert wq_b.shape == (c.num_heads * c.head_dim, c.q_lora_rank), wq_b.shape
        assert wk.shape == (c.head_dim, c.hidden_size) and w_proj.shape == (c.num_heads, c.hidden_size)
        self.w_q = linear(wq_b)
        # (key dims | head weights) per column of the mesh, so sharding the output dim hands a chip both.
        self.w_kw = linear(
            torch.cat([torch.cat(pair, dim=0) for pair in zip(wk.chunk(ctp, dim=0), w_proj.chunk(ctp, dim=0))], dim=0)
        )
        self.k_norm_weight = replicated(weights["k_norm.weight"], (1, 1, 1, c.head_dim))
        self.k_norm_bias = replicated(weights["k_norm.bias"], (1, 1, 1, c.head_dim))
        rope = c.qk_rope_head_dim
        self.rope_swap = replicated(frames.rope_swap(rope), (1, 1, rope, rope))
        self.hadamard = replicated(frames.hadamard(c.head_dim), (1, 1, c.head_dim, c.head_dim))

        self._slot_positions = {}  # slots -> [1, 1, 1, slots] fp32 per chip
        self._block_masks = {}  # S -> [1, 1, S * heads, S] fp32

    # ------------------------------------------------------------------------------------------
    # New tokens: index keys, queries, head weights
    # ------------------------------------------------------------------------------------------
    def project(self, x, q_resid):
        """x: [1,1,T,hidden/htp] split over the hidden axis; q_resid: [1,1,T,q_lora] whole on every chip.

        Returns, fp32 and whole on every chip:
            q  [1, T, heads, head_dim]   wq_b(q_resid), before RoPE
            k  [1, 1, T, head_dim]       wk(x), before the norm
            w  [1, 1, T, heads]          weights_proj(x) with both scales applied
        One all-reduce over the hidden axis and one all-gather over the head axis carry all three.
        """
        c, plan = self.cfg, self.plan
        T = x.shape[-2]
        q_p = ttnn.matmul(
            plan.partition(q_resid, 3, plan.hidden_axis), self.w_q, compute_kernel_config=self.ckc, dtype=ttnn.float32
        )
        kw_p = ttnn.matmul(x, self.w_kw, compute_kernel_config=self.ckc, dtype=ttnn.float32)
        both = ttnn.concat([q_p, kw_p], dim=-1)
        if plan.hidden_axis is not None:
            both = plan.all_reduce(both, plan.hidden_axis)
            both = plan.all_gather(both, dim=3, axis=plan.head_axis)
        # [.., ctp x (q | k | w)] -> one row per mesh column, then each piece back to its own width
        qw, kw, ww = self.q_width, self.k_width, self.w_width
        both = ttnn.reshape(both, [1, T, plan.head_tp, qw + kw + ww])
        q = ttnn.reshape(_cols(both, 0, qw), [1, T, c.num_heads, c.head_dim])
        k = ttnn.reshape(_cols(both, qw, qw + kw), [1, 1, T, c.head_dim])
        w = ttnn.reshape(_cols(both, qw + kw, qw + kw + ww), [1, 1, T, c.num_heads])
        return q, k, ttnn.multiply(w, c.weight_scale)

    def keys(self, k, cos, sin):
        """k: [1,1,T,head_dim] from project; cos / sin: [1,1,T,rope] at the tokens' positions.

        Returns the index keys [1,1,T,head_dim], bf16, whole on every chip: LayerNorm, RoPE on the
        leading rope dims, then the Hadamard rotation. This is the per-token cache entry of a full
        layer, as prefill stores it.
        """
        c = self.cfg
        k = _as(k, ttnn.float32)
        centred = ttnn.subtract(k, ttnn.mean(k, dim=-1, keepdim=True))
        var = ttnn.mean(ttnn.multiply(centred, centred), dim=-1, keepdim=True)
        k = ttnn.multiply(centred, ttnn.rsqrt(ttnn.add(var, c.norm_eps)))
        k = ttnn.add(ttnn.multiply(k, self.k_norm_weight), self.k_norm_bias)
        rope = c.qk_rope_head_dim
        k_rope = _rope(_cols(k, 0, rope), _as(cos, ttnn.float32), _as(sin, ttnn.float32), self.rope_swap, self.ckc)
        k = ttnn.concat([k_rope, _cols(k, rope, c.head_dim)], dim=-1)
        k = ttnn.matmul(k, self.hadamard, compute_kernel_config=self.ckc, dtype=ttnn.float32)
        return ttnn.typecast(k, ttnn.bfloat16)

    def queries(self, q, cos, sin):
        """q: [1,T,heads,head_dim] from project; cos / sin: [1,1,T,rope] at the queries' positions.

        Returns [1,T,heads,head_dim], bf16: RoPE on the leading rope dims of every head, then the
        Hadamard rotation the keys get.
        """
        c = self.cfg
        T = q.shape[1]
        q = _as(q, ttnn.float32)
        cos = ttnn.reshape(_as(cos, ttnn.float32), [1, T, 1, c.qk_rope_head_dim])
        sin = ttnn.reshape(_as(sin, ttnn.float32), [1, T, 1, c.qk_rope_head_dim])
        rope = c.qk_rope_head_dim
        q_rope = _rope(_cols(q, 0, rope), cos, sin, self.rope_swap, self.ckc)
        q = ttnn.concat([q_rope, _cols(q, rope, c.head_dim)], dim=-1)
        q = ttnn.matmul(q, self.hadamard, compute_kernel_config=self.ckc, dtype=ttnn.float32)
        return ttnn.typecast(q, ttnn.bfloat16)

    def forward(self, x, q_resid, cos, sin):
        """New tokens -> (keys [1,1,T,head_dim] bf16, q [1,T,heads,head_dim] bf16, w [1,1,T,heads] fp32).

        The keys go into the cache (the owner chip of each token's block keeps its row); q and w go
        to select() together with the stripes that already hold those keys.
        """
        q, k, w = self.project(x, q_resid)
        return self.keys(k, cos, sin), self.queries(q, cos, sin), w

    # ------------------------------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------------------------------
    def _on_chips(self, per_chip: torch.Tensor, dtype):
        """[num_chips, ...] -> a device tensor holding slice i, with a leading [1, 1], on chip i (row-major)."""
        plan = self.plan
        if plan.hidden_axis is None:
            return ttnn.from_torch(
                per_chip.reshape(1, 1, *per_chip.shape[1:]).contiguous(),
                layout=ttnn.TILE_LAYOUT,
                device=plan.mesh_device,
                dtype=dtype,
            )
        rows, cols = plan.mesh_shape
        return ttnn.from_torch(
            per_chip.reshape(rows, cols, *per_chip.shape[1:]).contiguous(),
            layout=ttnn.TILE_LAYOUT,
            device=plan.mesh_device,
            dtype=dtype,
            mesh_mapper=ttnn.ShardTensor2dMesh(plan.mesh_device, dims=(0, 1), mesh_shape=plan.mesh_shape),
        )

    def stripes_from_host(self, keys: torch.Tensor, slots: int | None = None):
        """Host index keys [T, head_dim] in position order -> the stripes select() takes: per chip
        [1, 1, slots, head_dim] bf16, laid out by ``self.layout``, zero where a slot has no position."""
        return self._on_chips(self.layout.split(keys.float(), dim=0, slots=slots), ttnn.bfloat16)

    def slot_positions(self, slots: int):
        """Per chip [1, 1, 1, slots] fp32: the sequence position of every slot of that chip's stripe."""
        if slots not in self._slot_positions:
            chip = torch.arange(self.num_chips)[:, None]
            position = self.layout.position(chip, torch.arange(slots)[None, :])
            assert int(position.max()) < 1 << 24, "positions are compared in fp32"
            self._slot_positions[slots] = self._on_chips(
                position.float().reshape(self.num_chips, 1, slots), ttnn.float32
            )
        return self._slot_positions[slots]

    def query_positions(self, positions):
        """Sequence positions of S queries (ints) -> [1, 1, S, 1] fp32, the same on every chip."""
        positions = torch.as_tensor(positions).reshape(1, 1, -1, 1)
        assert int(positions.max()) < 1 << 24, "positions are compared in fp32"
        return ttnn.from_torch(
            positions.float(), layout=ttnn.TILE_LAYOUT, device=self.plan.mesh_device, dtype=ttnn.float32
        )

    def _block_mask(self, S):
        """[1, 1, S * heads, S] fp32: 1 where row (s, h) belongs to query s."""
        if S not in self._block_masks:
            H = self.cfg.num_heads
            mask = torch.eye(S).repeat_interleave(H, dim=0).reshape(1, 1, S * H, S)
            self._block_masks[S] = ttnn.from_torch(
                mask, layout=ttnn.TILE_LAYOUT, device=self.plan.mesh_device, dtype=ttnn.float32
            )
        return self._block_masks[S]

    def scores(self, q, w, stripes):
        """Every chip scores its own stripe. q: [1,S,heads,head_dim]; w: [1,1,S,heads];
        stripes: [1,1,slots,head_dim] per chip. Returns [1,1,S,slots] fp32 per chip, unmasked."""
        c = self.cfg
        S, H = q.shape[1], c.num_heads
        if self.head_sum == "fp32":
            return self._scores_fp32_sum(q, w, stripes)
        q_t = ttnn.transpose(ttnn.reshape(q, [1, 1, S * H, c.head_dim]), -2, -1)  # [1,1,head_dim,S*H]
        per_head = ttnn.matmul(stripes, q_t, compute_kernel_config=self.ckc, dtype=ttnn.float32)
        per_head = ttnn.relu(per_head)  # [1,1,slots,S*H]
        w = ttnn.reshape(_as(w, ttnn.float32), [1, 1, S * H, 1])
        if S > 1:
            w = ttnn.multiply(w, self._block_mask(S))  # [1,1,S*H,S], block-diagonal
        scores = ttnn.matmul(per_head, w, compute_kernel_config=self.ckc, dtype=ttnn.float32)  # [1,1,slots,S]
        return ttnn.transpose(scores, -2, -1)

    def _scores_fp32_sum(self, q, w, stripes):
        """scores() with the head-weighted sum in elementwise fp32 (``head_sum="fp32"``): the stripe
        is scored against every query's heads in one batched matmul, [1,S,slots,heads], and the
        heads are weighted and summed without a second matmul."""
        c = self.cfg
        S, H, slots = q.shape[1], c.num_heads, stripes.shape[-2]
        q_t = ttnn.transpose(q, -2, -1)  # [1,S,head_dim,heads]
        per_head = ttnn.matmul(stripes, q_t, compute_kernel_config=self.ckc, dtype=ttnn.float32)
        per_head = ttnn.relu(per_head)  # [1,S,slots,heads]
        weighted = ttnn.multiply(per_head, ttnn.reshape(_as(w, ttnn.float32), [1, S, 1, H]))
        scores = ttnn.sum(weighted, dim=-1, keepdim=True)  # [1,S,slots,1]
        return ttnn.reshape(ttnn.transpose(scores, -2, -1), [1, 1, S, slots])

    def masked_scores(self, q, w, positions, stripes):
        """scores(), with NEG wherever the slot's position is after the query's (unfilled slots included)."""
        slots = stripes.shape[-2]
        scores = self.scores(q, w, stripes)
        visible = ttnn.le(self.slot_positions(slots), positions)  # [1,1,1,slots] vs [1,1,S,1]
        return ttnn.where(visible, scores, NEG)

    def _gather_all(self, x):
        """Concatenate every chip's ``x`` on the last dim, on every chip."""
        plan = self.plan
        if plan.hidden_axis is None:
            return x
        return plan.all_gather(plan.all_gather(x, dim=3, axis=plan.head_axis), dim=3, axis=plan.hidden_axis)

    def _top(self, scores, k, passes):
        """Top-k of fp32 ``scores`` [1,1,S,n] along the last dim through the bf16 top-k, ``passes`` times.

        Pass 1 ranks the scores rounded to bf16 and takes the k-th as the centre; every further pass
        ranks bf16(scores - centre) and moves the centre onto the new k-th. Returns
        (centre, cut, slots): ``slots`` [1,1,S,k] are the last pass's picks, best first, and an
        element is in the top-k exactly when bf16(score - centre) >= cut (centre is None after a
        single pass: the score itself is ranked). centre and cut are [1,1,S,1] fp32.
        """
        S = scores.shape[-2]
        centre = None
        for step in range(passes):
            ranked = scores if centre is None else ttnn.subtract(scores, centre)
            top, slots = ttnn.topk(ttnn.typecast(ranked, ttnn.bfloat16), k=k, dim=-1, largest=True, sorted=True)
            cut = ttnn.typecast(ttnn.slice(top, [0, 0, 0, k - 1], [1, 1, S, k]), ttnn.float32)
            if step < passes - 1:
                centre = cut if centre is None else ttnn.add(centre, cut)
        return centre, cut, slots

    def _select(self, q, w, positions, stripes) -> Picks:
        c, chips = self.cfg, self.num_chips
        S, slots = q.shape[1], stripes.shape[-2]
        masked = self.masked_scores(q, w, positions, stripes)  # [1,1,S,slots] fp32
        slot_position = self.slot_positions(slots)
        if S > 1:
            slot_position = ttnn.repeat(slot_position, [1, 1, S, 1])

        # the chip's candidates, as (score, position)
        if slots <= c.topk:
            cand, cand_position = masked, slot_position
        else:
            _, _, cand_slots = self._top(masked, c.topk, self.refine_passes)
            cand = ttnn.gather(masked, dim=-1, index=cand_slots)
            cand_position = ttnn.gather(slot_position, dim=-1, index=cand_slots)

        # one gather: every chip's candidates on every chip, [1,1,S,chips*K] each
        K = cand.shape[-1]
        both = self._gather_all(ttnn.concat([cand, cand_position], dim=-1))
        both = ttnn.reshape(both, [1, S, chips, 2 * K])
        scores = ttnn.reshape(ttnn.slice(both, [0, 0, 0, 0], [1, S, chips, K]), [1, 1, S, chips * K])
        picked = ttnn.reshape(ttnn.slice(both, [0, 0, 0, K], [1, S, chips, 2 * K]), [1, 1, S, chips * K])

        # merge: the global top-k of the candidates
        if chips * K > c.topk:
            _, _, index = self._top(scores, c.topk, self.refine_passes)
            scores = ttnn.gather(scores, dim=-1, index=index)
            picked = ttnn.gather(picked, dim=-1, index=index)
        valid = _as(ttnn.gt(scores, 0.5 * NEG), ttnn.float32)
        return Picks(positions=picked, valid=valid, num_queries=S)

    def select(self, q, w, positions, stripes) -> Picks:
        """The top-k positions of S queries over the context-parallel key stripes.

        q: [1,S,heads,head_dim] and w: [1,1,S,heads] from forward() (whole on every chip);
        positions: the queries' sequence positions, ints or a [1,1,S,1] fp32 device tensor;
        stripes: [1,1,slots,head_dim] per chip, the index-key cache of this layer, which must
        already hold the keys of every position up to the last query's.

        Decode is S = 1. A block of queries is processed in chunks that keep the per-head score
        intermediate under ``max_per_head_elements``.
        """
        if not isinstance(positions, ttnn.Tensor):
            positions = self.query_positions(positions)
        S, slots = q.shape[1], stripes.shape[-2]
        chunk = max(1, self.max_per_head_elements // (slots * self.cfg.num_heads))
        if S <= chunk:
            return self._select(q, w, positions, stripes)
        H, D = self.cfg.num_heads, self.cfg.head_dim
        parts = []
        for start in range(0, S, chunk):
            end = min(start + chunk, S)
            parts.append(
                self._select(
                    ttnn.slice(q, [0, start, 0, 0], [1, end, H, D]),
                    ttnn.slice(w, [0, 0, start, 0], [1, 1, end, H]),
                    ttnn.slice(positions, [0, 0, start, 0], [1, 1, end, 1]),
                    stripes,
                )
            )
        picks = Picks(
            positions=ttnn.concat([p.positions for p in parts], dim=2),
            valid=ttnn.concat([p.valid for p in parts], dim=2),
            num_queries=S,
        )
        for p in parts:
            p.deallocate()
        return picks

    # ------------------------------------------------------------------------------------------
    # Host views of a selection (tests, and anything that wants the indices on the host)
    # ------------------------------------------------------------------------------------------
    def picked(self, picks: Picks):
        """(positions int64 [S, K], valid bool [S, K]) read back from chip 0."""
        positions = ttnn.to_torch(ttnn.get_device_tensors(picks.positions)[0])[0, 0]
        valid = ttnn.to_torch(ttnn.get_device_tensors(picks.valid)[0])[0, 0]
        return positions.round().long(), valid > 0.5

    def locations(self, picks: Picks) -> list[torch.Tensor]:
        """Per query, the selected (chip, slot) pairs as an int64 [n, 2] tensor, read back to the host."""
        positions, valid = self.picked(picks)
        return [torch.stack([self.layout.chip(p[v]), self.layout.slot(p[v])], dim=-1) for p, v in zip(positions, valid)]

    def selection_mask(self, picks: Picks, num_positions: int) -> torch.Tensor:
        """The selected set as a host bool [S, num_positions] mask, the form the reference's
        selection_mask has. Raises if a valid pick is not below ``num_positions`` or is repeated."""
        return picked_mask(*self.picked(picks), num_positions)


class TtHy4IndexCache:
    """Hands a full layer's picks to the shared layers above it (HF's ``prev_topk_indices``).

    ``indexer_types`` is the config's per-layer "full" / "shared" list. Within one step (a decode
    token or a prefill block) the layers run bottom-up: a full layer runs its indexer and calls
    store(); every layer, full or shared, calls picks() for the selection its attention uses.
    The cache owns the picks: storing a full layer's picks frees the previous full layer's.
    """

    def __init__(self, indexer_types):
        self.source = picks_source(indexer_types)
        self._layer = None
        self._picks = None

    def is_full(self, layer: int) -> bool:
        return self.source[layer] == layer

    def store(self, layer: int, picks: Picks):
        assert self.is_full(layer), f"layer {layer} is a shared layer: it has no indexer of its own"
        if self._picks is not None and self._picks is not picks:
            self._picks.deallocate()
        self._layer, self._picks = layer, picks

    def picks(self, layer: int) -> Picks:
        source = self.source[layer]
        assert self._layer == source, (
            f"layer {layer} attends over layer {source}'s picks, but the cache holds "
            f"{'nothing' if self._layer is None else f'layer {self._layer}s'}"
        )
        return self._picks

    def clear(self):
        if self._picks is not None:
            self._picks.deallocate()
        self._layer = self._picks = None
