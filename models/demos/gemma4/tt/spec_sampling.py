# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device-sampled verify ids for the serving records (GEMMA4_SPEC_SAMPLED_VERIFY).

A serving record drafts K argmax ids per row and runs one packed target verify over
``[anchor, d1..dK]``; its ids are the target's argmax per candidate column. For a
row the runner samples (temperature > 0, top-k, top-p, an optional seed) the
argmax is not that row's distribution, so such rows are never drafted today and
decode plain.

With the knob on, the record's body ends with ONE target draw per candidate
column, made by the same ``ttnn.sampling`` op the plain decode's device sampler
runs, with the row's own temperature / top-k / top-p and a seed per (row,
column): the seed the plain path would push for that absolute position
(``_hash_request_seed_to_device_seed(seed, position + 1)``) for a seeded request,
fresh entropy for an unseeded one. Against the drafter's one-hot proposal the
plugin's unchanged id-equality walk is then exact rejection sampling: a draft is
accepted with probability p(draft), the first mismatch commits the draw of that
column, which is distributed p(x) / (1 - p(draft)) given the mismatch, and the
bonus after K accepts is the draw of the last column; every committed token is
one draw from the target distribution of its position. The columns' draws are
independent (one PRNG stream per core, one core per packed row).

Host side (``serving_sampling_rows``): the per-row parameters are broadcast to
the row's P packed rows, padded to 32-row chunks with the greedy triple and
encoded exactly as the plain path encodes them (``format_sampling_params``:
inverse temperature, top-k clamped to [1, 32], greedy = (temp 1, k 1, p 0)).
Device side (``SpecVerifySampler``): persistent per-chunk parameter and seed
rows, one ``[1, 1, 1, 32]`` draw output per chunk, and the in-record graph
top-k -> (TP gather | split concat) -> global indices -> manual_seed ->
sampling. Greedy rows keep the record's argmax ids by selection on the host
(``select_verify_ids``), so an all-greedy step is bit-identical to the knob off.

On the lane engine every sampler input is a lane frame: each mesh column
(lane) holds its own parameter rows, seed rows and draw output, sharded over
the lane axis the way the plain sampler shards its data-parallel groups, and
the plan is a ``LaneSamplingPlan`` (one ``SamplingRowsPlan`` per column over
that column's rows, ``serving_sampling_frame``); the draws come back per
column and ``select_verify_ids_frame`` reassembles the call order.

Not available on a target without the device top-k sampler (fractured
meshes) or on one whose sampler masks a padded vocabulary: ``serving_prepare``
refuses the knob there with a ValueError at warmup (the capability follows the
env; the refusal is loud).
"""

import itertools
import os
import secrets
from dataclasses import dataclass

import torch

import ttnn
from models.common.sampling.generator import (
    DEVICE_SEED_MAX,
    MAX_UINT32,
    _hash_request_seed_to_device_seed,
    format_sampling_params,
)
from models.common.sampling.sampling_params import SamplingParams
from models.common.sampling.tt_sampling import TIEBREAK_DELTA_FLOOR, TIEBREAK_DELTA_SCALE, TIEBREAK_INDEX_SENTINEL
from models.demos.gemma4.tt.ccl import ccl_allgather
from models.demos.gemma4.tt.lane_prefill import lane_row_groups

SPEC_SAMPLED_VERIFY_ENV = "GEMMA4_SPEC_SAMPLED_VERIFY"
SPEC_SAMPLED_ROW_TILE = 32
# ``ttnn.topk`` takes at most this many columns per call (models/common/sampling/tt_sampling.py TOPK_MAX_WIDTH):
# gathered logits wider than it are cut into equal power-of-two chunks, each topk'd on its own.
SPEC_SAMPLED_TOPK_MAX_WIDTH = 64 * 1024
SPEC_SAMPLED_MAX_TOP_K = 32
# The seed of a packed row whose draw is never read (greedy or padding rows): any valid device seed.
SPEC_SAMPLED_IDLE_SEED = MAX_UINT32  # the device SKIP sentinel: manual_seed leaves that core's PRNG untouched
# A replay's manual_seed runs on the target sampler's cores, the ones the plain decode draws on. Every packed row
# whose draw nobody reads (a greedy row, a padding row) carries the SKIP seed, so a replay re-initialises only
# the cores of its sampled rows; a plain unseeded row that shares one of those cores continues from the
# record's per-replay seed (fresh entropy or a position hash), a random state, not a fixed one.


def spec_sampled_verify_enabled(env=None):
    """True when GEMMA4_SPEC_SAMPLED_VERIFY holds one of the on-values the GEMMA4_* gates accept."""
    env = os.environ if env is None else env
    return str(env.get(SPEC_SAMPLED_VERIFY_ENV, "0")).lower() in ("1", "true", "yes")


def _row_value(value, row, default):
    """``value`` for ``row``: a scalar as is, a per-row sequence at ``row``, ``default`` past its end or None."""
    if value is None:
        return default
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return value.item()
    row = int(row)
    if row >= len(value):
        return default
    item = value[row]
    if isinstance(item, torch.Tensor):
        item = item.item()
    return default if item is None else item


def _fresh_seed():
    """An unseeded row's device seed: fresh entropy in the plain path's range (never the SKIP sentinel)."""
    return secrets.randbelow(DEVICE_SEED_MAX) + 1


@dataclass
class SamplingRowsPlan:
    """The device sampler's inputs for one replay of a ``(B, K)`` record, packed row ``b * P + j``.

    ``temperature`` / ``top_k`` / ``top_p`` hold the device encoding for every
    row of every 32-row chunk (``chunks * 32`` entries); ``seeds`` the per-row
    device seeds; ``sampled`` which packed rows take the sampler's draw (the
    rest keep the record's argmax id). ``row_params`` is what each live row
    asked for, ``(temperature, top_k, top_p, seed)`` as given, for the
    adapter's retained-row bookkeeping.
    """

    B: int
    P: int
    chunks: int
    temperature: list
    top_k: list
    top_p: list
    seeds: list
    sampled: list
    row_params: list

    @property
    def rows(self):
        return self.chunks * SPEC_SAMPLED_ROW_TILE

    @property
    def any_sampled(self):
        return any(self.sampled)

    def chunk(self, c):
        """``(top_k, top_p, temperature, seeds)`` of chunk ``c``, 32 values each."""
        lo, hi = c * SPEC_SAMPLED_ROW_TILE, (c + 1) * SPEC_SAMPLED_ROW_TILE
        return (self.top_k[lo:hi], self.top_p[lo:hi], self.temperature[lo:hi], self.seeds[lo:hi])

    def params_key(self, c):
        """The chunk's parameter rows as a hashable value (the upload skips a chunk whose rows did not change)."""
        k, p, t, _seeds = self.chunk(c)
        return (tuple(int(v) for v in k), tuple(float(v) for v in p), tuple(float(v) for v in t))


def serving_chunks(B, P):
    """32-row chunks a ``(B, P)`` record's packed rows take (one ``ttnn.sampling`` call each)."""
    return (int(B) * int(P) + SPEC_SAMPLED_ROW_TILE - 1) // SPEC_SAMPLED_ROW_TILE


def row_sampling_params(sampling, row):
    """``(temperature, top_k, top_p, seed)`` of ``row`` in the step's sampling params (None = host-sampled step).

    ``seed`` is None for an unseeded row (the plugin hands the model None for its sentinel); a
    negative seed is read as unseeded too. The values are what the runner sent, not the device
    encoding.
    """
    if sampling is None:
        return (0.0, 1, 1.0, None)
    temperature = float(_row_value(getattr(sampling, "temperature", None), row, 0.0))
    top_k = int(_row_value(getattr(sampling, "top_k", None), row, 1))
    top_p = float(_row_value(getattr(sampling, "top_p", None), row, 1.0))
    seed = _row_value(getattr(sampling, "seed", None), row, None)
    if seed is not None:
        seed = int(seed)
        if seed < 0:
            seed = None
    return (temperature, top_k, top_p, seed)


def row_is_sampled(sampling, row):
    """Whether ``row`` draws with a temperature above zero on the step's sampling params."""
    return row_sampling_params(sampling, row)[0] > 0.0


def serving_sampling_rows(
    shape, rows, positions, sampling, *, seed_hash=_hash_request_seed_to_device_seed, fresh_seed=_fresh_seed
):
    """The ``SamplingRowsPlan`` of one replay: live row ``i`` (row id ``rows[i]`` in ``sampling``, anchor at
    ``positions[i]``) broadcast to its P packed rows.

    A sampled row's column ``j`` (input position ``positions[i] + j``) gets the row's temperature / top-k /
    top-p and seed ``seed_hash(seed, positions[i] + j + 1, 0)`` for a seeded request (the device seed the plain
    decode pushes for that absolute position: counter = position + 1, salt 0 on the vLLM path) or
    ``fresh_seed()`` for an unseeded one. Greedy rows, padding rows and the chunk padding carry the greedy
    triple and the idle seed. The encoding is the plain path's (``format_sampling_params``).
    """
    B, K = (int(shape[0]), int(shape[1]))
    P = K + 1
    live = len(rows)
    if live > B:
        raise ValueError(f"shape {shape}: {live} live rows exceed B={B}")
    if len(positions) != live:
        raise ValueError(f"shape {shape}: {len(positions)} positions for {live} live rows")
    chunks = serving_chunks(B, P)
    n = chunks * SPEC_SAMPLED_ROW_TILE
    temperature = [0.0] * n
    top_k = [1] * n
    top_p = [1.0] * n
    seeds = [SPEC_SAMPLED_IDLE_SEED] * n
    sampled = [False] * n
    row_params = []
    for i in range(live):
        temp, k, p, seed = row_sampling_params(sampling, rows[i])
        row_params.append((temp, k, p, seed))
        if temp <= 0.0:
            continue
        for j in range(P):
            r = i * P + j
            temperature[r] = temp
            top_k[r] = k
            top_p[r] = p
            seeds[r] = int(seed_hash(seed, int(positions[i]) + j + 1, 0)) if seed is not None else int(fresh_seed())
            sampled[r] = True
    encoded = format_sampling_params(
        SamplingParams(temperature=list(temperature), top_k=list(top_k), top_p=list(top_p), seed=None), n
    )
    return SamplingRowsPlan(
        B=B,
        P=P,
        chunks=chunks,
        temperature=[float(v) for v in encoded.temperature],
        top_k=[int(v) for v in encoded.top_k],
        top_p=[float(v) for v in encoded.top_p],
        seeds=seeds,
        sampled=sampled,
        row_params=row_params,
    )


def greedy_sampling_rows(shape):
    """The plan of an all-greedy replay (the sampler stage runs on the greedy triple; nothing reads its draws)."""
    return serving_sampling_rows(shape, [], [], None)


@dataclass
class LaneSamplingPlan:
    """The lane engine's plan for one replay: ``plans[lane]`` is that column's ``SamplingRowsPlan`` over its own
    live rows (in the call's lane-major order, ``B`` rows each; the greedy plan on an idle column) and
    ``row_params`` what every live row asked for, in call order (``SamplingRowsPlan.row_params``)."""

    B: int
    P: int
    plans: list
    row_params: list

    @property
    def lanes(self):
        return len(self.plans)

    @property
    def chunks(self):
        return self.plans[0].chunks

    @property
    def any_sampled(self):
        return any(plan.any_sampled for plan in self.plans)


def serving_sampling_frame(shape, rows, positions, sampling, lanes, lane_slots, **kw):
    """The ``LaneSamplingPlan`` of one lane-engine replay: the global ``rows`` grouped per lane
    (``lane_row_groups``: lane-major, as the record call orders them and as its outputs come back), each
    column's ``serving_sampling_rows`` over its own rows and anchors (``kw`` reaches it)."""
    lanes = int(lanes)
    if lanes <= 1:
        raise ValueError(f"serving_sampling_frame needs lanes > 1, got {lanes} (one column: serving_sampling_rows)")
    if len(positions) != len(rows):
        raise ValueError(f"shape {shape}: {len(positions)} positions for {len(rows)} live rows")
    groups = lane_row_groups(rows, lanes, lane_slots)
    plans, row_params = [], [None] * len(rows)
    for g in groups:
        plan = serving_sampling_rows(shape, [rows[i] for i in g], [positions[i] for i in g], sampling, **kw)
        plans.append(plan)
        for j, i in enumerate(g):
            row_params[i] = plan.row_params[j]
    return LaneSamplingPlan(B=plans[0].B, P=plans[0].P, plans=plans, row_params=row_params)


def greedy_sampling_frame(shape, lanes):
    """The all-greedy ``LaneSamplingPlan`` of ``shape`` on ``lanes`` columns."""
    plans = [greedy_sampling_rows(shape) for _ in range(int(lanes))]
    return LaneSamplingPlan(B=plans[0].B, P=plans[0].P, plans=plans, row_params=[])


def select_verify_ids_frame(plan, argmax_ids, draws, groups):
    """``[live, P]`` host ids of a lane-engine replay in call order: ``argmax_ids`` ``[lanes, B, P]`` (every
    column's record argmax), ``draws`` ``[lanes][rows]`` (every column's sampler draws), ``groups`` the call's
    lane row groups; a column's live rows take ``select_verify_ids`` under the column's own plan."""
    if not isinstance(plan, LaneSamplingPlan) or plan.lanes != len(groups):
        raise ValueError(f"select_verify_ids_frame needs a LaneSamplingPlan of {len(groups)} lanes, got {plan!r}")
    parts = []
    for lane, g in enumerate(groups):
        if not g:
            continue
        column = None if draws is None else draws[lane]  # a greedy frame never reads the draws
        parts.append(select_verify_ids(plan.plans[lane], argmax_ids[lane], column)[: len(g)])
    return torch.cat(parts, dim=0)


def select_verify_ids(plan, argmax_ids, sampled_ids):
    """``[B, P]`` host ids: the sampler's draw on a sampled packed row, the record's argmax elsewhere."""
    B, P = plan.B, plan.P
    out = torch.as_tensor(argmax_ids).to(torch.int64).reshape(B, P).clone()
    if not plan.any_sampled:
        return out
    draws = torch.as_tensor(sampled_ids).to(torch.int64).reshape(-1)
    for b in range(B):
        for j in range(P):
            r = b * P + j
            if plan.sampled[r]:
                out[b, j] = int(draws[r])
    return out


class SpecVerifySampler:
    """The device half: persistent inputs per record shape, the in-record sampling graph, the uploads and readback.

    Built by ``SpeculativeDecoder.serving_prepare`` (knob on) from the decoder's target: its TP geometry and
    CCL manager (the sharded verify's gather), its sampler's user-id row and sub-core grid, and its invalid-
    vocabulary mask when the padded vocabulary has one. Everything on the device is allocated by ``alloc``
    (per shape, before any capture) or by the first eager body (``offsets``); a replay allocates nothing
    that outlives the body.
    """

    def __init__(self, decoder):
        self.decoder = decoder
        target = decoder.target
        self.tp = int(decoder._tp)
        self.mesh_config = target.mesh_config
        self.ccl_manager = target.ccl_manager
        tt_sampling = getattr(target.sampling, "tt_sampling", None)  # None on a target without device sampling
        if tt_sampling is None:
            raise ValueError(
                f"{SPEC_SAMPLED_VERIFY_ENV}=1 needs the target's device sampler (sample_on_device); this target has none"
            )
        if not getattr(tt_sampling, "_allow_topk_sampling", True):
            raise ValueError(
                f"{SPEC_SAMPLED_VERIFY_ENV}=1 needs the device top-k sampler, which this mesh disables "
                "(fractured weights): unset the knob (sampled rows then decode plain)"
            )
        self.user_ids = tt_sampling.user_ids_tt_tensor  # [32] uint32: core per packed row
        self.sub_core_grids = tt_sampling._sampling_sub_core_grids
        if tt_sampling.tt_invalid_vocab_mask is not None:
            # the record draws over the verify logits as published (the gathered vocabulary, or a chip's slice):
            # a padded vocabulary would need the plain sampler's per-chip mask re-laid for that width
            raise ValueError(
                f"{SPEC_SAMPLED_VERIFY_ENV}=1 does not mask a padded vocabulary (the target's sampler carries an "
                "invalid-vocabulary mask): unset the knob (sampled rows then decode plain)"
            )
        self.max_top_k = min(SPEC_SAMPLED_MAX_TOP_K, int(tt_sampling.max_top_k))
        # the lane engine: every per-row input below is a lane frame (one column's rows per chip), the plan a
        # LaneSamplingPlan and the draws read per column (``read_frame``); one column otherwise
        self.lanes = int(getattr(decoder, "_srv_lanes", 1) or 1)
        # (width, slice, block) -> [1, 1, 32, width] int32 TILE: candidate block d (``block`` = top-k columns from
        # one chip or one vocabulary chunk) holds the offset d * slice
        self._offsets = {}
        self._uploaded = {}  # (S's serial, chunk) -> the params_key of every column last uploaded
        self._serial = itertools.count(1)  # one per alloc: a recycled id() never reads as the same buffers
        self.ops = 0  # device programs the last body issued (diagnostics)

    # -- persistent buffers ----------------------------------------------------------------------------------
    def _from_torch(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=True):
        """A replicated tensor (the same on every chip)."""
        return self.decoder._pv_from_torch(t, dtype, layout=layout, device=device)

    def _column_buffer(self, t, dtype, lane_dim, layout=ttnn.ROW_MAJOR_LAYOUT):
        """A persistent per-column device buffer: ``t`` on every chip (under lanes: ``t`` stacked ``lanes`` times
        along ``lane_dim`` and sharded over the lane axis, so every column holds its own copy)."""
        if self.lanes > 1:
            return self.decoder._srv_from_torch(t, dtype, layout=layout, lane_dim=lane_dim, device=True)
        return self._from_torch(t, dtype, layout=layout)

    def _frame_host(self, frame, dtype, lane_dim, layout=ttnn.ROW_MAJOR_LAYOUT):
        """The host tensor of one upload: under lanes ``frame`` is the lane frame (the columns' values stacked
        along ``lane_dim``), sharded over the lane axis; one column otherwise."""
        if self.lanes > 1:
            return self.decoder._srv_frame_from_torch(frame, dtype, layout=layout, lane_dim=lane_dim, device=False)
        return self._from_torch(frame, dtype, layout=layout, device=False)

    def alloc(self, B, P):
        """The shape's per-chunk parameter rows, seed rows and draw outputs (``_srv_alloc_shape``)."""
        R = SPEC_SAMPLED_ROW_TILE
        chunks = serving_chunks(B, P)
        if chunks != 1:
            # spec_check_shape keeps every record within one 32-row argmax tile (spec_decode.py), so a record's
            # sampler stage is one chunk; the per-chunk lists stay for the plan's own chunking
            raise ValueError(f"a record of {B} x {P} packed rows spans {chunks} sampler chunks; one is supported")
        greedy = greedy_sampling_rows((B, P - 1))
        S = {
            "chunks": chunks,
            "serial": next(self._serial),
            "k": [],
            "p": [],
            "temp": [],
            "seeds": [],
            "sidx": [],
            "gcol": [],
        }
        for c in range(chunks):
            k, p, temp, seeds = greedy.chunk(c)
            # per column (a lane frame under lanes): the 32 parameter rows along dim 0, the draw output's rows
            # along dim 3, the tie-break mask's rows along dim 2
            S["k"].append(self._column_buffer(torch.tensor(k, dtype=torch.int32), ttnn.uint32, 0))
            # the top-1 tie-break's mask (1.0 on the rows whose k == 1), TILE so it broadcasts over the width
            S["gcol"].append(self._column_buffer(_top1_mask(k), ttnn.bfloat16, 2, layout=ttnn.TILE_LAYOUT))
            S["p"].append(self._column_buffer(torch.tensor(p, dtype=torch.float32), ttnn.bfloat16, 0))
            S["temp"].append(self._column_buffer(torch.tensor(temp, dtype=torch.float32), ttnn.bfloat16, 0))
            S["seeds"].append(self._column_buffer(torch.tensor(seeds, dtype=torch.int64), ttnn.uint32, 0))
            S["sidx"].append(self._column_buffer(torch.zeros(1, 1, 1, R, dtype=torch.int64), ttnn.uint32, 3))
        return S

    def offsets_for(self, width, vocab_slice, block):
        """Persistent ``[1, 1, 32, width]`` int32 TILE: the vocabulary offset of each ``block``-column candidate
        block (``block`` candidates per chip or per vocabulary chunk, ``width // block`` blocks)."""
        key = (int(width), int(vocab_slice), int(block))
        if int(width) % int(block):
            raise ValueError(f"candidate width {width} is not a multiple of the {block}-column block")
        t = self._offsets.get(key)
        if t is None:
            if getattr(self.decoder, "_srv_frozen", False):
                raise RuntimeError(
                    f"no sampler offsets for width {width} after serving_prepare: the eager bodies make them; a replay "
                    "never allocates"
                )
            R, W = SPEC_SAMPLED_ROW_TILE, int(block)
            host = torch.zeros(1, 1, R, int(width), dtype=torch.int32)
            for d in range(int(width) // W):
                host[:, :, :, d * W : (d + 1) * W] = d * int(vocab_slice)
            t = self._from_torch(host, ttnn.int32, layout=ttnn.TILE_LAYOUT)
            self._offsets[key] = t
        return t

    def release(self, shape_bufs):
        for S in shape_bufs:
            if S is None:
                continue
            for key in ("k", "p", "temp", "seeds", "sidx", "gcol"):
                self.decoder._free_tensor_tree(S.get(key))
        for t in self._offsets.values():
            self.decoder._free_tensor_tree(t)
        self._offsets = {}
        self._uploaded = {}

    # -- the in-record graph ---------------------------------------------------------------------------------
    def body(self, vlogits, S, rows, sharded):
        """The sampler stage over the verify logits ``[1, 1, rows, V_x]`` (each chip's ``V / tp`` slice when
        ``sharded``, else the gathered vocabulary): per 32-row chunk, the candidates (top-k per chip then the
        TP gather, or top-k per vocabulary chunk then the concat), their global indices, the chunk's seeds and
        one ``ttnn.sampling`` into the chunk's persistent draw output."""
        R = SPEC_SAMPLED_ROW_TILE
        self.ops = 0
        V_x = int(vlogits.shape[-1])
        for c in range(S["chunks"]):
            lo = c * R
            n = min(R, int(rows) - lo)
            # Every intermediate the chunk makes is freed before the next chunk; a ttnn op that hands back its
            # input under a new wrapper (pad / slice with nothing to do) must not free the caller's logits.
            owned = []

            def _own(t):
                if t is not vlogits and not _same_buffer(t, vlogits) and all(not _same_buffer(t, o) for o in owned):
                    owned.append(t)
                return t

            x = vlogits
            if S["chunks"] > 1:
                x = _own(ttnn.slice(vlogits, [0, 0, lo, 0], [1, 1, lo + n, V_x]))
            if n < R:
                x = _own(ttnn.pad(x, [(0, 0), (0, 0), (0, R - n), (0, 0)], value=0.0))
            values, indices, width, vocab_slice = self._candidates(x, V_x, sharded)
            for t in owned:
                t.deallocate(True)
            idx_i32 = indices if indices.dtype == ttnn.int32 else ttnn.typecast(indices, ttnn.int32)
            if idx_i32 is not indices:
                indices.deallocate(True)
            gidx = ttnn.add(
                self.offsets_for(width, vocab_slice, self.max_top_k), idx_i32, dtype=ttnn.uint32
            )  # [1, 1, 32, width]
            idx_i32.deallocate(True)
            adjusted = self._tiebreak_top1(values, gidx, S["gcol"][c])
            values.deallocate(True)
            values = adjusted
            gidx_rm = ttnn.untilize(gidx, use_multicore=True)
            gidx.deallocate(True)
            ttnn.manual_seed(seeds=S["seeds"][c], user_ids=self.user_ids, sub_core_grids=self.sub_core_grids)
            ttnn.sampling(
                values,
                gidx_rm,
                k=S["k"][c],
                p=S["p"][c],
                temp=S["temp"][c],
                sub_core_grids=self.sub_core_grids,
                output_tensor=S["sidx"][c],
            )
            values.deallocate(True)
            gidx_rm.deallocate(True)
            self.ops += 6

    def _tiebreak_top1(self, values, gidx, gcol):
        """The plain sampler's top-1 tie-break (``TTSampling._adjust_values_for_tiebreak``) on the record's
        candidates ``values`` [1, 1, 32, W] bf16 TILE and their global indices ``gidx`` [1, 1, 32, W] uint32 TILE:
        on the packed rows whose ``k == 1`` (``gcol`` [1, 1, 32, 1] bf16, 1.0 there) the single lowest-global-index
        candidate among the tied maxima is boosted just past the tie, so ``ttnn.sampling``'s argmax picks it
        whatever order the top-k network gave the ties, run to run and slot to slot; every other row's values
        are unchanged (a boost of 0). The reduce and the index arithmetic stay in int32 (the FPU truncates
        float32 indices above 2**11; see the plain sampler). Returns the adjusted values; ``values`` is the
        caller's to free."""
        scg = self.sub_core_grids
        maxv = ttnn.max(values, dim=3, keepdim=True, sub_core_grids=scg)  # [1, 1, 32, 1] bf16
        is_max = ttnn.eq(values, maxv, sub_core_grids=scg)
        not_max = ttnn.lt(values, maxv, sub_core_grids=scg)
        abs_max = ttnn.abs(maxv, sub_core_grids=scg)
        maxv.deallocate(True)
        delta_scaled = ttnn.multiply(abs_max, TIEBREAK_DELTA_SCALE, sub_core_grids=scg)
        abs_max.deallocate(True)
        delta = ttnn.add(delta_scaled, TIEBREAK_DELTA_FLOOR, sub_core_grids=scg)  # >= 2 bf16 ULP of the row max
        delta_scaled.deallocate(True)
        idx = ttnn.typecast(gidx, ttnn.int32, sub_core_grids=scg)
        offset = ttnn.multiply(not_max, TIEBREAK_INDEX_SENTINEL, sub_core_grids=scg)  # exact in bf16 (2**24)
        not_max.deallocate(True)
        offset_i32 = ttnn.typecast(offset, ttnn.int32, sub_core_grids=scg)
        offset.deallocate(True)
        masked_idx = ttnn.add(idx, offset_i32, sub_core_grids=scg)  # the non-maxima above every real index
        offset_i32.deallocate(True)
        lowest = ttnn.min(masked_idx, dim=3, keepdim=True, sub_core_grids=scg)  # [1, 1, 32, 1] int32
        masked_idx.deallocate(True)
        is_low_i32 = ttnn.eq(idx, lowest, sub_core_grids=scg)
        idx.deallocate(True)
        lowest.deallocate(True)
        is_low = ttnn.typecast(is_low_i32, ttnn.bfloat16, sub_core_grids=scg)
        is_low_i32.deallocate(True)
        winner = ttnn.multiply(is_max, is_low, sub_core_grids=scg)  # 1.0 at exactly one candidate of a row
        is_max.deallocate(True)
        is_low.deallocate(True)
        gated = ttnn.multiply(winner, gcol, sub_core_grids=scg)  # the k == 1 rows only
        winner.deallocate(True)
        boost = ttnn.multiply(gated, delta, sub_core_grids=scg)
        gated.deallocate(True)
        delta.deallocate(True)
        adjusted = ttnn.add(values, boost, sub_core_grids=scg)
        boost.deallocate(True)
        self.ops += 16
        return adjusted

    def _candidates(self, x, V_x, sharded):
        """``(values [1,1,32,W] bf16 TILE, indices [1,1,32,W], W, slice)``: the top-k candidates of every row."""
        k = self.max_top_k
        if sharded and self.tp > 1:
            values, indices = ttnn.topk(x, k=k, dim=-1, stable=False)  # [1, 1, 32, k] per chip
            idx_i32 = ttnn.typecast(indices, ttnn.int32)
            indices.deallocate(True)
            values_g = ccl_allgather(values, self.mesh_config, self.ccl_manager, dim=3)  # [1, 1, 32, k * tp]
            idx_g = ccl_allgather(idx_i32, self.mesh_config, self.ccl_manager, dim=3)
            values.deallocate(True)
            idx_i32.deallocate(True)
            self.ops += 4
            return values_g, idx_g, k * self.tp, V_x
        splits = 1
        while V_x // splits > SPEC_SAMPLED_TOPK_MAX_WIDTH:
            splits *= 2
        if V_x % splits or (V_x // splits) % SPEC_SAMPLED_ROW_TILE:
            raise ValueError(f"the verify logits' {V_x} columns cannot be cut into {splits} tile-aligned top-k chunks")
        if splits == 1:
            values, indices = ttnn.topk(x, k=k, dim=-1, stable=False)
            self.ops += 1
            return values, indices, k, V_x
        parts = ttnn.split(x, V_x // splits, dim=3)
        vals, idxs = [], []
        for part in parts:
            v, i = ttnn.topk(part, k=k, dim=-1, stable=False)
            vals.append(v)
            idxs.append(i)
            part.deallocate(True)
        values = ttnn.concat(vals, dim=3)  # [1, 1, 32, k * splits]
        indices = ttnn.concat(idxs, dim=3)
        for t in vals + idxs:
            t.deallocate(True)
        self.ops += 2 * splits + 2
        return values, indices, k * splits, V_x // splits

    # -- per replay ------------------------------------------------------------------------------------------
    def _column_plans(self, plan):
        """The per-column ``SamplingRowsPlan`` list of ``plan``: a ``LaneSamplingPlan`` of ``lanes`` columns on
        the lane engine, one ``SamplingRowsPlan`` otherwise."""
        if self.lanes > 1:
            if not isinstance(plan, LaneSamplingPlan) or plan.lanes != self.lanes:
                raise ValueError(
                    f"the lane engine's sampler takes a LaneSamplingPlan of {self.lanes} columns, got {plan!r}"
                )
            return list(plan.plans)
        if isinstance(plan, LaneSamplingPlan):
            raise ValueError("a LaneSamplingPlan on one column: serving_sampling_rows makes the plain plan")
        return [plan]

    def upload(self, S, plan):
        """The plan's parameter rows (chunks whose rows changed on any column) and seeds (every replay with a
        sampled row); on the lane engine each upload is the lane frame of every column's rows."""
        plans = self._column_plans(plan)
        if any(p.chunks != S["chunks"] for p in plans):
            raise ValueError(f"the plan has {[p.chunks for p in plans]} chunks, the record's sampler {S['chunks']}")
        R = SPEC_SAMPLED_ROW_TILE
        for c in range(S["chunks"]):
            key = (S["serial"], c)
            params = tuple(p.params_key(c) for p in plans)
            cols = [p.chunk(c) for p in plans]  # (k, p, temp, seeds) of chunk c per column
            if self._uploaded.get(key) != params:
                for which, dev, dtype, hdtype in (
                    (0, S["k"][c], ttnn.uint32, torch.int32),
                    (1, S["p"][c], ttnn.bfloat16, torch.float32),
                    (2, S["temp"][c], ttnn.bfloat16, torch.float32),
                ):
                    frame = torch.tensor([v for col in cols for v in col[which]], dtype=hdtype)
                    h = self._frame_host(frame, dtype, 0)
                    ttnn.copy_host_to_device_tensor(h, dev)
                    h.deallocate(True)
                gcol = torch.cat([_top1_mask(col[0]) for col in cols], dim=2)
                h = self._frame_host(gcol, ttnn.bfloat16, 2, layout=ttnn.TILE_LAYOUT)
                ttnn.copy_host_to_device_tensor(h, S["gcol"][c])
                h.deallocate(True)
                self._uploaded[key] = params
            if any(any(p.sampled[c * R : (c + 1) * R]) for p in plans):
                # every column's seed rows go up together: an idle column's rows carry the SKIP sentinel, so its
                # cores' PRNG stays untouched exactly as a greedy row's does
                frame = torch.tensor([v for col in cols for v in col[3]], dtype=torch.int64)
                h = self._frame_host(frame, ttnn.uint32, 0)
                ttnn.copy_host_to_device_tensor(h, S["seeds"][c])
                h.deallocate(True)

    def read(self, S, rows):
        """The sampler's draws of the first ``rows`` packed rows, as a host list (chip 0's replica; one column)."""
        if self.lanes > 1:
            raise ValueError("read() is the one-column readback; the lane engine reads every column (read_frame)")
        out = []
        for c in range(S["chunks"]):
            t = S["sidx"][c]
            host = ttnn.to_torch(ttnn.get_device_tensors(t)[0]) if self.tp > 1 else ttnn.to_torch(t)
            out.extend(int(v) for v in host.reshape(-1).tolist())
        return out[: int(rows)]

    def read_frame(self, S):
        """``[lanes][chunks * 32]`` host draws of every column (each lane's first chip, ``_srv_lane_read_u32``)."""
        if self.lanes <= 1:
            raise ValueError("read_frame() is the lane engine's readback; one column reads with read()")
        out = [[] for _ in range(self.lanes)]
        for c in range(S["chunks"]):
            per_lane = self.decoder._srv_lane_read_u32(S["sidx"][c], SPEC_SAMPLED_ROW_TILE)  # [lanes, 32]
            for lane in range(self.lanes):
                out[lane].extend(int(v) for v in per_lane[lane].reshape(-1).tolist())
        return out


def _top1_mask(k):
    """``[1, 1, 32, 1]`` bf16: 1.0 on the packed rows whose top-k is 1 (the plain sampler's argmax users)."""
    return (torch.as_tensor(list(k), dtype=torch.int64).reshape(1, 1, -1, 1) == 1).to(torch.bfloat16)


def _same_buffer(a, b):
    """Whether two device tensors share one buffer (a ttnn op that returns its input under a new wrapper).
    A device tensor answers ``buffer_address``; only a stand-in without it falls back to identity (a failing
    address lookup on a real tensor raises, because a wrong answer here frees the caller's logits)."""
    try:
        return a.buffer_address() == b.buffer_address()
    except AttributeError:
        return a is b
