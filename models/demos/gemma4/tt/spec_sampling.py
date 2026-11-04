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

Not available on the lane engine (per-lane sampling groups) or on a target
without the device top-k sampler (fractured meshes): ``serving_prepare``
refuses the knob there and the adapter reports the capability False.
"""

import os
import secrets
from dataclasses import dataclass

import torch

import ttnn
from models.common.sampling.generator import DEVICE_SEED_MAX, _hash_request_seed_to_device_seed, format_sampling_params
from models.common.sampling.sampling_params import SamplingParams
from models.demos.gemma4.tt.ccl import ccl_allgather

SPEC_SAMPLED_VERIFY_ENV = "GEMMA4_SPEC_SAMPLED_VERIFY"
SPEC_SAMPLED_ROW_TILE = 32
# ``ttnn.topk`` takes at most this many columns per call (models/common/sampling/tt_sampling.py TOPK_MAX_WIDTH):
# gathered logits wider than it are cut into equal power-of-two chunks, each topk'd on its own.
SPEC_SAMPLED_TOPK_MAX_WIDTH = 64 * 1024
SPEC_SAMPLED_MAX_TOP_K = 32
# The seed of a packed row whose draw is never read (greedy or padding rows): any valid device seed.
SPEC_SAMPLED_IDLE_SEED = 1


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
        self.mesh_config = getattr(target, "mesh_config", None)
        self.ccl_manager = getattr(target, "ccl_manager", None)
        sampling = getattr(target, "sampling", None)
        tt_sampling = getattr(sampling, "tt_sampling", None)
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
        self.sub_core_grids = getattr(tt_sampling, "_sampling_sub_core_grids", None)
        self.invalid_mask = getattr(tt_sampling, "tt_invalid_vocab_mask", None)
        self.max_top_k = min(SPEC_SAMPLED_MAX_TOP_K, int(getattr(tt_sampling, "max_top_k", SPEC_SAMPLED_MAX_TOP_K)))
        self._offsets = {}  # (width, slice) -> [1, 1, 32, width] int32 TILE: block d of 32 columns holds d * slice
        self._uploaded = {}  # (id(S), chunk) -> params_key last uploaded
        self.ops = 0  # device programs the last body issued (diagnostics)

    # -- persistent buffers ----------------------------------------------------------------------------------
    def _from_torch(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=True):
        return self.decoder._pv_from_torch(t, dtype, layout=layout, device=device)

    def alloc(self, B, P):
        """The shape's per-chunk parameter rows, seed rows and draw outputs (``_srv_alloc_shape``)."""
        R = SPEC_SAMPLED_ROW_TILE
        chunks = serving_chunks(B, P)
        greedy = greedy_sampling_rows((B, P - 1))
        S = {"chunks": chunks, "k": [], "p": [], "temp": [], "seeds": [], "sidx": []}
        for c in range(chunks):
            k, p, temp, seeds = greedy.chunk(c)
            S["k"].append(self._from_torch(torch.tensor(k, dtype=torch.int32), ttnn.uint32))
            S["p"].append(self._from_torch(torch.tensor(p, dtype=torch.float32), ttnn.bfloat16))
            S["temp"].append(self._from_torch(torch.tensor(temp, dtype=torch.float32), ttnn.bfloat16))
            S["seeds"].append(self._from_torch(torch.tensor(seeds, dtype=torch.int64), ttnn.uint32))
            S["sidx"].append(self._from_torch(torch.zeros(1, 1, 1, R, dtype=torch.int64), ttnn.uint32))
        return S

    def offsets_for(self, width, vocab_slice):
        """Persistent ``[1, 1, 32, width]`` int32 TILE: the vocabulary offset of each 32-column candidate block."""
        key = (int(width), int(vocab_slice))
        t = self._offsets.get(key)
        if t is None:
            if getattr(self.decoder, "_srv_frozen", False):
                raise RuntimeError(
                    f"no sampler offsets for width {width} after serving_prepare: the eager bodies make them; a replay "
                    "never allocates"
                )
            R = SPEC_SAMPLED_ROW_TILE
            host = torch.zeros(1, 1, R, int(width), dtype=torch.int32)
            for d in range(int(width) // R):
                host[:, :, :, d * R : (d + 1) * R] = d * int(vocab_slice)
            t = self._from_torch(host, ttnn.int32, layout=ttnn.TILE_LAYOUT)
            self._offsets[key] = t
        return t

    def release(self, shape_bufs):
        for S in shape_bufs:
            if S is None:
                continue
            for key in ("k", "p", "temp", "seeds", "sidx"):
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
            if self.invalid_mask is not None and tuple(int(d) for d in self.invalid_mask.shape) == (1, 1, R, V_x):
                x = _own(ttnn.add(x, self.invalid_mask))
            values, indices, width, vocab_slice = self._candidates(x, V_x, sharded)
            for t in owned:
                t.deallocate(True)
            idx_i32 = indices if indices.dtype == ttnn.int32 else ttnn.typecast(indices, ttnn.int32)
            if idx_i32 is not indices:
                indices.deallocate(True)
            gidx = ttnn.add(self.offsets_for(width, vocab_slice), idx_i32, dtype=ttnn.uint32)  # [1, 1, 32, width]
            idx_i32.deallocate(True)
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
    def upload(self, S, plan):
        """The plan's parameter rows (chunks whose rows changed) and seeds (every replay with a sampled row)."""
        if plan.chunks != S["chunks"]:
            raise ValueError(f"the plan has {plan.chunks} chunks, the record's sampler {S['chunks']}")
        for c in range(S["chunks"]):
            key = (id(S), c)
            params = plan.params_key(c)
            k, p, temp, seeds = plan.chunk(c)
            if self._uploaded.get(key) != params:
                for host, dev, dtype, hdtype in (
                    (k, S["k"][c], ttnn.uint32, torch.int32),
                    (p, S["p"][c], ttnn.bfloat16, torch.float32),
                    (temp, S["temp"][c], ttnn.bfloat16, torch.float32),
                ):
                    h = self._from_torch(torch.tensor(host, dtype=hdtype), dtype, device=False)
                    ttnn.copy_host_to_device_tensor(h, dev)
                    h.deallocate(True)
                self._uploaded[key] = params
            if any(plan.sampled[c * SPEC_SAMPLED_ROW_TILE : (c + 1) * SPEC_SAMPLED_ROW_TILE]):
                h = self._from_torch(torch.tensor(seeds, dtype=torch.int64), ttnn.uint32, device=False)
                ttnn.copy_host_to_device_tensor(h, S["seeds"][c])
                h.deallocate(True)

    def read(self, S, rows):
        """The sampler's draws of the first ``rows`` packed rows, as a host list (chip 0's replica)."""
        out = []
        for c in range(S["chunks"]):
            t = S["sidx"][c]
            host = ttnn.to_torch(ttnn.get_device_tensors(t)[0]) if self.tp > 1 else ttnn.to_torch(t)
            out.extend(int(v) for v in host.reshape(-1).tolist())
        return out[: int(rows)]


def _same_buffer(a, b):
    """Whether two device tensors share one buffer (a ttnn op that returns its input under a new wrapper)."""
    try:
        return a.buffer_address() == b.buffer_address()
    except Exception:
        return a is b
