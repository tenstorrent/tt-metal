# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-sampled verify ids (vllm-tt-plugin ``spec_ids_are_sampled``) for Qwen36MTPContractForCausalLM.

On a step the plugin samples on the device it sends the rows' ``sampling_params`` with the verify. For a sampled row
(temperature > 0, top-k above 1) the ``argmax_ids`` answer then carries one draw per verify column instead of the
target argmax: the token after input column j drawn from the target distribution at that column, under the row's own
temperature, top-k, top-p and seed. The plugin's unchanged id walk (accept draft j while it equals draw j, commit the
draw at the first mismatch, or the draw at column ``num_valid`` as the bonus) is then exact rejection sampling for the
point-mass drafts of the MTP drafter: draft j is accepted with probability p_j(d_j), and a rejected column commits a
draw of p_j conditioned on differing from d_j, which is the normalized residual (p_j - 1[d_j])+.

A greedy row, a row with top-k 1 and a padding row keep the verify's argmax ids: their answer is the argmax, which the
device sampler would return too (ttnn.argmax takes the lowest index among exact ties, as the plain sampler's top-1
tie-break does), so a step without a sampled row runs no sampler at all.

Seeds: a seeded row's column j draws with the device seed that a plain decode at that input position uses,
``_hash_request_seed_to_device_seed(seed, position + 1)``, so the draw at a position does not depend on how the
positions were grouped into verifies. An unseeded row's column takes fresh entropy. A row that draws nothing is
matched to no core (``manual_seed`` user id NO_CORE) and carries the device SKIP seed, so its core's random state stays
untouched. A chained verify (spec_chain) knows its positions only on the device: its plan carries no seeds
(``chained_sampling_plan``), and the chain program writes each drawing row's seed, picked by the predecessor's
accepted count from candidates the host hashes for every position the column can take.

Device graph (``VerifySampler.body``, a trace of its own after the verify trace): each device's vocabulary shard of
the verify's logits [1, 1, rows, V / shards] (at most 32 rows, one core each) is padded to a power-of-two width (32768
at V / shards = 31040) and gives its top-32 in one stable ``ttnn.topk`` call, lowest local id first among equal
values. The candidates' global ids are the local ids plus the shard's first vocabulary id; ``ttnn.all_gather`` along
the vocabulary puts the shards' candidates side by side in shard order (8 x 32 = 256 at TP = 8). Then
``ttnn.manual_seed`` and one ``ttnn.sampling`` with the per-row parameters write one draw per row. Top-k above 32 never
reaches it: the plugin samples such a row on the host (``max_device_top_k``).

Tie order: among equal logits the lowest token id comes first, as ``ttnn.argmax`` takes it for a greedy row.
``ttnn.sampling`` ranks equal values by their position among the candidates, and the candidates of equal value sit in
ascending id order (within a shard by the stable top-k, across shards by the shard order). So the sampler draws from
the vocabulary's top-k under the order (value descending, id ascending), the order of a stable torch sort of the whole
row: any token of that top-32 has fewer than 32 tokens before it in its own shard, so it is among its shard's
candidates. The stable top-k needs the power-of-two width for its multi-core path; at 31040 columns it runs on one
core.

A verify row bucket of b users (Qwen36Model._verify_bucket_body) has its own sampler trace: the same graph over the
bucket's b*T rows of the shard, whose pad fills the rows up to B*T, so the seeds, the per-row parameters and the
draws buffer are the full-width ones.
"""

import secrets
from dataclasses import dataclass, replace
from typing import NamedTuple

import torch

import ttnn
from models.common.sampling.generator import (
    DEVICE_SEED_MAX,
    MAX_UINT32,
    _hash_request_seed_to_device_seed,
    format_sampling_params,
)
from models.common.sampling.sampling_params import SamplingParams
from models.demos.blackhole.qwen36.tt.spec_argmax_fused import all_gather_dim3

SAMPLER_ROWS = 32  # one ttnn.sampling call: a row per core, B * T <= 32 verify rows
SAMPLER_TOP_K = 32  # candidates per vocabulary chunk; the plugin keeps a row with top-k above it on the host
# Widths of the stable ttnn.topk's multi-core path: a power of two, at least 8192 and below 65535. Its single-core
# path, which any other width takes, needs milliseconds for a 31040-column shard.
TOPK_MIN_WIDTH = 8 * 1024
TOPK_MAX_WIDTH = 32 * 1024
# Pad columns and rows of the top-k input: below any logit, so a real column always outranks a pad column. Finite:
# the stable top-k sorts bfloat16 values packed with their 16-bit index.
PAD_LOGIT = -1e30
SKIP_SEED = MAX_UINT32  # manual_seed leaves the core's random state untouched
NO_CORE = MAX_UINT32  # a manual_seed user id no core matches: the entry seeds nothing


def _row_value(value, row, default):
    """``value`` for ``row``: a scalar as is, a per-row sequence at ``row``, ``default`` past its end or for None."""
    if value is None:
        return default
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return value.item()
    if int(row) >= len(value):
        return default
    item = value[int(row)]
    if isinstance(item, torch.Tensor):
        item = item.item()
    return default if item is None else item


def row_sampling(sampling_params, row):
    """``(temperature, top_k, top_p, seed)`` the step's ``sampling_params`` give ``row``; seed None when unseeded."""
    temperature = float(_row_value(getattr(sampling_params, "temperature", None), row, 0.0))
    top_k = int(_row_value(getattr(sampling_params, "top_k", None), row, 1))
    top_p = float(_row_value(getattr(sampling_params, "top_p", None), row, 1.0))
    seed = _row_value(getattr(sampling_params, "seed", None), row, None)
    seed = None if seed is None or int(seed) < 0 else int(seed)
    return temperature, top_k, top_p, seed


def _fresh_seed():
    return secrets.randbelow(DEVICE_SEED_MAX) + 1


@dataclass
class SamplingPlan:
    """The sampler's inputs for one verify of B users by T columns, row u*T + j: the device encoding of top-k, top-p and
    inverse temperature (``format_sampling_params``), the device seed (None on a chained verify, whose chain program
    writes the seeds), and whether the row's draw replaces its argmax id."""

    k: list
    p: list
    temp: list
    seeds: list | None
    sampled: list

    @property
    def any_sampled(self):
        return any(self.sampled)

    @property
    def user_ids(self):
        """The manual_seed user id of each row: the row's own core for a drawing row, NO_CORE for the others."""
        return [r if s else NO_CORE for r, s in enumerate(self.sampled)]


def drawn_users(plan, B, T):
    """Per user, whether ``plan`` (or None) draws the user's columns."""
    return [bool(plan is not None and plan.sampled[u * T]) for u in range(B)]


def sampling_plan(sampling_params, positions, T, fresh_seed=_fresh_seed):
    """The plan of a verify of ``len(positions)`` users by T columns. ``positions[u]`` is the input position of user
    u's column 0, negative for a padding row. Every column of a row draws when the row's temperature is above 0 and
    its top-k above 1; the columns past a row's valid drafts are masked afterwards like its argmax ids."""
    B = len(positions)
    R = B * T
    if R > SAMPLER_ROWS:
        raise ValueError(f"{B} users x {T} columns exceed the sampler's {SAMPLER_ROWS} rows")
    temperature = [0.0] * R
    top_k = [1] * R
    top_p = [1.0] * R
    seeds = [SKIP_SEED] * R
    sampled = [False] * R
    for u in range(B):
        position = int(positions[u])
        temp, k, p, seed = row_sampling(sampling_params, u)
        if position < 0 or temp <= 0.0 or k == 1:
            continue
        if not 1 <= k <= SAMPLER_TOP_K:
            raise ValueError(
                f"row {u}: top_k {k} outside [1, {SAMPLER_TOP_K}]; such a row is sampled on the host (max_device_top_k)"
            )
        for j in range(T):
            r = u * T + j
            temperature[r], top_k[r], top_p[r] = temp, k, p
            seeds[r] = _hash_request_seed_to_device_seed(seed, position + j + 1) if seed is not None else fresh_seed()
            sampled[r] = True
    encoded = format_sampling_params(
        SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p, seed=None), SAMPLER_ROWS
    )
    return SamplingPlan(
        k=[int(v) for v in encoded.top_k[:R]],
        p=[float(v) for v in encoded.top_p[:R]],
        temp=[float(v) for v in encoded.temperature[:R]],
        seeds=seeds,
        sampled=sampled,
    )


def chained_sampling_plan(sampling_params, p0_prev, T, fresh_seed=_fresh_seed):
    """The plan of a chained verify of ``len(p0_prev)`` users by T columns, whose column-0 positions the chain program
    computes on the device: p0_prev[u] + c for the c tokens the predecessor committed for user u, a negative
    p0_prev[u] for a user the verify does not run. Returns (plan, candidates). The plan's top-k, top-p, temperature
    and drawing rows do not depend on c; its seeds are None. ``candidates`` float32 [T * B, T]: row (c - 1) * B + u
    holds user u's seeds if the predecessor committed c tokens, for a seeded user the seeds ``sampling_plan`` gives
    its columns at p0_prev[u] + c, for an unseeded drawing user the same fresh seed per column at every c, 0 for a
    user that draws nothing. Every seed is below 2**24, so float32 holds it exactly."""
    B = len(p0_prev)
    first = sampling_plan(sampling_params, [int(p) + 1 if int(p) >= 0 else -1 for p in p0_prev], T, fresh_seed)
    candidates = torch.zeros(T * B, T, dtype=torch.float64)
    for u in range(B):
        if not first.sampled[u * T]:
            continue
        seed = row_sampling(sampling_params, u)[3]
        for c in range(1, T + 1):
            for j in range(T):
                candidates[(c - 1) * B + u, j] = (
                    first.seeds[u * T + j]
                    if seed is None
                    else _hash_request_seed_to_device_seed(seed, int(p0_prev[u]) + c + j + 1)
                )
    if bool((candidates >= 2**24).any()):
        raise ValueError("a chained verify's device seed is at or above 2**24")
    return replace(first, seeds=None), candidates.to(torch.float32)


def with_draws(ids, draws, plan):
    """``ids`` [B, T] with each sampled row's draw in place of its argmax id, as int64."""
    # The verify's argmax reads back as torch.uint32, which has no index_put.
    ids = torch.as_tensor(ids).to(torch.int64, copy=True)
    B, T = ids.shape
    flat = ids.reshape(-1)
    draws = torch.as_tensor(draws).reshape(-1).to(torch.int64)
    rows = torch.tensor(plan.sampled[: B * T], dtype=torch.bool)
    flat[rows] = draws[: B * T][rows]
    return flat.reshape(B, T)


def _pow2_at_least(n):
    p = 1
    while p < n:
        p *= 2
    return p


def shard_chunks(width):
    """Fewest equal tile-aligned chunks, of at most TOPK_MAX_WIDTH columns each, of one device's vocabulary shard of
    ``width`` columns: each chunk is one row of the sampler's stable top-k."""
    chunks = -(-int(width) // TOPK_MAX_WIDTH)
    while width % chunks or (width // chunks) % ttnn.TILE_SIZE:
        chunks += 1
        if width // chunks < SAMPLER_TOP_K:
            raise ValueError(f"a vocabulary shard of {width} does not split into tile-aligned top-k chunks")
    return chunks


class VerifySampler:
    """The device half: persistent per-row inputs, the draw output, and the trace-capturable ``body``. ``shards``: the
    number of vocabulary shards of the logits over a 1 x shards mesh (the vocab-sharded LM head's), 1 when every device
    holds the whole vocabulary. ``gather`` all-gathers a [1, 1, rows, n] TILE tensor along dim 3 in mesh order, the
    order of the vocabulary shards."""

    def __init__(self, mesh, B, T, vocab, shards=1, gather=all_gather_dim3):
        if B * T > SAMPLER_ROWS:
            raise ValueError(f"{B} users x {T} columns exceed the sampler's {SAMPLER_ROWS} rows")
        self.mesh, self.rows, self.vocab, self.shards, self.gather = mesh, B * T, int(vocab), int(shards), gather
        if self.vocab % self.shards:
            raise ValueError(f"a vocabulary of {self.vocab} does not split into {self.shards} shards")
        self.width = self.vocab // self.shards
        self.chunks = shard_chunks(self.width)
        self.chunk_width = self.width // self.chunks
        self.topk_width = _pow2_at_least(max(self.chunk_width, TOPK_MIN_WIDTH))
        self.candidates_per_row = self.shards * self.chunks * SAMPLER_TOP_K
        tiles = self.candidates_per_row // ttnn.TILE_SIZE
        if tiles & (tiles - 1):
            raise ValueError(f"{self.candidates_per_row} candidates per row are not a power-of-two count of tiles")
        idle = sampling_plan(None, [-1] * B, T)
        self.k = self._device(torch.tensor(idle.k, dtype=torch.int32), ttnn.uint32)
        self.p = self._device(torch.tensor(idle.p), ttnn.bfloat16)
        self.temp = self._device(torch.tensor(idle.temp), ttnn.bfloat16)
        # The chain program (spec_chain.ChainInputs) writes the seeds of a chained verify.
        self.seeds = self._device(torch.tensor(idle.seeds, dtype=torch.int64), ttnn.uint32)
        self._staged = {}  # name -> host tensor its device buffer holds (stage)
        self.user_ids = self._device(torch.tensor(idle.user_ids, dtype=torch.int64), ttnn.uint32)
        self.draws = self._device(torch.zeros(1, 1, 1, self.rows, dtype=torch.int32), ttnn.uint32)
        self.offsets = self._offsets()

    def _offsets(self):
        """int32 TILE [1, 1, B*T, chunks * 32] per device: in chunk c's 32 candidate columns, the first vocabulary id of
        chunk c of the device's shard."""
        first = torch.arange(self.shards * self.chunks, dtype=torch.int32) * self.chunk_width
        cols = first.reshape(self.shards, self.chunks).repeat_interleave(SAMPLER_TOP_K, dim=1)
        offsets = cols.reshape(1, self.shards, 1, -1).expand(1, self.shards, self.rows, -1).contiguous()
        mapper = ttnn.ShardTensorToMesh(self.mesh, dim=1) if self.shards > 1 else ttnn.ReplicateTensorToMesh(self.mesh)
        host = ttnn.from_torch(offsets, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        return ttnn.to_device(host, self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _host(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))

    def _device(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.to_device(self._host(t, dtype, layout), self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def stage(self, plan):
        """Host -> device copies of the plan's parameters, user ids and seeds (addresses preserved); a chained plan has
        no seeds (the chain program wrote them). A parameter equal to the one staged last is not copied again: top-k,
        top-p, temperature and user ids change only with the rows' requests, the seeds every step."""
        staged = [
            ("k", torch.tensor(plan.k, dtype=torch.int32), self.k, ttnn.uint32),
            ("p", torch.tensor(plan.p), self.p, ttnn.bfloat16),
            ("temp", torch.tensor(plan.temp), self.temp, ttnn.bfloat16),
            ("user_ids", torch.tensor(plan.user_ids, dtype=torch.int64), self.user_ids, ttnn.uint32),
        ]
        if plan.seeds is not None:
            staged.append(("seeds", torch.tensor(plan.seeds, dtype=torch.int64), self.seeds, ttnn.uint32))
        for name, host, dev, dtype in staged:
            held = self._staged.get(name)
            if held is not None and torch.equal(held, host):
                continue
            ttnn.copy_host_to_device_tensor(self._host(host, dtype), dev)
            self._staged[name] = host

    def seeds_written(self):
        """The chain program wrote the seeds buffer: the next host plan copies its seeds again."""
        self._staged.pop("seeds", None)

    def candidates(self, logits):
        """The sampler's candidates of ``logits`` bfloat16 TILE, each device's vocabulary shard [1, 1, rows, vocab /
        shards] of the first ``rows`` <= B*T rows, or the whole vocabulary [1, 1, rows, vocab] on every device:
        (values [1, 1, B*T, shards * chunks * 32] bfloat16 TILE, global ids, the same shape, uint32 ROW_MAJOR). Shard d's
        chunk c holds its top-32 at columns 32 (d * chunks + c) .. + 31, in value order, equal values lowest id first.
        The rows past ``rows`` are the pad's: every candidate PAD_LOGIT."""
        rows, width = int(logits.shape[2]), int(logits.shape[3])
        if not 1 <= rows <= self.rows:
            raise ValueError(f"the sampler draws {self.rows} rows; logits of {rows} rows are not a prefix of them")
        x, owned = logits, None
        if width == self.vocab and self.shards > 1:
            x = owned = ttnn.mesh_partition(logits, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        elif width != self.width:
            raise ValueError(f"logits of width {width}: the sampler takes {self.width} (one of {self.shards} shards)")
        if self.chunks > 1:
            parts = ttnn.split(x, self.chunk_width, dim=3)
            stacked = ttnn.concat(parts, dim=1)
            for part in parts:
                ttnn.deallocate(part)
            if owned is not None:
                ttnn.deallocate(owned)
            x = owned = stacked
        padded = x
        if rows < self.rows or self.chunk_width < self.topk_width:
            pad = [(0, 0), (0, 0), (0, self.rows - rows), (0, self.topk_width - self.chunk_width)]
            padded = ttnn.pad(x, pad, value=PAD_LOGIT)
            if owned is not None:
                ttnn.deallocate(owned)
        v, i = ttnn.topk(padded, k=SAMPLER_TOP_K, dim=-1, largest=True, sorted=True, stable=True)
        if padded is not logits:
            ttnn.deallocate(padded)
        idx = ttnn.typecast(i, ttnn.int32)
        ttnn.deallocate(i)
        if self.chunks > 1:
            v, idx = (self._unstack(t) for t in (v, idx))
        gidx = ttnn.add(self.offsets, idx, dtype=ttnn.uint32)
        ttnn.deallocate(idx)
        if self.shards > 1:
            local = (v, gidx)
            v, gidx = (self.gather(t) for t in local)
            for t in local:
                ttnn.deallocate(t)
        gidx_rm = ttnn.untilize(gidx, use_multicore=True)
        ttnn.deallocate(gidx)
        return v, gidx_rm

    def _unstack(self, t):
        """[1, chunks, B*T, 32] -> [1, 1, B*T, chunks * 32], chunk c at columns 32c .. 32c + 31; frees ``t``."""
        parts = [ttnn.slice(t, (0, c, 0, 0), (1, c + 1, self.rows, SAMPLER_TOP_K)) for c in range(self.chunks)]
        ttnn.deallocate(t)
        out = ttnn.concat(parts, dim=3)
        for part in parts:
            ttnn.deallocate(part)
        return out

    def draw(self, vals, gidx_rm):
        """Seed the drawing rows' cores and write one draw per row of the candidates into ``draws``; frees both."""
        ttnn.manual_seed(seeds=self.seeds, user_ids=self.user_ids)
        ttnn.sampling(vals, gidx_rm, k=self.k, p=self.p, temp=self.temp, output_tensor=self.draws)
        ttnn.deallocate(vals)
        ttnn.deallocate(gidx_rm)
        return self.draws

    def body(self, logits):
        """``logits`` as ``candidates`` takes them, of all B*T rows or of the first ``rows`` (a verify row bucket's).
        Writes one draw per row of B*T into ``draws``. ttnn.sampling draws each row from its own candidates and its
        core's seed alone, and the top-k ranks each row on its own, so the first rows draw what they draw beside the
        full-width logits."""
        return self.draw(*self.candidates(logits))

    def read(self):
        """The draws of the last replay, one per row, from the device-0 replica."""
        out = ttnn.to_torch(ttnn.get_device_tensors(self.draws)[0]).reshape(-1)
        return out[: self.rows].to(torch.int64)


class SampledLogprobs(NamedTuple):
    """The logprob payload of a device-sampled step: each row's sampled token's raw logprob and its rank, the number
    of vocabulary scores at or above the sampled token's (vllm-tt-plugin ``build_device_logprobs``)."""

    sampled_logprobs: torch.Tensor  # [B] float32
    selected_token_ranks: torch.Tensor  # [B] int32


def enables_logprobs(sampling_params):
    """Whether any row of the step's ``sampling_params`` asks for logprobs."""
    enabled = getattr(sampling_params, "enable_log_probs", None)
    if enabled is None:
        return False
    if isinstance(enabled, (bool, int)):
        return bool(enabled)
    return bool(torch.as_tensor(enabled).any())


def sampled_token_logprobs(logits, tokens):
    """``SampledLogprobs`` of ``tokens`` [B] under the raw ``logits`` [B, V] (before temperature, as vLLM's default
    raw logprobs)."""
    logits = torch.as_tensor(logits).to(torch.float32)
    index = torch.as_tensor(tokens).to(torch.int64).reshape(-1, 1)
    chosen = logits.gather(1, index)
    return SampledLogprobs(
        sampled_logprobs=torch.log_softmax(logits, dim=-1).gather(1, index).reshape(-1),
        selected_token_ranks=(logits >= chosen).sum(dim=1).to(torch.int32),
    )
