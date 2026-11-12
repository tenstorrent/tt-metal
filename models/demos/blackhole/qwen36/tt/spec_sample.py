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
positions were grouped into verifies. An unseeded row's column takes fresh entropy. Every row that draws nothing
carries the device SKIP seed, which leaves its core's random state untouched.

Device graph (``VerifySampler.body``, a trace of its own after the verify trace): the verify's gathered logits
[1, 1, B*T, V] (at most 32 rows, one core each) are cut into equal vocabulary chunks of at most 64K columns, each chunk gives its
top-32 (``ttnn.topk``), the candidates' global ids are the local ids plus the chunk offset, then ``ttnn.manual_seed``
and one ``ttnn.sampling`` with the per-row parameters write one draw per row. Top-k above 32 never reaches it: the
plugin samples such a row on the host (``max_device_top_k``).
"""

import secrets
from dataclasses import dataclass
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

SAMPLER_ROWS = 32  # one ttnn.sampling call: a row per core, B * T <= 32 verify rows
SAMPLER_TOP_K = 32  # candidates per vocabulary chunk; the plugin keeps a row with top-k above it on the host
TOPK_MAX_WIDTH = 64 * 1024  # widest input one ttnn.topk call takes
SKIP_SEED = MAX_UINT32  # manual_seed leaves the core's random state untouched


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
    inverse temperature (``format_sampling_params``), the device seed, and whether the row's draw replaces its
    argmax id."""

    k: list
    p: list
    temp: list
    seeds: list
    sampled: list

    @property
    def any_sampled(self):
        return any(self.sampled)


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


def vocab_chunks(vocab):
    """Fewest equal vocabulary chunks, a power of two, of at most TOPK_MAX_WIDTH tile-aligned columns each."""
    chunks = 1
    while vocab // chunks > TOPK_MAX_WIDTH:
        chunks *= 2
    if vocab % chunks or (vocab // chunks) % ttnn.TILE_SIZE:
        raise ValueError(f"a vocabulary of {vocab} does not split into {chunks} tile-aligned top-k chunks")
    return chunks


class VerifySampler:
    """The device half: persistent per-row inputs, the draw output, and the trace-capturable ``body``."""

    def __init__(self, mesh, B, T, vocab):
        if B * T > SAMPLER_ROWS:
            raise ValueError(f"{B} users x {T} columns exceed the sampler's {SAMPLER_ROWS} rows")
        self.mesh, self.rows, self.vocab = mesh, B * T, int(vocab)
        self.chunks = vocab_chunks(self.vocab)
        width = self.vocab // self.chunks
        idle = sampling_plan(None, [-1] * B, T)
        self.k = self._device(torch.tensor(idle.k, dtype=torch.int32), ttnn.uint32)
        self.p = self._device(torch.tensor(idle.p), ttnn.bfloat16)
        self.temp = self._device(torch.tensor(idle.temp), ttnn.bfloat16)
        self.seeds = self._device(torch.tensor(idle.seeds, dtype=torch.int64), ttnn.uint32)
        self.user_ids = self._device(torch.arange(self.rows, dtype=torch.int32), ttnn.uint32)
        self.draws = self._device(torch.zeros(1, 1, 1, self.rows, dtype=torch.int32), ttnn.uint32)
        offsets = torch.arange(self.chunks, dtype=torch.int32).repeat_interleave(SAMPLER_TOP_K) * width
        self.offsets = self._device(
            offsets.reshape(1, 1, 1, -1).expand(1, 1, self.rows, -1).contiguous(), ttnn.int32, ttnn.TILE_LAYOUT
        )

    def _host(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))

    def _device(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.to_device(self._host(t, dtype, layout), self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def stage(self, plan):
        """Host -> device copies of the plan's parameters and seeds (addresses preserved)."""
        for host, dev, dtype in (
            (torch.tensor(plan.k, dtype=torch.int32), self.k, ttnn.uint32),
            (torch.tensor(plan.p), self.p, ttnn.bfloat16),
            (torch.tensor(plan.temp), self.temp, ttnn.bfloat16),
            (torch.tensor(plan.seeds, dtype=torch.int64), self.seeds, ttnn.uint32),
        ):
            ttnn.copy_host_to_device_tensor(self._host(host, dtype), dev)

    def body(self, logits):
        """``logits`` [1, 1, rows, vocab] bfloat16 TILE, replicated. Writes one draw per row into ``draws``."""
        parts = ttnn.split(logits, self.vocab // self.chunks, dim=3) if self.chunks > 1 else [logits]
        values, indices = [], []
        for part in parts:
            v, i = ttnn.topk(part, k=SAMPLER_TOP_K, dim=-1)
            values.append(v)
            indices.append(ttnn.typecast(i, ttnn.int32))
            ttnn.deallocate(i)
            if part is not logits:
                ttnn.deallocate(part)
        vals = ttnn.concat(values, dim=3) if self.chunks > 1 else values[0]
        idx = ttnn.concat(indices, dim=3) if self.chunks > 1 else indices[0]
        if self.chunks > 1:
            for t in values + indices:
                ttnn.deallocate(t)
        gidx = ttnn.add(self.offsets, idx, dtype=ttnn.uint32)
        ttnn.deallocate(idx)
        gidx_rm = ttnn.untilize(gidx, use_multicore=True)
        ttnn.deallocate(gidx)
        ttnn.manual_seed(seeds=self.seeds, user_ids=self.user_ids)
        ttnn.sampling(vals, gidx_rm, k=self.k, p=self.p, temp=self.temp, output_tensor=self.draws)
        ttnn.deallocate(vals)
        ttnn.deallocate(gidx_rm)
        return self.draws

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
