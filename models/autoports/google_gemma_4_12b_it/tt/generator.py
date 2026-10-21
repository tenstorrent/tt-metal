# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generator for the full ``google/gemma-4-12B-it`` TTNN model on a QB2 (``tt/model.py``).

Two API levels (``readiness_check.contract.Generator``):

* Low level, caller-owned state: ``prefill_forward(tokens, *, page_table, kv_cache, prompt_lens, ...)`` and
  ``decode_forward(tokens, start_pos, *, page_table, kv_cache, ...)``. The caller owns scheduling, page tables
  (``[B, 4096]`` rows of physical block ids, block 64 tokens), positions, prompt lengths and request slots; rows may be
  inactive (position -1). ``kv_cache`` is ``generator.kv_cache`` (``allocate_kv_cache()``), the model's paged caches.
* High level, generator-owned state: ``prefill_logits(prompt)`` and ``generate(prompt, max_new_tokens, ...)``, a thin
  deterministic loop over the low-level calls with an internal block allocator.

Token-out decode is the canonical split-sampling path: one captured **model trace** per (decode batch bucket,
full-layer attention placement) runs embedding -> 48 layers -> final norm -> LM head -> soft-cap into a persistent
vocab-sharded logits buffer and advances the persistent positions on device (``plus_one``); one captured **sampling
trace** runs the sampler (``tt/sampling.SplitTopKSampling``: ``models.common.sampling.TTSampling`` with a two-level
per-chip top-32 on a zero-copy view of the logits shard) on that buffer with ``tt_out_tok`` = the persistent decode
token buffer, so the sampled token is the next replay's input without a host round trip. Positions, tokens and page
tables are copied from the host only when the caller's values differ from what the device already holds (request
boundaries, teacher forcing, scheduler changes); ``decode_forward(tokens=None)`` decodes straight from the device
token buffer. Free-running ``generate`` is pipelined: it never waits for the step it just issued; each step's tokens
are copied to the host asynchronously (``read_decode_output`` / ``process_decode_output_host``, the split decode API)
and consumed ``PIPELINE_LAG`` steps later. ``counters`` records every refresh, replay and readback.

Prefill of a single-chunk request at position 0 (any prompt of at most 8192 tokens) replays a captured **prefill
trace** per bucket (128 ... 8192 rows: embedding -> 48 layers -> last-token terminal into the logits buffer) after
refreshing its persistent token buffer, page-table row and row ids; longer prompts, continuations and the
every-position logits API run the same programs eagerly, chunk by chunk.

All compiled programs (prefill buckets x chunk starts, every decode variant, the sampler, cache reset) are prepared in
``prepare()`` before the first capture, and traces are captured once and kept for the generator's lifetime.

``host_sampling=True`` (explicit compatibility mode for tests that sample on the host) replays only the model trace,
reads the logits and samples on the host; it is never the measured token-out path.

KV groups (serving, ``tt/generator_vllm.py``): ``kv_groups[i]`` names the page-table group of ``model.layers[i]``
(vLLM hybrid KV groups: one full-attention group and several sliding-window groups, each with its own block table).
Every persistent page-table input exists once per group and per bucket; page-table arguments are one table for all
groups or a list with one table per group. With ``windowed_groups`` a group whose layers are all sliding-window
layers may map the blocks below each row's window to the canary (vLLM frees them). Standalone use has one group.
"""

import random
import time
from collections import Counter, deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional

import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tt.functional_decoder import MAX_DECODE_BATCH
from models.autoports.google_gemma_4_12b_it.tt.model import HF_MODEL, HF_REVISION, PREFILL_BUCKETS, Gemma4Model
from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import NUM_LINKS
from models.autoports.google_gemma_4_12b_it.tt.sampling import SplitTopKSampling
from models.common.sampling.tt_sampling import TTSampling

SAMPLERS = {"split_topk": SplitTopKSampling, "ttsampling": TTSampling}

try:  # the readiness contract ABC (tt-model-bringup runtime on PYTHONPATH)
    from readiness_check.contract import Generator
except ImportError:  # pragma: no cover - serving / standalone use without the readiness package
    Generator = object

DECODE_BUCKETS = (1, 4, 8, 16, 32)
SUPPRESSED_LOGIT = -1.0e4  # soft-capped logits lie in (-30, 30)
SEED_RANGE = 1 << 30  # aligned seeds: base(seed) in [1, 2^30) plus the input position stay below 2^31
PIPELINE_LAG = 2  # free-running generate(): decode steps queued ahead of the oldest unread token
COUNTERS = (
    "model_replays",
    "sampling_replays",
    "eager_decode_steps",
    "token_refreshes",
    "position_refreshes",
    "page_table_refreshes",
    "page_table_resident_steps",
    "sampling_param_refreshes",
    "seed_refreshes",
    "sampling_state_resets",
    "token_readbacks",
    "logits_readbacks",
    "synchronizations",
    "prefill_requests",
    "prefill_chunks",
    "prefill_replays",
    "prefill_input_refreshes",
    "trace_captures",
    "trace_releases",
)


@dataclass(frozen=True)
class SamplingParams:
    """Per-request sampling parameters (scalars, or one value per row). ``top_k=1`` or ``temperature<=0`` is greedy;
    ``top_p`` 1.0 (or <= 0) disables nucleus filtering; ``top_k`` is capped at 32 (sampler candidate width)."""

    temperature: Any = 1.0
    top_k: Any = 1
    top_p: Any = 1.0
    seed: Any = None

    @property
    def greedy(self):
        ks = self.top_k if isinstance(self.top_k, (list, tuple)) else [self.top_k]
        temps = self.temperature if isinstance(self.temperature, (list, tuple)) else [self.temperature]
        return all(int(k) == 1 or float(t) <= 0 for k, t in zip(ks, temps * len(ks) if len(temps) == 1 else temps))


GREEDY = SamplingParams()


def _rows(value, n):
    if isinstance(value, (list, tuple)):
        assert len(value) == n, f"per-row sampling parameters need {n} values, got {len(value)}"
        return list(value)
    return [value] * n


class DeferredRead:
    """``read_decode_output(defer=True)``: the token copy of one decode step, enqueued only when the generator next
    queues device work (``decode_forward`` / ``prefill_forward``) or when ``process_decode_output_host`` needs it."""

    __slots__ = ("version", "read")

    def __init__(self, version):
        self.version = version
        self.read = None  # (host tensor, event, sample version) once issued


class Gemma4Generator(Generator):
    def __init__(
        self,
        model: Gemma4Model,
        *,
        decode_buckets=DECODE_BUCKETS,
        trace_prefill=True,
        sampler="split_topk",
        prepare=True,
        log=print,
        kv_groups=None,
        windowed_groups=False,
    ):
        from transformers import AutoTokenizer, GenerationConfig

        self.model = model
        self.mesh = model.mesh
        self.log = log
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_REVISION)
        gen_cfg = GenerationConfig.from_pretrained(HF_MODEL, revision=HF_REVISION)
        eos = gen_cfg.eos_token_id
        self.eos_token_ids = set(eos if isinstance(eos, list) else [eos])
        # generation_config.suppress_tokens (<image|>, <audio|>): HF generate() masks them in every sampling step
        self.suppress_ids = [int(t) for t in (gen_cfg.suppress_tokens or [])]
        self.buckets = tuple(sorted(decode_buckets))
        assert self.buckets[-1] <= MAX_DECODE_BATCH and self.buckets[0] >= 1
        self.max_batch_size = self.buckets[-1]
        self.counters = {k: 0 for k in COUNTERS}
        self.kv_cache = model.kv_cache()
        self.pages = model.pages_per_user
        # page-table groups (one per vLLM KV group; standalone: one) and the sliding window of windowed groups
        self.kv_groups = [0] * len(model.layers) if kv_groups is None else [int(g) for g in kv_groups]
        assert len(self.kv_groups) == len(model.layers)
        self.num_groups = max(self.kv_groups) + 1
        assert sorted(set(self.kv_groups)) == list(range(self.num_groups))
        self.group_window = [None] * self.num_groups
        if windowed_groups:
            for g in range(self.num_groups):
                kinds = {m.layer_type for m, k in zip(model.layers, self.kv_groups) if k == g}
                if kinds == {"sliding_attention"}:
                    self.group_window[g] = model.config.sliding_window
        canary = model.canary_block

        # ---- persistent device state, all allocated before any trace exists
        self.tokens = model.device_ints(torch.zeros(1, 1, 1, MAX_DECODE_BATCH, dtype=torch.int32), ttnn.uint32)
        self.logits = model.logits_buffer()
        # suppressed tokens: SplitTopKSampling masks them among its level-1 candidates (exact, no full-logits pass);
        # the stock sampler option gets a full-vocab additive mask in the sampling trace
        self.suppress_mask = None
        if self.suppress_ids and sampler != "split_topk":
            mask = torch.zeros(1, 1, MAX_DECODE_BATCH, model.vocab_size)
            mask[..., self.suppress_ids] = SUPPRESSED_LOGIT
            self.suppress_mask = ttnn.from_torch(
                mask,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=3),
            )
        self.row_ids = model.device_ints(torch.zeros(1, MAX_DECODE_BATCH, dtype=torch.int32), ttnn.uint32)
        # traced prefill (one trace per bucket for single-chunk requests at position 0): persistent token buffers
        # and the request's page-table row, rewritten before each replay
        self.prefill_trace_buckets = (
            tuple(b for b in PREFILL_BUCKETS if b <= model.prefill_chunk) if trace_prefill else ()
        )
        self.prefill_tokens = {
            b: model.device_ints(torch.zeros(1, b, dtype=torch.int32), ttnn.uint32) for b in self.prefill_trace_buckets
        }
        self.prefill_table = [
            model.device_ints(torch.full((1, self.pages), canary, dtype=torch.int32)) for _ in range(self.num_groups)
        ]
        self._prefill_table_host = [
            torch.full((1, self.pages), canary, dtype=torch.int32) for _ in range(self.num_groups)
        ]
        self._row_ids_host = torch.zeros(1, MAX_DECODE_BATCH, dtype=torch.int32)
        self.positions = {}
        self.tables = {}
        for b in self.buckets:
            self.positions[b] = model.device_ints(torch.full((b,), -1, dtype=torch.int32))
            self.tables[b] = [
                model.device_ints(torch.full((b, self.pages), canary, dtype=torch.int32))
                for _ in range(self.num_groups)
            ]
        args = SimpleNamespace(
            vocab_size=model.vocab_size,
            padded_vocab_size=model.vocab_size,
            max_batch_size=MAX_DECODE_BATCH,
            max_top_k=32,
            cluster_shape=tuple(self.mesh.shape),
            model_config={
                "SAMPLING_AG_CONFIG": {
                    "allow_force_argmax": False,
                    "num_links": NUM_LINKS,
                    "topology": ttnn.Topology.Ring,
                }
            },
        )
        self.sampler_name = sampler
        kw = {"suppress_ids": self.suppress_ids} if sampler == "split_topk" else {}
        self.sampler = SAMPLERS[sampler](mesh_device=self.mesh, tt_ccl=model.ccl, args=args, **kw)

        # token reads copy only device 0's replica of the replicated token buffer (a view of its shard): a read of the
        # mesh tensor pins a fresh host buffer per device shard first (PinnedMemoryCache::try_pin, 1-4 ms per read)
        self._tokens_dev0 = ttnn.get_device_tensors(self.tokens)[0]
        self._deferred_reads = []  # DeferredRead copies not enqueued yet

        # host mirrors of what the device buffers hold (None = unknown -> refresh on next use)
        self._tok_host = None
        # per token-buffer row: does it hold that slot's own latest sample (a valid on-device feedback input)?
        self._feedback_rows = torch.zeros(MAX_DECODE_BATCH, dtype=torch.bool)
        self._sample_version = 0  # increments whenever the device token buffer receives new samples
        self._pos_host = {b: torch.full((b,), -1, dtype=torch.int32) for b in self.buckets}
        self._table_host = {
            b: [torch.full((b, self.pages), canary, dtype=torch.int32) for _ in range(self.num_groups)]
            for b in self.buckets
        }
        # per-slot sampling state mirrored on the host (what the sampler's persistent tensors hold)
        self._row_k = torch.ones(MAX_DECODE_BATCH, dtype=torch.int64)
        self._row_p = torch.zeros(MAX_DECODE_BATCH)  # device convention: greedy rows p = 0, no filter p = 1
        self._row_t = torch.ones(MAX_DECODE_BATCH)  # temperature T (the device holds 1 / T)
        self._seed_counter = 0
        # host mirror of the sampler's seed tensor: the device holds (mirror + sampling replays since the last write)
        self._seed_host = torch.arange(MAX_DECODE_BATCH, dtype=torch.int64)
        self._seed_advance = 0
        self._rng = random.Random()
        self._table_cache = {}
        self.traces = {}
        self.trace_events = []  # capture / release events (key, trace id, capture seconds)
        self.replays = Counter()  # trace replays per key
        self.eager_chunks = Counter()  # eager prefill chunks per (chunk start, bucket)
        self.program_cache_entries_at_capture = None
        self.compiled = False
        self.prepared = False
        # standalone (generator-owned) block allocator
        self._free = list(range(model.first_block, model.block_limit))
        if prepare:
            self.prepare()

    # ------------------------------------------------------------------ setup
    def _decode_variants(self):
        out = []
        for b in self.buckets:
            out.append((b, False))
            if b % 4 == 0 and self.model.full_layer is not None:
                out.append((b, True))
        return out

    def prepare(self, capture=True):
        """Compile every program the generator can run, then capture every trace (two-phase warmup).
        ``capture=False`` runs only the compile phase (vLLM's eager warmup call); a later ``prepare()`` captures."""
        self._flush_reads()
        if self.prepared:
            return
        if not self.compiled:
            self._compile_all()
        if capture:
            self._capture_all()

    def _compile_all(self):
        m = self.model
        self.log("prepare: prefill programs")
        m.warm_prefill(log=self.log)
        self.log("prepare: cache reset, prefill terminal, decode variants, sampler")
        self.reset()  # compiles the in-place cache fill
        seeds = ttnn.to_torch(ttnn.get_device_tensors(self.sampler.seeds_tt_tensor)[0])
        x = m.select_rows(self._warm_hidden(), self.row_ids)
        m.decode_logits(x, self.logits)
        ttnn.deallocate(x)
        for b, split in self._decode_variants():
            self._write(self.positions[b], torch.zeros(b, dtype=torch.int32))
            for t in self.tables[b]:
                self._write(t, torch.full((b, self.pages), m.scratch_block, dtype=torch.int32))
            m.decode_step(
                self.tokens, self.positions[b], self._layer_list(self.tables[b]), self.logits, user_split=split
            )
            self._sample_step()
        self.sampler.reset_params(k=torch.ones(32), p=torch.zeros(32), temp=torch.ones(32))
        ttnn.synchronize_device(self.mesh)
        # restore the warmup-mutated state exactly: positions inactive, tables on the canary, tokens 0, seeds
        for b in self.buckets:
            self._write(self.positions[b], self._pos_host[b])
            for t, host in zip(self.tables[b], self._table_host[b]):
                self._write(t, host)
        self._write(self.tokens, torch.zeros(1, 1, 1, MAX_DECODE_BATCH, dtype=torch.int32))
        self._tok_host = torch.zeros(MAX_DECODE_BATCH, dtype=torch.int32)
        self._write(self.sampler.seeds_tt_tensor, seeds.reshape(-1))
        self._seed_host = seeds.reshape(-1).to(torch.int64)
        self._seed_advance = 0
        self._write(self.row_ids, self._row_ids_host)
        self.reset()
        ttnn.synchronize_device(self.mesh)
        self.compiled = True

    def _capture(self, key, body):
        t0 = time.perf_counter()
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        body()
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self.traces[key] = tid
        self.trace_events.append(
            {"event": "capture", "key": list(key), "trace_id": int(tid), "s": time.perf_counter() - t0}
        )
        self.counters["trace_captures"] += 1

    def _capture_all(self):
        m = self.model
        self.log("prepare: capturing traces")
        for b, split in self._decode_variants():
            self._capture(
                ("decode", b, split),
                lambda b=b, split=split: m.decode_step(
                    self.tokens, self.positions[b], self._layer_list(self.tables[b]), self.logits, user_split=split
                ),
            )
        self._capture(("sample",), self._sample_step)
        for b in self.prefill_trace_buckets:
            self._capture(
                ("prefill", b),
                lambda b=b: m.prefill_request_logits(
                    self.prefill_tokens[b], self._layer_list(self.prefill_table), self.row_ids, self.logits
                ),
            )
        self.program_cache_entries_at_capture = self.mesh.num_program_cache_entries()
        self.prepared = True
        self.log(f"prepare: done, traces {sorted(self.traces, key=str)}")

    def _warm_hidden(self):
        local = self.model.hidden_size // 4
        return ttnn.from_torch(
            torch.zeros(1, 1, 128, local * 4),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=3),
        )

    def _layer_list(self, per_group):
        """Per-group page-table tensors -> the model's page-table argument (one tensor when there is one group)."""
        if self.num_groups == 1:
            return per_group[0]
        return [per_group[g] for g in self.kv_groups]

    def _sample_step(self):
        """Sampling-trace body: sample every row of the logits buffer into the decode token buffer with the
        generation-config suppressed tokens masked (as HF generate does; inside SplitTopKSampling, or a full-vocab
        additive mask for the stock sampler), advance the seeds."""
        logits = self.logits if self.suppress_mask is None else ttnn.add(self.logits, self.suppress_mask)
        self.sampler(logits, tt_out_tok=self.tokens)
        if logits is not self.logits:
            ttnn.deallocate(logits)
        ttnn.plus_one(self.sampler.seeds_tt_tensor)

    def _write(self, dest, values):
        host = ttnn.from_torch(values.to(torch.int32).contiguous(), dtype=dest.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, dest)

    def teardown(self):
        for key, tid in self.traces.items():
            ttnn.release_trace(self.mesh, tid)
            self.trace_events.append({"event": "release", "key": list(key), "trace_id": int(tid)})
            self.counters["trace_releases"] += 1
        self.traces = {}
        self.prepared = False

    def _memory(self, kind):
        v = ttnn.get_memory_view(self.mesh, getattr(ttnn.BufferType, kind))
        return {
            "allocated": v.total_bytes_allocated_per_bank * v.num_banks,
            "free": v.total_bytes_free_per_bank * v.num_banks,
            "largest_free_block_per_bank": v.largest_contiguous_bytes_free_per_bank,
        }

    def trace_report(self):
        """Trace registry and execution signatures (serving evidence: captures, retirements, replays per key, eager
        prefill chunks per (chunk start, bucket), program-cache growth since the captures)."""
        return {
            "traces": {"/".join(map(str, k)): int(v) for k, v in self.traces.items()},
            "events": list(self.trace_events),
            "replays_by_key": {"/".join(map(str, k)): n for k, n in sorted(self.replays.items(), key=str)},
            "eager_prefill_chunks": {f"start{c0}/bucket{b}": n for (c0, b), n in sorted(self.eager_chunks.items())},
            "program_cache_entries": self.mesh.num_program_cache_entries(),
            "memory_per_chip": {kind: self._memory(kind) for kind in ("DRAM", "L1")},
            "program_cache_entries_at_capture": self.program_cache_entries_at_capture,
            "counters": dict(self.counters),
        }

    # ------------------------------------------------------------------ cache / allocator
    def allocate_kv_cache(self):
        """The model-owned paged caches (``[[k, v] per layer]``); pass them as ``kv_cache`` to the low-level API.
        Blocks 0 (zero canary) and 1 (padding scratch) are reserved; callers allocate from
        ``range(model.first_block, model.block_limit)``. Serving attaches vLLM-owned caches instead
        (``Gemma4Model.attach_kv_cache``)."""
        return self.kv_cache

    def _check_cache(self, kv_cache):
        if kv_cache is None:
            return
        assert len(kv_cache) == len(self.kv_cache), "kv_cache must be generator.kv_cache"
        for got, own in zip(kv_cache, self.kv_cache):
            assert all(
                g is o or g.buffer_address() == o.buffer_address() for g, o in zip(got, own)
            ), "kv_cache must be generator.kv_cache (allocate_kv_cache)"

    def _alloc_row(self, length):
        need = -(-length // self.model.block_size)
        if need > len(self._free):
            raise RuntimeError(f"KV cache has {len(self._free)} free blocks, request needs {need}")
        row = torch.full((self.pages,), self.model.canary_block, dtype=torch.int32)
        blocks = [self._free.pop(0) for _ in range(need)]
        row[:need] = torch.tensor(blocks, dtype=torch.int32)
        return row

    def reset(self):
        """Zero every KV cache in place, forget request state; buffers, programs and traces survive."""
        self._flush_reads()
        for k, v in self.kv_cache:
            ttnn.fill(k, 0.0, output_tensor=k)
            ttnn.fill(v, 0.0, output_tensor=v)
        self._free = list(range(self.model.first_block, self.model.block_limit))
        self._feedback_rows[:] = False
        # finish the cache fill here, so the next request's latency does not absorb it
        ttnn.synchronize_device(self.mesh)
        self.counters["synchronizations"] += 1
        for b in self.buckets:
            if not torch.equal(self._pos_host[b], torch.full((b,), -1, dtype=torch.int32)):
                self._pos_host[b] = torch.full((b,), -1, dtype=torch.int32)
                self._write(self.positions[b], self._pos_host[b])

    # ------------------------------------------------------------------ sampling parameters
    def set_sampling_params(
        self, params: Optional[SamplingParams] = None, rows=MAX_DECODE_BATCH, *, slots=None, reseed=True
    ):
        """Set the sampling parameters of decode slots ``slots`` (default ``0..rows-1``); every other slot keeps its
        own. The sampler's persistent tensors are rewritten in place (the sampling trace stays valid), only when a
        value changes. ``reseed``: a request seed re-seeds those slots' device RNG (request start); unseeded slots
        keep the device RNG stream, which the sampling trace advances once per sampled token."""
        params = params or GREEDY
        slots = list(range(rows)) if slots is None else [int(v) for v in slots]
        n = len(slots)
        k = torch.tensor([int(v) for v in _rows(params.top_k, n)], dtype=torch.int64)
        p = torch.tensor([float(v) for v in _rows(params.top_p, n)])
        t = torch.tensor([float(v) for v in _rows(params.temperature, n)])
        greedy = (t <= 0) | (k == 1)
        k = torch.where(greedy, torch.ones_like(k), torch.where(k < 1, torch.full_like(k, 32), k.clamp(max=32)))
        p = torch.where(greedy, torch.zeros_like(p), torch.where((p <= 0) | (p > 1), torch.ones_like(p), p))
        t = torch.where(greedy, torch.ones_like(t), t)
        idx = torch.tensor(slots)
        new_k, new_p, new_t = self._row_k.clone(), self._row_p.clone(), self._row_t.clone()
        new_k[idx], new_p[idx], new_t[idx] = k, p, t
        if not (
            torch.equal(new_k, self._row_k) and torch.equal(new_p, self._row_p) and torch.equal(new_t, self._row_t)
        ):
            self.sampler.reset_params(k=new_k, p=new_p, temp=1.0 / new_t)
            self._row_k, self._row_p, self._row_t = new_k, new_p, new_t
            self.counters["sampling_param_refreshes"] += 1
        if reseed and params.seed is not None:
            seeds = self._device_seeds()
            seeds[idx] = torch.tensor([int(v) % 1_000_000 + 1 for v in _rows(params.seed, n)], dtype=torch.int64)
            self._write_seeds(seeds)

    def _device_seeds(self):
        """What the sampler's seed tensor holds now: the host mirror advanced by every sampling replay since the
        last write (each replay adds one per row)."""
        return (self._seed_host + self._seed_advance) % SEED_RANGE

    def _write_seeds(self, seeds):
        self._write(self.sampler.seeds_tt_tensor, seeds)
        self._seed_host = seeds.clone()
        self._seed_advance = 0
        self.counters["seed_refreshes"] += 1

    def reset_sampling_state(self, seeds, positions, slots):
        """Rebuild the RNG state of decode slots ``slots`` for a new request layout (vLLM ``reset_sampling_state``).

        A seeded row's seed is aligned to its input position, ``base(seed) + position``; the sampling trace adds one
        per sampled token, so a seeded stream depends only on (seed, position), not on batch composition, slot or
        readback timing. An unseeded row (seed None) gets a fresh random seed. Other slots keep their state.
        """
        cur = self._device_seeds()
        for slot, seed, pos in zip(slots, seeds, positions):
            if seed is None:
                cur[int(slot)] = self._rng.randrange(1, SEED_RANGE)
            else:
                base = (int(seed) * 2654435761) % (SEED_RANGE - (1 << 20)) + 1
                cur[int(slot)] = base + max(int(pos), 0)
        self._write_seeds(cur)
        self.counters["sampling_state_resets"] += 1

    # ------------------------------------------------------------------ decode
    def _bucket(self, batch):
        for b in self.buckets:
            if b >= batch:
                return b
        raise ValueError(f"decode batch {batch} exceeds the largest bucket {self.buckets[-1]}")

    def _group_tables(self, page_table):
        """A page-table argument -> one entry per group (a single table serves every group; None keeps the
        resident tables)."""
        if isinstance(page_table, (list, tuple)):
            assert len(page_table) == self.num_groups, f"one page table per KV group ({self.num_groups})"
            return list(page_table)
        return [page_table] * self.num_groups

    def _sanitize_table(self, table, rows, bucket, positions, group=0):
        """``[bucket, 4096]`` device page table of one group: the caller's rows, padded with the canary;
        out-of-range entries map to the canary. Every block an active row reads or writes must be a request block
        in ``[model.first_block, model.block_limit)``: blocks ``0..position // 64``, or, in a windowed group, the blocks
        of the row's sliding window. The table is rebuilt only when the caller's rows change (``table`` None: the
        resident table); each step checks the positions against the cached per-row counts of invalid entries."""
        m = self.model
        key = (bucket, group)
        cached = self._table_cache.get(key)
        if table is not None:
            raw = torch.as_tensor(table, dtype=torch.int32)
            assert raw.shape[0] >= rows and raw.shape[-1] <= self.pages
            raw = raw[:rows]
            if cached is None or cached[0].shape != raw.shape or not torch.equal(cached[0], raw):
                t = torch.full((bucket, self.pages), m.canary_block, dtype=torch.int32)
                t[:rows, : raw.shape[-1]] = raw
                t[(t < 0) | (t >= m.block_limit)] = m.canary_block
                invalid = ((t < m.first_block) | (t >= m.block_limit)).to(torch.int32)
                cached = (raw.clone(), t, invalid.cumsum(-1))  # invalid entries up to and including each block
                self._table_cache[key] = cached
        assert cached is not None, "no resident page table for this bucket: pass one"
        _, t, inv = cached
        pos = positions[:rows].to(torch.int64)
        active = pos >= 0
        hi = (pos // m.block_size).clamp(min=0)
        window = self.group_window[group]
        lo = ((pos - window + 1).clamp(min=0) // m.block_size) if window else torch.zeros_like(pos)
        inv = inv[:rows]
        count = inv.gather(1, hi[:, None])[:, 0] - torch.where(
            lo > 0, inv.gather(1, (lo - 1).clamp(min=0)[:, None])[:, 0], torch.zeros_like(lo, dtype=inv.dtype)
        )
        bad = active & (count > 0)
        assert not bad.any(), (
            f"page-table group {group} rows {bad.nonzero().flatten().tolist()}: every block a row attends up to its "
            f"position must be a request block in [{m.first_block}, {m.block_limit})"
        )
        return t

    def _refresh_inputs(self, tokens, positions, tables, bucket):
        """Copy decode inputs to the device only where they differ from what it already holds. ``tokens`` None:
        on-device feedback, the token buffer already holds every active row's input (the previous step's samples).
        ``tables``: the sanitized table of each group."""
        rows = positions.numel() if tokens is None else tokens.numel()
        active = positions[:rows] >= 0  # an inactive row's token is never read (its position is -1)
        if tokens is None:
            pass
        elif self._tok_host is None or not torch.equal(self._tok_host[:rows][active], tokens[active]):
            full = self._tok_host.clone() if self._tok_host is not None else torch.zeros(32, dtype=torch.int32)
            full[:rows] = torch.where(active, tokens, full[:rows])
            self._write(self.tokens, full.reshape(1, 1, 1, MAX_DECODE_BATCH))
            self._tok_host = full
            self._sample_version += 1  # older async copies no longer describe the buffer
            self.counters["token_refreshes"] += 1
        if not torch.equal(self._pos_host[bucket], positions):
            self._write(self.positions[bucket], positions)
            self._pos_host[bucket] = positions.clone()
            self.counters["position_refreshes"] += 1
        written = 0
        for g, table in enumerate(tables):
            host = self._table_host[bucket][g]
            if host is not table and not torch.equal(host, table):
                self._write(self.tables[bucket][g], table)
                written += 1
            self._table_host[bucket][g] = table  # cached by _sanitize_table, never mutated
        self.counters["page_table_refreshes"] += written

    def resident_positions(self, rows):
        """The positions the device holds for a ``rows``-row decode (advanced on device by every replay): the
        input positions of a decode that keeps its inputs resident (vLLM steady decode)."""
        return self._pos_host[self._bucket(rows)][:rows].clone()

    def _issue_read(self, blocking):
        """Copy device 0's replica of the token buffer to the host behind everything queued so far: ``(host tensor,
        event, sample version)``, the event None when blocking (the copy is complete)."""
        host = self._tokens_dev0.cpu(blocking=blocking)
        event = None if blocking else ttnn.record_event(self.mesh, 0)
        self.counters["token_readbacks"] += 1
        return host, event, self._sample_version

    def _finish_read(self, read):
        """Tokens ``[32]`` (int32) of an ``_issue_read`` copy (waits for its event)."""
        host, event, version = read
        if event is not None:
            ttnn.event_synchronize(event)
        t = ttnn.to_torch(host).reshape(-1).to(torch.int32).clone()
        if version == self._sample_version:  # the latest samples: the host mirror is exact again
            self._tok_host = t.clone()
        return t

    def read_tokens(self):
        """Read the sampled tokens of every row (one device's replica of the replicated token buffer)."""
        self._flush_reads()
        return self._finish_read(self._issue_read(blocking=True))

    def _flush_reads(self):
        """Enqueue the deferred token copies (before any device work that could change the token buffer)."""
        for d in self._deferred_reads:
            assert d.version == self._sample_version, "the token buffer changed under a deferred read"
            d.read = self._issue_read(blocking=False)
        self._deferred_reads.clear()

    def _mark_samples(self, valid_rows):
        """The token buffer just received new samples; ``valid_rows`` (leading rows) are their slots' own."""
        self._sample_version += 1
        self._feedback_rows[:] = False
        self._feedback_rows[: valid_rows.numel()] = valid_rows

    def read_decode_output(self, async_read=True, defer=False):
        """Split decode (``decode_forward(..., read_from_device=False)``): enqueue the copy of the sampled-token buffer
        to the host behind the queued replays, without waiting. Returns ``(host tensor, event, sample version)`` for
        ``process_decode_output_host``; the device keeps running the next queued steps meanwhile.
        ``async_read=False``: the tokens, read now (``(tokens, None, version)``). ``defer=True``: a ``DeferredRead``;
        the copy and its event are enqueued with the next device work instead (serving: the first decode step of a
        request is submitted before vLLM returns the request's first token, so enqueueing the copy there, about
        1-3 ms of host dispatch, would delay the first token)."""
        if not async_read:
            return self.read_tokens(), None, self._sample_version
        if defer:
            d = DeferredRead(self._sample_version)
            self._deferred_reads.append(d)
            return d
        return self._issue_read(blocking=False)

    def process_decode_output_host(self, output, rows=None):
        """Wait for one ``read_decode_output`` copy and return its tokens ``[rows]`` (int32)."""
        if isinstance(output, DeferredRead):
            if output.read is None:
                self._flush_reads()
            output = output.read
        host, event, version = output if len(output) == 3 else (*output, None)
        if event is None:  # a synchronous read: host tokens already
            return host[:rows]
        t = self._finish_read((host, event, version))
        return t if rows is None else t[:rows]

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache=None,
        sampling_params: Optional[SamplingParams] = None,
        host_sampling=False,
        return_logits=False,
        read_from_device=True,
        enable_trace=True,
        user_split=None,
        **kwargs,
    ):
        """One decode step for ``B = len(start_pos)`` rows (fixed slots; position -1 = inactive row).

        tokens ``[B, 1]`` / ``[B]``, or None to decode from the device token buffer (on-device feedback of the previous
        on-device sampled step: no host token input at all); start_pos ``[B]``; page_table ``[B, <=4096]`` (one for
        every KV group, a list with one per group, or None to keep the resident tables of this bucket). Returns
        sampled tokens ``[B]`` (on-device split sampling), None with ``read_from_device=False`` (then
        ``read_decode_output`` / ``process_decode_output_host``), or, with ``return_logits`` / ``host_sampling``,
        logits ``[B, vocab]`` (host sampling also returns the host-sampled tokens: ``(tokens, logits)``).
        """
        self._check_cache(kv_cache)
        self._flush_reads()
        positions = torch.as_tensor(start_pos, dtype=torch.int32).reshape(-1)
        rows = positions.numel()
        bucket = self._bucket(rows)
        active = positions >= 0
        if tokens is None:
            assert not host_sampling, "host sampling needs host input tokens"
            assert self._feedback_rows[:rows][active].all(), (
                "tokens=None decodes each active row's previous on-device sample; rows "
                f"{(active & ~self._feedback_rows[:rows]).nonzero().flatten().tolist()} hold none (pass host tokens)"
            )
            tok = None
        else:
            tok = torch.as_tensor(tokens, dtype=torch.int64).reshape(-1)[:rows].to(torch.int32)
            assert tok.numel() == rows, "one input token per decode row"
            assert (
                (tok[active] >= 0) & (tok[active] < self.model.vocab_size)
            ).all(), "token ids outside the vocabulary"
        assert (positions < self.model.max_context).all() and (positions >= -1).all(), "positions outside the context"
        pos = torch.full((bucket,), -1, dtype=torch.int32)
        pos[:rows] = positions
        given = self._group_tables(page_table)
        tables = [self._sanitize_table(t, rows, bucket, pos, g) for g, t in enumerate(given)]
        if page_table is None:
            self.counters["page_table_resident_steps"] += 1
        if sampling_params is not None:
            self.set_sampling_params(sampling_params, rows, reseed=False)  # uploads only changed values
        self._refresh_inputs(tok, pos, tables, bucket)
        if user_split is None:  # the measured break-even of the full-layer attention placement
            split = bucket % 4 == 0 and self.model.use_user_split(bucket, int(pos.max()))
        else:  # explicit override (both traces exist for every bucket that is a multiple of 4)
            assert not user_split or bucket % 4 == 0, "the user split needs a bucket that is a multiple of 4"
            split = bool(user_split) and self.model.full_layer is not None
        if enable_trace:
            if not self.prepared:
                raise RuntimeError("traced decode requested before prepare()")
            ttnn.execute_trace(self.mesh, self.traces[("decode", bucket, split)], cq_id=0, blocking=False)
            self.replays[("decode", bucket, split)] += 1
            self.counters["model_replays"] += 1
        else:
            self.model.decode_step(
                self.tokens,
                self.positions[bucket],
                self._layer_list(self.tables[bucket]),
                self.logits,
                user_split=split,
            )
            self.counters["eager_decode_steps"] += 1
        self._pos_host[bucket] = torch.where(pos >= 0, pos + 1, pos)  # mirrors the in-trace plus_one
        if host_sampling or return_logits:
            logits = self.model.logits_to_torch(self.logits, rows)
            self.counters["logits_readbacks"] += 1
            if not host_sampling:
                self._feedback_rows[:] = False  # no sampling ran: the buffer still holds this step's inputs
                return logits
            out = self._host_sample(logits)
            full = self._tok_host.clone() if self._tok_host is not None else torch.zeros(32, dtype=torch.int32)
            full[:rows] = out.to(torch.int32)
            self._write(self.tokens, full.reshape(1, 1, 1, MAX_DECODE_BATCH))
            self._tok_host = full
            self.counters["token_refreshes"] += 1
            self._mark_samples(pos >= 0)
            return out, logits
        self._run_sampling(enable_trace)
        self._mark_samples(pos >= 0)
        if not read_from_device:
            self._tok_host = None  # unknown until the copy of this step is processed
            return None
        return self.read_tokens()[:rows]

    def _run_sampling(self, enable_trace=True):
        """Sample every row of the logits buffer into the token buffer (sampling trace replay, or eagerly)."""
        if enable_trace:
            ttnn.execute_trace(self.mesh, self.traces[("sample",)], cq_id=0, blocking=False)
            self.replays[("sample",)] += 1
            self.counters["sampling_replays"] += 1
        else:
            self._sample_step()
        self._seed_advance += 1  # the sampling body advances every row's seed by one

    def _host_sample(self, logits):
        """Host-sampling compatibility mode (greedy argmax or temperature / top-k / top-p with the active params)."""
        logits = logits.clone()
        logits[:, self.suppress_ids] = SUPPRESSED_LOGIT
        rows = logits.shape[0]
        ks, ps, ts = self._row_k[:rows], self._row_p[:rows], self._row_t[:rows]
        out = []
        for row, k, top_p, t in zip(logits, ks, ps, ts):
            if int(k) == 1:
                out.append(row.argmax())
                continue
            vals, idx = row.topk(int(k))
            probs = torch.softmax(vals / float(t), -1)
            if float(top_p) < 1:
                keep = probs.cumsum(-1) - probs < float(top_p)
                probs = torch.where(keep, probs, torch.zeros_like(probs))
                probs = probs / probs.sum()
            out.append(idx[torch.multinomial(probs, 1)][0])
        return torch.stack(out)

    # ------------------------------------------------------------------ prefill
    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache=None,
        prompt_lens,
        return_all_logits=False,
        start_pos=None,
        slots=None,
        sample_on_device=False,
        sampling_params: Optional[SamplingParams] = None,
        enable_trace=True,
        seed_positions=None,
        **kwargs,
    ):
        """Prefill ``B`` requests (one after another) and fill their pages of ``kv_cache``.

        tokens ``[B, S_padded]`` (row b's first ``prompt_lens[b]`` entries are its prompt, or its continuation when
        ``start_pos[b] > 0``; continuations start on a prefill-chunk boundary, 8192); page_table ``[B, <=4096]``
        (or a list with one per KV group) maps the request's blocks for ``[0, start_pos + prompt_len)``. ``slots[b]``
        is the decode row whose sampling parameters a request uses (default ``b``). ``seed_positions[b]`` (with
        ``sample_on_device``): align row b's RNG to that input position right before its sample
        (``reset_sampling_state``; per-row seeds may be None). Returns logits at the last prompt position
        ``[B, 1, vocab]`` (float32), every position ``[B, S, vocab]`` with ``return_all_logits``, or the
        on-device sampled first tokens ``[B]`` with ``sample_on_device``. A single-chunk request at position 0 (any
        prompt of at most 8192 tokens) replays its bucket's prefill trace; ``enable_trace=False`` forces the eager
        bucketed path (same programs, same result).
        """
        self._check_cache(kv_cache)
        self._flush_reads()
        B = len(prompt_lens)
        tokens = torch.as_tensor(tokens).reshape(B, -1)
        groups = [
            torch.as_tensor(t, dtype=torch.int32).reshape(-1, torch.as_tensor(t).shape[-1])
            for t in self._group_tables(page_table)
        ]
        slot_list = list(range(B)) if slots is None else [int(v) for v in slots]
        if sampling_params is not None:
            self.set_sampling_params(sampling_params, slots=slot_list, reseed=seed_positions is None)
        seeds = _rows(sampling_params.seed, B) if seed_positions is not None else None
        outs = []
        for b, n in enumerate(prompt_lens):
            n = int(n)
            s0 = 0 if start_pos is None else int(start_pos[b])
            slot = slot_list[b]
            rows = []
            for t in groups:
                row = torch.full((self.pages,), self.model.canary_block, dtype=torch.int32)
                row[: t.shape[-1]] = t[b]
                rows.append(row)
            if seeds is not None and sample_on_device:
                self.reset_sampling_state([seeds[b]], [int(seed_positions[b])], [slot])
            outs.append(
                self._prefill_one(tokens[b, :n], rows, s0, slot, return_all_logits, sample_on_device, enable_trace)
            )
        if sample_on_device:  # each row's token copy was queued behind its sample: wait for them only now
            return torch.stack([self._finish_read(read)[slot] for read, slot in outs])
        if return_all_logits:
            if len(outs) == 1:
                return outs[0][None]  # a view: no second copy of the all-position logits
            width = max(o.shape[0] for o in outs)
            return torch.stack([torch.nn.functional.pad(o, (0, 0, 0, width - o.shape[0])) for o in outs])
        return torch.stack(outs)[:, None, :]

    def _set_row_ids(self, ids):
        if not torch.equal(self._row_ids_host, ids):
            self._write(self.row_ids, ids)
            self._row_ids_host = ids.clone()
            self.counters["prefill_input_refreshes"] += 1

    def _valid_from(self, group, start):
        """First position whose block a prefill from ``start`` reads unmasked in ``group`` (its sliding window)."""
        window = self.group_window[group]
        return max(0, start - window + 1) if window else 0

    def _prefill_traced(self, ids, rows, bucket):
        """Single-chunk request at position 0: refresh the persistent inputs that changed, replay the bucket's
        prefill trace (48 layers + last-token terminal into the logits buffer), without blocking."""
        m = self.model
        n = ids.numel()
        chunk = torch.zeros(1, bucket, dtype=torch.int32)
        chunk[0, :n] = ids
        self._write(self.prefill_tokens[bucket], chunk)
        for g, row in enumerate(rows):
            table = m.prefill_page_table(row, n, bucket)
            if not torch.equal(self._prefill_table_host[g], table):
                self._write(self.prefill_table[g], table)
                self._prefill_table_host[g] = table
                self.counters["prefill_input_refreshes"] += 1
        self._set_row_ids(torch.full((1, MAX_DECODE_BATCH), n - 1, dtype=torch.int32))
        ttnn.execute_trace(self.mesh, self.traces[("prefill", bucket)], cq_id=0, blocking=False)
        self.replays[("prefill", bucket)] += 1
        self.counters["prefill_replays"] += 1
        self.counters["prefill_chunks"] += 1

    def _prefill_one(self, ids, rows, start, slot, all_logits, sample, enable_trace=True):
        m = self.model
        ids = ids.to(torch.int32)
        n = ids.numel()
        plan = m.chunk_plan(start, n)
        last_c0, _, last_bucket = plan[-1]
        self.counters["prefill_requests"] += 1
        # a new request in this slot: its row of the token buffer is no valid feedback until a sample lands there
        # (on-device first-token sampling below, or the caller's host token on the first decode step)
        self._feedback_rows[slot] = False
        traced = enable_trace and self.prepared and not all_logits and len(plan) == 1 and start == 0
        if traced and ("prefill", last_bucket) in self.traces:
            self._prefill_traced(ids, rows, last_bucket)
            return self._prefill_result(slot, sample)
        tables = [
            m.device_ints(m.prefill_page_table(row, start + n, last_c0 + last_bucket, self._valid_from(g, start)))
            for g, row in enumerate(rows)
        ]
        # every position's logits, written in place (one host copy: an 8K prompt is ~8.8 GB of float32 logits)
        out = torch.empty(n, m.vocab_size, dtype=torch.float32) if all_logits else None
        hidden = None
        for c0, real, bucket in plan:
            chunk = torch.zeros(bucket, dtype=torch.int32)
            chunk[:real] = ids[c0 - start : c0 - start + real]
            if hidden is not None:
                ttnn.deallocate(hidden)
            hidden = m.prefill_chunk_hidden(chunk, self._layer_list(tables), c0)
            self.eager_chunks[(c0, bucket)] += 1
            self.counters["prefill_chunks"] += 1
            if all_logits:
                for r0 in range(0, real, MAX_DECODE_BATCH):
                    idx = torch.arange(r0, r0 + MAX_DECODE_BATCH, dtype=torch.int32).clamp(max=bucket - 1)
                    k = min(MAX_DECODE_BATCH, real - r0)
                    o = c0 - start + r0
                    out[o : o + k] = self._rows_logits(hidden, idx)[:k]
        for t in tables:
            ttnn.deallocate(t)
        if all_logits:
            ttnn.deallocate(hidden)
            return out
        last = (start + n - 1) - last_c0
        self._set_row_ids(torch.full((1, MAX_DECODE_BATCH), last, dtype=torch.int32))
        x = m.select_rows(hidden, self.row_ids)
        ttnn.deallocate(hidden)
        m.decode_logits(x, self.logits)  # every row holds this request's last-token logits
        ttnn.deallocate(x)
        return self._prefill_result(slot, sample)

    def _prefill_result(self, slot, sample):
        """The request's first token sampled on device (sampling trace on the logits buffer): ``(read, slot)``,
        its copy to the host queued behind the sample without waiting, so the next row's prefill follows on the
        device without a host round trip. Or its logits."""
        m = self.model
        if sample:
            self._run_sampling(self.prepared)
            # every row now holds a sample of this request's logits: only the request's own slot is valid feedback
            valid = torch.zeros(MAX_DECODE_BATCH, dtype=torch.bool)
            valid[slot] = True
            self._mark_samples(valid)
            return self._issue_read(blocking=False), slot
        self.counters["logits_readbacks"] += 1
        return m.logits_to_torch(self.logits, 1)[0]

    def _rows_logits(self, hidden, idx):
        self._set_row_ids(idx.reshape(1, MAX_DECODE_BATCH).to(torch.int32))
        x = self.model.select_rows(hidden, self.row_ids)
        self.model.decode_logits(x, self.logits)
        ttnn.deallocate(x)
        self.counters["logits_readbacks"] += 1
        return self.model.logits_to_torch(self.logits)

    # ------------------------------------------------------------------ high level
    def prefill_logits(self, prompt_token_ids: List[int]) -> torch.Tensor:
        """Logits of every prompt position ``[1, S, vocab]`` (generator-owned cache and page table)."""
        n = len(prompt_token_ids)
        row = self._alloc_row(n)
        try:
            out = self.prefill_forward(
                torch.tensor([prompt_token_ids]), page_table=row[None], prompt_lens=[n], return_all_logits=True
            )
        finally:
            self._release(row)
        return out[:, :n]

    def _release(self, row):
        used = [int(b) for b in row if int(b) >= self.model.first_block]
        self._free = used + self._free

    def _generate_pipelined(self, preds, first_tokens, pos, done, table, max_new_tokens, stop, lag=PIPELINE_LAG):
        """Free-running token-out decode with no per-token host synchronization.

        Step s is issued (model + sampling replays, non-blocking) with on-device token feedback (``tokens=None``) and
        positions that the device already holds (advanced in the trace), then its token copy to the host is enqueued
        behind it; the host waits only for step ``s - lag``'s copy, so the device always has ``lag`` queued steps.
        A row that emits EOS is deactivated (position -1) from the next issued step; the at most ``lag`` steps
        already in flight for it are discarded (they write only the row's own reserved pages).
        """
        B = len(preds)
        pending = deque()
        timing = self.last_generate_timing = {"decode_steps": 0}

        def collect():
            out, active = pending.popleft()
            tokens = self.process_decode_output_host(out, B)
            for b in range(B):
                if active[b] and not done[b]:
                    p = int(tokens[b])
                    preds[b].append(p)
                    done[b] = stop and p in self.eos_token_ids

        step = 1
        timing["t_first_issue"] = time.perf_counter()
        while step < max_new_tokens and not all(done):
            active = [not d for d in done]
            start = torch.tensor([(pos[b] if active[b] else -1) for b in range(B)])
            self.decode_forward(
                torch.tensor(first_tokens) if step == 1 else None, start, page_table=table, read_from_device=False
            )
            pending.append((self.read_decode_output(async_read=True), active))
            for b in range(B):
                pos[b] += active[b]
            step += 1
            timing["decode_steps"] += 1
            if len(pending) > lag:
                collect()
        while pending:
            collect()
        timing["t_done"] = time.perf_counter()

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens: int,
        *,
        next_input=None,
        enable_trace: bool = True,
        sampling_params: Optional[SamplingParams] = None,
        host_sampling: bool = False,
        stop_on_eos: Optional[bool] = None,
        **kwargs,
    ):
        """HF-style generation for one prompt (list of ids) or a batch (list of lists, one decode row each).

        Prefill samples the first token on device; every later token comes from the traced model + sampling
        replays with on-device token feedback. ``next_input(step, predicted)`` (teacher forcing) overrides the next
        input; the returned lists hold the model's own predictions. Stops at EOS when free-running
        (``stop_on_eos`` default) and never when ``next_input`` is given.
        """
        batched = len(prompt_token_ids) > 0 and isinstance(prompt_token_ids[0], (list, tuple))
        prompts = [list(p) for p in prompt_token_ids] if batched else [list(prompt_token_ids)]
        B = len(prompts)
        stop = (next_input is None) if stop_on_eos is None else stop_on_eos
        params = sampling_params or GREEDY
        self.set_sampling_params(params, B)
        assert next_input is None or B == 1, "teacher forcing drives one prompt"
        rows = [self._alloc_row(len(p) + max_new_tokens) for p in prompts]
        table = torch.stack(rows)
        width = max(map(len, prompts))
        try:
            first = self.prefill_forward(
                torch.tensor([p + [0] * (width - len(p)) for p in prompts]),
                page_table=table,
                prompt_lens=[len(p) for p in prompts],
                sample_on_device=not host_sampling,
            )
            if host_sampling:
                first = self._host_sample(first[:, 0, :])
            preds = [[int(first[b])] for b in range(B)]
            nxt = [next_input(0, preds[b][0]) if next_input is not None else preds[b][0] for b in range(B)]
            pos = [len(p) for p in prompts]
            done = [stop and preds[b][0] in self.eos_token_ids for b in range(B)]
            if next_input is None and not host_sampling and enable_trace:
                self._generate_pipelined(preds, nxt, pos, done, table, max_new_tokens, stop)
            else:
                for step in range(1, max_new_tokens):
                    if all(done):
                        break
                    start = [(-1 if done[b] else pos[b]) for b in range(B)]
                    out = self.decode_forward(
                        torch.tensor(nxt),
                        torch.tensor(start),
                        page_table=table,
                        host_sampling=host_sampling,
                        enable_trace=enable_trace,
                    )
                    out = out[0] if host_sampling else out
                    for b in range(B):
                        if done[b]:
                            continue
                        p = int(out[b])
                        preds[b].append(p)
                        pos[b] += 1
                        nxt[b] = next_input(step, p) if next_input is not None else p
                        done[b] = stop and p in self.eos_token_ids
        finally:
            for r in rows:
                self._release(r)
        return preds if batched else preds[0]


def build_generator(model_dir, mesh_device, **kwargs) -> Gemma4Generator:
    """Readiness factory: the full 48-layer model on the open QB2 ``[1, 4]`` ring mesh.

    kwargs: ``num_blocks`` (KV blocks per layer, default one 262144-token request + 2 reserved), ``max_context``,
    ``layer_indices`` (reduced model), ``precision`` (precision config: default the datatype sweep's
    ``doc/datatype_sweep/selected_precision_config.json``, ``"baseline"`` the stage-7 policy; ``tt/precision.py``),
    ``policy`` / ``lm_head`` overrides on top of it, ``decode_buckets``, ``trace_prefill``, ``sampler`` (default: the
    precision config's ``sampling.sampler``, ``"split_topk"``; ``"ttsampling"`` the stock module), ``prepare``.
    """
    gen_keys = {"decode_buckets", "trace_prefill", "sampler", "prepare", "log"}
    gen_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in gen_keys}
    model = Gemma4Model(mesh_device, **kwargs)
    gen_kwargs.setdefault("sampler", model.precision.sampler)
    return Gemma4Generator(model, **gen_kwargs)
