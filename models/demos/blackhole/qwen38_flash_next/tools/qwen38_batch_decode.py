# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P1b driver: batched greedy decode of the PLAIN lane body, one token per lane, on the 1x4 Blackhole mesh.

``Qwen38PlainLaneDecoder`` sits beside an open vLLM adapter (``Qwen38ForCausalLM``, one resident slot).  It allocates a
B-lane state on the adapter's model, builds the lane pagers and the one-slot host pool and warms every program
eagerly (``prepare``, the chain's warm hook: the lane tensors must exist before the chain captures its traces), then
captures the ONE residue-free lane body (``capture`` after the chain opened) and checks the trace allocation tracker.
A prompt is prefilled on the adapter's B=1 chain and moved into a lane (``admit``): paged lanes by the device-direct
``Qwen38TTNNLaneImporter`` (no host round trip, no lane reset: the writes cover every buffer the reset would zero, its GDN
ring slots rotated into the canonical fixed-phase order by the choice of source slot), the others (and
``QWEN38_IMPORT_HOST=1``) through the host pager: evicted into the host slot, rotated, the lane reset and re-admitted;
``step`` replays the one lane trace and returns one new greedy token per lane.

Decode-width buckets (W3; ``buckets=(R1, ..., 32)``, 32 paged lanes only): besides the full 32-lane trace the decoder
captures one lane trace (and its sampling traces) per smaller width R over a bucket view of the lane state
(``model.allocate_bucket_lane_state``).  Each step replays the smallest bucket holding every live lane; the invariant
is that every live lane is below R (idle lanes inside the bucket run the warm token on the scratch page as today), and
a step returns -1 for the lanes R and up.  Without ``buckets`` there is one bucket, the full one: today's behavior.

Residue-free: the lanes sit at any positions (the GDN lane state keeps a per-lane conv shift register, see
``Qwen38TTNNGDNState.fixed_phase``).  No MTP.  ``Qwen38ContinuousBatch`` (host-only, below) is the continuous-batching policy over this decoder (P1d).

Paged KV (P1c-B, ``paging=Qwen38LanePaging(...)``): the lanes' QSA KV is one page pool per layer plus one page table
(``model.write_lane_page_table``).  ``Qwen38PageAllocator`` hands a lane ``ceil((P + decode_steps + 1) / page_size)``
pages from a seeded-shuffled free list over pages ``1 .. num_pages - 1`` (page 0 is the scratch, never handed out), so
the pages are non-contiguous and interleaved across lanes.  Admission writes the table, scatters the B=1 prefill's KV
into the lane's pages (``model.admit_lane_kv_paged``) and moves the other families through the host pager (the pager
layouts exclude the KV family).

Multi-user pack (P3d-3; the decoder's ``pack`` argument, else ``QWEN38_PREFILL_PACK=1``: off here by default, the vLLM
batch adapter turns it on with at least 8 lanes, P3e; ``QWEN38_PACK_BUCKETS`` default ``256,512,1024`` rows per
user with at most 8 lanes, ``128,256,512,1024`` with more, P3d-4): ``prepare`` also allocates the pack state (``ttnn/pack.py``), runs one eager pack per bucket and builds and warms the
landing (``ttnn/pack_landing.py``) inside the chain's warm hook, before any capture; ``capture`` captures one pack trace per
bucket after the lane trace; ``admit_pack(lanes, prompts, rows_per_user)`` prefills the prompts as ONE pack, lands each user
into its lane (every family, device only; the lane then holds position ``P``) and returns the users' first-token logits.
"""

from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import torch
from loguru import logger
from ttnn.tools.trace_allocation_tracker import (
    TraceAllocationTracker,
    acknowledge_corruptible,
    corruptible_allocation_scope,
)

import ttnn
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_pack_plan as pack_plan
from models.demos.blackhole.qwen38_flash_next.tools import resident_decode
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import Qwen38TracedChain
from models.demos.blackhole.qwen38_flash_next.ttnn import pack as pack_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROWS, TP_SIZE, replicate_tensor_2d_mesh_mapper
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import LOCAL_VOCAB_SIZE, SAMPLING_CANDIDATE_ROW_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.lanes import (
    Qwen38LaneHostPool,
    Qwen38TTNNLaneImporter,
    Qwen38TTNNLanePager,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.pack_landing import Qwen38TTNNPackLander
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging

WARM_TOKEN_ID = 1000  # any in-vocabulary id: the warm passes write garbage into lane state that a reset clears


PREFILL_PACK_VARIABLE = "QWEN38_PREFILL_PACK"
# P3e (2026-10-26): with the knob unset the vLLM batch adapter packs with at least this many lanes.  GPQA-Diamond
# through the server at B=32, every eligible row packed against the pack off: 85.4 % against 83.8 %, paired difference
# +1.5 [-2.0, +5.1] points.
PACK_DEFAULT_MIN_LANES = 8
PACK_BUCKETS_VARIABLE = "QWEN38_PACK_BUCKETS"
PACK_MIB = 1 << 20
# DRAM per device of the pack (P3d-2 / P3d-3 measurements, 2026-10-26): the persistent state is 187.6 MiB plus 226.5 MiB per
# GDN prims launch (36 layers x [96,128,128] fp32; a bucket of 128 rows per user runs two launches, the others one), the
# landing's select buffers 10.9 MiB (18.8 MiB measured at context 32768 minus the zero tails below); one trace
# per enabled bucket (the whole pack body: 74.0 / 63.0 / 61.5 / 60.7 MiB for 128 / 256 / 512 / 1024 rows per user).
# The landing's compressed zero tail of a bucket scales with the allocated context: (context / 4 - S / 4) rows x 128 x bf16
# (P3d-4: 7.9 MiB over all four buckets at 32768).  Total with all four buckets: 918.6 MiB (measured at B=8 and B=32, 32768).
PACK_FIXED_BYTES = int((187.6 + 10.9) * PACK_MIB)
PACK_LAUNCH_BYTES = int(226.5 * PACK_MIB)
PACK_TRACE_BYTES = {128: 74 * PACK_MIB, 256: 63 * PACK_MIB, 512: 62 * PACK_MIB, 1024: 61 * PACK_MIB}
PACK_TAIL_ROW_BYTES = 128 * 2  # one compressed index row: INDEX_HEAD_DIM bf16


def prefill_pack_enabled(environ=None, lanes: int | None = None) -> bool:
    """``QWEN38_PREFILL_PACK``: "1" = on, "0" = off, anything else is refused.  Unset: on with at least
    ``PACK_DEFAULT_MIN_LANES`` ``lanes`` (the vLLM batch adapter passes its lane count), off without ``lanes`` (the P1
    drivers and text_demo)."""

    value = (os.environ if environ is None else environ).get(PREFILL_PACK_VARIABLE)
    if value is None:
        return lanes is not None and lanes >= PACK_DEFAULT_MIN_LANES
    if value not in ("0", "1"):
        raise ValueError(f"{PREFILL_PACK_VARIABLE} must be 0 or 1, got {value!r}")
    return value == "1"


def pack_buckets_from_env(environ=None, lanes: int | None = None) -> tuple[int, ...]:
    """``QWEN38_PACK_BUCKETS``: comma-separated rows per user, each one of ``pack.PACK_BUCKETS``, returned ascending without
    duplicates.  Unset: :func:`qwen38_pack_plan.default_pack_buckets` of ``lanes`` (at most 8 lanes: 256, 512, 1024; more
    or unknown: all four)."""

    text = (os.environ if environ is None else environ).get(PACK_BUCKETS_VARIABLE)
    if text is None:
        return pack_plan.default_pack_buckets(lanes)
    try:
        values = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise ValueError(f"{PACK_BUCKETS_VARIABLE} must be comma-separated integers, got {text!r}") from None
    if not values or any(v not in pack_module.PACK_BUCKETS for v in values):
        raise ValueError(
            f"{PACK_BUCKETS_VARIABLE} must list rows per user from {pack_module.PACK_BUCKETS}, got {text!r}"
        )
    return tuple(sorted(set(values)))


def pack_dram_bytes(buckets: Sequence[int], context: int = 32768) -> int:
    """Per device DRAM the pack holds: its persistent state, the bucket traces and the landing's compressed zero tails at
    the allocated ``context`` (pool admission counts it)."""

    launches = max(len(pack_module.pack_bucket(rows).launches) for rows in buckets)
    tails = sum((context // 4 - rows // 4) * PACK_TAIL_ROW_BYTES for rows in buckets)
    return PACK_FIXED_BYTES + launches * PACK_LAUNCH_BYTES + sum(PACK_TRACE_BYTES[rows] for rows in buckets) + tails


def row_read_buckets(lanes: int) -> tuple[int, ...]:
    """P3g-3: the sizes of the row-read traces of a ``lanes``-lane decoder: the powers of two up to ``lanes``.  A
    fallback step reads ``k`` lanes with the smallest bucket >= ``k`` (``None`` above the largest: the full gather).
    Measured at B=32 (scratch test, trace replay plus one concat read of the bf16 row blocks): 0.16 / 0.23 / 0.32 /
    0.55 / 0.98 ms for 1 / 2 / 4 / 8 / 16 lanes against 7.5 ms for the full gather (p3g3 REPORT.md, read options)."""

    buckets, size = [], 1
    while size <= lanes:
        buckets.append(size)
        size *= 2
    return tuple(buckets)


def row_read_bucket(count: int, buckets: Sequence[int]) -> int | None:
    """The smallest bucket holding ``count`` lanes, ``None`` when ``count`` is above the largest (or below 1)."""

    return next((size for size in buckets if size >= count > 0), None)


def lane_page_count(prompt_len: int, decode_steps: int, page_size: int) -> int:
    """Pages of one lane: its prompt, the ``decode_steps`` decoded tokens and the one token consumed by the last step."""

    return -(-(prompt_len + decode_steps + 1) // page_size)


def page_table_rows(page_rows: Sequence[Sequence[int]], lanes: int) -> list[list[int]]:
    """The host page-table rows ``write_lane_page_table`` takes: row u = lane u's pages (empty for a free lane)."""

    if len(page_rows) != lanes:
        raise ValueError(f"expected {lanes} page rows, got {len(page_rows)}")
    return [list(row) for row in page_rows]


class Qwen38PageAllocator:
    """Host free list over pages ``1 .. num_pages - 1`` in a seeded-shuffled order (page 0, the scratch, is never
    handed out): consecutive allocations are not contiguous and interleave across lanes."""

    def __init__(self, num_pages: int, seed: int = 0) -> None:
        if num_pages < 2:
            raise ValueError(f"the allocator needs at least 2 pages, got {num_pages}")
        self.num_pages = num_pages
        self._rng = random.Random(seed)
        self.free: list[int] = list(range(1, num_pages))
        self._rng.shuffle(self.free)

    def allocate(self, count: int) -> list[int]:
        if count < 1 or count > len(self.free):
            raise RuntimeError(f"cannot allocate {count} pages, {len(self.free)} free of {self.num_pages - 1}")
        pages = [self.free.pop() for _ in range(count)]
        return pages

    def release(self, pages: Sequence[int]) -> None:
        owned = set(self.free)
        for page in pages:
            if not 1 <= page < self.num_pages or page in owned:
                raise RuntimeError(f"page {page} is not allocated")
            owned.add(page)
        self.free.extend(pages)
        self._rng.shuffle(self.free)


def greedy_sampling_params() -> SimpleNamespace:
    """The plugin's ``TTSamplingParams`` for one greedy row (as demo/text_demo.py builds it)."""

    return SimpleNamespace(
        temperature=[0.0],
        top_k=[1],
        top_p=[0.0],
        presence_penalty=[0.0],
        frequency_penalty=[0.0],
        repetition_penalty=[1.0],
        seed=[None],
        num_logprobs=[0],
        enable_log_probs=[False],
    )


@contextmanager
def chain_warm_hook(hook):
    """Inject ``warm_hook`` into ``Qwen38TracedChain.open`` for the adapter's ``initialize_vllm_model`` (which has no
    parameter for it); the chain's own open calls it after its warm pass, before its captures."""

    original = Qwen38TracedChain.__dict__["open"]
    function = original.__func__

    def open_with_hook(cls, construction, **kwargs):
        kwargs.setdefault("warm_hook", hook)
        return function(cls, construction, **kwargs)

    Qwen38TracedChain.open = classmethod(open_with_hook)
    try:
        yield
    finally:
        Qwen38TracedChain.open = original


@dataclass
class Qwen38LaneBucket:
    """Everything that depends on the lane count of one lane trace: the lane state (the parent's for the full bucket,
    a bucket view otherwise), its capture and trace, and the sampling buffers and traces over that capture's logits."""

    lanes: int
    ls: Any = None
    capture: Any = None
    trace_id: int | None = None
    candidate_buffer: Any = None
    candidate_trace_id: int | None = None
    candidate_scratch: Any = None
    normalizer_inv_temperature: Any = None
    normalizer_buffer: Any = None
    normalizer_trace_id: int | None = None
    normalizer_scratch: tuple | None = None
    row_read_buckets: tuple[int, ...] = ()
    row_read_indices: dict[int, Any] = field(default_factory=dict)
    row_read_blocks: dict[int, Any] = field(default_factory=dict)
    row_read_traces: dict[int, int] = field(default_factory=dict)


class Qwen38PlainLaneDecoder:
    """B lanes of the plain lane body over one adapter; see the module docstring.  ``buckets``: the decode widths (each
    1..lanes, the largest ``lanes``; more than one needs 32 paged lanes); ``step`` replays the smallest bucket that
    holds every live lane (``bucket_for_step``), ``active`` is the last step's bucket and the reads follow it."""

    def __init__(
        self,
        lanes: int,
        paging: Qwen38LanePaging | None = None,
        decode_steps: int = 32,
        seed: int = 0,
        indexer_form: str | None = None,
        pack: bool | None = None,
        candidates: bool = False,
        normalizer: bool = False,
        row_read: bool = False,
        buckets: Sequence[int] | None = None,
    ) -> None:
        self.lanes = lanes
        self.buckets: tuple[int, ...] = (lanes,) if buckets is None else tuple(sorted({int(r) for r in buckets}))
        if not self.buckets or self.buckets[0] < 1 or self.buckets[-1] != lanes:
            raise ValueError(f"buckets must be widths in 1..{lanes} whose largest is {lanes}, got {buckets!r}")
        if len(self.buckets) > 1 and (lanes != CHUNK_ROWS or paging is None):
            raise ValueError(f"more than one bucket needs {CHUNK_ROWS} paged lanes, got {lanes} lanes, paging {paging}")
        self.force_bucket: int | None = None  # test hook: step replays this bucket (it must hold every live lane)
        self.bucket_steps: dict[int, int] = {r: 0 for r in self.buckets}
        # QSA lane indexer form, "wide" or "per_lane"; the argument beats QWEN38_LANE_INDEXER_FORM (default "wide")
        self.indexer_form = indexer_form or os.environ.get("QWEN38_LANE_INDEXER_FORM", "wide")
        self.paging = paging
        self.decode_steps = decode_steps
        self.allocator = None if paging is None else Qwen38PageAllocator(paging.num_pages, seed)
        self.page_rows: list[list[int]] = [[] for _ in range(lanes)]
        self.vm = self.chain = self.mesh = self.model = None
        self.positions: list[int | None] = [None] * lanes  # P_u: the prompt length, the lane's start position
        self.contexts: list[Any] = [None] * lanes
        self.ls = None
        self.pager_g = self.pager_l = self.pool = None
        self.importer: Qwen38TTNNLaneImporter | None = None  # paged lanes: the device-direct import
        # QWEN38_IMPORT_HOST=1: the host-pager import (evict, rotate, admit_lane reset, readmit) of P1 / P2: the
        # reference of the bitwise import test and the fallback
        self.host_import = os.environ.get("QWEN38_IMPORT_HOST", "0") == "1"
        self.captures: list[Any] = []  # over every bucket in capture order, the full bucket first
        self.trace_ids: list[int] = []
        self.timing: dict[str, Any] = {}
        self.released = False
        self.page_table_dirty = False  # a page row changed since the last table write (grow, release)
        self.last_admit: dict[str, float] = {}
        self.lane_moves = 0  # W5: lanes moved down by ``move_lane`` (compaction)
        self.last_move: dict[str, Any] = {}
        # P2a (vLLM adapter): compile the full-vocabulary gather of the lane logits in ``prepare`` (eager, misses allowed)
        # so ``read_full_logits`` runs under the closed miss guard; off for the P1 drivers (they read greedy ids only).
        self.warm_logits_gather = False
        # P3d-3: the multi-user packed prefill; the argument beats QWEN38_PREFILL_PACK (unset: off, the P1 drivers; the
        # vLLM adapter passes its own default, on with at least PACK_DEFAULT_MIN_LANES lanes)
        self.pack_enabled = prefill_pack_enabled() if pack is None else pack
        self.pack_buckets: tuple[int, ...] = pack_buckets_from_env(lanes=lanes) if self.pack_enabled else ()
        if self.pack_enabled:
            pack_plan.pack_force_enabled()  # QWEN38_PACK_FORCE is read per prefill call: refuse a bad value at start
        # P3g-1: the candidate rows of every lane (``sampling_candidates`` over the lane logits, ``[1,1,lanes,256]`` fp32
        # ROW_MAJOR) in a small second trace replayed after the lane replay, only on steps with a sampled row.  The buffer is
        # allocated and the programs compiled in ``prepare``; ``capture`` captures the trace after the lane trace.
        self.candidates = candidates
        # P3g-2: the full-row normalizer of the sampled lanes (per shard LSE of logits / T over the real vocabulary, one fp32 per
        # lane and shard), for top-k-off rows.  Its trace is the candidate trace plus the normalizer ops (one replay, two
        # buffers), chosen per step; the inverse temperatures are a persistent input the host writes before the replay.
        self.normalizer = bool(candidates and normalizer)
        self.normalizer_mask = None  # lane independent: one for every bucket
        # P3g-3: the logits rows of chosen lanes for the host's fallback draws.  One small trace per bucket size
        # (``row_read_buckets``) selects the rows into a bf16 block per shard; its lane indices are a persistent input
        # the host writes before the replay.  Only the sampled path uses it, so it needs the candidate rows (the step
        # that found a fallback row).
        self.row_read = bool(candidates and row_read)
        # W3: one bucket per decode width; the full one holds the parent lane state (``self.ls``, set in ``prepare``)
        self.bucket_sets: dict[int, Qwen38LaneBucket] = {
            r: Qwen38LaneBucket(r, row_read_buckets=row_read_buckets(r) if self.row_read else ()) for r in self.buckets
        }
        self.active: Qwen38LaneBucket = self.bucket_sets[lanes]
        self.pack_state = None
        self.pack_traces: dict[int, tuple[int, Any]] = {}
        self.lander: Qwen38TTNNPackLander | None = None
        self.pack_dram: dict[str, Any] = {}
        self.last_pack: dict[str, Any] = {}

    # -- the two build phases ----------------------------------------------------------------------

    def prepare(self, chain: Any) -> None:
        """The chain's warm hook (after its warm pass, before its own captures, misses allowed): the lane state, the
        pagers and host slot, the warm admissions and one eager body pass.  The lane tensors must
        exist before the chain's traces are captured, or the trace allocation tracker refuses those traces."""

        self.chain = chain
        self.mesh = mesh = chain.mesh
        self.model = model = chain.built_target.model
        lanes = self.lanes
        started = time.perf_counter()
        paging = self.paging
        self.ls = ls = model.allocate_lane_state(lanes, indexer_form=self.indexer_form, residue_free=True, paged=paging)
        full = self.bucket_sets[lanes]
        full.ls = ls
        gl = model.generic_lane_layout(chain.state, exclude_kv=paging is not None)
        ll = model.lane_layout(ls)
        self.pager_g = Qwen38TTNNLanePager(mesh, model.mesh_contract, gl)
        self.pager_l = Qwen38TTNNLanePager(mesh, model.mesh_contract, ll, buffers=self.pager_g)
        self.pool = Qwen38LaneHostPool(mesh, gl, slots=1)
        if paging is not None:
            self.importer = Qwen38TTNNLaneImporter(self.pager_l, gl)
        slot = self.pool.slots[0]
        self.pager_g.warm_moves(slot)
        self.pager_l.warm_moves(slot)
        ttnn.synchronize_device(mesh)
        self.timing["allocate_s"] = time.perf_counter() - started

        # Eager warm: every lane admitted once (compiles the admission programs), then one body pass.
        started = time.perf_counter()
        zeros = [0] * lanes
        model.reset_lane_state_inplace(ls, zeros)
        for lane in range(lanes):
            model.admit_lane(ls, lane, 0, rope_shift=0)
        if paging is not None:
            # compile the scatter on the all-zero table (every write lands in the scratch page 0)
            for lane in sorted({0, lanes - 1}):
                model.admit_lane_kv_paged(ls, chain.state, lane, ttnn.TILE_SIZE)
            ttnn.synchronize_device(mesh)
        if self.importer is not None:
            # compile the device-direct import for every lane (its fill_cache batch index is a runtime argument; the
            # warm pass compiles them all) and every ring phase
            for lane in range(lanes):
                self.importer.import_lane(lane, lane % 4)
            ttnn.synchronize_device(mesh)
        model.reset_lane_state_inplace(ls, zeros)
        contexts: Sequence[Any] = [None] * lanes
        tokens = [WARM_TOKEN_ID] * lanes
        contexts = model.write_lane_ple_rows(ls, tokens, contexts)
        model.write_lane_tokens(ls, tokens)
        output = model.forward_decode_lanes(ls)
        candidates, token_row = model.resolve_lane_tokens(output, ls)
        ttnn.synchronize_device(mesh)
        if self.warm_logits_gather:
            chain.sampling.read_full_logits_rows(output.logits)
        if self.candidates:
            self._prepare_candidates(chain, output.logits, full)
        self._release_warm_pass(output, token_row, candidates)
        model.reset_lane_state_inplace(ls, zeros)
        ttnn.synchronize_device(mesh)
        self.timing["warm_s"] = time.perf_counter() - started
        # W3: one eager pass per smaller bucket over its view (compiles the view's programs, builds its sampling
        # buffers); the passes wrote lanes 0..R-1 of the parent, so the parent is reset after them
        small = [r for r in self.buckets if r < lanes]
        for r in small:
            started, before = time.perf_counter(), self._dram_free()
            view = model.allocate_bucket_lane_state(ls, r)
            bucket = self.bucket_sets[r]
            bucket.ls = view
            self.timing.setdefault("bucket_alloc_bytes", {})[r] = before - self._dram_free()
            tokens = [WARM_TOKEN_ID] * r
            model.write_lane_ple_rows(view, tokens, [None] * r)
            model.write_lane_tokens(view, tokens)
            output = model.forward_decode_lanes(view)
            candidates, token_row = model.resolve_lane_tokens(output, view)
            ttnn.synchronize_device(mesh)
            if self.warm_logits_gather:
                chain.sampling.read_full_logits_rows(output.logits)
            if self.candidates:
                self._prepare_candidates(chain, output.logits, bucket)
            self._release_warm_pass(output, token_row, candidates)
            ttnn.synchronize_device(mesh)
            self.timing.setdefault("bucket_warm_s", {})[r] = time.perf_counter() - started
        if small:
            model.reset_lane_state_inplace(ls, zeros)
            ttnn.synchronize_device(mesh)
        if self.pack_enabled:
            self._prepare_pack(chain)

    @staticmethod
    def _release_warm_pass(output: Any, token_row: Any, candidates: Any) -> None:
        output.release_tensors()
        for tensor in (
            token_row,
            candidates.local_indices,
            candidates.local_values,
            candidates.packed,
            candidates.shard_row,
        ):
            if isinstance(tensor, ttnn.Tensor):
                ttnn.deallocate(tensor)

    def _prepare_candidates(self, chain: Any, logits: Any, bucket: Qwen38LaneBucket) -> None:
        """The bucket's candidate buffer (before any capture, like the pack state) and one eager ``sampling_candidates``
        over the warm pass's lane logits: every program of the candidate trace compiles here, and the host reads the row
        once to compile its readback."""

        mesh, lanes = self.mesh, bucket.lanes
        shape = (1, 1, lanes, SAMPLING_CANDIDATE_ROW_SHAPE[3])
        bucket.candidate_buffer = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh),
        )
        lm_head = self.model.model_io.lm_head
        row = lm_head.sampling_candidates(logits, chain.sampling.constants, into=bucket.candidate_buffer)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(row)
        self._read_candidate_buffer(bucket)
        if self.normalizer:
            bucket.normalizer_inv_temperature = ttnn.from_torch(
                torch.ones(1, 1, lanes, 1, dtype=torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh),
            )
            if self.normalizer_mask is None:
                self.normalizer_mask = lm_head.sampling_normalizer_mask()
            bucket.normalizer_buffer = ttnn.from_torch(
                torch.zeros(1, 1, lanes, TP_SIZE, dtype=torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh),
            )
            lse = lm_head.sampling_normalizer(
                logits, bucket.normalizer_inv_temperature, self.normalizer_mask, into=bucket.normalizer_buffer
            )
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(lse)
            self._read_normalizer_buffer(bucket)
        if self.row_read:
            self._prepare_row_reads(logits, bucket)

    def _prepare_row_reads(self, logits: Any, bucket: Qwen38LaneBucket) -> None:
        """The index inputs and row blocks of every row-read size of the bucket (before any capture) and one eager
        selection and readback each: every program of the row-read traces compiles here."""

        mesh, lm_head = self.mesh, self.model.model_io.lm_head
        for size in bucket.row_read_buckets:
            bucket.row_read_indices[size] = ttnn.from_torch(
                torch.zeros(1, 1, size, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh),
            )
            bucket.row_read_blocks[size] = ttnn.from_torch(
                torch.zeros(1, size, LOCAL_VOCAB_SIZE, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh),
            )
            lm_head.select_logits_rows(logits, bucket.row_read_indices[size], into=bucket.row_read_blocks[size])
            ttnn.synchronize_device(mesh)
            self._read_row_block(size, bucket)

    def _read_row_block(self, size: int, bucket: Qwen38LaneBucket) -> torch.Tensor:
        """The bucket's row block of the four shards, one readback: bf16 ``[size, VOCAB_SIZE]`` (slot j = the lane the
        j-th index named)."""

        host = ttnn.to_torch(bucket.row_read_blocks[size], mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1))
        return host.reshape(size, -1)

    def _read_candidate_buffer(self, bucket: Qwen38LaneBucket) -> torch.Tensor:
        host = ttnn.to_torch(ttnn.get_device_tensors(bucket.candidate_buffer)[0])
        return host.reshape(bucket.lanes, SAMPLING_CANDIDATE_ROW_SHAPE[3]).to(torch.float32)

    def _read_normalizer_buffer(self, bucket: Qwen38LaneBucket) -> torch.Tensor:
        host = ttnn.to_torch(ttnn.get_device_tensors(bucket.normalizer_buffer)[0])
        return host.reshape(bucket.lanes, TP_SIZE).to(torch.float32)

    def _bucket_timing(self, bucket: Qwen38LaneBucket, name: str, value: Any) -> None:
        """The full bucket keeps the plain timing keys, a smaller one lands in ``bucket_<name>[lanes]``."""

        if bucket.lanes == self.lanes:
            self.timing[name] = value
        else:
            self.timing.setdefault(f"bucket_{name}", {})[bucket.lanes] = value

    def capture_candidates(self, bucket: Qwen38LaneBucket) -> None:
        """The bucket's candidate trace, after the lane capture: ``sampling_candidates`` over the lane trace's
        persistent logits into the persistent buffer (the programs compiled in :meth:`_prepare_candidates`; nothing
        allocates that lives past the capture but the retained gather row)."""

        chain, mesh, model = self.chain, self.mesh, self.model
        logits = bucket.capture.output.logits
        guard = lambda label: resident_decode.forbid_trace_body_host_io_and_sync(
            phase=f"candidates {label}"
        )  # noqa: E731
        started = time.perf_counter()
        before = self._dram_free()
        with corruptible_allocation_scope(mesh):
            trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
            with guard("capture"):
                row = model.model_io.lm_head.sampling_candidates(
                    logits, chain.sampling.constants, into=bucket.candidate_buffer
                )
            ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh)
        bucket.candidate_trace_id, bucket.candidate_scratch = trace_id, row
        acknowledge_corruptible(bucket.candidate_buffer)
        acknowledge_corruptible(row)
        self._bucket_timing(bucket, "candidates_capture_s", time.perf_counter() - started)
        self._bucket_timing(bucket, "candidates_trace_bytes", before - self._dram_free())
        if self.normalizer:
            started, before = time.perf_counter(), self._dram_free()
            with corruptible_allocation_scope(mesh):
                trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                with guard("normalizer capture"):
                    row = model.model_io.lm_head.sampling_candidates(
                        logits, chain.sampling.constants, into=bucket.candidate_buffer
                    )
                    lse = model.model_io.lm_head.sampling_normalizer(
                        logits, bucket.normalizer_inv_temperature, self.normalizer_mask, into=bucket.normalizer_buffer
                    )
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh)
            bucket.normalizer_trace_id, bucket.normalizer_scratch = trace_id, (row, lse)
            for tensor in (
                bucket.normalizer_buffer,
                bucket.normalizer_inv_temperature,
                self.normalizer_mask,
                row,
                lse,
            ):
                acknowledge_corruptible(tensor)
            self._bucket_timing(bucket, "normalizer_capture_s", time.perf_counter() - started)
            self._bucket_timing(bucket, "normalizer_trace_bytes", before - self._dram_free())
        if self.row_read:
            self.capture_row_reads(logits, guard, bucket)

    def capture_row_reads(self, logits: Any, guard: Any, bucket: Qwen38LaneBucket) -> None:
        """One trace per row-read size of the bucket, after the candidate traces: the row selection of
        :meth:`_prepare_row_reads` over the lane trace's persistent logits into the size's block (the programs
        compiled there)."""

        mesh, lm_head = self.mesh, self.model.model_io.lm_head
        started, before = time.perf_counter(), self._dram_free()
        for size in bucket.row_read_buckets:
            with corruptible_allocation_scope(mesh):
                trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                with guard(f"row read {size} capture"):
                    lm_head.select_logits_rows(logits, bucket.row_read_indices[size], into=bucket.row_read_blocks[size])
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh)
            bucket.row_read_traces[size] = trace_id
            acknowledge_corruptible(bucket.row_read_indices[size])
            acknowledge_corruptible(bucket.row_read_blocks[size])
        self._bucket_timing(bucket, "row_read_capture_s", time.perf_counter() - started)
        self._bucket_timing(bucket, "row_read_trace_bytes", before - self._dram_free())

    # The sampling state of the ACTIVE bucket (the last step's), read only.
    @property
    def candidate_buffer(self) -> Any:
        return self.active.candidate_buffer

    @property
    def candidate_trace_id(self) -> int | None:
        return self.active.candidate_trace_id

    @property
    def normalizer_trace_id(self) -> int | None:
        return self.active.normalizer_trace_id

    @property
    def normalizer_buffer(self) -> Any:
        return self.active.normalizer_buffer

    @property
    def normalizer_inv_temperature(self) -> Any:
        return self.active.normalizer_inv_temperature

    @property
    def row_read_buckets(self) -> tuple[int, ...]:
        return self.active.row_read_buckets

    @property
    def row_read_indices(self) -> dict[int, Any]:
        return self.active.row_read_indices

    @property
    def row_read_blocks(self) -> dict[int, Any]:
        return self.active.row_read_blocks

    @property
    def row_read_traces(self) -> dict[int, int]:
        return self.active.row_read_traces

    @property
    def has_row_read(self) -> bool:
        return bool(self.active.row_read_traces)

    def read_logits_rows(self, lanes: Sequence[int]) -> torch.Tensor:
        """The last lane replay's logits of the given lanes (distinct lanes below the active bucket's width, in this
        order), fp32 ``[len(lanes), VOCAB_SIZE]``: the rows :meth:`read_full_logits` returns for them, bit for bit (the
        LM head's padding columns are not masked here either).  Writes the lane indices into the smallest row-read
        bucket's input (unused slots repeat the first lane), replays its trace and reads the row blocks.  More lanes than
        the largest bucket: ``ValueError`` (use the full gather).
        """

        active = self.active
        lanes = [int(lane) for lane in lanes]
        if not active.row_read_traces:
            raise RuntimeError("the row-read traces are not captured")
        if not lanes or len(set(lanes)) != len(lanes) or not all(0 <= lane < active.lanes for lane in lanes):
            raise ValueError(f"row read needs distinct lanes in 0..{active.lanes - 1}, got {lanes}")
        size = row_read_bucket(len(lanes), active.row_read_buckets)
        if size is None:
            raise ValueError(f"{len(lanes)} lanes are above the largest row-read bucket {active.row_read_buckets[-1]}")
        padded = lanes + [lanes[0]] * (size - len(lanes))
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor(padded, dtype=torch.int32).reshape(1, 1, size),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh),
            ),
            active.row_read_indices[size],
        )
        ttnn._ttnn_execute_trace(self.mesh, active.row_read_traces[size], cq_id=0, blocking=True)
        return self._read_row_block(size, active)[: len(lanes)].to(torch.float32)

    @property
    def has_candidates(self) -> bool:
        return self.active.candidate_trace_id is not None

    @property
    def has_normalizer(self) -> bool:
        return self.active.normalizer_trace_id is not None

    def read_candidates_normalized(self, inv_temperatures: Sequence[float]) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`read_candidates` plus the full-row normalizer: writes the lanes' inverse temperatures (one per lane,
        ``lanes`` or the active bucket's width of them, the first ``active.lanes`` are used; 1.0 for a lane that needs
        no normalizer), replays the candidate + normalizer trace and reads both buffers.  Returns ``(candidate rows
        fp32 [active.lanes, 256], shard LSEs fp32 [active.lanes, 4])``: entry (u, d) = logsumexp over shard d's real
        vocabulary columns of lane u's logits x ``inv_temperatures[u]``."""

        active = self.active
        if active.normalizer_trace_id is None:
            raise RuntimeError("the normalizer trace is not captured")
        if len(inv_temperatures) not in (self.lanes, active.lanes):
            raise ValueError(
                f"need one inverse temperature per lane: got {len(inv_temperatures)}, expected {self.lanes} "
                f"or {active.lanes}"
            )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([float(v) for v in inv_temperatures[: active.lanes]], dtype=torch.float32).reshape(
                    1, 1, active.lanes, 1
                ),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh),
            ),
            active.normalizer_inv_temperature,
        )
        ttnn._ttnn_execute_trace(self.mesh, active.normalizer_trace_id, cq_id=0, blocking=True)
        return self._read_candidate_buffer(active), self._read_normalizer_buffer(active)

    def read_candidates(self) -> torch.Tensor:
        """Replay the active bucket's candidate trace over the last lane replay's logits and read the buffer: fp32
        ``[active.lanes, 256]``, row u = lane u's ``[values | global ids]`` packs of the four shards
        (``Qwen38CandidateRows`` parses it)."""

        active = self.active
        if active.candidate_trace_id is None:
            raise RuntimeError("the candidate trace is not captured")
        ttnn._ttnn_execute_trace(self.mesh, active.candidate_trace_id, cq_id=0, blocking=True)
        return self._read_candidate_buffer(active)

    def warm_bucket_reads(self) -> None:
        """Replay every bucket's candidate trace, normalizer trace (inverse temperatures 1.0) and row-read traces once
        with readback (the adapter's warmup), then restore the full bucket as active."""

        try:
            for r in self.buckets:
                bucket = self.active = self.bucket_sets[r]
                if bucket.candidate_trace_id is not None:
                    self.read_candidates()
                if bucket.normalizer_trace_id is not None:
                    self.read_candidates_normalized([1.0] * bucket.lanes)
                for size in bucket.row_read_traces:
                    self.read_logits_rows(list(range(size)))
        finally:
            self.active = self.bucket_sets[self.lanes]

    def _prepare_pack(self, chain: Any) -> None:
        """The multi-user pack inside the chain's warm hook (before any capture: its persistent buffers must exist before
        the chain's traces, P3d-2 problem 1): the pack state, one eager pack per enabled bucket (every program of the body
        compiles here), the landing's select buffers and every landing program.  The pack traces are captured in
        :meth:`capture`."""

        if self.paging is None:
            raise RuntimeError("the prefill pack lands into paged lanes")
        if chain.slab_state is None or chain.slab_state.rows != pack_module.PACK_ROWS:
            raise RuntimeError(f"the prefill pack runs on the {pack_module.PACK_ROWS}-row slab: the chain holds none")
        model, mesh, ls = self.model, self.mesh, self.ls
        started = time.perf_counter()
        before = self._dram_free()
        self.pack_state = model.allocate_pack_state(chain.state, chain.slab_state, buckets=self.pack_buckets)
        ttnn.synchronize_device(mesh)
        after_state = self._dram_free()
        tokens = [[(7 * i + 13) % 5000 + 1 for i in range(40)]]
        for rows in self.pack_buckets:
            model.run_pack(self.pack_state, rows, chain.state, tokens)
        self.lander = Qwen38TTNNPackLander(model, self.pack_state, chain.state, ls, self.pager_l)
        after_lander = self._dram_free()
        self.lander.warm()
        model.reset_lane_state_inplace(ls, [0] * self.lanes)
        ttnn.synchronize_device(mesh)
        self.pack_dram.update(
            before_state=before,
            state_bytes=before - after_state,
            lander_bytes=after_state - after_lander,
            estimate_bytes=pack_dram_bytes(self.pack_buckets, int(model.allocated_context)),
        )
        self.timing["pack_prepare_s"] = time.perf_counter() - started
        logger.info(
            f"prefill pack prepared: buckets {self.pack_buckets}, state {self.pack_dram['state_bytes'] / PACK_MIB:.1f} MiB, "
            f"landing buffers {self.pack_dram['lander_bytes'] / PACK_MIB:.2f} MiB, {self.timing['pack_prepare_s']:.1f} s"
        )

    def _dram_free(self) -> int:
        view = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
        return int(view.num_banks) * int(view.total_bytes_free_per_bank)

    def capture_pack(self) -> None:
        """The pack traces, one per enabled bucket, after the lane capture (the bodies were run eagerly in
        :meth:`_prepare_pack`, so nothing compiles here)."""

        chain, mesh, model = self.chain, self.mesh, self.model
        guard = lambda label: resident_decode.forbid_trace_body_host_io_and_sync(phase=f"pack {label}")  # noqa: E731
        before = self._dram_free()
        for rows in self.pack_buckets:
            started = time.perf_counter()
            trace_id, output = model.capture_prefill_pack(self.pack_state, rows, chain.state, guard=guard)
            ttnn.synchronize_device(mesh)
            self.pack_traces[rows] = (trace_id, output)
            self.pack_dram.setdefault("trace_bytes", {})[rows] = before - self._dram_free()
            before = self._dram_free()
            self.pack_dram.setdefault("capture_s", {})[rows] = time.perf_counter() - started
        self.pack_dram["traces_total_bytes"] = sum(self.pack_dram["trace_bytes"].values())
        logger.info(f"prefill pack traces captured: {self.pack_dram}")

    def capture(self, vm: Any) -> None:
        """After the chain opened (its traces captured, the miss guard closed): the capture of the lane body from
        reset positions (the full bucket first, then each smaller bucket over its view), the sampling traces of every
        bucket, the tracker over every trace, the lane mirrors back at position 0."""

        self.vm = vm
        chain, mesh, model, ls = self.chain, self.mesh, self.model, self.ls
        if vm.chain is not chain or ls is None:
            raise RuntimeError("capture follows prepare on the same chain")
        if not chain.misses_forbidden:
            raise RuntimeError("the lane captures run under the chain's closed miss guard")
        started = time.perf_counter()
        ttnn.synchronize_device(mesh)
        guard = lambda label: resident_decode.forbid_trace_body_host_io_and_sync(phase=f"lanes {label}")  # noqa: E731
        order = [self.bucket_sets[self.lanes]] + [self.bucket_sets[r] for r in self.buckets if r < self.lanes]
        for bucket in order:
            bucket_started, before = time.perf_counter(), self._dram_free()
            epilogue = lambda out, s=bucket.ls: model.resolve_lane_tokens(out, s)  # noqa: E731
            cap = model.capture_decode_lanes(bucket.ls, guard=guard, epilogue=epilogue, phase_observer=None, cq_id=0)
            bucket.capture, bucket.trace_id = cap, next(iter(cap.trace_ids.values()))
            self.captures.append(cap)
            self.trace_ids.append(bucket.trace_id)
            ttnn.synchronize_device(mesh)
            if bucket.ls is ls:
                for tensor in (
                    ls.token_row,
                    ls.ple_rows.embedding_rows,
                    ls.position.row,
                    ls.position.shift_row,
                    *self.pager_g.packs.values(),
                    *self.pager_g.kv_stagings,
                ):
                    acknowledge_corruptible(tensor)
            else:  # the view's own persistent input; its token and position rows are the parent's
                acknowledge_corruptible(bucket.ls.ple_rows.embedding_rows)
            self.timing.setdefault("bucket_capture_s", {})[bucket.lanes] = time.perf_counter() - bucket_started
            self.timing.setdefault("bucket_trace_bytes", {})[bucket.lanes] = before - self._dram_free()
        if self.candidates:
            for bucket in order:
                self.capture_candidates(bucket)
        if self.pack_enabled:
            self.capture_pack()
        sampling_ids = []
        for bucket in order:
            sampling_ids += [i for i in (bucket.candidate_trace_id, bucket.normalizer_trace_id) if i is not None]
            sampling_ids += list(bucket.row_read_traces.values())
        for trace_id in (*chain.trace_ids(), *self.trace_ids, *sampling_ids, *self.pack_trace_ids()):
            TraceAllocationTracker.verify_before_replay(mesh, trace_id)
        entries = resident_decode.program_cache_count(mesh)
        logger.info(f"program cache entries after the lane captures {entries} (chain {chain.program_cache_entries})")
        if entries != chain.program_cache_entries:
            raise RuntimeError(f"lane captures compiled programs: {entries} vs {chain.program_cache_entries}")
        model.reset_lane_state_inplace(ls, [0] * self.lanes)  # the mirrors advanced during capture
        ttnn.synchronize_device(mesh)
        self.active = self.bucket_sets[self.lanes]
        self.timing["capture_s"] = time.perf_counter() - started
        logger.info(f"plain lane decoder ready: {self.lanes} lanes, timing {self.timing}")

    def pack_trace_ids(self) -> list[int]:
        return [trace_id for trace_id, _ in self.pack_traces.values()]

    # -- admission ---------------------------------------------------------------------------------

    @property
    def paged(self) -> bool:
        return self.paging is not None

    @property
    def page_size(self) -> int:
        return 0 if self.paging is None else self.paging.page_size

    def free_pages(self) -> int:
        return 0 if self.allocator is None else len(self.allocator.free)

    def admit(self, lane: int, prompt_ids: Sequence[int], page_count: int | None = None) -> int:
        """Prefill ``prompt_ids`` on the B=1 chain, move its state into lane ``lane``; returns the first token t0
        (the greedy token after the prompt, to be consumed by the lane's first step).  Paged: ``page_count`` pages are
        allocated (default: the whole ``decode_steps`` run, ``lane_page_count``; the continuous batch passes the
        prompt's pages and grows the lane later with ``grow_lane``)."""

        vm, model, ls = self.vm, self.model, self.ls
        length = len(prompt_ids)
        started = time.perf_counter()
        out, _ = vm.prefill_forward(
            tokens=torch.tensor([list(prompt_ids)], dtype=torch.int32),
            page_table=None,
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[length],
            start_pos=[0],
            empty_slots=[0],
            sampling_params=greedy_sampling_params(),
        )
        first_token = int(out.reshape(-1)[0])
        context = vm.slot.ple_context
        if len(vm.slot.committed) != length:
            raise RuntimeError(f"adapter committed {len(vm.slot.committed)} tokens, expected {length}")
        prefilled = time.perf_counter()
        if self.paging is not None:
            if self.page_rows[lane]:
                raise RuntimeError(f"lane {lane} still holds pages: release_lane first")
            if page_count is None:
                page_count = lane_page_count(length, self.decode_steps, self.paging.page_size)
            self.page_rows[lane] = self.allocator.allocate(page_count)
            model.write_lane_page_table(ls, page_table_rows(self.page_rows, self.lanes))
            self.page_table_dirty = False  # the whole table, other lanes' pending changes included
        self.move_into_lane(lane, length)
        ttnn.synchronize_device(self.mesh)
        finished = time.perf_counter()
        self.timing.setdefault("admit_s", []).append(finished - prefilled)
        self.timing.setdefault("admit_with_prefill_s", []).append(finished - started)
        self.last_admit = {"prefill_s": prefilled - started, "import_s": finished - prefilled}
        vm.release_request(0)
        self.positions[lane] = length
        self.contexts[lane] = context
        return first_token

    def admit_pack(
        self,
        lanes: Sequence[int],
        prompts: Sequence[Sequence[int]],
        rows_per_user: int,
        pages: Sequence[Sequence[int]] | None = None,
        users: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Prefill ``prompts`` (1..S tokens each, text) as ONE pack of bucket ``rows_per_user`` and land user ``users[i]``
        (default: every prompt, in order) in lane ``lanes[i]``: pack replay, then the device-only landing of every family (``Qwen38TTNNPackLander``), the position
        rows, the PLE contexts and the page rows.  Returns the users' first-token logits fp32 ``[len(prompts), VOCAB]`` (the
        last prompt token is inside the pack; the caller picks / samples the token t0 and feeds it to the lane's first step).
        ``pages``: page rows of the lanes (vLLM's block ids, external mode); omitted: allocated here like :meth:`admit`
        (``decode_steps`` ahead).  The caller owns the lanes (free: no page row, no position)."""

        if not self.pack_enabled or self.pack_state is None or rows_per_user not in self.pack_traces:
            raise RuntimeError(f"no pack trace for {rows_per_user} rows per user (QWEN38_PREFILL_PACK=1, buckets)")
        lanes = [int(lane) for lane in lanes]
        prompts = [[int(t) for t in prompt] for prompt in prompts]
        users = list(range(len(lanes))) if users is None else [int(u) for u in users]
        bucket = self.pack_state.buckets[rows_per_user]
        if not lanes or len(lanes) != len(users) or len(set(lanes)) != len(lanes) or len(set(users)) != len(users):
            raise ValueError(f"{len(users)} users and {len(lanes)} lanes must be equal in number and distinct")
        if not users or len(prompts) > bucket.users or min(users) < 0 or max(users) >= len(prompts):
            raise ValueError(f"users {users} outside the {len(prompts)} prompts of a {bucket.users}-user pack")
        for prompt in prompts:
            if not 1 <= len(prompt) <= rows_per_user:
                raise ValueError(f"{len(prompt)} tokens outside the {rows_per_user}-row bucket")
        for lane in lanes:
            if not 0 <= lane < self.lanes:
                raise ValueError(f"lane {lane} outside the {self.lanes} lanes")
            if self.page_rows[lane] or self.positions[lane] is not None:
                raise RuntimeError(f"lane {lane} is still live: release it first")
        landed = [prompts[u] for u in users]
        model, ls, mesh = self.model, self.ls, self.mesh
        started = time.perf_counter()
        allocated = []
        try:
            for index, (lane, prompt) in enumerate(zip(lanes, landed)):
                if pages is None:
                    self.page_rows[lane] = self.allocator.allocate(
                        lane_page_count(len(prompt), self.decode_steps, self.paging.page_size)
                    )
                    allocated.append(lane)
                else:
                    self.set_page_row(lane, pages[index])
            model.write_lane_page_table(ls, page_table_rows(self.page_rows, self.lanes))
            self.page_table_dirty = False  # the whole table, other lanes' pending changes included
            trace_id, output = self.pack_traces[rows_per_user]
            TraceAllocationTracker.verify_before_replay(mesh, trace_id)
            timing: dict[str, float] = {}
            info: dict[str, Any] = {}
            logits = model.run_pack(
                self.pack_state,
                rows_per_user,
                self.chain.state,
                prompts,
                trace_id=trace_id,
                output=output,
                timing=timing,
                info=info,
            )
            packed = time.perf_counter()
            for lane, prompt in zip(lanes, landed):
                model.admit_lane_position(ls, lane, len(prompt), rope_shift=0)
            self.lander.land([(lane, u, len(prompts[u])) for lane, u in zip(lanes, users)], rows_per_user)
            ttnn.synchronize_device(mesh)
        except BaseException:
            for lane in lanes:
                if pages is None and lane in allocated:
                    self.allocator.release(self.page_rows[lane])
                self.page_rows[lane] = []
            self.page_table_dirty = True
            raise
        finished = time.perf_counter()
        for lane, u in zip(lanes, users):
            self.positions[lane] = len(prompts[u])
            self.contexts[lane] = info["contexts"][u]
        pack_ms = (packed - started) * 1e3
        self.last_pack = {
            "users": len(prompts),
            "landed": len(users),
            "rows_per_user": rows_per_user,
            "pack_ms": pack_ms,
            "prepare_ms": timing["prepare_ms"],
            "upload_ms": timing["upload_ms"],
            "replay_ms": timing["replay_ms"],
            "logits_ms": pack_ms - timing["prepare_ms"] - timing["upload_ms"] - timing["replay_ms"],
            "land_ms": (finished - packed) * 1e3,
            "land_ms_per_user": (finished - packed) * 1e3 / len(users),
            "total_ms": (finished - started) * 1e3,
        }
        self.last_admit = {"prefill_s": pack_ms / 1e3, "import_s": (finished - packed) / 1e3}
        self.timing.setdefault("admit_pack_s", []).append(finished - started)
        return logits[users]

    def move_into_lane(self, lane: int, length: int) -> None:
        """The chain's generic state (after ``length`` tokens) into lane ``lane``: the page table row is already written.
        Paged lanes import device to device (``Qwen38TTNNLaneImporter``: the lane's position row and PLE context, the KV
        scatter, every other family written whole, no keep-mask reset: the writes cover what the reset zeroes); the
        rest, and ``QWEN38_IMPORT_HOST=1``, go through the host pager (the generic image evicted, its ring slots rotated
        into the canonical order, the lane reset, the image re-admitted)."""

        model, ls = self.model, self.ls
        if self.importer is not None and not self.host_import:
            model.admit_lane_position(ls, lane, length, rope_shift=0)
            model.admit_lane_kv_paged(ls, self.chain.state, lane, length)
            self.importer.import_lane(lane, length)
            return
        slot = self.pool.slots[0]
        self.pager_g.evict(0, slot)
        self.pager_g.rotate_ring_canonical(slot, length)
        slot.position = length
        model.admit_lane(ls, lane, length, rope_shift=0)
        if self.paging is not None:
            model.admit_lane_kv_paged(ls, self.chain.state, lane, length)
        self.pager_l.readmit(slot, lane, position_row=None)

    def release_lane(self, lane: int, write_table: bool = True) -> None:
        """Free lane ``lane``: paged, its pages go back to the allocator and its page-table row is zeroed (the lane then
        reads and writes the scratch page until it is admitted again; ``write_table=False`` defers the upload to
        ``sync_page_table``); unpaged, only the host state is cleared.  The lane's device state stays as it is:
        ``admit`` resets it at the next admission."""

        if self.paging is not None:
            self.allocator.release(self.page_rows[lane])
            self.page_rows[lane] = []
            self.page_table_dirty = True
            if write_table:
                self.sync_page_table()
        self.positions[lane] = None
        self.contexts[lane] = None

    def grow_lane(self, lane: int) -> None:
        """Allocate one more page for lane ``lane`` (its next step writes into a page it does not hold yet); the table
        upload is left to ``sync_page_table``."""

        if self.paging is None:
            raise RuntimeError("grow_lane allocates pages: the decoder is not paged")
        if len(self.page_rows[lane]) >= self.paging.max_blocks:
            raise RuntimeError(f"lane {lane} already fills its page table ({self.paging.max_blocks} blocks)")
        self.page_rows[lane] += self.allocator.allocate(1)
        self.page_table_dirty = True

    def sync_page_table(self) -> bool:
        """Upload the whole page table if a row changed since the last upload; returns whether it wrote."""

        if self.paging is None or not self.page_table_dirty:
            return False
        self.model.write_lane_page_table(self.ls, page_table_rows(self.page_rows, self.lanes))
        self.page_table_dirty = False
        return True

    def park_idle(self, lanes: Iterable[int]) -> bool:
        """Park idle lanes at position 0 before a step (the trace advanced them): one position-row write, only when an
        idle lane's mirror is off 0; returns whether it wrote."""

        lanes = list(lanes)
        position = self.ls.position
        if not any(position.positions[lane] or position.shifts[lane] for lane in lanes):
            return False
        position.park(lanes)
        return True

    # -- vLLM-owned pages (P2a) --------------------------------------------------------------------
    #
    # The vLLM adapter owns the block ids (the scheduler's), so it never calls ``allocator``: it writes the page rows
    # itself (``set_page_row``), imports a prefill into a lane with the rows it was given (``import_lane``) and frees a
    # lane without returning pages to the unused allocator (``release_lane_external``).

    def lane_position(self, lane: int) -> int:
        """The lane's position mirror: the position its next step consumes (the device row's host twin)."""

        return int(self.ls.position.positions[lane])

    def set_page_row(self, lane: int, pages: Sequence[int]) -> bool:
        """Lane ``lane``'s page row (vLLM block ids, no padding); marks the table dirty only when the row changed.
        Returns whether it changed; the upload is left to ``sync_page_table``."""

        if self.paging is None:
            raise RuntimeError("set_page_row writes the page table: the decoder is not paged")
        pages = [int(page) for page in pages]
        if len(pages) > self.paging.max_blocks:
            raise ValueError(f"lane {lane} holds {len(pages)} pages, the table {self.paging.max_blocks}")
        if pages == self.page_rows[lane]:
            return False
        self.page_rows[lane] = pages
        self.page_table_dirty = True
        return True

    def import_lane(self, vm: Any, lane: int, length: int, context: Any, pages: Sequence[int]) -> None:
        """Move the B=1 chain's just-prefilled state (``length`` tokens, PLE context ``context``) into lane ``lane``
        with the page row ``pages``: ``admit`` without its prefill and without the allocator (the pages are vLLM's)."""

        model, ls = self.model, self.ls
        started = time.perf_counter()
        if self.page_rows[lane] or self.positions[lane] is not None:
            raise RuntimeError(f"lane {lane} is still live: release_lane_external first")
        self.page_rows[lane] = []
        self.set_page_row(lane, pages)
        model.write_lane_page_table(ls, page_table_rows(self.page_rows, self.lanes))
        self.page_table_dirty = False  # the whole table, other lanes' pending changes included
        self.move_into_lane(lane, length)
        ttnn.synchronize_device(self.mesh)
        self.positions[lane] = length
        self.contexts[lane] = context
        self.last_admit = {"prefill_s": 0.0, "import_s": time.perf_counter() - started}

    def release_lane_external(self, lane: int) -> bool:
        """Free lane ``lane`` (host state cleared, page row zeroed; the upload is left to ``sync_page_table``, the
        parking to ``park_idle``); the blocks are vLLM's to free.  Returns whether the lane was live."""

        was_live = self.positions[lane] is not None or bool(self.page_rows[lane])
        if self.page_rows[lane]:
            self.page_rows[lane] = []
            self.page_table_dirty = True
        self.positions[lane] = None
        self.contexts[lane] = None
        return was_live

    def read_full_logits(self) -> torch.Tensor:
        """The last replay's logits of every lane of the active bucket, gathered over the vocabulary shards: fp32
        ``[active.lanes, VOCAB]`` (the LM head's padding columns are not masked here).  The capture retains the logits
        tensor."""

        active = self.active
        return self.chain.sampling.read_full_logits_rows(active.capture.output.logits)[: active.lanes]

    # -- decode ------------------------------------------------------------------------------------

    def bucket_for_step(self) -> int:
        """The width the next step replays: the smallest bucket above the highest live lane (the smallest bucket when
        no lane is live); ``force_bucket`` overrides it and must still hold every live lane."""

        live = [u for u in range(self.lanes) if self.positions[u] is not None]
        need = max(live) + 1 if live else 1
        if self.force_bucket is None:
            return next(r for r in self.buckets if r >= need)
        if self.force_bucket not in self.bucket_sets:
            raise ValueError(f"force_bucket {self.force_bucket} is not one of the buckets {self.buckets}")
        if self.force_bucket < need:
            raise ValueError(f"force_bucket {self.force_bucket} does not hold live lane {need - 1}")
        return self.force_bucket

    def compaction_moves(self) -> list[tuple[int, int]]:
        """The ``(src, dst)`` lane moves that let the next step replay the smallest bucket that holds the live lanes'
        count: ``[]`` with one bucket or when every live lane is below that bucket ``b``; else each live lane >= ``b``
        (highest first) is paired with the lowest free lane < ``b`` (no request, no page row), in order.  Enough free
        lanes exist below ``b``: the live count is at most ``b``."""

        if len(self.buckets) == 1:
            return []
        live = sorted(u for u in range(self.lanes) if self.positions[u] is not None)
        b = next(r for r in self.buckets if r >= max(len(live), 1))
        high = [u for u in reversed(live) if u >= b]
        if not high:
            return []
        free = [u for u in range(b) if self.positions[u] is None and not self.page_rows[u]]
        return list(zip(high, free))

    def move_lane(self, src: int, dst: int) -> None:
        """Move live lane ``src`` into the free lane ``dst`` (paged decoder): every per-lane device tensor through the
        local pager's pack of ``src`` and unpack into ``dst`` (device only; the paged KV stays in its pages), the
        position row and rope shift, the page-table row (the caller's ``sync_page_table`` uploads it) and the host
        mirrors.  ``src`` is parked and free afterwards."""

        if self.paging is None or self.pager_l is None:
            raise RuntimeError("move_lane needs a prepared paged decoder")
        for name, lane in (("src", src), ("dst", dst)):
            if isinstance(lane, bool) or not isinstance(lane, int) or not 0 <= lane < self.lanes:
                raise ValueError(f"move_lane {name} must be a lane in [0, {self.lanes}), got {lane!r}")
        if src == dst:
            raise ValueError(f"move_lane needs two lanes, got {src} twice")
        if self.positions[src] is None:
            raise ValueError(f"move_lane: lane {src} holds no request")
        if self.positions[dst] is not None or self.page_rows[dst]:
            raise ValueError(f"move_lane: lane {dst} is not free")
        model, ls, mesh = self.model, self.ls, self.mesh
        started = time.perf_counter()
        with corruptible_allocation_scope(mesh):  # as the pager's evict and readmit: no transient outlives the move
            self.pager_l._run_pack(src)
            self.pager_l._fence()
            self.pager_l._run_unpack(dst)
            self.pager_l._fence()
        position, shift = ls.position.positions[src], ls.position.shifts[src]
        model.admit_lane_position(ls, dst, position, rope_shift=shift)
        ls.position.park([src])
        self.page_rows[dst], self.page_rows[src] = self.page_rows[src], []
        self.page_table_dirty = True
        self.positions[dst], self.positions[src] = self.positions[src], None
        self.contexts[dst], self.contexts[src] = self.contexts[src], None
        ttnn.synchronize_device(mesh)
        self.lane_moves += 1
        self.last_move = {"src": src, "dst": dst, "ms": (time.perf_counter() - started) * 1e3}

    def step(self, tokens: Sequence[int]) -> list[int]:
        """One lane-body replay of the smallest bucket holding every live lane, consuming ``tokens`` (one per lane; the
        bucket reads the first R); returns the lanes' next greedy tokens, -1 for the lanes R and up."""

        model, ls, mesh = self.model, self.ls, self.mesh
        r = self.bucket_for_step()
        bucket = self.bucket_sets[r]
        idle = [u for u in range(self.lanes) if self.positions[u] is None]  # no request: PLE context None
        for u in idle:
            self.contexts[u] = None
        self.contexts[:r] = list(model.write_lane_ple_rows(bucket.ls, tokens[:r], self.contexts[:r]))
        for u in idle:
            self.contexts[u] = None
        model.write_lane_tokens(bucket.ls, tokens[:r])
        ttnn._ttnn_execute_trace(mesh, bucket.trace_id, cq_id=0, blocking=True)
        ls.position.advance_replayed()  # the shared position row: the trace added 1 to all 32 entries
        self.active = bucket
        self.bucket_steps[r] += 1
        row = ttnn.to_torch(ttnn.get_device_tensors(ls.token_row)[0]).reshape(-1)[:r]
        return [int(v) for v in row.to(torch.float32).round().to(torch.int64).tolist()] + [-1] * (self.lanes - r)

    # -- release -----------------------------------------------------------------------------------

    def release(self) -> None:
        """Lane traces, pagers, lane state, capture objects; the caller then calls ``vm.release_persistent_capture``."""

        if self.released or self.mesh is None:
            return
        self.released = True
        mesh = self.mesh
        ttnn.synchronize_device(mesh)
        mesh.set_program_cache_misses_allowed(True)
        try:
            for trace_id in (*self.trace_ids, *self.pack_trace_ids()):
                ttnn.release_trace(mesh, trace_id)
            self.trace_ids.clear()
            for bucket in self.bucket_sets.values():
                for trace_id in (
                    *bucket.row_read_traces.values(),
                    bucket.candidate_trace_id,
                    bucket.normalizer_trace_id,
                ):
                    if trace_id is not None:
                        ttnn.release_trace(mesh, trace_id)
                bucket.row_read_traces.clear()
                bucket.candidate_trace_id = bucket.normalizer_trace_id = None
                for tensor in (
                    bucket.candidate_scratch,
                    *(bucket.normalizer_scratch or ()),
                    bucket.candidate_buffer,
                    bucket.normalizer_buffer,
                    bucket.normalizer_inv_temperature,
                    *bucket.row_read_indices.values(),
                    *bucket.row_read_blocks.values(),
                ):
                    if tensor is not None:
                        ttnn.deallocate(tensor)
                bucket.row_read_indices.clear()
                bucket.row_read_blocks.clear()
                bucket.candidate_scratch = bucket.normalizer_scratch = bucket.candidate_buffer = None
                bucket.normalizer_buffer = bucket.normalizer_inv_temperature = None
                bucket.capture = None
            if self.normalizer_mask is not None:
                ttnn.deallocate(self.normalizer_mask)
                self.normalizer_mask = None
            for _trace_id, output in self.pack_traces.values():
                output.release_tensors()
            self.pack_traces.clear()
            if self.lander is not None:
                self.lander.release()
                self.lander = None
            if self.pack_state is not None:
                self.model.release_pack_state(self.pack_state)
                self.pack_state = None
            for pager in (self.pager_l, self.pager_g):
                if pager is not None:
                    pager.release()
            for r in self.buckets:  # the views before their parent
                bucket = self.bucket_sets[r]
                if bucket.ls is not None and bucket.ls is not self.ls:
                    self.model.release_bucket_lane_state(bucket.ls)
                bucket.ls = None
            if self.ls is not None:
                self.model.release_lane_state(self.ls)
                self.ls = None
            self.captures.clear()
        finally:
            mesh.set_program_cache_misses_allowed(False)


# -- P1d: continuous batching -----------------------------------------------------------------------


@dataclass
class Qwen38Request:
    """One request of a continuous batch: arrives at replay step ``arrival_step`` and wants ``max_new_tokens`` tokens
    (the prefill's t0 counts as the first)."""

    request_id: Any
    prompt_ids: Sequence[int]
    max_new_tokens: int
    arrival_step: int = 0


@dataclass
class _ActiveLane:
    request: Qwen38Request
    tokens: list[int]
    budget: int  # pages reserved for the whole request (0 unpaged)
    record: dict[str, Any]

    @property
    def done(self) -> bool:
        return len(self.tokens) >= self.request.max_new_tokens

    @property
    def next_row(self) -> int:
        """The cache row the next step writes: the prompt rows plus the steps taken so far."""

        return len(self.request.prompt_ids) + len(self.tokens) - 1


class Qwen38ContinuousBatch:
    """Host-only vLLM-style continuous batching over a lane decoder (``Qwen38PlainLaneDecoder`` or a stand-in).

    Loop per iteration: release finished lanes, admit arrived requests into free lanes (lowest lane first, FIFO, no
    skipping), grow the page rows that reach a page boundary, park idle lanes, replay one step when a lane is active,
    record the tokens.  Paged mode admits a request only when ``free pages - outstanding reservations`` covers the
    request's full budget ``ceil((P + max_new_tokens + 1) / page_size)``; the pages themselves are allocated
    lazily (``ceil((P + 1) / page_size)`` at admission, one more when a step first writes into a new page).

    The decoder interface used: ``lanes``, ``paged``, ``page_size``, ``page_rows``, ``free_pages()``, ``admit(lane,
    prompt_ids, page_count)``, ``last_admit`` (dict ``prefill_s``/``import_s``), ``grow_lane(lane)``,
    ``release_lane(lane, write_table=False)``, ``sync_page_table()``, ``park_idle(lanes)``, ``step(tokens)``.
    """

    def __init__(self, decoder: Any, requests: Sequence[Qwen38Request | tuple], clock=time.perf_counter) -> None:
        self.decoder = decoder
        self.clock = clock
        parsed = [r if isinstance(r, Qwen38Request) else Qwen38Request(*r) for r in requests]
        ids = [r.request_id for r in parsed]
        if len(set(ids)) != len(ids):
            raise ValueError("request ids must be unique")
        for r in parsed:
            if r.max_new_tokens < 1 or len(r.prompt_ids) < 1 or r.arrival_step < 0:
                raise ValueError(f"bad request {r.request_id}: need a prompt, max_new_tokens >= 1, arrival_step >= 0")
        self.waiting: list[Qwen38Request] = sorted(parsed, key=lambda r: r.arrival_step)  # stable: FIFO per arrival
        self.paged = bool(decoder.paged)
        self.page_size = decoder.page_size if self.paged else 0
        self.capacity_pages = decoder.free_pages() if self.paged else 0
        if self.paged:
            for r in parsed:
                if self.budget(r) > self.capacity_pages:
                    raise ValueError(
                        f"request {r.request_id} needs {self.budget(r)} pages, the pool {self.capacity_pages}"
                    )
        self.active: dict[int, _ActiveLane] = {}
        self.step_no = 0
        self.records: dict[Any, dict[str, Any]] = {}
        self.step_log: list[dict[str, Any]] = []
        self.page_owners: dict[int, set] = {}
        self.max_pages_in_use = 0
        self.admission_order: list[tuple[Any, int]] = []  # (request id, lane)

    # -- policy pieces -----------------------------------------------------------------------------

    def budget(self, request: Qwen38Request) -> int:
        return lane_page_count(len(request.prompt_ids), request.max_new_tokens, self.page_size) if self.paged else 0

    def outstanding_reservation(self) -> int:
        return sum(a.budget - len(self.decoder.page_rows[lane]) for lane, a in self.active.items())

    def free_lanes(self) -> list[int]:
        return [lane for lane in range(self.decoder.lanes) if lane not in self.active]

    def _note_pages(self, lane: int, request_id: Any) -> None:
        for page in self.decoder.page_rows[lane]:
            self.page_owners.setdefault(page, set()).add(request_id)
        in_use = sum(len(rows) for rows in self.decoder.page_rows)
        self.max_pages_in_use = max(self.max_pages_in_use, in_use)

    def _release_finished(self) -> bool:
        released = False
        for lane in sorted(self.active):
            a = self.active[lane]
            if not a.done:
                continue
            a.record["pages_used"] = a.record["pages_peak"]
            a.record["tokens"] = list(a.tokens)
            a.record["finish_step"] = self.step_no
            self.decoder.release_lane(lane, write_table=False)
            del self.active[lane]
            released = True
        return released

    def _admit_arrived(self) -> bool:
        admitted = False
        while self.waiting and self.waiting[0].arrival_step <= self.step_no:
            free = self.free_lanes()
            request = self.waiting[0]
            if not free:
                break
            budget = self.budget(request)
            if self.paged and self.decoder.free_pages() - self.outstanding_reservation() < budget:
                break
            self.waiting.pop(0)
            lane = free[0]
            prompt_len = len(request.prompt_ids)
            count = lane_page_count(prompt_len, 0, self.page_size) if self.paged else None
            t0 = self.decoder.admit(lane, request.prompt_ids, count)
            timing = dict(getattr(self.decoder, "last_admit", {}))
            record = {
                "request_id": request.request_id,
                "lane": lane,
                "prompt_len": prompt_len,
                "max_new_tokens": request.max_new_tokens,
                "arrival_step": request.arrival_step,
                "admit_step": self.step_no,
                "prefill_s": timing.get("prefill_s"),
                "import_s": timing.get("import_s"),
                "budget_pages": budget,
                "pages_peak": len(self.decoder.page_rows[lane]) if self.paged else 0,
            }
            self.records[request.request_id] = record
            self.active[lane] = _ActiveLane(request, [int(t0)], budget, record)
            self.admission_order.append((request.request_id, lane))
            if self.paged:
                self._note_pages(lane, request.request_id)
            admitted = True
        return admitted

    def _grow_pages(self) -> None:
        if not self.paged:
            return
        shift = self.page_size.bit_length() - 1
        for lane in sorted(self.active):
            a = self.active[lane]
            while a.next_row >> shift >= len(self.decoder.page_rows[lane]):
                self.decoder.grow_lane(lane)
            a.record["pages_peak"] = max(a.record["pages_peak"], len(self.decoder.page_rows[lane]))
            self._note_pages(lane, a.request.request_id)

    # -- the loop ----------------------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        decoder = self.decoder
        started = self.clock()
        while True:
            while True:  # a request with max_new_tokens == 1 finishes at admission: free its lane before the step
                self._release_finished()
                self._admit_arrived()
                if not any(a.done for a in self.active.values()):
                    break
            if not self.active:
                if not self.waiting:
                    break
                if self.waiting[0].arrival_step > self.step_no:
                    self.step_no = self.waiting[0].arrival_step  # nothing to replay: jump to the next arrival
                    continue
                raise RuntimeError(f"request {self.waiting[0].request_id} cannot be admitted into an empty batch")
            step_started = self.clock()
            self._grow_pages()
            wrote_table = bool(decoder.sync_page_table()) if self.paged else False
            idle = [lane for lane in range(decoder.lanes) if lane not in self.active]
            parked = bool(decoder.park_idle(idle)) if idle else False
            tokens = [
                self.active[lane].tokens[-1] if lane in self.active else WARM_TOKEN_ID for lane in range(decoder.lanes)
            ]
            out = decoder.step(tokens)
            for lane, a in self.active.items():
                a.tokens.append(int(out[lane]))
            self.step_log.append(
                {
                    "step": self.step_no,
                    "ms": 1000.0 * (self.clock() - step_started),
                    "active": len(self.active),
                    "table_write": wrote_table,
                    "parked": parked,
                }
            )
            self.step_no += 1
        wall = self.clock() - started
        return self.report(wall)

    def report(self, wall_s: float) -> dict[str, Any]:
        generated = sum(len(r["tokens"]) for r in self.records.values())
        peak = max((s["active"] for s in self.step_log), default=0)
        steady = [s["ms"] for s in self.step_log if s["active"] == peak]
        reuse = sum(1 for owners in self.page_owners.values() if len(owners) >= 2)
        return {
            "wall_s": wall_s,
            "generated_tokens": generated,
            "effective_tok_s": generated / wall_s if wall_s > 0 else 0.0,
            "steps": len(self.step_log),
            "steady_step_ms": sum(steady) / len(steady) if steady else None,
            "peak_active_lanes": peak,
            "max_pages_in_use": self.max_pages_in_use,
            "page_reuses": reuse,
            "table_writes_in_steps": sum(1 for s in self.step_log if s["table_write"]),
            "requests": list(self.records.values()),
            "step_log": self.step_log,
        }
