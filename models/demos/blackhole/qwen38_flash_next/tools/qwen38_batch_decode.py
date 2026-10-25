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

Residue-free: the lanes sit at any positions (the GDN lane state keeps a per-lane conv shift register, see
``Qwen38TTNNGDNState.fixed_phase``).  No MTP.  ``Qwen38ContinuousBatch`` (host-only, below) is the continuous-batching policy over this decoder (P1d).

Paged KV (P1c-B, ``paging=Qwen38LanePaging(...)``): the lanes' QSA KV is one page pool per layer plus one page table
(``model.write_lane_page_table``).  ``Qwen38PageAllocator`` hands a lane ``ceil((P + decode_steps + 1) / page_size)``
pages from a seeded-shuffled free list over pages ``1 .. num_pages - 1`` (page 0 is the scratch, never handed out), so
the pages are non-contiguous and interleaved across lanes.  Admission writes the table, scatters the B=1 prefill's KV
into the lane's pages (``model.admit_lane_kv_paged``) and moves the other families through the host pager (the pager
layouts exclude the KV family).
"""

from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import torch
from loguru import logger
from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker, acknowledge_corruptible

import ttnn
from models.demos.blackhole.qwen38_flash_next.tools import resident_decode
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import Qwen38TracedChain
from models.demos.blackhole.qwen38_flash_next.ttnn.lanes import (
    Qwen38LaneHostPool,
    Qwen38TTNNLaneImporter,
    Qwen38TTNNLanePager,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging

WARM_TOKEN_ID = 1000  # any in-vocabulary id: the warm passes write garbage into lane state that a reset clears


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


class Qwen38PlainLaneDecoder:
    """B lanes of the plain lane body over one adapter; see the module docstring."""

    def __init__(
        self,
        lanes: int,
        paging: Qwen38LanePaging | None = None,
        decode_steps: int = 32,
        seed: int = 0,
        indexer_form: str | None = None,
    ) -> None:
        self.lanes = lanes
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
        self.captures: list[Any] = []
        self.trace_ids: list[int] = []
        self.timing: dict[str, Any] = {}
        self.released = False
        self.page_table_dirty = False  # a page row changed since the last table write (grow, release)
        self.last_admit: dict[str, float] = {}
        # P2a (vLLM adapter): compile the full-vocabulary gather of the lane logits in ``prepare`` (eager, misses allowed)
        # so ``read_full_logits`` runs under the closed miss guard; off for the P1 drivers (they read greedy ids only).
        self.warm_logits_gather = False

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
        model.reset_lane_state_inplace(ls, zeros)
        ttnn.synchronize_device(mesh)
        self.timing["warm_s"] = time.perf_counter() - started

    def capture(self, vm: Any) -> None:
        """After the chain opened (its traces captured, the miss guard closed): the one capture of the lane body from
        reset positions, the tracker over every trace, the lane mirrors back at position 0."""

        self.vm = vm
        chain, mesh, model, ls = self.chain, self.mesh, self.model, self.ls
        if vm.chain is not chain or ls is None:
            raise RuntimeError("capture follows prepare on the same chain")
        if not chain.misses_forbidden:
            raise RuntimeError("the lane captures run under the chain's closed miss guard")
        started = time.perf_counter()
        ttnn.synchronize_device(mesh)
        guard = lambda label: resident_decode.forbid_trace_body_host_io_and_sync(phase=f"lanes {label}")  # noqa: E731
        epilogue = lambda out: model.resolve_lane_tokens(out, ls)  # noqa: E731
        cap = model.capture_decode_lanes(ls, guard=guard, epilogue=epilogue, phase_observer=None, cq_id=0)
        self.captures.append(cap)
        self.trace_ids.append(next(iter(cap.trace_ids.values())))
        ttnn.synchronize_device(mesh)
        for tensor in (
            ls.token_row,
            ls.ple_rows.embedding_rows,
            ls.position.row,
            ls.position.shift_row,
            *self.pager_g.packs.values(),
            *self.pager_g.kv_stagings,
        ):
            acknowledge_corruptible(tensor)
        for trace_id in (*chain.trace_ids(), *self.trace_ids):
            TraceAllocationTracker.verify_before_replay(mesh, trace_id)
        entries = resident_decode.program_cache_count(mesh)
        logger.info(f"program cache entries after the lane captures {entries} (chain {chain.program_cache_entries})")
        if entries != chain.program_cache_entries:
            raise RuntimeError(f"lane captures compiled programs: {entries} vs {chain.program_cache_entries}")
        model.reset_lane_state_inplace(ls, [0] * self.lanes)  # the mirrors advanced during capture
        ttnn.synchronize_device(mesh)
        self.timing["capture_s"] = time.perf_counter() - started
        logger.info(f"plain lane decoder ready: {self.lanes} lanes, timing {self.timing}")

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
        """The last replay's logits of every lane, gathered over the vocabulary shards: fp32 ``[lanes, VOCAB]`` (the
        LM head's padding columns are not masked here).  The capture retains the logits tensor."""

        logits = self.captures[0].output.logits
        return self.chain.sampling.read_full_logits_rows(logits)[: self.lanes]

    # -- decode ------------------------------------------------------------------------------------

    def step(self, tokens: Sequence[int]) -> list[int]:
        """One lane-body replay consuming ``tokens`` (one per lane); returns the lanes' next greedy tokens."""

        model, ls, mesh = self.model, self.ls, self.mesh
        idle = [u for u in range(self.lanes) if self.positions[u] is None]  # no request: PLE context None
        for u in idle:
            self.contexts[u] = None
        self.contexts = list(model.write_lane_ple_rows(ls, tokens, self.contexts))
        for u in idle:
            self.contexts[u] = None
        model.write_lane_tokens(ls, tokens)
        ttnn._ttnn_execute_trace(mesh, self.trace_ids[0], cq_id=0, blocking=True)
        ls.position.advance_replayed()
        row = ttnn.to_torch(ttnn.get_device_tensors(ls.token_row)[0]).reshape(-1)[: self.lanes]
        return [int(v) for v in row.to(torch.float32).round().to(torch.int64).tolist()]

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
            for trace_id in self.trace_ids:
                ttnn.release_trace(mesh, trace_id)
            self.trace_ids.clear()
            for pager in (self.pager_l, self.pager_g):
                if pager is not None:
                    pager.release()
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
