# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generator for zai-org/GLM-5.3-Flash: readiness `Generator` contract over a model backend (stage 06a skeleton).

Two layers (normative description: $GLM53_WS/doc/ttnn/06a-full-model-prep/contract.md in the program workspace):

* `ModelBackend` - the model-level device contract Stage 6 implements in `tt/model.py` (embedding -> 45 decoder layers
  honouring Stage 1's `FunctionalDecoder` contract -> mean of the 4 mHC streams -> RMSNorm -> lm_head -> sampler) and
  that the host-only mocks in `tests/mock_model.py` implement on CPU. It owns device buffers, per-physical-chunk
  prefill, device-only decode, trace capture/replay and the on-device sampler; it never plans chunks, pads prompts
  or tracks request state.
* `Glm53Generator` - token-level logic, identical for every backend: request slots, prompt chunking and padding for
  any logical length, output slicing, positions, page tables, KV/state ownership, stop tokens, max_new_tokens,
  teacher forcing, two-phase warmup, split-sampling trace orchestration, host-work counters.

API levels (readiness_check.contract.Generator):
  low level  `prefill_forward(tokens [B, P], *, page_table, kv_cache, prompt_lens, empty_slots=None, start_pos=None,
             return_all_logits=False, sampling_params=None)` and
             `decode_forward(tokens [B, 1], start_pos [B], *, page_table, kv_cache, slots=None, enable_trace=True,
             sampling_params=None, read_from_device=True)` - the caller (vLLM) owns cache, page table, positions,
             prompt lengths and slots; rows map to fixed request slots; `start_pos == -1` marks an inactive row.
  high level `prefill_logits(ids) -> [1, T, V]`, `generate(ids, max_new_tokens, *, next_input=None, enable_trace=True,
             ...)`, `generate_batch(prompts, max_new_tokens, ...)` - the generator owns cache, page table and slots.

Sampling modes (split sampling): decode trace -> sampler-ready logits; sampler trace (greedy / sample) writes the token
into the persistent decode token input (`tt_out_tok`) and positions advance on device, so the steady-state token-out
loop is replay + replay + one token readback with zero token / position / page-table refreshes (greedy and unseeded
sampling; explicitly seeded requests also refresh their seed each token in models/common/sampling). `host_sampling=True`
is the explicit, optional compatibility mode (logits read back, `tt/sampling.HostSampler`, token written back); it is
never the default and never the measured path.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import json
import os
import time as _time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import torch

from models.autoports.zai_org_glm_5_3_flash.tt.sampling import GREEDY, HostSampler, SamplingParams

# the mock cluster (GLM53_MOCK=1: image rehearsals) runs no kernels and reads zeros from device buffers: the speculative
# step's device-cursor invariants take the host's values there instead of failing (a device never sets it)
_MOCK_DEVICE = os.environ.get("GLM53_MOCK") == "1"

try:  # the plugin's readiness contract (on PYTHONPATH via the bringup environment)
    from readiness_check.contract import Generator as _ReadinessGenerator
except ImportError:  # pragma: no cover - same abstract surface, used only when the plugin runtime is absent

    class _ReadinessGenerator(ABC):
        tokenizer: Any = None

        @abstractmethod
        def prefill_forward(self, tokens, *, page_table, kv_cache, prompt_lens, return_all_logits=False, **kw):
            ...

        @abstractmethod
        def decode_forward(self, tokens, start_pos, *, page_table, kv_cache, **kw):
            ...

        @abstractmethod
        def prefill_logits(self, prompt_token_ids):
            ...

        @abstractmethod
        def generate(self, prompt_token_ids, max_new_tokens, *, next_input=None, enable_trace=True, **kw):
            ...

        @abstractmethod
        def reset(self):
            ...


NextInputFn = Callable[[int, int], int]

# Stage 1 constants (tt/functional_decoder.py; asserted equal by tests/test_generator.py without importing ttnn)
PREFILL_CHUNK = 2048
BLOCK_SIZE = 64
PREFILL_ALIGN = {"kda": 32, "mla": 64}
MODEL_PREFILL_ALIGN = 64  # one token chunk feeds every layer: lcm of the per-kind alignments
MAX_DECODE_SLOTS = 32  # FusedKDA.allocate_state cap; ttnn.sampling users per call
DECODE_BUCKETS = (1, 2, 4, 8, 16, 32)  # decode trace widths (qwen36 / tt_transformers convention)
MAX_IMPLICIT_SEQ_LEN = 65536  # largest generator-owned context allocated without an explicit max_seq_len
SEED_SKIP = -2  # DecodeIO.seeds row: keep the slot's RNG state
SEED_UNSEEDED = -1  # DecodeIO.seeds row: fresh unseeded RNG state for a new request


def decode_widths(max_batch: int, bucketed: bool | Sequence[int] = True) -> tuple[int, ...]:
    """Decode trace widths for `max_batch` slots: the power-of-two buckets below it plus max_batch itself; only
    max_batch when the backend's width-w trace does the full-width work anyway (`ModelBackend.width_bucketed_decode`
    False), so neither extra captures nor slot compaction can save anything; the given widths below max_batch plus
    max_batch when the backend names the widths that have their own traces (a sequence)."""
    if not bucketed:
        return (max_batch,)
    if bucketed is True:
        return tuple(w for w in DECODE_BUCKETS if w < max_batch) + (max_batch,)
    return tuple(sorted({int(w) for w in bucketed if 1 <= int(w) < max_batch})) + (max_batch,)


def round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def row_align(env=None) -> int:
    """GLM53_PF_ROW_ALIGN (cf3h-k2 fix (a), coordinator 2026-10-26 13:38:01Z; 9's K2-M1 pre-check): "bucket" = the cache
    row aligned to the LARGEST prefill bucket (max of GLM53_PREFILL_BUCKETS, else 2,048) instead of MODEL_PREFILL_ALIGN:
    a non-aligned continuation chunk near max_seq_len replays a whole bucket past its start, so the row must cover
    max_seq_len + bucket - 1 (1M: 1,049,088 -> 1,050,624; 64K: 66,048 -> 67,584) and TPPrefill captures its top
    window at that row; an int = that alignment; unset / "0" = MODEL_PREFILL_ALIGN (the k line, the host tests)"""
    env = os.environ if env is None else env
    v = (env.get("GLM53_PF_ROW_ALIGN", "") or "").strip().lower()
    if v in ("", "0"):
        return MODEL_PREFILL_ALIGN
    if v == "bucket":
        bs = [int(x) for x in (env.get("GLM53_PREFILL_BUCKETS", "") or "").split(",") if x.strip()]
        return max([MODEL_PREFILL_ALIGN] + (bs or [2048]))
    return max(MODEL_PREFILL_ALIGN, int(v))


def blocks_per_seq(max_seq_len: int, block_size: int = BLOCK_SIZE, align: int | None = None) -> int:
    """Stage 1 page-table width: the padded tail of a prompt's last chunk may end up to align-1 tokens past
    max_seq_len, and rows are a multiple of 8 int32 entries (32-byte sticks). align None = row_align() (env)."""
    align = row_align() if align is None else int(align)
    return round_up(round_up(max_seq_len + align - 1, block_size) // block_size, 8)


WINDOW_DENSE_MAX = 2051  # the dense window's bound (tt/mk_mla.py DSA_DENSE_MAX; dsa_common.hpp DENSE_MAX)


@dataclass(frozen=True)
class ChunkPlan:
    start: int  # absolute position of the chunk's first token
    offset: int  # offset in the caller's token list
    n_valid: int
    physical: int  # padded length passed to the backend (multiple of MODEL_PREFILL_ALIGN)


def plan_prefill(seq_len: int, start_pos: int = 0, chunk: int = PREFILL_CHUNK, align: int = MODEL_PREFILL_ALIGN):
    """Chunk plan for `seq_len` new tokens starting at absolute position `start_pos`: full `chunk`-token chunks and
    one partial last chunk padded to `align`. Any logical length >= 1 is accepted."""
    if seq_len < 1:
        raise ValueError("prefill needs at least one token")
    plans = []
    for off in range(0, seq_len, chunk):
        n = min(chunk, seq_len - off)
        plans.append(ChunkPlan(start=start_pos + off, offset=off, n_valid=n, physical=round_up(n, align)))
    return plans


# ---- Stage 16 slice-boundary mitigation (coordinator 2026-10-22T23:13:59Z, Stage 16's TP8 per-row probe): in a
# sequence-parallel (SP) prefill bucket chip r holds rows [r R, (r + 1) R), R = bucket rows / chips; a chunk whose LAST
# valid row is the FIRST row of a chip's slice ((L - 1) mod R == 0: golden[:97] in the 256 bucket, chip 3's first row)
# gets a wrong first token. Replicated buckets are unaffected. The rule (slice_plan) moves such a chunk to the next
# larger bucket, else chunks SLICE_SPLIT_FRONT tokens off its front and runs the rest in its smallest safe bucket (the
# 2,048 bucket). Row 0 (a 1-token chunk: chip 0's first row, the chunk's own first row) is NOT counted as a boundary:
# every SP bucket and every packed layout puts it there, so no bucket choice can move it (counted apart, see
# Glm53Generator._slice_safe), and the split never creates such a chunk.
SLICE_SPLIT_FRONT = 128


def slice_boundary(last_row: int, rows_per_chip: int) -> bool:
    """the chunk row `last_row` (a chunk's or packed segment's last valid row) is the first row of chip r >= 1's slice
    of a sequence-parallel chunk with `rows_per_chip` rows per chip (0: replicated -- never a boundary)"""
    return rows_per_chip > 0 and last_row > 0 and last_row % rows_per_chip == 0


def slice_safe_bucket(n_valid: int, natural: int, buckets: Sequence[int], rows: Callable[[int], int],
                      runs: Callable[[int], bool]) -> Optional[int]:  # fmt: skip
    """the smallest bucket >= `natural` (the chunk's own bucket) whose geometry keeps a chunk of n_valid tokens off
    the slice boundary: `natural` itself when safe, else the smallest larger bucket that `runs(c)` (a captured key
    that holds the chunk) with rows(c) rows per chip; None when there is none"""
    if not slice_boundary(n_valid - 1, rows(natural)):
        return natural
    for c in sorted(int(c_) for c_ in buckets):
        if c > natural and runs(c) and not slice_boundary(n_valid - 1, rows(c)):
            return c
    return None


def slice_plan(chunk_start: int, n_valid: int, front: int, choose: Callable[[int, int], tuple]) -> tuple[list, str]:
    """The slice-boundary rule for ONE chunk -> (lengths of the chunks it runs as, in order; the branch that fired).
    `choose(start, n)` -> (natural bucket, the bucket the backend runs the chunk in = slice_safe_bucket, or None),
    `front` = the tokens split off (SLICE_SPLIT_FRONT rounded up to the continuation alignment). Branches:
      ""            the natural bucket is safe (no firing)
      "larger"      ONE chunk in the smallest larger bucket that runs it safely: the next larger bucket whenever that
                    one is safe, else a further one when it replays no more rows than the split (buckets 256,512,2048:
                    L = 65 / 129 / 193 -> 2,048; the split is infeasible at 65 / 129 and costs 256 + 2,048 rows at 193)
      "split"       [front, L - front]: the front chunk (head-less; its last row front - 1 is odd, never a boundary) and
                    the rest in its smallest safe bucket (the 2,048 bucket: L - front - 1 = -front mod R, safe iff
                    R > front); taken when no larger bucket is safe or the safe one costs more rows (8,192 for a
                    513-token chunk vs 256 + 2,048) -- never with a 1-token rest (row 0, L = front + 1)
      "unavoidable" no safe geometry (no larger safe bucket, split infeasible): the natural bucket"""
    nat, safe = choose(chunk_start, n_valid)
    if safe == nat:
        return [n_valid], ""
    split = None
    if n_valid - front >= 2:  # never a 1-token rest (row 0)
        _, f_safe = choose(chunk_start, front)
        _, r_safe = choose(chunk_start + front, n_valid - front)
        if f_safe is not None and r_safe is not None:
            split = f_safe + r_safe  # bucket rows replayed
    if safe is not None and (split is None or safe <= split):
        return [n_valid], "larger"
    if split is not None:
        return [front, n_valid - front], "split"
    return [n_valid], "unavoidable"


# ------------------------------------------------------------------------------------------------ backend contract
@dataclass
class DecodeIO:
    """Persistent device buffers of the decode / sampling traces (allocated once per KV cache, before capture).

    `tokens` - the token input the NEXT decode step consumes (on the layer pipeline: chip 0's embedding input, i.e. the
    buffer the per-step D2H -> H2D socket hop fills). Generator-visible semantics: after a sampler step it holds, for
    every row, that step's sampled token (tt_out_tok, delivered by the hop before the next step is submitted); a host
    `write(tokens)` issued after a step and before the next one takes precedence over that step's delivered value
    (teacher forcing, host sampling, request admission). The generator never READS `tokens`.
    `sampled` - the sampler's output of the most recent sampler step (last chip; the caller-visible readback copy); the
    generator reads it, never `tokens`, to learn sampled values. The two differ exactly on rows the host rewrote.
    `cur_pos` (-1 = inactive slot) is advanced on device by the decode trace for active rows; `page_table` is shared by
    every MLA layer; `logits` is the decode trace's sampler-ready output (rows of inactive slots must be finite: the
    backend masks them, see ModelBackend.decode); `k/p/temp/seeds` are the sampler's per-slot parameter buffers
    (seeds: SEED_SKIP / SEED_UNSEEDED / seed >= 0 per row, the write itself re-seeds like ttnn.manual_seed; this
    generator writes them at request boundaries only - models/common/sampling additionally re-hashes explicit request
    seeds every token, see contract.md section 7)."""

    tokens: Any
    cur_pos: Any
    page_table: Any
    logits: Any
    sampled: Any
    k: Any
    p: Any
    temp: Any
    seeds: Any


def _export_state(fn, kv_cache, slot_or_entry: int, n_tokens: Optional[int], pools: bool = True, source: str = "live"):
    """call a backend's export_kda_state / export_snapshot, passing n_tokens only to one that takes it (backends without
    DSA state, e.g. the host mocks, keep the two-argument form); pools=False (the pooled keys ride in the LMCache pages)
    only to one that takes it; source != "live" (the in-replay snapshot at the cut n_tokens) only to one that takes it,
    else ValueError (a backend without the snapshot has no state at the cut)"""
    if source != "live":
        try:
            takes = "source" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            takes = False
        if not takes or n_tokens is None:
            raise ValueError(f"this backend cannot export the KDA state from the {source} (n_tokens {n_tokens})")
        kw = {} if pools else {"pools": False}
        return fn(kv_cache, slot_or_entry, n_tokens=int(n_tokens), source=source, **kw)
    if not pools:
        try:
            if "pools" in inspect.signature(fn).parameters:
                return fn(kv_cache, slot_or_entry, n_tokens=None if n_tokens is None else int(n_tokens), pools=False)
        except (TypeError, ValueError):
            pass
    if n_tokens is not None:
        try:
            takes = "n_tokens" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            takes = False
        if takes:
            return fn(kv_cache, slot_or_entry, n_tokens=int(n_tokens))
    return fn(kv_cache, slot_or_entry)


class ModelBackend(ABC):
    """Model-level contract the generator drives. Stage 6 implements it on the mesh (tt/model.py); tests/mock_model.py
    implements it on CPU (MockModel over Stage-1-contract MockDecoders, GoldenReplayModel over the fp32 goldens).

    Static facts the generator reads:
      vocab_size (154,880), logits_width (>= vocab_size; TP-padded columns are masked by the sampler), max_batch
      (decode slots, <= 32), supported_context (min over layer kinds: 2,051 while MLA/DSA runs the dense-exact regime,
      1,048,576 target), block_size (64), prefill_chunk_tokens (2048), prefill_align (64), continuation_align (64: MLA chunk
      starts are page multiples), layer_kinds.
    """

    vocab_size: int
    logits_width: int
    max_batch: int
    supported_context: int
    block_size: int = BLOCK_SIZE
    prefill_chunk_tokens: int = PREFILL_CHUNK
    prefill_align: int = MODEL_PREFILL_ALIGN
    continuation_align: int = BLOCK_SIZE
    layer_kinds: tuple = ()
    #: a width-w decode trace runs only slots 0..w-1 (False: every width replays the full max_batch pipeline; a
    #: sequence: only these widths have their own traces, the others replay the full-width one)
    width_bucketed_decode: bool | tuple = True
    #: copy_slot moves the slot's device sampler RNG with it (False: the destination row is re-seeded, so a slot move
    #: restarts an explicitly seeded request's device stream and serving must not move such requests)
    copy_slot_keeps_sampler_rng: bool = True
    #: the decode trace also runs the device sampler over its own logits (io.k / io.p / io.temp / io.seeds rows; greedy
    #: rows are k=1) into io.tokens / io.sampled, after writing io.logits (still read on host-route steps): ONE trace
    #: replay per decode step (the Stage 14 persistent program, PRODUCTION_PATH.md M5). The generator then captures and
    #: replays no sampler trace; `sample()` may do nothing (the Stage 14b stack's eager decode step samples as well).
    #: On a host-route step the fused step has drawn from the device RNG of device-sampled rows too (their seeded
    #: streams advance one draw per such step) and has overwritten every token row, so the generator handles every
    #: decode step's token rows (traced or eager) like a sampler step's.
    samples_in_decode_trace: bool = False

    # ---- state ownership
    @abstractmethod
    def allocate_kv_cache(self, *, max_batch: int, max_seq_len: int, num_blocks: Optional[int] = None) -> Any:
        """Per-layer caches for `max_batch` slots (KDA: S [slots,64,128,128] fp32 + conv rows; MLA: paged latent
        [num_blocks, 1, 64, 512], default num_blocks = slots * blocks_per_seq; a serving pool may be smaller or
        larger). Returned object is opaque to the generator; it exposes `num_blocks`."""

    def latent_pages(self, kv_cache) -> dict[int, Any]:
        """{decoder layer index: that MLA/DSA layer's paged latent cache of `kv_cache` ([num_blocks, 1, 64, 512] bf16,
        page b = vLLM block b)}: what a vLLM KV connector moves (vllm-tt-plugin `kv_cache_connector_views`, LMCache).
        A backend that does not expose its pages refuses connectors."""
        raise NotImplementedError(f"{type(self).__name__} does not expose its latent pages to KV connectors")

    def export_kda_state(self, kv_cache, slot: int, n_tokens: Optional[int] = None) -> Any:
        """Host copy of `slot`'s per-request state that is not paged (KDA recurrent S + conv rows of every KDA layer,
        and in the sparse regime the DSA pooled keys of the first ceil(n_tokens / 128) units + ring): the checkpoint a
        KV-connector prefix hit restores together with the prefix's latent pages (doc/ttnn/15-lmcache/page_contract.md
        section 5). A backend without it refuses connectors."""
        raise NotImplementedError(f"{type(self).__name__} cannot export per-slot KDA state")

    def import_kda_state(self, kv_cache, slot: int, states: Any) -> None:
        """Inverse of export_kda_state: write a checkpoint into `slot` (no other slot changes)."""
        raise NotImplementedError(f"{type(self).__name__} cannot import per-slot KDA state")

    # ---- prefix-cache snapshots (tt/prefix_cache.py; doc/ttnn/09-vllm/srv/prefixcache): a device snapshot pool of
    # `snapshot_entries` entries holding one slot's non-paged state each (KDA S + conv rows of every KDA layer, the DSA
    # pooled keys + ring in the sparse regime). Entry k is addressed as the virtual slot max_batch + k by the same
    # device programs that move state between slots: plain DMA (SlotOps) between steps, or INSIDE the prefill trace
    # (state_src / state_snap of prefill_chunk: the chunk starts from the entry's state instead of the slot's and / or
    # its final state is also written to an entry, at no host-dispatched copy). Never eager model execution.
    snapshot_entries: int = 0
    #: prefill_chunk honours state_src / state_snap itself (inside its trace); False: the generator copies with
    #: restore_slot / snapshot_slot around the chunk
    prefill_state_io: bool = False

    def decode_tee(self, kv_cache) -> bool:
        """True: the decode trace itself copies crossing slots into snapshot entries (set_decode_tee before the step)"""
        return False

    def set_decode_tee(self, kv_cache, pairs) -> None:
        """the next decode step's in-trace snapshots [(slot, entry, t0, t1)]: the entry already holds the slot's per-token
        state (DSA pooled keys) of positions < t0, the step ends at t1 tokens; [] clears. Only when decode_tee()."""
        raise NotImplementedError(f"{type(self).__name__} has no decode tee")

    def snapshot_slot(self, kv_cache, slot: int, entry: int, upto: Optional[int] = None) -> None:
        """Copy slot's non-paged state into snapshot entry `entry` (device DMA; the slot keeps its state). upto: the
        tokens the state covers (paged per-token state past it, e.g. DSA pooled keys, need not move)."""
        raise NotImplementedError(f"{type(self).__name__} has no snapshot pool")

    def restore_slot(self, kv_cache, entry: int, slot: int) -> None:
        """Copy snapshot entry `entry` into slot's state (device DMA; the entry keeps its state)."""
        raise NotImplementedError(f"{type(self).__name__} has no snapshot pool")

    def export_snapshot(self, kv_cache, entry: int, n_tokens: Optional[int] = None) -> Any:
        """Host blob of a snapshot entry (the host tier): export_kda_state's layout at the virtual slot (n_tokens: the
        entry's boundary, bounds the DSA rows exported in the sparse regime)."""
        return _export_state(self.export_kda_state, kv_cache, self.max_batch + int(entry), n_tokens)

    def import_snapshot(self, kv_cache, entry: int, states: Any) -> None:
        self.import_kda_state(kv_cache, self.max_batch + int(entry), states)

    @abstractmethod
    def allocate_decode_io(self, kv_cache) -> DecodeIO:
        """Persistent decode inputs/outputs bound to `kv_cache` (page table initialised to the identity mapping)."""

    @abstractmethod
    def write(self, buf, value: torch.Tensor) -> None:
        """Host -> device copy INTO the existing buffer (ttnn.copy_host_to_device_tensor; same shape and dtype)."""

    @abstractmethod
    def read(self, buf) -> torch.Tensor:
        """Device -> host copy (the only readback path)."""

    def read_async(self, buf) -> tuple[Any, list]:
        """Enqueue the device -> host copy of `buf` behind the work submitted so far without waiting for it: returns
        (pending, events). After every event completed, `finish_read(buf, pending)` gives what `read(buf)` would have
        returned at the enqueue point, even if later steps were submitted meanwhile. Default: a blocking read."""
        return self.read(buf), []

    def finish_read(self, buf, pending) -> torch.Tensor:
        return pending

    @abstractmethod
    def set_page_table_host(self, kv_cache, page_table: torch.Tensor) -> None:
        """Host copy of the page table used by prefill chunk tables (Stage 1 `DecoderCache.page_table_host`) for every
        MLA layer; the device copy lives in DecodeIO.page_table and is written by the generator."""

    def release_slot(self, kv_cache, slot: int) -> None:
        """Optional: the serving adapter released `slot` (its request finished, was aborted or preempted). The
        generator already stops decoding the slot and drops its page coverage; the backend needs no device work
        (a fresh prefill from position 0 re-attaches the slot's state), so the default is a no-op. A backend may
        record it (the CPU mock's slot ledger) or zero state eagerly."""
        return None

    @abstractmethod
    def reset_slot(self, kv_cache, slot: int) -> None:
        """Fresh request in `slot` (zero KDA state / conv rows; MLA needs nothing)."""

    @abstractmethod
    def copy_slot(self, kv_cache, src: int, dst: int) -> None:
        """Copy one slot's per-request device state to another slot (vLLM slot_remap): KDA S + conv rows and the
        sampler's per-slot RNG / seed state (models/common/sampling: SeedManager.apply_slot_remap). MLA latent pages
        are not copied: they follow the page-table row, which the generator moves."""

    @abstractmethod
    def reset_all(self, kv_cache) -> None:
        """Zero every slot's per-request state (generator reset); buffers, compiled programs and traces survive."""

    # ---- prefill (eager per physical chunk; the generator plans chunks, pads and slices)
    @abstractmethod
    def prepare_prefill(self, kv_cache) -> None:
        """Warmup phase 1: compile every prefill program (every physical bucket of every layer kind, embedding,
        last-row and all-rows heads, prefill sampler). Overwrites slot 0; called before any trace capture."""

    @abstractmethod
    def prefill_chunk(
        self,
        tokens: torch.Tensor,
        *,
        kv_cache,
        slot: int,
        chunk_start: int,
        n_valid: int,
        logits_rows: str,
        sample: Optional[SamplingParams] = None,
        io: Optional[DecodeIO] = None,
        state_src: Optional[int] = None,
        state_snap: Optional[int] = None,
    ):
        """One physical chunk: tokens [physical] int (rows >= n_valid are padding, masked by the decoder contract),
        `chunk_start == 0` starts a fresh request in `slot`. Returns host logits for `logits_rows` in {"last" -> [1, W]
        (row n_valid - 1), "all" -> [physical, W], "none" -> None}; with `sample` (and logits_rows == "last") the
        backend samples the last row on device, writes the token into io.tokens[slot] and returns it as an int.
        Prefix cache (`prefill_state_io` backends): `state_src` = the snapshot entry whose state the chunk starts from
        (a fresh request attaching at chunk_start > 0; the slot's own state is ignored), `state_snap` = an entry that
        also receives the state after the chunk."""

    # ---- decode (device-only over DecodeIO). Decode is bucketed by width: a width-w step reads rows 0..w-1 of
    # io.tokens / io.cur_pos / io.page_table, runs slots 0..w-1 (the KDA state of a slot-prefix view), writes
    # io.logits rows 0..w-1 and advances cur_pos rows 0..w-1 that are >= 0; rows >= w (state, cache, inputs) are
    # untouched. One trace per width (DECODE_BUCKETS below max_batch, plus max_batch) so a single user does not pay
    # for 32 rows.
    # Inactive rows inside the width (cur_pos == -1: finished / paused / between prefill chunks) are routine with
    # buckets; the decoders leave their output rows undefined (Stage 1 contract; Stage 2 review C-9: the fused MLA row
    # is not even reproducible and its garbage router ids enter the fused_experts union), so the backend MUST zero
    # those rows at every layer boundary (at least at each MLA layer's output / MoE router input and at the head
    # input), keep their router ids out of the expert union, and leave io.logits rows of inactive slots finite (or
    # -inf-masked except one column) for every row the sampler reads: ttnn.topk / ttnn.sampling have no defined
    # behaviour on NaN/Inf. The generator never consumes inactive rows' outputs.
    # Token feedback on the layer pipeline: io.tokens is the stage-0 (embedding chip) token input, the sampler and
    # tt_out_tok live on the last chip; the backend owns the per-step hop between them (dsv4-style D2H / H2D PCIe
    # sockets captured inside the per-chip traces, no Python per-token copy) and must honour the DecodeIO token
    # semantics: the hop delivers step n's samples into io.tokens before step n+1 runs, and a host write(io.tokens)
    # issued between the two steps wins over the delivered value; io.sampled holds step n's samples for read().
    @abstractmethod
    def prepare_decode(self, io: DecodeIO, kv_cache, widths: Sequence[int]) -> None:
        """Warmup phase 1: compile decode for EVERY width in `widths`, the position advance and both sampler modes
        over `io` - all before the first capture."""

    @abstractmethod
    def decode(self, io: DecodeIO, kv_cache, width: int) -> None:
        """Eager width-`width` decode step (debug path): same device work as that width's trace."""

    @abstractmethod
    def sample(self, io: DecodeIO, mode: str) -> None:
        """Eager on-device sampler over io.logits for every slot -> io.tokens (tt_out_tok) and io.sampled."""

    @abstractmethod
    def capture_decode(self, io: DecodeIO, kv_cache, width: int) -> Any:
        """Capture the width-`width` decode trace: io.tokens / io.cur_pos / io.page_table rows 0..width-1 ->
        io.logits rows 0..width-1, then cur_pos += 1 on those rows that are >= 0 (ttnn.plus_one skip_negative_entries).
        """

    @abstractmethod
    def capture_sampling(self, io: DecodeIO, mode: str) -> Any:
        """Capture the sampler trace for `mode` in {"greedy", "sample"}: io.logits -> io.tokens (tt_out_tok), io.sampled."""

    @abstractmethod
    def replay(self, trace) -> None:
        """ttnn.execute_trace."""

    @abstractmethod
    def release(self, trace) -> None:
        """ttnn.release_trace (teardown only)."""

    def close(self) -> None:
        """Model unload, after every trace was released and before the caller closes the mesh (serving shutdown)."""


# ------------------------------------------------------------------------------------------------ generator
@dataclass
class SlotState:
    active: bool = False  # holds a live request
    seq_len: int = 0  # tokens in the cache = the next decode position
    params: SamplingParams = GREEDY
    stop_ids: frozenset = frozenset()
    max_new_tokens: int = 0
    generated: list = field(default_factory=list)
    finished: bool = False


@dataclass
class DecodeOutput:
    """Device-resident result of `decode_forward(read_from_device=False)` (async split). The sampler output buffer
    is shared by all steps, so the handle is valid until the next decode step on the same KV cache: read it with
    `read_decode_output` before submitting the next step (a stale handle raises), then format it with
    `process_decode_output_host`."""

    binding: Any
    step: int
    rows: list
    live: list
    is_tokens: bool = True


@dataclass
class PendingTokens:
    """Host side of `read_decode_output(..., async_read=True)`: the copy of the sampler output was enqueued behind that
    step's sampler, so it holds the step's tokens even when the caller submits more steps before formatting it with
    `process_decode_output_host` (after waiting on the returned events)."""

    binding: Any
    step: int
    rows: list
    live: list
    pending: Any


class _CacheBinding:
    """Generator bookkeeping for one KV cache: DecodeIO, host shadows of the persistent device inputs, traces, slots.

    Decode token input per slot: `tok_val[s]` = the slot's next decode input if known on the host (-1 unknown),
    `dev_ok[s]` = the device token row already holds it (delivered there by tt_out_tok or written by the host).
    Invariant: dev_ok and tok_val == -1 only for rows sampled by the most recent sampler step and not read yet, so the
    value is `io.sampled[s]`. The sampler writes every row, so a slot that does not decode in a step loses its device
    row (dev_ok False) but keeps its host value (read back from `io.sampled` first if it only lived on device).
    Page coverage: `coverage[s]` = page-table entries supplied for the slot; positions >= coverage * block_size are
    not addressable (a decode there raises instead of reading another request's blocks)."""

    def __init__(
        self,
        kv_cache,
        io: DecodeIO,
        max_batch: int,
        max_seq_len: int,
        bps: int,
        num_blocks: int,
        owned: bool,
        bucketed: bool | Sequence[int] = True,
        null_block: Optional[int] = None,
    ):
        self.kv_cache = kv_cache
        self.io = io
        self.owned = owned
        # a block no request ever holds (vLLM's BlockPool.null_block): the tail of every caller row past its coverage
        # (None: the tail repeats the row's last entry)
        self.null_block = null_block
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.blocks_per_seq = bps
        self.num_blocks = num_blocks
        self.identity_ok = num_blocks >= max_batch * bps
        self.pt_shadow = (torch.arange(max_batch * bps, dtype=torch.int64) % num_blocks).reshape(max_batch, bps)
        self.coverage = torch.full((max_batch,), bps if self.identity_ok else 0, dtype=torch.int64)
        self.tok_val = torch.full((max_batch,), -1, dtype=torch.int64)
        self.dev_ok = torch.zeros(max_batch, dtype=torch.bool)
        self.pos_shadow = torch.full((max_batch,), -1, dtype=torch.int64)
        self.params_shadow: list = [None] * max_batch
        self.slots = [SlotState() for _ in range(max_batch)]
        self.step = 0
        self.widths = decode_widths(max_batch, bucketed)
        self.decode_traces: dict = {}  # width -> trace
        self.width_replays: dict = {w: 0 for w in self.widths}
        self.sampling_traces: dict = {}
        self.prepared_prefill = False
        self.prepared_decode = False
        # the sequence-sharded DSA pool (backend.dsa_shard_owners, GLM53_DSA_SHARD): pt_shadow stays LOGICAL (vLLM's
        # blocks); obm maps them to owner-aware PHYSICAL blocks, phys = the table the backend is given
        self.obm = None
        self.phys: Optional[torch.Tensor] = None
        self.null_row: Optional[torch.Tensor] = None

    def identity_row(self, slot: int) -> torch.Tensor:
        return torch.arange(slot * self.blocks_per_seq, (slot + 1) * self.blocks_per_seq, dtype=torch.int64)


def _mock_dram_watch(backend, label: str):
    """tt/mock_dram.py's DRAM headroom probe / ballast around a warm-up phase on the image mock (GLM53_MOCK=1 with a
    mesh backend); a no-op on the device and for host-only backends"""
    if os.environ.get("GLM53_MOCK") != "1" or getattr(backend, "mesh", None) is None:
        return contextlib.nullcontext()
    from models.autoports.zai_org_glm_5_3_flash.tt import mock_dram

    return mock_dram.watch(backend.mesh, label)


def _mock_dram_point(backend, label: str):
    """tt/mock_dram.point on the image mock (GLM53_MOCK=1 with a mesh backend); a no-op on the device"""
    if os.environ.get("GLM53_MOCK") != "1" or getattr(backend, "mesh", None) is None:
        return None
    from models.autoports.zai_org_glm_5_3_flash.tt import mock_dram

    return mock_dram.point(backend.mesh, label)


class Glm53Generator(_ReadinessGenerator):
    def __init__(
        self,
        backend: ModelBackend,
        *,
        tokenizer=None,
        max_batch: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        host_sampling: bool = False,
        stop_token_ids: Optional[Sequence[int]] = None,
        sampler_seed: int = 0,
    ):
        self.backend = backend
        self.tokenizer = tokenizer
        self.max_batch = min(max_batch or backend.max_batch, backend.max_batch)
        if not 1 <= self.max_batch <= MAX_DECODE_SLOTS:
            raise ValueError(f"max_batch {self.max_batch} outside [1, {MAX_DECODE_SLOTS}]")
        # The generator-owned cache holds max_batch x max_seq_len positions: without an explicit max_seq_len the default is
        # the backend's supported context, refused above MAX_IMPLICIT_SEQ_LEN (1,048,576 once the DSA sparse path lands
        # would silently size a max_batch x 1M-token cache).
        if max_seq_len is None and backend.supported_context > MAX_IMPLICIT_SEQ_LEN:
            raise ValueError(
                f"backend supports {backend.supported_context} tokens: pass max_seq_len explicitly (the generator-owned "
                f"cache holds max_batch x max_seq_len positions)"
            )
        self.max_seq_len = max_seq_len or backend.supported_context
        #: decode trace widths (pow2 buckets below max_batch + max_batch, the backend's own trace widths + max_batch,
        #: or max_batch alone when the backend's traces are full-width); serving keeps live requests in the lowest
        #: slots (compaction) when there is more than one, so a lone request replays the width-1 trace
        bucketed = getattr(backend, "width_bucketed_decode", True)
        self.bucketed_decode = bucketed if isinstance(bucketed, bool) else tuple(bucketed)
        self.decode_buckets: tuple[int, ...] = decode_widths(self.max_batch, self.bucketed_decode)
        self.host_sampling = host_sampling
        self.fused_sampling = bool(getattr(backend, "samples_in_decode_trace", False))
        #: prefix caching: page-table rows of live slots may alias fully computed pages (set by the serving adapter
        #: when its PrefixStore is on; the generator-owned path keeps every block exclusive)
        self.shared_pages = False
        if stop_token_ids is None:
            stop_token_ids = getattr(tokenizer, "stop_token_ids", ())
        self.stop_token_ids = frozenset(int(t) for t in stop_token_ids)
        self.host_sampler = HostSampler(self.max_batch, base_seed=sampler_seed)
        self.counters = {
            k: 0
            for k in (
                "prefill_chunks",
                "prefix_attaches",
                "prefix_snapshots",
                "snapshot_copies",
                "prefill_chunks_headless",  # chunks run without the head (intermediate: no read, no host wait)
                "prefill_runs",  # prefill calls that carried > 1 chunk of one request (chunk pipelining)
                "prefill_run_chunks_max",
                "prefill_logits_readback_rows",
                "decode_trace_captures",
                "sampling_trace_captures",
                "decode_trace_replays",
                "sampling_trace_replays",
                "fused_sampling_steps",
                "eager_decode_steps",
                "token_refresh",
                "pos_refresh",
                "page_table_refresh",
                "page_rows_unchanged",  # reloaded rows equal to the published ones: no page-table work (_rows_unchanged)
                "sampling_param_refresh",
                "token_readback",
                "logits_readback",
                "host_sampling_steps",
                "teacher_forced_overrides",
                "seed_writes",
                "prefix_state_imports",
                "prefill_unsampled_rows",  # head-less intermediate chunks (per-row prefill sampling)
                "prefill_packed_steps",  # whole vLLM steps in one packed replay (_packed_step)
                "slot_releases",  # release_slot calls (the backend's release hook ran once each)
                # Stage 16 slice-boundary rule (slice_plan): geometries avoided = larger + split + regrouped
                "prefill_slice_boundary_avoided",
                "prefill_slice_boundary_larger",  # a chunk moved to a larger bucket
                "prefill_slice_boundary_split",  # a chunk split into [front, rest]
                "prefill_slice_boundary_regrouped",  # a packed / batched segment sent to the per-user path
                "prefill_slice_boundary_unavoidable",  # no safe geometry: ran in its natural bucket
                "prefill_slice_row0_chunks",  # 1-token chunks of a sequence-parallel bucket (row 0: not movable)
            )
        }
        self._bindings: dict[int, _CacheBinding] = {}
        self._own: Optional[_CacheBinding] = None

    # ================================================================================================ context / slots
    @property
    def supported_context(self) -> int:
        return min(self.backend.supported_context, self.max_seq_len)

    @property
    def vocab_size(self) -> int:
        return self.backend.vocab_size

    def check_context(self, prompt_len: int, max_new_tokens: int = 1, start_pos: int = 0):
        """The cache holds every prompt token and every generated token but the last: needs
        start_pos + prompt_len + max_new_tokens - 1 <= supported context."""
        if prompt_len < 1:
            raise ValueError("prompt must contain at least one token")
        need = start_pos + prompt_len + max(max_new_tokens, 1) - 1
        if need > self.supported_context:
            raise ValueError(
                f"request needs {need} cache positions (start {start_pos} + prompt {prompt_len} + "
                f"{max_new_tokens} new - 1) > supported context {self.supported_context} "
                f"(backend {self.backend.supported_context}, allocated {self.max_seq_len})"
            )

    def allocate_kv_cache(
        self,
        max_batch: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        num_blocks: Optional[int] = None,
        null_block: Optional[int] = None,
    ):
        """Caller-owned cache (serving): allocate and register a binding; pass it back as `kv_cache=`. `num_blocks`
        sizes the MLA page pool (vLLM block budget); default slots * blocks_per_seq. `null_block`: a block of the pool
        no request ever holds (vLLM's null block 0), the tail of every page-table row past its coverage: the TP prefill
        writes a bucket's pad pages through that tail (tt/prefill_tp.py write_inputs), so it must not repeat a real
        page."""
        mb = min(max_batch or self.max_batch, self.max_batch)
        ms = min(max_seq_len or self.max_seq_len, self.backend.supported_context)
        kv = self.backend.allocate_kv_cache(max_batch=mb, max_seq_len=ms, num_blocks=num_blocks)
        b = self._binding(kv, mb, ms, owned=False)
        if null_block is not None:
            if not 0 <= int(null_block) < b.num_blocks:
                raise ValueError(f"null block {null_block} outside [0, {b.num_blocks})")
            b.null_block = int(null_block)
        return kv

    def latent_pages(self, kv_cache) -> dict:
        """The MLA/DSA paged latent caches of a serving cache (ModelBackend.latent_pages), for KV connectors. On the
        sequence-sharded pool (GLM53_DSA_SHARD) each tensor is every chip's LOCAL pool [nb_local, 1, 64, 512] (one mesh
        tensor, per-chip content) and its page index is the LOCAL block: a connector addresses vLLM's logical block b
        as (owner chip g % C, local block g // C) of its physical block g through connector_shard_map (doc/ttnn/
        14-production-tp8-megakernel/dsa/shard_tp8_design.md section 6.3; Stage 12's LMCache pages per owner)."""
        return self.backend.latent_pages(kv_cache)

    def dsa_shard_map(self, kv_cache):
        """the logical -> owner-aware physical block map of a sequence-sharded cache (tt/owner_blocks.py), else None"""
        return self._binding(kv_cache).obm

    def connector_shard_map(self, kv_cache) -> Optional[dict]:
        """GLM53_DSA_SHARD, for the KV connector (vllm-tt-plugin kv_connector.TTShardedPagedCacheHandle): {"C" owner
        chips, "nb_local" physical blocks per chip (local 0 = the chip's null block), "bpu" blocks per 128-token unit,
        "num_blocks" vLLM's logical pool, "map" the OwnerBlockMap: physical(b) for a STORE (None = no device content),
        admit([(b, column)]) for a LOAD (maps the hit's blocks on their columns' owners before the row exists)};
        None on a replicated pool"""
        b = self._binding(kv_cache)
        if b.obm is None:
            return None
        return dict(C=b.obm.C, nb_local=b.obm.nb_local, bpu=b.obm.bpu, num_blocks=b.num_blocks, map=b.obm)

    def export_slot_state(self, kv_cache, slot: int, n_tokens: Optional[int] = None, pools: bool = True,
                          source: str = "live") -> Any:  # fmt: skip
        """`slot`'s per-request state that is not paged (backend.export_kda_state): a KV-connector prefix checkpoint,
        valid after exactly the slot's cached tokens (n_tokens: that count, bounds the DSA rows in the sparse regime).
        prefill_forward(prefix_states=) restores it. source="snapshot": the KDA state at the cut n_tokens < the slot's
        tokens, from the in-replay snapshot of the prefill that covered it (StackBackend._kda_state_src)."""
        b = self._binding(kv_cache)
        if not 0 <= slot < b.max_batch:
            raise ValueError(f"slot {slot} outside [0, {b.max_batch})")
        self._check_not_speculating(b, "export_slot_state")
        return _export_state(self.backend.export_kda_state, kv_cache, slot, n_tokens, pools=pools, source=source)

    def export_slot_state_async(self, kv_cache, slot: int, n_tokens: Optional[int] = None, dma=None, pools: bool = True,
                                source: str = "live"):  # fmt: skip
        """export_slot_state off the first-token path (Stage 12): the backend snapshots the slot on device and the
        page-DMA saves it while the next steps run; returns a zero-argument callable -> export_slot_state's value (any
        thread), or None (the backend / DMA cannot: export synchronously)"""
        fn = getattr(self.backend, "export_kda_state_async", None)
        if fn is None or dma is None:
            return None
        b = self._binding(kv_cache)
        if not 0 <= slot < b.max_batch:
            raise ValueError(f"slot {slot} outside [0, {b.max_batch})")
        self._check_not_speculating(b, "export_slot_state_async")
        kw = {} if pools else {"pools": False}  # GLM53_KV_POOLED_PAGES: the checkpoint carries no pooled-key rows
        if source != "live":  # the in-replay snapshot at the cut (StackBackend._kda_state_src)
            kw["source"] = source
        return fn(kv_cache, slot, n_tokens=None if n_tokens is None else int(n_tokens), dma=dma, **kw)

    # ---- prefix-cache snapshots (device DMA between steps; the prefill-time variants ride inside the prefill trace
    # through prefill_forward's prefix_attach / snapshot_after)
    def decode_tee(self, kv_cache) -> bool:
        """the decode trace tees crossing slots into their snapshot entries (the backend's DecodeTee)"""
        return bool(getattr(self.backend, "decode_tee", lambda kv: False)(kv_cache))

    def set_decode_tee(self, kv_cache, pairs) -> None:
        """the next decode step's in-trace G snapshots [(slot, entry, t0, t1)]; [] clears (written only on change)"""
        b = self._binding(kv_cache)
        for slot, entry, _, _ in pairs:
            self._check_entry(b, slot, entry)
        if pairs:  # a tee copies the slot's row-0 state: never while slot 0 speculates (its state is in row sel)
            self._check_not_speculating(b, "set_decode_tee")
        self.backend.set_decode_tee(kv_cache, list(pairs))
        self.counters["snapshot_tees"] = self.counters.get("snapshot_tees", 0) + len(pairs)

    def snapshot_slot(self, kv_cache, slot: int, entry: int, upto: Optional[int] = None) -> None:
        """upto: the tokens the slot's state covers (the snapshot's boundary): only the DSA pooled-key pages of
        positions < upto move (None: the whole store row)"""
        b = self._binding(kv_cache)
        self._check_entry(b, slot, entry)
        self._check_not_speculating(b, "snapshot_slot")
        self.backend.snapshot_slot(kv_cache, slot, int(entry), **({} if upto is None else {"upto": int(upto)}))
        self.counters["snapshot_copies"] = self.counters.get("snapshot_copies", 0) + 1

    def restore_slot(self, kv_cache, entry: int, slot: int) -> None:
        b = self._binding(kv_cache)
        self._check_entry(b, slot, entry)
        if b.slots[slot].active:
            raise ValueError(f"restore into slot {slot}, which holds a live request")
        self.backend.restore_slot(kv_cache, int(entry), slot)
        self.counters["snapshot_copies"] = self.counters.get("snapshot_copies", 0) + 1

    def export_snapshot(self, kv_cache, entry: int, n_tokens: Optional[int] = None) -> Any:
        self._check_entry(self._binding(kv_cache), 0, entry)
        return _export_state(self.backend.export_snapshot, kv_cache, int(entry), n_tokens)

    def import_snapshot(self, kv_cache, entry: int, states: Any) -> None:
        self._check_entry(self._binding(kv_cache), 0, entry)
        self.backend.import_snapshot(kv_cache, int(entry), states)

    @staticmethod
    def _check_not_speculating(b: _CacheBinding, what: str) -> None:
        """Slot state is read only between plain steps. While slot 0 speculates, its committed KDA state / conv
        history sits in candidate row n of the shared state until spec_leave (row 0 holds a candidate), so a copy of
        slot 0 would store the wrong state (doc/ttnn/17-mtp-spec-decode/lmcache_spec_contract.md)."""
        if getattr(b, "spec", None) is not None:
            raise RuntimeError(f"{what} while slot 0 speculates: call spec_leave first")

    def _check_entry(self, b: _CacheBinding, slot: int, entry: int) -> None:
        if not 0 <= slot < b.max_batch:
            raise ValueError(f"slot {slot} outside [0, {b.max_batch})")
        if not 0 <= int(entry) < int(getattr(self.backend, "snapshot_entries", 0)):
            raise ValueError(f"snapshot entry {entry} outside [0, {getattr(self.backend, 'snapshot_entries', 0)})")

    def _binding(self, kv_cache, max_batch=None, max_seq_len=None, owned=False) -> _CacheBinding:
        b = self._bindings.get(id(kv_cache))
        if b is None:
            if max_batch is None:
                raise ValueError("unknown kv_cache: allocate it with Glm53Generator.allocate_kv_cache")
            io = self.backend.allocate_decode_io(kv_cache)
            bps = blocks_per_seq(max_seq_len, self.backend.block_size)
            nb = getattr(kv_cache, "num_blocks", 0) or max_batch * bps
            b = _CacheBinding(kv_cache, io, max_batch, max_seq_len, bps, nb, owned, bucketed=self.bucketed_decode)
            owners = getattr(self.backend, "dsa_shard_owners", None)
            spec = owners(kv_cache) if callable(owners) else None
            if spec is not None:  # (C, physical blocks per chip, blocks per unit)
                from models.autoports.zai_org_glm_5_3_flash.tt.owner_blocks import OwnerBlockMap

                b.obm = OwnerBlockMap(*spec)
                b.null_row = torch.tensor([b.obm.null(j) for j in range(bps)], dtype=torch.int64)
                b.phys = b.null_row.expand(max_batch, bps).clone()
                b.phys, done, err = self._physical_rows(b, b.pt_shadow, b.coverage, range(max_batch))
                if err is not None:
                    raise err
                self.backend.write(io.page_table, b.phys.to(torch.int32))
            self.backend.set_page_table_host(kv_cache, (b.pt_shadow if b.obm is None else b.phys).to(torch.int32))
            self._bindings[id(kv_cache)] = b
        return b

    def _own_binding(self) -> _CacheBinding:
        if self._own is None:
            kv = self.backend.allocate_kv_cache(max_batch=self.max_batch, max_seq_len=self.max_seq_len)
            self._own = self._binding(kv, self.max_batch, self.max_seq_len, owned=True)
        return self._own

    # ================================================================================================ warmup / traces
    def warmup(self, kv_cache=None, *, capture: bool = True):
        """Two-phase warmup ($tt-enable-tracing). Phase 1 (once per KV cache) prepares every prefill bucket, decode,
        the position advance and both sampler modes without capturing; Stage 1's `prepare_prefill` runs real chunks on
        slot 0, so phase 1 must run at setup, before any request (raises if a request is live), and the idle request
        state is restored after it. Phase 2 captures the decode trace and both sampler traces; capture records only and
        leaves request state alone. Repeated calls are no-ops."""
        b = self._binding(kv_cache) if kv_cache is not None else self._own_binding()
        self._prepare(b)
        if capture:
            self._ensure_traces(b)
        return b

    def warmup_serving(self, kv_cache, *, phase: str = "capture", prefill: bool = True, decode: bool = True, **_):
        """Plugin warm-up callbacks: "prepare" = phase 1, "capture" = phase 2 (prepares first if needed)."""
        if phase not in ("prepare", "capture"):
            raise ValueError(f"unknown warm-up phase {phase!r}")
        b = self._binding(kv_cache)
        with _mock_dram_watch(self.backend, "prepare"):
            self._prepare(b, prefill=prefill, decode=decode)
        if phase == "capture":
            with _mock_dram_watch(self.backend, "capture"):
                self._ensure_traces(b)
            _mock_dram_point(self.backend, "after_capture")

    def _prepare(self, b: _CacheBinding, prefill: bool = True, decode: bool = True):
        need_p, need_d = prefill and not b.prepared_prefill, decode and not b.prepared_decode
        if not (need_p or need_d):
            return
        if any(st.active for st in b.slots):
            raise RuntimeError("warmup phase 1 (program preparation) must run before any request is admitted")
        if need_p:
            self.backend.prepare_prefill(b.kv_cache)
            b.prepared_prefill = True
        if need_d:
            self.backend.prepare_decode(b.io, b.kv_cache, b.widths)
            b.prepared_decode = True
        # preparation may have run real passes (slot 0 / all-inactive decode): restore the idle state
        self.backend.reset_all(b.kv_cache)
        self._write_positions(b, torch.full((b.max_batch,), -1, dtype=torch.int64), force=True)
        self.backend.write(b.io.tokens, torch.zeros(b.max_batch, dtype=torch.int32))  # known idle rows (see reset())
        b.tok_val[:], b.dev_ok[:] = -1, False
        b.slots = [SlotState() for _ in range(b.max_batch)]

    def _ensure_traces(self, b: _CacheBinding):
        if not (b.prepared_prefill and b.prepared_decode):
            raise RuntimeError("capture requested before warmup phase 1 (prepare_prefill / prepare_decode)")
        for w in b.widths:  # every width was compiled in phase 1; captures follow, none compiles
            if w not in b.decode_traces:
                b.decode_traces[w] = self.backend.capture_decode(b.io, b.kv_cache, w)
                self.counters["decode_trace_captures"] += 1
        for mode in () if self.fused_sampling else ("greedy", "sample"):  # fused: the decode trace samples
            if mode not in b.sampling_traces:
                b.sampling_traces[mode] = self.backend.capture_sampling(b.io, mode)
                self.counters["sampling_trace_captures"] += 1

    def teardown(self):
        for b in self._bindings.values():
            for t in list(b.decode_traces.values()) + list(b.sampling_traces.values()):
                self.backend.release(t)
            b.decode_traces, b.sampling_traces = {}, {}

    def release_traces(self) -> None:
        self.teardown()

    def close(self) -> None:
        """Model unload (serving shutdown, mesh still open): release every trace, then the backend's own resources."""
        self.teardown()
        self.backend.close()

    # ================================================================================================ input refresh
    def _write_positions(self, b: _CacheBinding, pos: torch.Tensor, force: bool = False):
        if force or not torch.equal(pos, b.pos_shadow):
            self.backend.write(b.io.cur_pos, pos.to(torch.int32))
            b.pos_shadow = pos.clone()
            self.counters["pos_refresh"] += 1

    def _read_tokens(self, b: _CacheBinding):
        """Recover next inputs that exist only on device: by the binding invariant they are outputs of the most recent
        sampler step, read from `io.sampled` (the last-chip sampler output), never from `io.tokens` (chip 0's input
        copy, which may hold a host-written value)."""
        dev = self.backend.read(b.io.sampled).to(torch.int64).reshape(-1)
        self.counters["token_readback"] += 1
        fill = b.dev_ok & (b.tok_val < 0)
        b.tok_val = torch.where(fill, dev, b.tok_val)

    def _sync_tokens(self, b: _CacheBinding, want: dict[int, int]):
        """Make the device token buffer hold, for every slot decoding this step (`want`), its next input:
        `want[s] >= 0` = that token, `-1` = the slot's current next input (its device row if still valid, else the
        host value; neither -> ValueError). The buffer is written only when a decoding row needs it; rows whose only
        copy is on device are read back first, so the whole-buffer write loses nothing."""
        target = {}
        for s, t in want.items():
            if t < 0:
                if b.dev_ok[s]:
                    continue  # already in place (tt_out_tok)
                if b.tok_val[s] < 0:
                    raise ValueError(f"slot {s}: next decode token unknown (its device row was overwritten); pass it")
                target[s] = int(b.tok_val[s])
            elif not (b.dev_ok[s] and b.tok_val[s] == t):
                target[s] = int(t)
        if not target:
            return
        if any(bool(b.dev_ok[s]) and b.tok_val[s] < 0 and s not in target for s in range(b.max_batch)):
            self._read_tokens(b)  # the whole-buffer write below must reproduce rows that live only on device
        buf = torch.zeros(b.max_batch, dtype=torch.int64)
        for s in range(b.max_batch):
            if s in target:
                buf[s] = target[s]
            elif b.tok_val[s] >= 0:
                buf[s] = b.tok_val[s]  # in place already (dev_ok) or put back (free: the buffer is written anyway)
            # else 0: a valid embedding index for a row nobody reads
        self.backend.write(b.io.tokens, buf.to(torch.int32))
        self.counters["token_refresh"] += 1
        for s, t in target.items():
            b.tok_val[s] = t
        b.dev_ok = (b.tok_val >= 0) & (buf == b.tok_val)

    def _preserve_offstep_tokens(self, b: _CacheBinding, live: Sequence[int]):
        """The sampler overwrites every token row: fetch the values of slots that are not decoding this step and whose
        next input exists only on device (async step not read yet)."""
        if any(bool(b.dev_ok[s]) and b.tok_val[s] < 0 for s in range(b.max_batch) if s not in live):
            self._read_tokens(b)

    def _after_sample(self, b: _CacheBinding, live: Sequence[int], tok: Optional[torch.Tensor]):
        for s in range(b.max_batch):
            if s in live:
                b.dev_ok[s] = True
                b.tok_val[s] = int(tok[s]) if tok is not None else -1
            else:
                b.dev_ok[s] = False

    def _write_page_rows(
        self,
        b: _CacheBinding,
        rows: dict[int, torch.Tensor],
        used: Optional[dict[int, int]] = None,
        keep: Optional[dict[int, int]] = None,
    ):
        """Rewrite page-table rows (slot -> physical block ids, a caller row may be padded with negatives) in place in
        the bound device page table (a live decode trace keeps reading the same buffer; the host copy feeds the prefill
        chunk tables). The supplied entries set the slot's coverage; the row's tail is the cache's null block (a vLLM
        cache, allocate_kv_cache(null_block=)) or else repeats its last entry. Decode never addresses the tail
        (positions beyond the coverage are rejected), but the TP prefill writes a bucket's pad pages through it
        (tt/prefill_tp.py write_inputs: a 161-token prompt in the 512 bucket writes 5 pad pages): a repeated real page
        would take the pad rows' latents (int24svc 2026-10-22: API TF prefill top-1 0.75). `used[slot]` = positions
        the slot must address now. Checks: entries in [0, num_blocks), never the null block; no block shared by two
        live slots within their used ranges. keep[slot] (the sequence-sharded pool only): positions the slot's blocks
        already hold (a prefix hit's cached tokens; default a live slot's length, else 0): those blocks must still
        map to their physical block (tt/owner_blocks.py refuses a fresh one)."""
        if rows and self._rows_unchanged(b, rows, used):
            self.counters["page_rows_unchanged"] += 1
            return
        bs = self.backend.block_size
        new, cov = b.pt_shadow.clone(), b.coverage.clone()
        for slot, row in rows.items():
            row = torch.as_tensor(row, dtype=torch.int64).reshape(-1)
            neg = torch.nonzero(row < 0)
            valid = row[: int(neg[0])] if neg.numel() else row
            valid = valid[: b.blocks_per_seq]
            if b.obm is not None and valid.numel():
                # the sharded pool: a block listed again later in the row (a caller's pad-by-repeat tail) cannot be
                # one physical block on several owner chips -> the row's coverage ends at its first repeat
                seen_b = set()
                for k, v in enumerate(valid.tolist()):
                    if v in seen_b:
                        valid = valid[:k]
                        break
                    seen_b.add(v)
            if valid.numel() == 0:
                raise ValueError(f"empty page-table row for slot {slot}")
            if (valid >= b.num_blocks).any():
                raise ValueError(f"slot {slot}: page-table entries outside [0, {b.num_blocks})")
            if b.null_block is not None and (valid == b.null_block).any():
                raise ValueError(f"slot {slot}: page-table row holds the null block {b.null_block}")
            if used is not None and slot in used:
                need = min(-(-round_up(max(used[slot], 1), self.backend.prefill_align) // bs), b.blocks_per_seq)
                if valid.numel() < need:
                    raise ValueError(f"slot {slot}: page-table row covers {valid.numel()} blocks < the {need} it needs")
            tail = valid[-1:] if b.null_block is None else torch.tensor([b.null_block], dtype=valid.dtype)
            new[slot] = torch.cat([valid, tail.expand(b.blocks_per_seq - valid.numel())])
            cov[slot] = valid.numel()
        owner = {}
        for slot in range(b.max_batch) if not self.shared_pages else ():
            # prefix caching (shared_pages, set by the serving adapter): live requests share the fully computed pages
            # of a cached prefix read-only (vLLM's block pool refcounts them; a prefill writes only the blocks vLLM
            # allocated for its uncached suffix), so this uniqueness check is off there
            if used is not None and slot in used:
                n_pos = used[slot]
            elif b.slots[slot].active:
                n_pos = b.slots[slot].seq_len + 1
            else:
                continue
            need = min(-(-round_up(max(n_pos, 1), self.backend.prefill_align) // bs), int(cov[slot]))
            for blk in new[slot, :need].tolist():
                if owner.setdefault(blk, slot) != slot:
                    raise ValueError(f"physical block {blk} mapped by live slots {owner[blk]} and {slot}")
        err = None
        if b.obm is not None:
            bs = self.backend.block_size
            kc = {s: -(-int((keep or {}).get(s, b.slots[s].seq_len if b.slots[s].active else 0)) // bs) for s in rows}
            phys, done, err = self._physical_rows(b, new, cov, list(rows), kc)
            if err is not None:  # a refused row (an owner chip out of physical blocks, or a live block listed at
                # another owner's column): the rows mapped before it are published, it and the rest keep their rows
                keep = [s for s in rows if s not in done]
                new[keep], cov[keep] = b.pt_shadow[keep], b.coverage[keep]
        b.coverage = cov
        self._publish_page_table(b, new, phys if b.obm is not None else None)
        if err is not None:
            raise err

    def _rows_unchanged(self, b: _CacheBinding, rows: dict[int, torch.Tensor], used: Optional[dict[int, int]]) -> bool:
        """every caller row lists exactly the slot's published entries (same coverage) and its used positions fit that
        coverage: _write_page_rows would publish nothing. The serving adapter passes every live slot's whole row on each
        reloaded decode step; at 1M context the full path's Python passes over 16,392 columns (the repeat scan, the
        live-block uniqueness scan, the sharded pool's set_row) cost ~35 ms per step (n29h7p2, 2026-10-23), against
        O(1) here. A changed row (a new block every 64 tokens, a new request) takes the full path and its checks."""
        bs = self.backend.block_size
        for slot, row in rows.items():
            row = torch.as_tensor(row).reshape(-1)
            c = int(b.coverage[slot])
            if c == 0 or row.numel() < c or (row.numel() > c and int(row[c]) >= 0):
                return False
            if not torch.equal(row[:c].to(torch.int64), b.pt_shadow[slot, :c]):
                return False
            if used is not None and slot in used:
                if -(-round_up(max(int(used[slot]), 1), self.backend.prefill_align) // bs) > c:
                    return False  # past the coverage: the full path raises
        return True

    def _publish_page_table(self, b: _CacheBinding, new: torch.Tensor, phys: Optional[torch.Tensor]):
        """the logical table `new` (and, for the sharded DSA pool, its physical form) -> the device page table and the
        backend's host copy, when either changed"""
        dev_new = new if b.obm is None else phys
        dev_old = b.pt_shadow if b.obm is None else b.phys
        if b.obm is not None:
            # every chip's local block 0 is its null / scratch block: KV-connector loads write a non-owner chip's rows
            # there (lmshard N1). It is safe only if no slot's COVERED column points at a null block, so readers never
            # see it; the tail past the coverage must be exactly the column owners' null blocks (never another block)
            live = torch.arange(dev_new.shape[1])[None, :] < b.coverage[:, None]
            null_live = live & (dev_new >= 0) & (dev_new < b.obm.C)  # = (dev_new // C == 0), without the int64 divide
            tail_bad = ~live & (dev_new != b.null_row[None, :])
            if bool(null_live.any()) or bool(tail_bad.any()):
                bad = torch.nonzero(null_live | tail_bad)[:4].tolist()
                raise RuntimeError(
                    f"sharded pool: page-table [slot, column] {bad} breaks the null-block rule (a covered "
                    f"column on a null / scratch block, or a tail column off its owner's null block)"
                )
        if not torch.equal(new, b.pt_shadow) or not torch.equal(dev_new, dev_old):
            self.backend.write(b.io.page_table, dev_new.to(torch.int32))
            self.backend.set_page_table_host(b.kv_cache, dev_new.to(torch.int32))
            b.pt_shadow = new
            if b.obm is not None:
                b.phys = dev_new
            self.counters["page_table_refresh"] += 1

    @staticmethod
    def _physical_rows(b: _CacheBinding, table: torch.Tensor, cov: torch.Tensor, slots, keep: Optional[dict] = None):
        """(the sequence-sharded DSA pool) the LOGICAL rows `slots` of `table`, `cov[s]` covered columns each, ->
        owner-aware physical rows (tt/owner_blocks.py: the block of column j lives on chip owner(j) = (j // bpu) % C as
        physical g with g % C == owner(j), local block g // C; past the coverage the column owner's null block).
        Returns (the physical table, the slots mapped, the error that refused the next slot or None): each row is
        mapped atomically, a refused row leaves its previous mapping."""
        phys, done = b.phys.clone(), []
        slots = list(slots)
        # rows only IDLE slots hold (an owned cache's identity rows, a finished request's) are reclaimable: a row being
        # written that lists one of their blocks releases them (the idle slot loses its coverage; a later admission
        # writes its rows again)
        incoming = {int(v) for s in slots for v in table[s, : int(cov[s])].tolist()}
        for s2 in range(b.max_batch):
            r2 = b.obm.rows.get(s2)
            if s2 in slots or not r2 or b.slots[s2].active or not incoming.intersection(r2):
                continue
            b.obm.release_slot(s2)
            phys[s2], cov[s2] = b.null_row, 0
        for s in slots:
            c = int(cov[s])
            try:
                if c > 0:
                    k = (keep or {}).get(s, 0)
                    # the covered head only: past the coverage set_row returns the column owners' null blocks, which is
                    # b.null_row (cf3h-d lever A: no 16,392-column list round trip per changed row)
                    phys[s, :c] = torch.tensor(b.obm.set_row(s, table[s, :c].tolist(), c, keep=k), dtype=torch.int64)
                    phys[s, c:] = b.null_row[c:]
                else:
                    b.obm.release_slot(s)
                    phys[s] = b.null_row
            except (RuntimeError, ValueError) as e:
                return phys, done, e
            done.append(s)
        return phys, done, None

    def _ensure_identity_rows(self, b: _CacheBinding, slots: Sequence[int], used: Optional[dict[int, int]] = None):
        """Generator-owned cache: a request admitted without a caller page table gets the slot's own blocks back."""
        rows = {
            s: b.identity_row(s)
            for s in slots
            if not torch.equal(b.pt_shadow[s], b.identity_row(s)) or b.coverage[s] < b.blocks_per_seq
        }
        if rows:
            self._write_page_rows(b, rows, used)

    def _write_params(self, b: _CacheBinding):
        vals = [b.slots[s].params.device_values() if b.slots[s].active else (1, 0.0, 1.0) for s in range(b.max_batch)]
        if vals != b.params_shadow:
            self.backend.write(b.io.k, torch.tensor([v[0] for v in vals], dtype=torch.int32))
            self.backend.write(b.io.p, torch.tensor([v[1] for v in vals], dtype=torch.float32))
            self.backend.write(b.io.temp, torch.tensor([v[2] for v in vals], dtype=torch.float32))
            b.params_shadow = vals
            self.counters["sampling_param_refresh"] += 1

    def _write_seed(self, b: _CacheBinding, slot: int, seed: Optional[int]):
        """Request boundary: re-seed one slot's device RNG (SEED_SKIP rows keep their state; SEED_UNSEEDED draws a
        fresh state); the sampler then advances its RNG on device. (models/common/sampling hashes explicit request
        seeds per token on the host instead - see contract.md, sampling.)"""
        seeds = torch.full((b.max_batch,), SEED_SKIP, dtype=torch.int64)
        seeds[slot] = SEED_UNSEEDED if seed is None else int(seed)
        self.backend.write(b.io.seeds, seeds)
        self.counters["seed_writes"] += 1

    def _new_request_rng(self, b: _CacheBinding, slot: int, params: SamplingParams):
        self.host_sampler.reset_slot(slot, params.seed)
        self._write_seed(b, slot, params.seed)

    # ================================================================================================ prefill
    def _count_slice(self, branch: str) -> None:
        for br in branch.split("+"):  # "clamp+split": the max_seq clamp (TPPrefill.clamp_lengths) + a piece's branch
            if br == "clamp":
                self.counters["prefill_max_seq_clamped"] = self.counters.get("prefill_max_seq_clamped", 0) + 1
                continue
            key = "prefill_slice_row0_chunks" if br == "row0" else f"prefill_slice_boundary_{br}"
            self.counters[key] = self.counters.get(key, 0) + 1
            if br in ("larger", "split", "regrouped"):
                self.counters["prefill_slice_boundary_avoided"] += 1

    def _slice_rows(self, rows: int) -> int:
        """rows per chip of a `rows`-row prefill replay when the backend runs it sequence-parallel (0: replicated, or
        a backend without the TP prefill's slice geometry)"""
        fn = getattr(self.backend, "prefill_slice_rows", None)
        return int(fn(int(rows))) if fn is not None else 0

    def _slice_safe(self, plans: list) -> list:
        """Stage 16's slice-boundary rule (slice_plan) over a request's chunk plan, on backends that pick a chunk's
        bucket with it (`prefill_slice_plan`: the TP prefill, TPPrefill.slice_plan): a chunk whose last valid row would
        be the first row of a chip's slice runs in a larger bucket (the backend's own choice at replay) or as
        [front, rest] (SLICE_SPLIT_FRONT tokens rounded up to continuation_align, so the rest starts on an aligned
        position); every firing is counted per branch. The chunks still cover the request exactly, in order."""
        rule = getattr(self.backend, "prefill_slice_plan", None)
        if rule is None:
            return plans
        front = round_up(SLICE_SPLIT_FRONT, self.backend.continuation_align)
        out = []
        for pl in plans:
            lens, branch = rule(pl.start, pl.n_valid, front)
            if branch:
                self._count_slice(branch)
            off = 0
            for n in lens:
                out.append(ChunkPlan(start=pl.start + off, offset=pl.offset + off, n_valid=int(n),
                                     physical=round_up(int(n), self.backend.prefill_align)))  # fmt: skip
                off += int(n)
            if off != pl.n_valid:
                raise RuntimeError(f"slice plan {lens} does not cover the chunk [{pl.start}, {pl.start + pl.n_valid})")
        return out

    def _slice_safe_group(self, group: list, lens: list, al: int, sizes: list) -> list:
        """Stage 16's slice-boundary rule for ONE packed replay: segment k's last row o_k + n_k - 1 (o_k = the al-row
        aligned back-to-back offsets of TPPrefill.packed_layout) must not be the first row of a chip's slice of the
        T-row bucket the group replays (the smallest packed T that holds it). Drops the first offending segment -- it
        runs on the per-user path, whose rule handles its own chunk -- and re-lays the rest until none offends; returns
        the kept members in order (the caller skips groups left with < 2)."""
        group, lens = list(group), [int(n) for n in lens]
        while len(group) >= 2:
            need = sum(round_up(n, al) for n in lens)
            t = min(t_ for t_, k_ in sizes if t_ >= need and k_ >= len(group))
            r, o, bad = self._slice_rows(t), 0, None
            for j, n in enumerate(lens):
                if slice_boundary(o + n - 1, r):
                    bad = j
                    break
                o += round_up(n, al)
            if bad is None:
                break
            del group[bad], lens[bad]
            self._count_slice("regrouped")
        return group

    def _admit_batch(self, b: _CacheBinding, users: list) -> dict:
        """Stage 16 batched admission: users = [(key, slot, ids, SlotState)] fresh requests with device sampling. Groups
        of exactly nb (the backend's captured batched-prefill trace, GLM53_PF_BATCH) whose prompts fit its bucket and
        whose page rows cover them are prefilled by ONE replay each; returns {key: first token} of the admitted users
        (their SlotState installed, seq_len / decode token set). The rest stay for the per-user path."""
        done = self._admit_packed(b, users)  # mixed lengths first (GLM53_PF_PACKED), the rest below
        users = [u for u in users if u[0] not in done]
        cap, nb = getattr(self.backend, "prefill_batch_size", lambda: (0, 0))()
        if nb < 2 or self.host_sampling:
            return done
        lim = min(cap, self.backend.supported_context)
        ok = [u for u in users if 1 <= int(u[2].numel()) <= lim
              and -(-round_up(int(u[2].numel()), self.backend.prefill_align) // self.backend.block_size)
              <= b.coverage[u[1]]]  # fmt: skip
        # segment j of a group sits at rows [j cap, j cap + n) of the nb x cap replay (TPPrefill.write_batch): a user
        # whose last row would land on a chip slice's first row there takes the per-user path (slice-boundary rule)
        r_chip, groups, cur = self._slice_rows(nb * cap), [], []
        for u in ok:
            if slice_boundary(len(cur) * cap + int(u[2].numel()) - 1, r_chip):
                self._count_slice("regrouped")
                continue
            cur.append(u)
            if len(cur) == nb:
                groups.append(cur)
                cur = []
        for g in groups:
            for _, slot, _, st in g:
                b.slots[slot] = st
                self._new_request_rng(b, slot, st.params)
            firsts = self.backend.prefill_batch_tp([(ids, slot, int(ids.numel())) for _, slot, ids, _ in g],
                                                   kv_cache=b.kv_cache, io=b.io, sample=[st.params for *_, st in g])  # fmt: skip
            self.counters["prefill_batches"] = self.counters.get("prefill_batches", 0) + 1
            for (key, slot, ids, st), tok in zip(g, firsts):
                st.seq_len = int(ids.numel())
                b.tok_val[slot], b.dev_ok[slot] = int(tok), True
                done[key] = int(tok)
        return done

    def _packed_step(self, b: _CacheBinding, segs: list) -> dict:
        """ONE packed replay for a vLLM step's rows (backend prefill_packed_caps: continuation + no_head): segs =
        [(key, slot, ids, start, head, SamplingParams)] in row order -- fresh prompts (start 0), continuations (start =
        the slot's cached length, a multiple of continuation_align) and head-less intermediate chunks (no sample: the
        slot's RNG does not advance). Packs the longest prefix of rows whose segment-aligned lengths (the backend's
        prefill_packed_align: 32, 128 with the DSA indexer) fit the largest packed bucket T with K <= KMAX (K >= 2; the
        rest run below), then replays the smallest bucket that holds the group (prefill_packed_sizes, as _admit_packed);
        a segment whose last row would be a chip slice's first row runs below too (_slice_safe_group);
        returns {key: first token, or -1 for a head-less row}."""
        sizes = getattr(self.backend, "prefill_packed_sizes", None)
        sizes = sizes() if sizes is not None else [getattr(self.backend, "prefill_packed_size", lambda: (0, 0))()]
        sizes = sorted((int(t_), int(k_)) for t_, k_ in sizes if int(k_) >= 2)
        if not sizes:
            return {}
        t, kmax = sizes[-1]
        seg_al = getattr(self.backend, "prefill_packed_align", lambda: 32)()
        bs, al = self.backend.block_size, self.backend.prefill_align
        group, rows = [], 0
        for key, slot, ids, start, head, sp in segs:
            n = int(ids.numel())
            st = b.slots[slot]
            ok = (1 <= n <= t and start % self.backend.continuation_align == 0
                  and (start == 0 or (st.active and st.seq_len == start))
                  and -(-round_up(start + n, al) // bs) <= b.coverage[slot]
                  and start + n <= min(b.max_seq_len, self.backend.supported_context))  # fmt: skip
            r = round_up(n, seg_al)
            if not ok or rows + r > t or len(group) == kmax:
                break
            group.append((key, slot, ids, start, head, sp))
            rows += r
        group = self._slice_safe_group(group, [int(g_[2].numel()) for g_ in group], seg_al, sizes)
        if len(group) < 2:
            return {}
        rows = sum(round_up(int(g_[2].numel()), seg_al) for g_ in group)
        t = min(t_ for t_, k_ in sizes if t_ >= rows and k_ >= len(group))
        for key, slot, ids, start, head, sp in group:
            if start == 0:
                b.slots[slot] = SlotState(active=True, params=sp)
                self._new_request_rng(b, slot, sp)
            else:
                b.slots[slot].active = True
        firsts = self.backend.prefill_packed_tp([(ids, slot, int(ids.numel())) for _, slot, ids, *_ in group],
                                                kv_cache=b.kv_cache, io=b.io, t=t,
                                                sample=[sp if head else None for *_, head, sp in group],
                                                starts=[start for _, _, _, start, _, _ in group],
                                                heads=[head for *_, head, _ in group])  # fmt: skip
        self.counters["prefill_packed_steps"] = self.counters.get("prefill_packed_steps", 0) + 1
        done = {}
        for (key, slot, ids, start, head, sp), tok in zip(group, firsts):
            st = b.slots[slot]
            st.seq_len = start + int(ids.numel())
            if head:
                b.tok_val[slot], b.dev_ok[slot] = int(tok), True
                done[key] = int(tok)
            else:
                b.tok_val[slot], b.dev_ok[slot] = -1, False
                done[key] = -1
                self.counters["prefill_unsampled_rows"] = self.counters.get("prefill_unsampled_rows", 0) + 1
        return done

    def _admit_packed(self, b: _CacheBinding, users: list) -> dict:
        """Stage 16 PACKED admission (mixed ISL, coordinator 2026-10-22 03:15Z): fresh device-sampled users of DIFFERENT
        prompt lengths packed back to back (32-row aligned) into ONE replay of the backend's (T, K) packed trace
        (GLM53_PF_PACKED), greedily in arrival order while the rounded lengths fit T and K <= KMAX (groups of >= 2; a
        segment on a chip slice's first row leaves its group: _slice_safe_group);
        returns {key: first token} of the admitted users (SlotState installed, seq_len / decode token set)."""
        sizes = getattr(self.backend, "prefill_packed_sizes", None)
        sizes = sizes() if sizes is not None else [getattr(self.backend, "prefill_packed_size", lambda: (0, 0))()]
        sizes = sorted((int(t_), int(k_)) for t_, k_ in sizes if int(k_) >= 2)
        if not sizes or self.host_sampling:
            return {}
        t, kmax = sizes[-1]  # groups against the largest packed bucket; each replays the smallest bucket that holds it
        lim = min(t, self.backend.supported_context)
        ok = [u for u in users if 1 <= int(u[2].numel()) <= lim
              and -(-round_up(int(u[2].numel()), self.backend.prefill_align) // self.backend.block_size)
              <= b.coverage[u[1]]]  # fmt: skip
        al = getattr(self.backend, "prefill_packed_align", lambda: 32)()  # 128 with the DSA indexer
        groups, cur, rows = [], [], 0
        for u in ok:
            r = round_up(int(u[2].numel()), al)
            if cur and (rows + r > t or len(cur) == kmax):
                groups.append(cur)
                cur, rows = [], 0
            cur.append(u)
            rows += r
        if cur:
            groups.append(cur)
        done = {}
        for g in [g_ for g_ in groups if len(g_) >= 2]:
            g = self._slice_safe_group(g, [int(ids.numel()) for _, _, ids, _ in g], al, sizes)
            if len(g) < 2:
                continue
            for _, slot, _, st in g:
                b.slots[slot] = st
                self._new_request_rng(b, slot, st.params)
            rows_g = sum(round_up(int(ids.numel()), al) for _, _, ids, _ in g)
            t_g = min(t_ for t_, k_ in sizes if t_ >= rows_g and k_ >= len(g))
            firsts = self.backend.prefill_packed_tp([(ids, slot, int(ids.numel())) for _, slot, ids, _ in g],
                                                    kv_cache=b.kv_cache, io=b.io, t=t_g,
                                                    sample=[st.params for *_, st in g])  # fmt: skip
            self.counters["prefill_packed"] = self.counters.get("prefill_packed", 0) + 1
            self.counters[f"prefill_packed_t{t_g}"] = self.counters.get(f"prefill_packed_t{t_g}", 0) + 1
            for (key, slot, ids, st), tok in zip(g, firsts):
                st.seq_len = int(ids.numel())
                b.tok_val[slot], b.dev_ok[slot] = int(tok), True
                done[key] = int(tok)
        return done

    def _prefill_user(
        self,
        b: _CacheBinding,
        slot: int,
        ids: torch.Tensor,
        *,
        start_pos: int = 0,
        all_logits: bool = False,
        sample: Optional[SamplingParams] = None,
        no_head: bool = False,
        state_src: Optional[int] = None,
        state_snap: Optional[int] = None,
        snap_at: Optional[int] = None,
    ):
        """Chunked prefill of one user into `slot`: plan, pad, run, slice. Returns logits [n, V] (all), [1, V] (last)
        or the first token sampled on device (int) when `sample` is given and host sampling is off; with `no_head`
        (an intermediate chunk of a longer prompt: its token would be discarded) no chunk produces logits or samples
        and it returns None. Updates the slot's cache length and decode-token state.
        `state_src`: the snapshot entry the first chunk starts from (a prefix-cache attach at start_pos > 0);
        `state_snap`: the entry that also receives the state after the last chunk (a prefix-cache snapshot)."""
        n = int(ids.numel())
        in_trace = bool(getattr(self.backend, "prefill_state_io", False))
        if state_src is not None and not in_trace:  # plain DMA before the chunk (no in-trace state I/O)
            self.backend.restore_slot(b.kv_cache, int(state_src), slot)
            self.counters["snapshot_copies"] = self.counters.get("snapshot_copies", 0) + 1
        if start_pos % self.backend.continuation_align:
            raise ValueError(
                f"continuation start {start_pos} must be a multiple of {self.backend.continuation_align} "
                "(MLA latent page); restart the request from 0 instead"
            )
        st = b.slots[slot]
        if start_pos and start_pos != st.seq_len:
            raise ValueError(f"slot {slot} holds {st.seq_len} tokens; KDA state cannot continue from {start_pos}")
        if start_pos + n > b.max_seq_len or start_pos + n > self.backend.supported_context:
            raise ValueError(
                f"prefill end {start_pos + n} > supported context {min(b.max_seq_len, self.backend.supported_context)}"
            )
        need = -(-round_up(start_pos + n, self.backend.prefill_align) // self.backend.block_size)
        if need > b.coverage[slot]:
            raise ValueError(
                f"slot {slot}: prefill to {start_pos + n} needs {need} page-table entries, has {int(b.coverage[slot])}"
            )
        if start_pos == 0:
            st.seq_len = 0
        plans = plan_prefill(n, start_pos, self.backend.prefill_chunk_tokens, self.backend.prefill_align)
        if not all_logits:  # all-rows logits run the decode-step prefill (no sequence-parallel bucket)
            plans = self._slice_safe(plans)
        outs, result = [], None
        if no_head and (all_logits or sample is not None):
            raise ValueError("no_head excludes all_logits and sample")
        dev_sample = sample if (sample is not None and not self.host_sampling and not all_logits) else None
        if len(plans) > 1:  # a run: several chunks of one request in one call (chunk pipelining)
            self.counters["prefill_runs"] += 1
            self.counters["prefill_run_chunks_max"] = max(self.counters["prefill_run_chunks_max"], len(plans))
        # W1 (GLM53_PF_GROUP_MAX = N >= 2, default 0 = off; design/w1_grpmax_design.md §8): up to N consecutive FULL
        # chunks in ONE layer-major replay (StackBackend.prefill_group: the MoE / FFN
        # of every grouped layer once over the group); a group the backend cannot serve (no served chain, its trace not
        # prewarmed) returns NotImplemented and runs chunk by chunk below
        gmax = int(os.environ.get("GLM53_PF_GROUP_MAX", "0") or 0)
        gdone = 0  # plans [0, gdone) already run by a group

        # group sizes from a fixed set (GLM53_PF_GROUP_SIZES, default 16,8,4,2, capped at N): one GroupStack (per-chunk
        # residuals + the n x c scratch) per size, so a run is decomposed greedily instead of building every n
        gsizes = sorted({int(x) for x in os.environ.get("GLM53_PF_GROUP_SIZES", "16,8,4,2").split(",") if x.strip()
                         and 2 <= int(x) <= gmax}, reverse=True)  # fmt: skip
        def _chunk(i, pl):  # one chunk through the backend's per-chunk path (its key's captured trace)
            nonlocal result
            last = i == len(plans) - 1 and not no_head
            chunk = torch.zeros(pl.physical, dtype=torch.int64)
            chunk[: pl.n_valid] = ids[pl.offset : pl.offset + pl.n_valid]
            rows = "all" if all_logits else ("last" if last else "none")
            sio = {}
            if in_trace:
                if state_src is not None and i == 0:
                    sio["state_src"] = int(state_src)
                if state_snap is not None and i == len(plans) - 1:
                    sio["state_snap"] = int(state_snap)
            if snap_at is not None and pl.start < int(snap_at) <= pl.start + pl.n_valid:
                sio["snap_at"] = int(snap_at)  # one-replay store: this chunk also writes the KDA state at the cut
                self.counters["kda_snapshot_chunks"] = self.counters.get("kda_snapshot_chunks", 0) + 1
            out = self.backend.prefill_chunk(
                chunk,
                kv_cache=b.kv_cache,
                slot=slot,
                chunk_start=pl.start,
                n_valid=pl.n_valid,
                logits_rows=rows,
                sample=dev_sample if last else None,
                io=b.io,
                **sio,
            )
            self.counters["prefill_chunks"] += 1
            if rows == "none":
                self.counters["prefill_chunks_headless"] += 1
            if all_logits:
                outs.append(out[: pl.n_valid, : self.backend.vocab_size])
                self.counters["prefill_logits_readback_rows"] += pl.n_valid
            elif last:
                result = out
                if dev_sample is None:
                    self.counters["prefill_logits_readback_rows"] += 1

        if gsizes and not all_logits and hasattr(self.backend, "prefill_group"):
            gc = int(self.backend.prefill_chunk_tokens)
            i = 0
            while i < len(plans):
                j = i  # the longest groupable run from i: FULL chunks only (a partial chunk may plan to a smaller bucket;
                # it runs chunk by chunk, with the head when it is the request's last)
                # nor the chunk carrying a store-A cut (snap_at): it runs alone, its segment inputs hold no cut scalar
                while j < len(plans) and plans[j].n_valid == gc and not (
                        snap_at is not None and plans[j].start < int(snap_at) <= plans[j].start + gc):  # fmt: skip
                    j += 1
                fit = [x for x in gsizes if x <= j - i]
                if not fit:
                    break  # the rest runs chunk by chunk
                j = i + fit[0]
                grp = plans[i:j]
                last_g = j == len(plans) and not no_head
                toks = torch.zeros(len(grp) * gc, dtype=torch.int64)
                for k_, pl in enumerate(grp):
                    toks[k_ * gc : k_ * gc + pl.n_valid] = ids[pl.offset : pl.offset + pl.n_valid]
                sio = {}
                if in_trace:
                    if state_src is not None and i == 0:
                        sio["state_src"] = int(state_src)
                    if state_snap is not None and j == len(plans):
                        sio["state_snap"] = int(state_snap)
                out = self.backend.prefill_group(toks, kv_cache=b.kv_cache, slot=slot, chunk_start=grp[0].start,
                                                 n_valid=[p_.n_valid for p_ in grp], logits_rows="last" if last_g else "none",
                                                 sample=dev_sample if last_g else None, io=b.io, **sio)  # fmt: skip
                if out is NotImplemented:
                    # this group is not served (no prewarmed key for its windows, e.g. a request's first group at chunk 0):
                    # its chunks run one by one IN ORDER and grouping continues after it -- a cold group no longer turns
                    # the rest of the run per-chunk (coordinator 2026-10-25 20:58Z: 14's G2 rows, prefill_groups 0)
                    for k_ in range(i, j):
                        _chunk(k_, plans[k_])
                    gdone = i = j
                    continue
                self.counters["prefill_groups"] = self.counters.get("prefill_groups", 0) + 1
                self.counters["prefill_chunks"] += len(grp)
                if not last_g:
                    self.counters["prefill_chunks_headless"] += len(grp)
                else:
                    result = out
                    if dev_sample is None:
                        self.counters["prefill_logits_readback_rows"] += 1
                gdone = i = j
        for i, pl in enumerate(plans):
            if i < gdone:
                continue
            _chunk(i, pl)
        st.seq_len = start_pos + n
        if state_snap is not None and not in_trace:  # plain DMA after the chunk
            self.backend.snapshot_slot(b.kv_cache, slot, int(state_snap))
            self.counters["snapshot_copies"] = self.counters.get("snapshot_copies", 0) + 1
        if dev_sample is not None:
            tok = int(result)  # the device sampler wrote it into the slot's decode token row (tt_out_tok)
            b.tok_val[slot], b.dev_ok[slot] = tok, True
            return tok
        b.tok_val[slot], b.dev_ok[slot] = -1, False  # the caller (or the host sampler) supplies the next input
        if no_head:
            return None
        if all_logits:
            return torch.cat(outs)
        return result[..., : self.backend.vocab_size].reshape(1, -1)

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        prompt_lens: List[int],
        return_all_logits: bool = False,
        empty_slots: Optional[Sequence[int]] = None,
        start_pos: Optional[Sequence[int]] = None,
        sampling_params: Optional[Sequence[SamplingParams]] = None,
        prefix_states: Optional[dict] = None,
        sample_rows: Optional[Sequence[bool]] = None,
        prefix_attach: Optional[dict] = None,
        snapshot_after: Optional[dict] = None,
        head_rows: Optional[Sequence[bool]] = None,
        snapshot_at: Optional[dict] = None,
        **kwargs: Any,
    ):
        """Low level: prefill `tokens [B, P_padded]` (real lengths `prompt_lens`) into fixed slots (`empty_slots`, or
        `slots=`, default row i -> slot i) of the caller's `kv_cache` using the caller's page-table rows `page_table
        [B, blocks]` (each row's supplied entries are the request's blocks). Users are prefilled one at a time in
        chunks of <= 2048 tokens (any logical length; `start_pos[i]` continues the slot's cached prefix at a 64-token
        page boundary). A request prefilled from position 0 replaces whatever the slot held. `prefix_states[i]` (KV
        connector prefix hit): row i is a NEW request whose first start_pos[i] tokens were loaded into its blocks by the
        connector; the slot takes it as fresh with that cached length after restoring the prefix's checkpoint
        (export_slot_state of a request at the same boundary). Returns logits [B, 1, V] (last prompt position),
        [B, max_len, V] with `return_all_logits` (rows past a user's length are zero), or the first tokens [B] with
        `sampling_params` (sampled on device into the slot's decode token row; host sampler in compatibility mode).
        `sample_rows[i]` False (with `sampling_params`): row i is an intermediate chunk of a longer prompt (vLLM
        discards its token), so it is prefilled without a head or a sample -- its slot's RNG does not advance -- and
        its output is -1; such rows never join a batched admission.
        Prefix cache (tt/prefix_cache.py): `prefix_attach[i]` = the device snapshot entry row i (a NEW request)
        starts from at start_pos[i] (its first start_pos[i] tokens are the entry's prefix, their pages shared through
        the row's page table); `snapshot_after[i]` = the entry that receives row i's state after this chunk (its end
        must be a snapshot boundary). Both ride inside the prefill trace on backends with `prefill_state_io`, else
        they are plain device copies around the chunk; neither row joins a batched admission.
        `head_rows[i]` False (chunk pipelining, doc/ttnn/09-vllm/chunk_pipelining.md): the same head-less row as
        `sample_rows[i]` False, also without `sampling_params` (its result row is then zero logits)."""
        b = self._binding(kv_cache)
        if os.environ.get("GLM53_MOCK") == "1":  # [mock-dram-point] after the first 1 / 8 / 32 served prefill calls
            self._mock_pf_calls = getattr(self, "_mock_pf_calls", 0) + 1
            if self._mock_pf_calls - 1 in (1, 8, 32):
                _mock_dram_point(self.backend, f"after_served_prefill_calls_{self._mock_pf_calls - 1}")
        if not b.prepared_prefill:
            raise RuntimeError("prefill before warmup: call Glm53Generator.warmup(kv_cache) at setup")
        if return_all_logits and sampling_params is not None:
            raise ValueError("return_all_logits and sampling_params are exclusive")
        tokens = torch.as_tensor(tokens)
        if tokens.dim() == 1:
            tokens = tokens[None]
        batch = tokens.shape[0]
        if len(prompt_lens) != batch:
            raise ValueError("one prompt length per row")
        if empty_slots is None and kwargs.get("slots") is not None:
            empty_slots = kwargs["slots"]  # 09a adapter spelling
        slots = list(empty_slots) if empty_slots is not None else list(range(batch))
        if len(slots) != batch or len(set(slots)) != batch or any(not 0 <= s < b.max_batch for s in slots):
            raise ValueError(f"slots {slots} must be distinct and in [0, {b.max_batch})")
        starts = [int(x) for x in start_pos] if start_pos is not None else [0] * batch
        lens = [int(n) for n in prompt_lens]
        for i, n in enumerate(lens):
            if not 1 <= n <= tokens.shape[1]:
                raise ValueError(f"prompt_lens[{i}] = {n} outside [1, {tokens.shape[1]}]")
        if sample_rows is not None:
            sample_rows = [bool(v) for v in sample_rows]
            if len(sample_rows) != batch or sampling_params is None:
                raise ValueError("sample_rows needs sampling_params and one flag per row")
        skip = [sample_rows is not None and not sample_rows[i] for i in range(batch)]  # intermediate chunks: no head
        if head_rows is not None:  # chunk pipelining's spelling of the same mask (also valid for logits rows)
            if len(head_rows) != batch:
                raise ValueError("one head_rows entry per row")
            if return_all_logits and not all(head_rows):
                raise ValueError("return_all_logits and head_rows are exclusive")
            skip = [skip[i] or not bool(head_rows[i]) for i in range(batch)]
        prefix_states = dict(prefix_states or {})
        for i in prefix_states:
            if not 0 <= i < batch or starts[i] <= 0 or starts[i] % self.backend.continuation_align:
                raise ValueError(
                    f"prefix_states row {i}: a prefix checkpoint continues at a positive multiple of "
                    f"{self.backend.continuation_align} (start_pos {starts[i] if 0 <= i < batch else None})"
                )
            if b.slots[slots[i]].active:
                raise ValueError(f"prefix_states row {i}: slot {slots[i]} holds a live request")
        prefix_attach = {int(k): int(v) for k, v in (prefix_attach or {}).items()}
        snapshot_after = {int(k): int(v) for k, v in (snapshot_after or {}).items()}
        n_entries = int(getattr(self.backend, "snapshot_entries", 0))
        for i, ent in prefix_attach.items():
            if not 0 <= i < batch or starts[i] <= 0 or starts[i] % self.backend.continuation_align:
                raise ValueError(
                    f"prefix_attach row {i}: an attach continues at a positive multiple of "
                    f"{self.backend.continuation_align} (start_pos {starts[i] if 0 <= i < batch else None})"
                )
            if i in prefix_states:
                raise ValueError(f"prefix_attach row {i} also has a host checkpoint (prefix_states)")
            if b.slots[slots[i]].active:
                raise ValueError(f"prefix_attach row {i}: slot {slots[i]} holds a live request")
            if not 0 <= ent < n_entries:
                raise ValueError(f"prefix_attach row {i}: snapshot entry {ent} outside [0, {n_entries})")
        for i, ent in snapshot_after.items():
            if not 0 <= i < batch or not 0 <= ent < n_entries:
                raise ValueError(f"snapshot_after row {i}: snapshot entry {ent} outside [0, {n_entries})")
            if skip[i]:
                pass  # an intermediate chunk may well end on a boundary: its state is a valid snapshot
        page_table = torch.as_tensor(page_table)
        self._write_page_rows(
            b,
            {slots[i]: page_table[i] for i in range(batch)},
            {slots[i]: starts[i] + lens[i] for i in range(batch)},
            keep={slots[i]: starts[i] for i in range(batch)},  # a prefix hit / continuation: [0, start) is computed
        )
        # Stage 16 batched admission: groups of exactly nb fresh, single-chunk, device-sampled requests in ONE replay
        # of the backend's batched-prefill trace (GLM53_PF_BATCH); every other request runs the per-user path below
        heads = [True] * batch if head_rows is None else [bool(h) for h in head_rows]
        if len(heads) != batch:
            raise ValueError("one head_rows entry per row")
        if return_all_logits and not all(heads):
            raise ValueError("return_all_logits and head_rows are exclusive")
        done = {}
        # one-replay store A rows (snapshot_at: a storing row's KDA checkpoint cut inside its chunk) bypass the packed and
        # batched groups too: their replays write no per-segment snapshot, so the export would find none
        snaps = {int(k) for k, v in (snapshot_at or {}).items() if v}
        caps = getattr(self.backend, "prefill_packed_caps", lambda: {})() or {}
        if (sampling_params is not None and not return_all_logits and not self.host_sampling and batch >= 2
                and caps.get("continuation") and caps.get("no_head")):  # fmt: skip
            # the operator's packed step (2026-10-22 07:32:59Z): EVERY row of vLLM's step -- fresh prompts, a long
            # prompt's intermediate chunk (no head), continuations -- in ONE packed replay when they fit (T, KMAX).
            # Prefix-cache rows (prefix_attach / snapshot_after) bypass it: the packed trace has no per-segment state
            # source / snapshot target (prefill_packed_tp reads and writes each segment's own slot only), so they run
            # the per-user path below, whose chunk does the attach / snapshot inside its trace (state_io)
            done = self._packed_step(b, [(i, slots[i], tokens[i, : lens[i]], starts[i], not skip[i],
                                          sampling_params[i] or GREEDY) for i in range(batch)
                                         if i not in prefix_states and i not in prefix_attach
                                         and i not in snapshot_after and i not in snaps])  # fmt: skip
        if sampling_params is not None and not return_all_logits:
            done.update(self._admit_batch(b, [(i, slots[i], tokens[i, : lens[i]],
                                          SlotState(active=True, params=sampling_params[i] or GREEDY))
                                         for i in range(batch)
                                         if starts[i] == 0 and not skip[i] and i not in done
                                         and i not in snapshot_after and i not in snaps]))  # fmt: skip
        outs = []
        for i in range(batch):
            if i in done:
                outs.append(done[i])
                continue
            slot, n = slots[i], lens[i]
            st = b.slots[slot]
            sp = sampling_params[i] if sampling_params is not None else None
            if starts[i] == 0 or i in prefix_states or i in prefix_attach:
                b.slots[slot] = st = SlotState(active=True, params=sp or GREEDY)
                self._new_request_rng(b, slot, st.params)
                if i in prefix_states:  # a fresh request whose prefix the connector loaded: restore its state
                    self.backend.import_kda_state(b.kv_cache, slot, prefix_states[i])
                    st.seq_len = starts[i]
                    self.counters["prefix_state_imports"] += 1
                elif i in prefix_attach:  # a fresh request attaching to a device snapshot: the chunk reads it
                    st.seq_len = starts[i]
                    self.counters["prefix_attaches"] = self.counters.get("prefix_attaches", 0) + 1
                # GLM53_DSA_DENSE_WINDOW (dsa/dense_window_restore_design.md W2): the restored prefix's window rows are
                # its latent pages; a prompt < DENSE_MAX (the only kind whose decode reads the window) gets them copied
                # into its seat's window before its first chunk (which writes the rows >= start)
                rebuild = getattr(self.backend, "rebuild_window", None)
                if (rebuild is not None and (i in prefix_states or i in prefix_attach) and starts[i] > 0
                        and starts[i] + lens[i] < WINDOW_DENSE_MAX):  # fmt: skip
                    phys = b.phys[slot] if b.obm is not None else b.pt_shadow[slot]
                    if rebuild(b.kv_cache, slot, starts[i], phys):
                        self.counters["window_rebuilds"] = self.counters.get("window_rebuilds", 0) + 1
            else:
                st.active = True
            sio = dict(state_src=prefix_attach.get(i), state_snap=snapshot_after.get(i))
            if snapshot_at and snapshot_at.get(i):  # one-replay store A: the KDA state at row i's checkpoint cut
                if not getattr(self.backend, "kda_checkpoint_in_replay", False):
                    raise ValueError("snapshot_at needs a kda_checkpoint_in_replay backend (GLM53_PF_KDA_SNAPSHOT)")
                sio["snap_at"] = int(snapshot_at[i])
            if i in snapshot_after:
                self.counters["prefix_snapshots"] = self.counters.get("prefix_snapshots", 0) + 1
            if skip[i]:  # an intermediate chunk: no head, no sample (the slot's RNG does not advance)
                self._prefill_user(b, slot, tokens[i, :n], start_pos=starts[i], no_head=True, **sio)
                self.counters["prefill_unsampled_rows"] = self.counters.get("prefill_unsampled_rows", 0) + 1
                outs.append(-1 if sp is not None else torch.zeros(1, self.backend.vocab_size))
                continue
            out = self._prefill_user(
                b, slot, tokens[i, :n], start_pos=starts[i], all_logits=return_all_logits, sample=sp, **sio
            )
            if not heads[i]:
                outs.append(-1 if sp is not None else torch.zeros(1, self.backend.vocab_size))
                continue
            if sp is not None and self.host_sampling:
                out = int(self.host_sampler.sample(out.float(), [sp], [slot])[0])
                self.counters["host_sampling_steps"] += 1
                b.tok_val[slot] = out  # written at the next decode
            outs.append(out)
        if sampling_params is not None:
            return torch.tensor([int(o) for o in outs], dtype=torch.int64)
        if return_all_logits:
            width = max(o.shape[0] for o in outs)
            res = torch.zeros(batch, width, self.backend.vocab_size, dtype=outs[0].dtype)
            for i, o in enumerate(outs):
                res[i, : o.shape[0]] = o
            return res
        return torch.stack([o.reshape(-1) for o in outs]).reshape(batch, 1, -1)

    # ================================================================================================ decode
    @staticmethod
    def decode_width(b: _CacheBinding, live: Sequence[int]) -> int:
        """Smallest decode bucket covering every decoding slot (slots 0..w-1). Keeping live requests in the lowest
        slots (generate_batch does; serving can compact with slot_remap) keeps single-user decode at width 1."""
        top = max(live) + 1 if live else 1
        return next(w for w in b.widths if w >= top)

    def _check_decode_rows(self, b: _CacheBinding, pos: dict[int, int]):
        limit = min(b.max_seq_len, self.backend.supported_context)
        for s, p in pos.items():
            if p >= limit:
                raise ValueError(f"slot {s}: decode position {p} >= supported context {limit}")
            if p != b.slots[s].seq_len:
                raise ValueError(
                    f"slot {s}: decode position {p} != cache length {b.slots[s].seq_len} (no rewind / skip)"
                )
            if p // self.backend.block_size >= b.coverage[s]:
                raise ValueError(
                    f"slot {s}: position {p} is outside its page-table coverage ({int(b.coverage[s])} blocks); "
                    "pass the request's page-table row"
                )

    def _decode_step(
        self,
        b: _CacheBinding,
        live: Sequence[int],
        *,
        enable_trace: bool,
        mode: Optional[str],
        read_tokens: bool = True,
    ):
        """One decode step over every slot (rows not in `live` have cur_pos -1). mode None -> logits only (read
        back); 'greedy' / 'sample' -> on-device sampler (tt_out_tok feedback) and token readback; host mode -> logits
        readback + HostSampler (the token is written at the next step). Returns [max_batch] tokens or
        [max_batch, V] logits, None when the tokens stay on device."""
        device_sampler = mode is not None and not self.host_sampling
        # a fused backend's decode step samples on every route (traced, and eagerly on the Stage 14b stack): it
        # overwrites every token row like a sampler
        overwrites_tokens = device_sampler or self.fused_sampling
        if overwrites_tokens:
            self._preserve_offstep_tokens(b, live)
        width = self.decode_width(b, live)
        # Stage 17 (n29h6): a lone SPECULABLE request decodes at the speculative plain width (the smallest width >= K+1,
        # SpecMTP.plain_w: its verify's lossless twin); set per step by the adapter, 0 = the plain bucket rule
        spw = int(getattr(self, "spec_lone_width", 0) or 0)
        if spw and spw in b.widths and len(live) == 1 and spw > width:
            width = spw
        if enable_trace:
            if width not in b.decode_traces:
                raise RuntimeError("traced decode requested but no decode trace is captured (call warmup())")
            self.backend.replay(b.decode_traces[width])
            self.counters["decode_trace_replays"] += 1
        else:
            self.backend.decode(b.io, b.kv_cache, width)
            self.counters["eager_decode_steps"] += 1
        b.width_replays[width] += 1
        b.step += 1
        b.pos_shadow = torch.where(b.pos_shadow >= 0, b.pos_shadow + 1, b.pos_shadow)  # advanced on device
        if not device_sampler:
            logits = self.backend.read(b.io.logits)[:, : self.backend.vocab_size]
            self.counters["logits_readback"] += 1
            if overwrites_tokens:  # the fused step (traced or eager) left its own samples in every token row
                b.dev_ok[:] = False
            if mode is None:
                return logits
            tok = torch.zeros(b.max_batch, dtype=torch.int64)
            if live:
                tok[list(live)] = self.host_sampler.sample(
                    logits[list(live)].float(), [b.slots[s].params for s in live], list(live)
                )
            self.counters["host_sampling_steps"] += 1
            for s in live:
                b.tok_val[s], b.dev_ok[s] = int(tok[s]), False
            return tok
        if enable_trace and self.fused_sampling:
            self.counters["fused_sampling_steps"] += 1  # the decode replay above already sampled
        elif enable_trace:
            self.backend.replay(b.sampling_traces[mode])
            self.counters["sampling_trace_replays"] += 1
        else:
            self.backend.sample(b.io, mode)
        if not read_tokens:
            self._after_sample(b, live, None)
            return None
        tok = self.backend.read(b.io.sampled).to(torch.int64).reshape(-1)
        self.counters["token_readback"] += 1
        self._after_sample(b, live, tok)
        return tok

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: Optional[torch.Tensor],
        kv_cache: Any,
        slots: Optional[Sequence[int]] = None,
        enable_trace: bool = True,
        sampling_params: Optional[Sequence[SamplingParams]] = None,
        read_from_device: bool = True,
        slot_remap: Any = None,
        reload_inputs: Optional[bool] = None,
        reload_page_table: Optional[bool] = None,
        reload_sampling_params: Optional[bool] = None,
        reset_sampling_state: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Low level: one decode step. tokens [B, 1] / start_pos [B] (absolute position of each token = the slot's
        cache length; -1 = inactive row) for the rows' fixed slots (`slots`, default row i -> slot i); slots not listed
        do not decode this step and their cache / state is left untouched. `tokens[r] == -1` = the slot's current next
        input (the token the device sampled for it, or the host value if its device row was overwritten).
        `page_table` [B, blocks] rows for the listed slots (None = unchanged; positions beyond a slot's coverage
        raise). Only inputs that differ from the device copies are written. Returns logits [B, V] without
        `sampling_params`, tokens [B] sampled on device, or a `DecodeOutput` handle with read_from_device=False.
        Serving commands (09a adapter): `slot_remap` is applied first; `reload_inputs=False` decodes every live slot
        (active, not finished) from its device-resident token and cache length, ignoring `tokens` / `start_pos`."""
        b = self._binding(kv_cache)
        # GLM53_PF_HOST_TIMING (Stage 12, n29h7p2 2026-10-23: an admission's first decode submit took 1.1 s on the 1M
        # line vs 0.07 s steady): per-phase host ms of the input-reload steps -> backend.counters["decode_reload_host_ms"]
        tm = ([("start", _time.perf_counter())]
              if reload_inputs is not False and os.environ.get("GLM53_PF_HOST_TIMING") == "1" else None)  # fmt: skip

        def mark(label):
            if tm is not None:
                tm.append((label, _time.perf_counter()))

        if slot_remap is not None:
            self.remap_slots(kv_cache, self._remap_dict(slot_remap))
        mark("remap")
        if reload_inputs is False:
            rows = [s for s in range(b.max_batch) if b.slots[s].active and not b.slots[s].finished]
            tokens = torch.full((len(rows),), -1, dtype=torch.int64)
            start_pos = torch.tensor([b.slots[s].seq_len for s in rows], dtype=torch.int64)
            if not reload_page_table:
                page_table = None
            if sampling_params is not None and not reload_sampling_params:
                sampling_params = [b.slots[s].params for s in rows]
        else:
            tokens = torch.as_tensor(tokens).reshape(-1).to(torch.int64)
            start_pos = torch.as_tensor(start_pos).reshape(-1).to(torch.int64)
            rows = list(slots) if slots is not None else list(range(tokens.numel()))
        if len(rows) != tokens.numel() or len(set(rows)) != len(rows) or start_pos.numel() != len(rows):
            raise ValueError("tokens, start_pos and slots must have one entry per row (distinct slots)")
        if any(not 0 <= s < b.max_batch for s in rows):
            raise ValueError(f"slots {rows} outside [0, {b.max_batch})")
        pos = torch.full((b.max_batch,), -1, dtype=torch.int64)
        want, page_rows, used = {}, {}, {}
        pt = torch.as_tensor(page_table) if page_table is not None else None
        for r, s in enumerate(rows):
            p = int(start_pos[r])
            if p < 0:
                continue
            pos[s], want[s] = p, int(tokens[r])
            b.slots[s].active = True
            if sampling_params is not None:
                b.slots[s].params = sampling_params[r]
            if pt is not None:
                page_rows[s], used[s] = pt[r], p + 1
        if reset_sampling_state and sampling_params is not None:
            for s in want:
                self._new_request_rng(b, s, b.slots[s].params)
        mark("rows")
        if page_rows:
            self._write_page_rows(b, page_rows, used)
        mark("page_rows")
        self._check_decode_rows(b, {s: int(pos[s]) for s in want})
        mark("check_rows")
        self._write_positions(b, pos)
        mark("positions")
        self._sync_tokens(b, want)
        mark("tokens")
        mode = None
        if sampling_params is not None:
            self._write_params(b)
            mark("params")
            mode = "greedy" if all(b.slots[s].params.is_greedy for s in want) else "sample"
        elif not read_from_device:
            raise ValueError("read_from_device=False needs sampling_params (device-resident tokens)")
        live = list(want)
        out = self._decode_step(b, live, enable_trace=enable_trace, mode=mode, read_tokens=read_from_device)
        mark("decode_step")
        if tm is not None:
            bc = getattr(
                self.backend, "counters", None
            )  # beside pf_host_ms (the generator's counters stay flat numbers)
            acc = (bc if isinstance(bc, dict) else self.__dict__.setdefault("host_ms", {})).setdefault(
                "decode_reload_host_ms", {})  # fmt: skip
            for (_a, t0), (lab, t1) in zip(tm, tm[1:]):
                acc[lab] = acc.get(lab, 0.0) + (t1 - t0) * 1e3
            acc["n"] = acc.get("n", 0) + 1
        for s in live:
            b.slots[s].seq_len = int(pos[s]) + 1
        if out is None:
            return DecodeOutput(binding=b, step=b.step, rows=rows, live=live)
        return out[rows]

    @staticmethod
    def _remap_dict(slot_remap) -> dict[int, int]:
        """slot_remap as {src: dst}, or a sequence where entry i names the slot whose request moves to slot i
        (-1 / i = unchanged)."""
        if isinstance(slot_remap, dict):
            return {int(k): int(v) for k, v in slot_remap.items()}
        seq = [int(x) for x in torch.as_tensor(slot_remap).reshape(-1).tolist()]
        return {src: dst for dst, src in enumerate(seq) if src >= 0 and src != dst}

    def read_decode_output(self, tt_out: DecodeOutput, async_read: bool = False):
        """Read the tokens a `read_from_device=False` decode left on device (they are already the next step's input;
        reading is for the caller). Must be issued before the next decode step on the same cache (stale handle
        raises). `async_read=True` only enqueues the copy behind the step and returns (PendingTokens, events): the
        caller waits on the events, then `process_decode_output_host` yields the tokens - the vLLM async split, which
        may submit the next step in between."""
        b = tt_out.binding
        if tt_out.step != b.step:
            raise RuntimeError(
                f"stale DecodeOutput (step {tt_out.step}, cache at step {b.step}): read before the next decode"
            )
        self.counters["token_readback"] += 1
        if async_read:
            pending, events = self.backend.read_async(b.io.sampled)
            handle = PendingTokens(binding=b, step=tt_out.step, rows=tt_out.rows, live=tt_out.live, pending=pending)
            return handle, events
        tok = self.backend.read(b.io.sampled).to(torch.int64).reshape(-1)
        self._note_sampled(b, tt_out.step, tt_out.live, tok)
        return tok[tt_out.rows]

    @staticmethod
    def _note_sampled(b: _CacheBinding, step: int, live: Sequence[int], tok: torch.Tensor):
        """Host copy of a sampler step's tokens. Only while no later sampler step ran: afterwards the device rows hold
        newer tokens and these values are no slot's next input (the binding invariant keeps them device-only)."""
        if step != b.step:
            return
        for s in live:
            if b.dev_ok[s] and b.tok_val[s] < 0:
                b.tok_val[s] = int(tok[s])

    def process_decode_output_host(self, host_out, is_tokens: bool = True) -> torch.Tensor:
        """Host formatting boundary: tokens [B] -> [B, 1] int64; logits pass through. A `PendingTokens` from an async
        read is completed first (its events must have been waited on)."""
        if isinstance(host_out, PendingTokens):
            b = host_out.binding
            tok = self.backend.finish_read(b.io.sampled, host_out.pending).to(torch.int64).reshape(-1)
            self._note_sampled(b, host_out.step, host_out.live, tok)
            host_out = tok[host_out.rows]
        return host_out.reshape(-1, 1).to(torch.int64) if is_tokens else host_out

    # ============================================================================ Stage 17 speculative decoding
    # The backend's speculative step (tt/spec_mtp.py, GLM53_SPEC_K) for the lone live request in slot 0: the bootstrap
    # drafts on device after a plain step / prefill, every spec step verifies the K drafts (the main model on K+1 rows),
    # accepts on device and drafts the next K; leaving restores the plain decode inputs (KDA candidate row -> row 0,
    # positions). These keep the binding's host shadows (cache length, token rows, positions) consistent.
    def spec_k(self) -> int:
        sp = getattr(self.backend, "spec", None)
        return int(sp.K) if sp is not None else 0

    def _spec_limit(self, b) -> int:
        """the positions a speculative step may touch: the context, and the drafter's own cache capacity when the step
        has one (SpecMTP.pos_cap: GLM53_DF2_CACHE_CAP)"""
        limit = min(b.max_seq_len, self.backend.supported_context)
        cap = getattr(getattr(self.backend, "spec", None), "pos_cap", None)
        return min(limit, int(cap)) if cap else limit

    def spec_room(self, kv_cache) -> bool:
        """the next speculative step of slot 0 fits (its verify + the next drafts: positions up to L + 2K)"""
        b = self._binding(kv_cache)
        sp = getattr(self.backend, "spec", None)
        return sp is not None and int(b.slots[0].seq_len) + 2 * int(sp.K) + 1 <= self._spec_limit(b)

    def spec_enter(self, kv_cache, slot: int = 0) -> list:
        """start speculating for the request in `slot` (must be 0, the only live slot): its next input is made resident
        in device row 0, the plain position row 0 = its cache length; the bootstrap replay drafts -> the K drafts"""
        b = self._binding(kv_cache)
        sp = getattr(self.backend, "spec", None)
        if sp is None or slot != 0:
            raise ValueError(f"speculative step: needs the backend's spec step and slot 0 (slot {slot})")
        live = [s for s in range(b.max_batch) if b.slots[s].active and not b.slots[s].finished]
        if live != [0]:
            raise ValueError(f"speculative step: slot 0 must be the only live slot (live {live})")
        L = int(b.slots[0].seq_len)
        limit = self._spec_limit(b)
        if L < 1 or L + sp.K + 1 > limit:
            raise ValueError(f"speculative step at length {L}: positions up to {L + sp.K} outside [1, {limit})")
        if getattr(sp, "pos_cap", None) and L + 2 * sp.K + 1 > limit:  # (the first step must fit the drafter's cache)
            raise ValueError(f"speculative step at length {L}: the drafter's cache holds {limit} positions")
        if -(-(L + sp.K + 1) // self.backend.block_size) > int(b.coverage[0]):
            raise ValueError(
                f"slot 0: page-table coverage {int(b.coverage[0])} blocks < the spec step's {L + sp.K + 1}"
            )
        if b.obm is not None and os.environ.get("GLM53_SPEC_ON_SHARD", "0") != "1":
            raise NotImplementedError("speculative decode on the sequence-sharded DSA pool (GLM53_DSA_SHARD) needs "
                                      "GLM53_SPEC_ON_SHARD=1")  # fmt: skip
        self._sync_tokens(b, {0: -1})
        pos = torch.full((b.max_batch,), -1, dtype=torch.int64)
        pos[0] = L
        self._write_positions(b, pos)
        # the sharded pool: slot 0's PHYSICAL row (owner-aware; spec_mtp localizes it per chip). Its columns through
        # L + K are covered (the check above), so the OwnerBlockMap has mapped them; a rollback releases no block
        row = b.pt_shadow[0] if b.obm is None else b.phys[0]
        sp.set_page_table(row)
        sp.enter(row)
        rec = sp.read()
        if rec["c"] != L - 1 and _MOCK_DEVICE:
            rec = dict(rec, c=L - 1)  # mock: no kernel ran (zeros)
        if rec["c"] != L - 1:
            raise RuntimeError(f"speculative bootstrap at c {rec['c']}, expected {L - 1}")
        b.spec = dict(drafts=list(rec["drafts"]), bonus=int(rec["a"][0]))
        self.counters["spec_enters"] = self.counters.get("spec_enters", 0) + 1
        return list(rec["drafts"])

    def spec_step(self, kv_cache, page_row: Optional[torch.Tensor] = None) -> dict:
        """one speculative step (verify + accept + draft, ONE trace replay) -> the record: n accepted drafts, the
        committed tokens a[0..n], the next drafts; slot 0's cache length += n + 1. page_row: slot 0's page-table row
        when the caller's changed (it must cover the K+1 verify positions)"""
        b = self._binding(kv_cache)
        sp = self.backend.spec
        if getattr(b, "spec", None) is None:
            raise RuntimeError("spec_step without spec_enter")
        L = int(b.slots[0].seq_len)
        if page_row is not None:
            self._write_page_rows(b, {0: page_row}, {0: L + sp.K + 1})
        if L + 2 * sp.K + 1 > self._spec_limit(b):
            raise ValueError(f"speculative step at length {L}: the next step's positions pass the context")
        sp.step()
        rec = sp.read()
        if rec["c"] != L and _MOCK_DEVICE:
            # mock: no kernel ran (zeros) -> no accepted draft, the bonus token only; that token must REJECT the block's
            # first draft (zeros too), else the runner's accept walk takes the K drafts the device did not commit and the
            # request runs K positions ahead of slot 0 (the next decode's slot check fails)
            d0 = int(b.spec["drafts"][0]) if b.spec.get("drafts") else -1
            a = [int(v) for v in rec["a"]]
            if a[0] == d0:
                a[0] = d0 + 1 if d0 + 1 < int(getattr(self.backend, "vocab_size", 1 << 17)) else d0 - 1
            rec = dict(rec, c=L, n=0, a=a)
        if rec["c"] != L:
            raise RuntimeError(f"speculative step base {rec['c']} != slot 0's cache length {L}")
        n = int(rec["n"])
        b.slots[0].seq_len = L + n + 1
        b.tok_val[0], b.dev_ok[0] = int(rec["a"][n]), True  # the bonus = the next input, in device row 0
        b.dev_ok[1:] = False  # token rows 1..K hold the drafts
        b.pos_shadow[0] = -2  # the plain position row was not advanced by the speculative step
        b.step += 1
        b.spec = dict(drafts=list(rec["drafts"]), bonus=int(rec["a"][n]))
        for k in ("spec_steps", "spec_tokens"):
            self.counters[k] = self.counters.get(k, 0) + (1 if k == "spec_steps" else n + 1)
        return rec

    def spec_abort(self, kv_cache) -> None:
        """the speculating request ended (released): forget the speculation (its candidate rows are dead state)"""
        b = self._binding(kv_cache)
        b.spec = None
        if getattr(self.backend, "spec", None) is not None:
            from models.autoports.zai_org_glm_5_3_flash.tt.spec_mtp import write_rows

            write_rows(self.backend.mesh, self.backend.spec.sel, torch.zeros(1, 32, dtype=torch.int32))

    def spec_leave(self, kv_cache) -> None:
        """back to plain decode: KDA state / conv history of the committed candidate row -> row 0, plain position row 0
        = the next input's position (token row 0 already holds the bonus)"""
        b = self._binding(kv_cache)
        if getattr(b, "spec", None) is None:
            return
        rec = self.backend.spec.leave()
        b.pos_shadow = self.backend.read(b.io.cur_pos).to(torch.int64).reshape(-1)[: b.max_batch].clone()
        if _MOCK_DEVICE and int(b.pos_shadow[0]) != int(b.slots[0].seq_len):
            b.pos_shadow[0] = int(b.slots[0].seq_len)  # mock: no kernel ran (zeros)
            rec = dict(rec, c=int(b.slots[0].seq_len) - int(rec["n"]) - 1)
        if int(b.pos_shadow[0]) != rec["c"] + rec["n"] + 1 or int(b.pos_shadow[0]) != int(b.slots[0].seq_len):
            raise RuntimeError(
                f"spec_leave: plain position {int(b.pos_shadow[0])} != slot 0's length {b.slots[0].seq_len}"
            )
        b.dev_ok[1:] = False
        b.spec = None
        self.counters["spec_leaves"] = self.counters.get("spec_leaves", 0) + 1

    def remap_slots(self, kv_cache, mapping: dict[int, int]) -> None:
        """Scheduler slot remap (vLLM `slot_remap`): move each live request from slot `src` to slot `dst` - its
        recurrent state and per-slot sampler state (backend.copy_slot), host sampler RNG, page-table row and coverage,
        next-input token and bookkeeping. Destinations must be free or themselves moved in the same mapping; chains are
        ordered automatically, cycles need a free scratch slot. Positions / token rows are refreshed by the next
        decode."""
        b = self._binding(kv_cache)
        mapping = {s: d for s, d in mapping.items() if s != d}
        if not mapping:
            return
        srcs, dsts = list(mapping), list(mapping.values())
        if len(set(dsts)) != len(dsts) or any(not 0 <= x < b.max_batch for x in srcs + dsts):
            raise ValueError("slot_remap destinations must be distinct slots")
        for d in dsts:
            if b.slots[d].active and d not in srcs:
                raise ValueError(f"slot_remap destination {d} holds a live request")
        if any(bool(b.dev_ok[s]) and b.tok_val[s] < 0 for s in srcs):
            self._read_tokens(b)  # device-only next inputs: keep their values before the rows change owner
        state = {
            s: (
                b.slots[s],
                b.pt_shadow[s].clone(),
                int(b.coverage[s]),
                int(b.tok_val[s]),
                self.host_sampler._gens[s].get_state(),
            )
            for s in srcs
        }
        pending = dict(mapping)
        while pending:  # copy a source only once no pending move still needs to read its destination
            ready = [s for s, d in pending.items() if d not in pending]
            if not ready:
                raise ValueError("cyclic slot_remap needs a free scratch slot")
            for s in ready:
                self.backend.copy_slot(b.kv_cache, s, pending.pop(s))
        new_pt, cov = b.pt_shadow.clone(), b.coverage.clone()
        phys = None
        if b.obm is not None:  # the rows keep their physical blocks (the latents stay where they are)
            for d in set(dsts) - set(srcs):  # an idle destination's stale row (e.g. an owned cache's identity row) goes
                b.obm.release_slot(d)
            b.obm.remap(mapping)
            phys = b.phys.clone()
            for s in srcs:
                if s not in dsts:
                    phys[s] = b.null_row
            for s, d in mapping.items():
                phys[d] = b.phys[s]
        for s in srcs:
            if s not in dsts:
                b.slots[s] = SlotState()
                cov[s], b.tok_val[s] = 0, -1
            b.dev_ok[s] = False
        for s, d in mapping.items():
            st, row, c, tv, rng = state[s]
            b.slots[d] = st
            new_pt[d], cov[d], b.tok_val[d], b.dev_ok[d] = row, c, tv, False
            self.host_sampler._gens[d].set_state(rng)
        b.coverage = cov
        self._publish_page_table(b, new_pt, phys)

    def release_slot(self, kv_cache, slot: int) -> None:
        """Request finished / preempted: the slot stops decoding (cur_pos -1 from the next step) and loses its page
        coverage, so its former blocks can be given to another request."""
        b = self._binding(kv_cache)
        b.slots[slot] = SlotState()
        b.coverage[slot], b.tok_val[slot], b.dev_ok[slot] = 0, -1, False
        if b.obm is not None:  # its physical blocks go back to their owners' free lists (the stale device row of an
            # idle slot is never read; the next publish writes the null row)
            b.obm.release_slot(slot)
            b.phys[slot] = b.null_row
        # the backend's release hook, called exactly once (ModelBackend default: no-op; StackBackend: canonical idle
        # row, GLM53_IDLE_CANON; the lifecycle mock: its slot ledger)
        release = getattr(self.backend, "release_slot", None)
        if callable(release):
            release(b.kv_cache, slot)
        self.counters["slot_releases"] += 1

    # ================================================================================================ high level
    def prefill_logits(self, prompt_token_ids: List[int]) -> torch.Tensor:
        """Readiness prefill driver: generator-owned cache, slot 0 (its own blocks), logits for every logical
        position [1, T, V]."""
        ids = torch.as_tensor(list(prompt_token_ids), dtype=torch.int64)
        self.check_context(ids.numel(), 1)
        b = self._own_binding()
        if not b.prepared_prefill:
            self.warmup(capture=False)
        if b.slots[0].active:
            raise ValueError("slot 0 holds a live request")
        self._ensure_identity_rows(b, [0], {0: ids.numel()})
        b.slots[0] = SlotState(active=True)
        try:
            logits = self._prefill_user(b, 0, ids, all_logits=True)
        finally:
            b.slots[0].active = False
        return logits.reshape(1, ids.numel(), -1)

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn] = None,
        enable_trace: bool = True,
        sampling: Optional[SamplingParams] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
        stop_on_eos: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[int]:
        """Readiness high-level driver (one user, slot 0): prefill -> p0 -> decode loop. Returns the model's own
        predictions (also under teacher forcing). Greedy unless `sampling` is given. Stops at a stop token only when
        `stop_on_eos` (default: True without `next_input`, False under teacher forcing, which must run all steps).
        `enable_trace=True` runs every decode step through the captured trace and fails if none can be captured."""
        if stop_on_eos is None:
            stop_on_eos = next_input is None
        outs = self.generate_batch(
            [list(prompt_token_ids)],
            max_new_tokens,
            sampling=[sampling or GREEDY],
            stop_token_ids=stop_token_ids,
            stop_on_eos=stop_on_eos,
            next_input=None if next_input is None else (lambda user, step, pred: next_input(step, pred)),
            enable_trace=enable_trace,
        )
        return outs[0]

    def generate_batch(
        self,
        prompts: Sequence[Sequence[int]],
        max_new_tokens: int | Sequence[int],
        *,
        sampling: Optional[Sequence[SamplingParams]] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
        stop_on_eos: bool = True,
        next_input: Optional[Callable[[int, int, int], int]] = None,
        enable_trace: bool = True,
        slots: Optional[Sequence[int]] = None,
        page_table: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Mixed-length batch generation over fixed slots of the generator-owned cache. Each prompt is prefilled into
        its slot (any length), then all live slots decode together; a user that hits a stop token or its
        max_new_tokens becomes an inactive row (cur_pos -1) whose state the remaining steps leave untouched.
        `next_input(user, step, predicted) -> token` teacher-forces inputs (called once per prediction). `page_table`
        [max_batch, blocks] gives the listed slots these block rows (rewritten in place under the live trace);
        without it each slot gets its own identity blocks back."""
        n = len(prompts)
        if n == 0:
            return []
        b = self._own_binding()
        if not (b.prepared_prefill and b.prepared_decode) or (enable_trace and not b.decode_traces):
            self.warmup(capture=enable_trace)
        slots = list(slots) if slots is not None else list(range(n))
        if len(slots) != n or len(set(slots)) != n or any(not 0 <= s < b.max_batch for s in slots):
            raise ValueError(f"need {n} distinct slots in [0, {b.max_batch})")
        if any(b.slots[s].active for s in slots):
            raise ValueError("slot already holds a live request")
        mnt = [max_new_tokens] * n if isinstance(max_new_tokens, int) else [int(m) for m in max_new_tokens]
        params = list(sampling) if sampling is not None else [GREEDY] * n
        stops = frozenset(int(t) for t in stop_token_ids) if stop_token_ids is not None else self.stop_token_ids
        for i, p in enumerate(prompts):
            if mnt[i] < 1:
                raise ValueError("max_new_tokens must be >= 1")
            self.check_context(len(p), mnt[i])
        used = {s: len(prompts[i]) + mnt[i] - 1 for i, s in enumerate(slots)}  # positions the request writes
        if page_table is not None:
            pt = torch.as_tensor(page_table)
            self._write_page_rows(b, {s: pt[s] for s in slots}, used)
        else:
            self._ensure_identity_rows(b, slots, used)
        outputs: list[list[int]] = [[] for _ in range(n)]
        user_of = {s: i for i, s in enumerate(slots)}
        feed: dict[int, int] = {}  # slot -> token fed at the next decode step (the prediction or a forced one)
        try:
            # ---- prefill every user into its slot; first prediction p0 from the last prompt position (Stage 16:
            # groups of fresh users admitted by one batched-prefill replay when the backend captured one)
            mk = lambda i: SlotState(active=True, params=params[i], max_new_tokens=mnt[i],  # noqa: E731
                                     stop_ids=stops if stop_on_eos else frozenset())  # fmt: skip
            admitted = self._admit_batch(b, [(i, s, torch.as_tensor(list(p), dtype=torch.int64), mk(i))
                                             for i, (p, s) in enumerate(zip(prompts, slots))])  # fmt: skip
            for i, (p, s) in enumerate(zip(prompts, slots)):
                if i in admitted:
                    self._emit(b, s, outputs[i], admitted[i], next_input, user_of, feed)
                    continue
                b.slots[s] = st = mk(i)
                self._new_request_rng(b, s, params[i])
                ids = torch.as_tensor(list(p), dtype=torch.int64)
                if self.host_sampling:
                    logits = self._prefill_user(b, s, ids, sample=None)
                    p0 = int(self.host_sampler.sample(logits.float(), [params[i]], [s])[0])
                    self.counters["host_sampling_steps"] += 1
                else:
                    p0 = self._prefill_user(b, s, ids, sample=params[i])
                self._emit(b, s, outputs[i], p0, next_input, user_of, feed)
            # ---- decode loop: replay decode + sampler; refresh only scheduler changes and forced tokens
            while any(b.slots[s].active and not b.slots[s].finished for s in slots):
                pos = torch.full((b.max_batch,), -1, dtype=torch.int64)
                want = {}
                for s in range(b.max_batch):
                    st = b.slots[s]
                    if st.active and not st.finished:
                        pos[s], want[s] = st.seq_len, feed[s]
                self._check_decode_rows(b, {s: int(pos[s]) for s in want})
                self._write_positions(b, pos)
                self._sync_tokens(b, want)
                self._write_params(b)
                live = list(want)
                mode = "greedy" if all(b.slots[s].params.is_greedy for s in live) else "sample"
                sampled = self._decode_step(b, live, enable_trace=enable_trace, mode=mode)
                for s in live:
                    b.slots[s].seq_len += 1
                    self._emit(b, s, outputs[user_of[s]], int(sampled[s]), next_input, user_of, feed)
        finally:
            for s in slots:
                b.slots[s].active = False
                b.slots[s].finished = True
        return outputs

    def _emit(self, b, slot, out, token, next_input, user_of, feed):
        """Record one prediction of `slot`; decide the token fed next (teacher forcing may override it)."""
        st = b.slots[slot]
        step = len(out)
        out.append(int(token))
        st.generated.append(int(token))
        nxt = int(token)
        if next_input is not None:
            nxt = int(next_input(user_of[slot], step, int(token)))
            if nxt != int(token):
                self.counters["teacher_forced_overrides"] += 1
        feed[slot] = nxt
        if len(out) >= st.max_new_tokens or int(token) in st.stop_ids:
            st.finished = True

    def reset(self) -> None:
        """Wipe per-request state of every cache (KDA state, positions, next-input tokens, bookkeeping) and the
        generator-owned page table (identity blocks again; caller-owned caches lose their page coverage until the
        caller supplies rows). Buffers, compiled programs and traces survive."""
        for b in self._bindings.values():
            self.backend.reset_all(b.kv_cache)
            b.slots = [SlotState() for _ in range(b.max_batch)]
            b.tok_val[:], b.dev_ok[:] = -1, False
            # every device token row to one known value: an idle row's leftover token (the previous request set's)
            # changed the numerics of the live rows sharing its row chunk (StackBackend, 2026-10-20 #67 / #75) -> the
            # same requests diverged run to run
            self.backend.write(b.io.tokens, torch.zeros(b.max_batch, dtype=torch.int32))
            self._write_positions(b, torch.full((b.max_batch,), -1, dtype=torch.int64))
            if b.owned and b.identity_ok:
                self._ensure_identity_rows(b, range(b.max_batch))
            else:
                b.coverage[:] = 0
                if b.obm is not None:
                    for s in range(b.max_batch):
                        b.obm.release_slot(s)
                    b.phys = b.null_row.expand(b.max_batch, b.blocks_per_seq).clone()

    def host_work_report(self) -> dict:
        return dict(self.counters)

    def device_sync(self) -> bool:
        """Wait until the device finished every enqueued command (the adapter's host-overhead probes) through the
        backend's optional `synchronize()`; False = the backend has none (a host backend: no probe)."""
        sync = getattr(self.backend, "synchronize", None)
        if not callable(sync):
            return False
        sync()
        return True

    def counters_report(self) -> dict:
        """trace_report()'s counters without device queries (the serving events' hot-path report): the generator's
        counters, decode replays per width and live slots per binding, the backend's counters (+ its TP prefill's)."""
        out = {"counters": dict(self.counters), "bindings": []}
        for b in self._bindings.values():
            out["bindings"].append(
                {"steps": b.step, "decode_replays_by_width": dict(b.width_replays),
                 "live_slots": [s for s in range(b.max_batch) if b.slots[s].active]}  # fmt: skip
            )
        counters = getattr(self.backend, "counters", None)
        if isinstance(counters, dict):
            out["backend"] = dict(counters)
            tpp_counters = getattr(getattr(self.backend, "tpp", None), "counters", None)
            if isinstance(tpp_counters, dict):
                out["backend"]["prefill"] = dict(tpp_counters)
        return out

    def trace_report(self) -> dict:
        """Captures / replays per trace and host-work counters (Stage 9 mixed-shape trace-reuse evidence)."""
        out = {"counters": dict(self.counters), "bindings": []}
        for b in self._bindings.values():
            out["bindings"].append(
                {
                    "owned": b.owned,
                    "max_batch": b.max_batch,
                    "num_blocks": b.num_blocks,
                    "blocks_per_seq": b.blocks_per_seq,
                    "steps": b.step,
                    "prepared_prefill": b.prepared_prefill,
                    "prepared_decode": b.prepared_decode,
                    "decode_widths": list(b.widths),
                    "decode_traces": sorted(b.decode_traces),
                    "decode_replays_by_width": dict(b.width_replays),
                    "sampling_traces": sorted(b.sampling_traces),
                    "live_slots": [s for s in range(b.max_batch) if b.slots[s].active],
                    "coverage": b.coverage.tolist(),
                }
            )
        report = getattr(self.backend, "trace_report", None)
        if callable(report):
            out["backend"] = report()
        return out


# ------------------------------------------------------------------------------------------------ factory
class BorrowedGenerator:
    """A live Glm53Generator lent to a readiness runner (`build_generator(generator=...)`): every attribute forwards to
    the owner's generator except `teardown` / `release_traces`, which are no-ops — traces live as long as the owner's
    model (the $tt-enable-tracing lifetime rule), the owner tears down at model unload."""

    def __init__(self, generator):
        object.__setattr__(self, "_owner", generator)

    def __getattr__(self, name):
        return getattr(self._owner, name)

    def __setattr__(self, name, value):
        setattr(self._owner, name, value)

    def teardown(self):
        return None

    def release_traces(self):
        return None


# The device backend's module: `build_model_backend(model_dir, mesh_device, **kw) -> ModelBackend` (and, for the vLLM
# adapter's fabric capability, `serving_fabric_config()` or `pipeline_fabric_config()`). Default: Stage 6's 8-chip
# layer pipeline; the Stage 14 TP8 megakernel model plugs in here (PRODUCTION_PATH.md M5 / M6) with the same contract.
BACKEND_MODULE_ENV = "GLM53_BACKEND_MODULE"
DEFAULT_BACKEND_MODULE = "models.autoports.zai_org_glm_5_3_flash.tt.model"
# JSON keyword arguments for that module's build_model_backend on top of the caller's (serving profiles: e.g.
# {"native_t": true} = the Stage 14b stack's native 32-row decode step)
BACKEND_KW_ENV = "GLM53_BACKEND_KW"
# Stage 12 for the TP stack (tt/tp_image.py, Stage 14b int8): with GLM53_WEIGHT_SOURCE=image these backend modules'
# weights come from a per-chip DRAM image (resident restart: only stale / clobbered blocks move) instead of the tensor
# loader; the image of the newest matching allocation map wins (weights manifest, GLM53_TP_LAYOUT, mesh shape, then
# the replay of every upload at its recorded address). GLM53_IMAGE_REQUIRED=1: no usable image raises instead of
# loading tensors (coordinator 2026-10-21 00:28:13Z: no tensor loads in a mesh window).
TP_IMAGE_MODULES = ("models.autoports.zai_org_glm_5_3_flash.tt.stack_model",)
WEIGHT_SOURCE_ENV = "GLM53_WEIGHT_SOURCE"
IMAGE_REQUIRED_ENV = "GLM53_IMAGE_REQUIRED"
IMAGE_PIN_ENV = "GLM53_IMAGE_PIN"


def check_image_required(path_taken: bool, where: str) -> None:
    """GLM53_IMAGE_REQUIRED=1 holds for every construction path: one that does not restore a TP image raises before
    anything loads (it would load tensors)."""
    if os.environ.get(IMAGE_REQUIRED_ENV) == "1" and not path_taken:
        raise ValueError(f"{IMAGE_REQUIRED_ENV}=1 but {where} does not restore a TP DRAM image (needs "
                         f"{WEIGHT_SOURCE_ENV}=image and a {BACKEND_MODULE_ENV} in TP_IMAGE_MODULES)")  # fmt: skip


def restore_tp_backend(build, mesh_device, *, signature: dict, on_fail=None):
    """`build()` (a TP_IMAGE_MODULES backend construction) through tt/tp_image.py: restored from an image, else (unless
    GLM53_IMAGE_REQUIRED=1) the tensor loader under the recorder (GLM53_IMAGE_BUILD=1: and the image written). The
    backend gets `weight_source_info` (source, image, stale / clobbered blocks, restore_s, tried images).
    GLM53_IMAGE_PIN=<image dir name or path>: only that image is tried (the one a mock replay validated).
    on_fail: called after every failed restore (e.g. clear the shared weight cache the failed replay filled with
    layers whose DRAM was never restored, as tests/s14b/m5common.serving does), then gc."""
    import gc

    from models.autoports.zai_org_glm_5_3_flash.tt import tp_image as TI

    teardown = lambda b: b.close()  # noqa: E731

    def failed():
        if on_fail is not None:
            on_fail()
        gc.collect()

    info = {}
    if os.environ.get(IMAGE_REQUIRED_ENV) == "1":
        root = os.environ.get("GLM53_TP_IMAGE_ROOT", TI.DEFAULT_ROOT)
        pin = os.environ.get(IMAGE_PIN_ENV)
        cands = TI.candidates(root, TI.weights_identity(), mesh_device)
        if pin:
            cands = [d for d in cands if pin in (d, os.path.basename(d))]
        tried = []
        for d in cands:
            sub = {}
            try:
                backend = TI.restore_backend(build, mesh_device, d, teardown=teardown, timeline=sub)
            except (TI.ImageMismatch, RuntimeError) as e:  # as tp_image.build_or_restore: the next image
                tried.append((os.path.basename(d), f"{type(e).__name__}: {str(e)[:300]}"))
                failed()
                continue
            info.update(source="image", image=d, tried=tried, **sub)
            break
        else:
            raise RuntimeError(f"{IMAGE_REQUIRED_ENV}=1: no usable TP DRAM image under {root}"
                               f"{f' pinned {pin}' if pin else ''} (tried {tried}); the tensor loader is not allowed "
                               "in this window")  # fmt: skip
    else:
        backend, info = TI.build_or_restore(build, mesh_device, signature=signature, teardown=teardown, timeline=info,
                                            on_fail=failed)  # fmt: skip
    backend.weight_source_info = info
    return backend


def build_generator(model_dir: str | Path, mesh_device, **kwargs) -> Glm53Generator:
    """Readiness factory. `backend=` (a ModelBackend) selects the model implementation explicitly (host-only
    tests pass the CPU mocks); otherwise Stage 6's `tt/model.py::build_model_backend(model_dir, mesh_device, **kw)`
    builds the device model. `tokenizer=` defaults to `tt/tokenizer_chat.load_tokenizer()`."""
    existing = kwargs.pop("generator", None)
    if existing is not None:
        # one long-lived device process runs every readiness phase (DEVICE RULES: <= 1 mesh open per 30 min; the
        # 45-layer load takes minutes): the runners borrow the already-built generator (request state reset); the
        # runner's end-of-run teardown() must not end the owner's trace lifetime (no recapture between phases)
        existing.reset()
        return BorrowedGenerator(existing)
    backend = kwargs.pop("backend", None)
    tokenizer = kwargs.pop("tokenizer", None)
    gen_kw = {
        k: kwargs.pop(k)
        for k in ("max_batch", "max_seq_len", "host_sampling", "stop_token_ids", "sampler_seed")
        if k in kwargs
    }
    if backend is None:
        if mesh_device is None:
            raise ValueError("build_generator needs a mesh_device (device model) or backend= (host-only mock)")
        module = os.environ.get(BACKEND_MODULE_ENV) or DEFAULT_BACKEND_MODULE
        try:
            build_model_backend = importlib.import_module(module).build_model_backend
        except (ImportError, AttributeError) as e:
            raise NotImplementedError(
                f"{module}.build_model_backend is not available ({BACKEND_MODULE_ENV}); pass backend= for host-only runs"
            ) from e
        extra = json.loads(os.environ.get(BACKEND_KW_ENV) or "{}")
        clash = sorted(set(extra) & set(kwargs))
        if clash:  # a profile keyword never silently overrides the caller's
            raise ValueError(f"{BACKEND_KW_ENV} sets {clash}, which the caller passes too")
        image = module in TP_IMAGE_MODULES and os.environ.get(WEIGHT_SOURCE_ENV) == "image"
        check_image_required(image, f"tt/generator.py build_generator ({module})")
        if image:
            kw = {**kwargs, **extra}
            wc = kw.get("weight_cache")
            # JSON-safe (written into the image manifest): the shared weight cache only as its presence
            signature = dict(B=gen_kw.get("max_batch"), module=module,
                             kw={k: (v if k != "weight_cache" else isinstance(v, dict)) for k, v in kw.items()})  # fmt: skip
            backend = restore_tp_backend(lambda: build_model_backend(model_dir, mesh_device, **kwargs, **extra),
                                         mesh_device, signature=signature,
                                         on_fail=wc.clear if isinstance(wc, dict) else None)  # fmt: skip
        else:
            backend = build_model_backend(model_dir, mesh_device, **kwargs, **extra)
    if tokenizer is None:
        from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

        tokenizer = load_tokenizer()
    return Glm53Generator(backend, tokenizer=tokenizer, **gen_kw)
