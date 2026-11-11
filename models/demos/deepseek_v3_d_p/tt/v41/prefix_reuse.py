# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash prefill-side prefix reuse, H1 (port plan ``v41ax_logs/prefill_reuse_design.md`` on host 29, 2026-11-11; GLM-5.3's G53F-0311
ported): the runner skips the chunks of a prompt that its device still holds (same slot, same prefix as the previous
prompt there) and resumes at the first new chunk. Opt-in: ``V41_PREFILL_PREFIX_REUSE=1`` (unset = the runner exactly as
before: ``V41PrefillRuntime`` never builds this). ``V41_PREFILL_PREFIX_CKPTS`` = checkpoint slots per user slot (default 16).

What a chunk boundary needs (``V41Prefill``, ``kv_export``):

* position-indexed and kept between prompts (``V41Prefill.reset`` zeroes only the counters): every KV source's compressed
  entries, index keys and path-A slab rows (layers 2 / 8 / 14, ratio 2) and layer 20's (ratio 1), the Engram n-gram history
  (``EngramHost`` slot cache, written per position). V4.1's compressor carries NO state across chunks (non-overlapping
  groups, ``V41Compressor.empty_prior``), so a chunk start is always a clean boundary.
* NOT position-indexed: the 128-row window carry (``state.sliding_carry``) of each of the 21 attention states (layers 0..19
  + layer 20 KV-only) -- a checkpoint copies those (21 x 128 KiB = 2.6 MiB per chip), in device DRAM allocated before the
  trace-island capture.
* the counters and what derives from them: ``kv_actual`` / ``entry_count`` of every state, ``V41Prefill.kv_actual``, and the
  A3 islands' shared score mask, whose entry columns [0, entries) a cold prefill's glue has opened by the chunk at ``s``.
  ``resume_at`` re-opens them in ONE-CHUNK pieces with the zero block allocated before the capture: the glue would open
  ``[mask_zeroed_upto, entry_count)`` in a single ``_zeros_block_tile(S_l, entries)`` -- a new persistent tensor per resume
  point (328 MiB per chip at s = 41 chunks), allocated after the capture (DS4F-0262 class). Design risk R2.
* the export (what the KV Manager ships once per layer at the burst's seal, tt-d-gen row-addressed tables): a cold prompt
  starts with ``zero_slot``; a resumed one keeps its entry rows and gets ``zero_windows`` + the window rings at the resume
  point + ``zero_export_rows`` past it (``kv_export.py``), i.e. the export a cold prefill has after its chunk ending there.

The decisions (skip / resume / checkpoints) are the model-independent ``PrefixRetention``
(``models/demos/common/prefill/prefix_reuse.py``). Chunk headers marked by tt-d-gen ``device.prefill.mark_last_chunk``
(slot word bits 29 / 30) let mid-prompt chunks skip freely; unmarked, a full chunk is skipped only where its end carries are
held (and a skipped last chunk then leaves the export's rows past it as the older prompt wrote them: never shipped as KV,
but not byte-identical to cold in the <= 256 index-key rows the seal moves past the keys -- use marked headers).

A PARTIAL last chunk is always computed (resumed at its chunk start), never skipped -- device check 11-11 (host 30, run1,
S.17): A2 sent again (41600 = 5 x 8192 + 640, every chunk matching) exported different window rings than cold. A cold export's
256-row ring after a partial chunk ending at E = 128 (mod 256) holds TWO carries: the chunk's own (rows [0, 128)) and the
previous chunk boundary's (rows [128, 256), written by the chunk before); a skip rebuilt only the first. Computing that one
partial chunk costs at most one chunk on an exact re-send and writes the ring as a cold prefill does (a resume at a boundary
already restores the boundary's carry: _on_resume). Skips ending ON a chunk boundary (one carry) are unaffected.

Image prompts: a chunk with a virtual image-span id (>= vocab) is never matched (the same span layout gives the same ids for
different images): it is computed -- resumed at its start first, like any computed chunk, when a skipped / restored prefix
precedes it -- and nothing of the prompt from it on is retained (nor stored, H2). The text chunks before it depend on their own
ids only, so they are reused like any other.

H2, the host-RAM chunk store (``V41_PREFILL_PREFIX_STORE_GB`` > 0, on top of H1; 0 / unset = H1 exactly, call for call;
plan ``v41ax_logs/prefill_reuse_design.md`` section 3; GLM-5.3's G53F-0311 phase 2 / G53F-0256 / G53F-0315 ported). Every
FULL chunk the runner computes is read into the store after its sync: the position-indexed rows a restore must write back --
per KV source (2 / 8 / 14 at ratio 2, layer 20 at ratio 1) the bf16 entries ``compressed_kv`` and index keys ``index_k`` of
its ``C / ratio`` entry rows (the indexer scores the bf16 state, not the bfp8 export), chip 0 (the states are replicated):
25 MiB per 8192-token chunk. A chunk the device does not hold but the store does is RESTORED (``_restore_rows``), through
the writers a computed chunk uses: replicated uploads -> ``fill_cache_for_user_`` into ``compressed_kv`` / ``index_k`` (as
``TtCSA.forward`` / ``kv_only``) -> the path-A slab's entry rows at ``chunk + s / ratio`` (``_slab_write``, a block source
with ``sparse_path``; layer 20 KV-only has no slab) -> the export's entry / key rows repacked by ``_export_entries``
(``kv_export.export_entry_rows``) -> the Engram n-gram history (``EngramHost.hash_ids``, host only); a restore at chunk 0
first does what a cold chunk 0 does (``pf.reset`` + ``zero_slot``). The window rings, counters and score mask are written
where H1 writes them (``_on_resume`` / ``_export_carries`` / ``resume_at``). Tip carries: when the device leaves a prompt,
the device checkpoints of its ``V41_PREFILL_PREFIX_SPILL_TIPS`` (default 2) newest boundaries are read into the store's tip
arena (21 x 128 x 512 bf16 = 2.625 MiB each), after the burst's last chunk (``after_acks``); a later turn resumes on them
(``_load_carries``: replicated upload + ``ttnn.copy`` INTO the live carry buffers, whose addresses the islands captured).
Memory: two fixed arenas (``SlotArena``), RSS <= the budget whatever the eviction history (G53F-0315), plus the
``V41_PREFILL_PREFIX_HOST_RESERVE_GB`` MemAvailable floor (default 32). One store per user slot (the budget split evenly).
"""

from __future__ import annotations

import os
import time
from functools import partial

import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.prefix_reuse import ChunkStore, PrefixRetention

from .kv_export import export_entry_rows, export_windows, export_zero_blocks, zero_export_rows

ENV_FLAG = "V41_PREFILL_PREFIX_REUSE"
ENV_CKPTS = "V41_PREFILL_PREFIX_CKPTS"
# H2 (host chunk store; needs ENV_FLAG): the store's RAM budget (GiB, all slots; 0 / unset = off), the part of it for tip carries
# (default 1/16), the host-memory floor (MemAvailable GiB under which the store gives memory back), the boundaries spilled per
# prompt the device leaves
ENV_STORE_GB = "V41_PREFILL_PREFIX_STORE_GB"
ENV_STORE_TIP_GB = "V41_PREFILL_PREFIX_STORE_TIP_GB"
ENV_RESERVE_GB = "V41_PREFILL_PREFIX_HOST_RESERVE_GB"
ENV_SPILL_TIPS = "V41_PREFILL_PREFIX_SPILL_TIPS"
TAG = "v41-prefix"


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0") == "1"


def n_checkpoints() -> int:
    return int(os.environ.get(ENV_CKPTS, "16"))


def store_gb() -> float:
    return float(os.environ.get(ENV_STORE_GB, "0") or 0)


def entry_sources(pf) -> list:
    """(layer, attention, CURRENT-slot state, compress ratio, is a block) of every KV source whose position-indexed rows a
    chunk writes: the encoder's full layers (2 / 8 / 14), then layer 20 KV-only -- ``export_chunk``'s order."""
    out = [
        (blk.layer, blk.attn, blk.state, pf.cfg.role(blk.layer).compress_ratio, True)
        for blk in pf.blocks
        if pf.cfg.role(blk.layer).mode == "full"
    ]
    if pf.kv_attn is not None:
        out.append((pf.kv_only_layer, pf.kv_attn, pf.kv_state, pf.cfg.role(pf.kv_only_layer).compress_ratio, False))
    return out


def carry_states(pf, slot: int) -> list:
    """The attention states of ``slot`` whose window carry a checkpoint holds: the blocks in order, then layer 20 KV-only."""
    out = [blk.states[slot] for blk in pf.blocks]
    if pf.kv_attn is not None:
        out.append(pf.kv_states[slot])
    return out


def _island_masks(pf):
    """(block, its CURRENT-slot state) of every block whose attention replays islands and holds a score mask -- the same
    blocks ``V41Prefill.reset`` re-initialises the mask of."""
    return [
        (blk, blk.state)
        for blk in pf.blocks
        if getattr(blk.state, "score_mask", None) is not None and blk._attn_islands_enabled()
    ]


def _zeros_piece_key(pf, blk, mask) -> tuple:
    ratio = pf.cfg.role(blk.layer).compress_ratio
    return ("zeros", int(mask.shape[2]), int(pf.chunk) // ratio)


def open_score_masks(pf, pos: int) -> int:
    """After ``pf.reset()`` (every entry column -inf): open entry columns [0, pos / ratio) of each island source's score
    mask -- what a cold prefill's glue has opened before its chunk at ``pos`` -- in one-chunk pieces, and set every sharing
    state's ``mask_zeroed_upto``. Never allocates (the piece is built by ``precreate_mask_pieces``). Returns the writes.
    """
    opened, writes = {}, 0
    for blk, st in _island_masks(pf):
        mask = st.score_mask
        ratio = pf.cfg.role(blk.layer).compress_ratio
        n = int(pos) // ratio
        if id(mask) not in opened:
            key = _zeros_piece_key(pf, blk, mask)
            piece = key[2]
            assert n % piece == 0, (pos, ratio, piece)
            zeros = blk.attn.indexer._cut_masks.get(key)
            assert zeros is not None, f"layer {blk.layer}: zero block {key} was not built before the capture"
            S_l = key[1]
            for c0 in range(0, n, piece):
                ttnn.experimental.slice_write(
                    zeros, mask, start=[0, 0, 0, c0], end=[1, 1, S_l, c0 + piece], step=[1, 1, 1, 1]
                )
                writes += 1
            opened[id(mask)] = n
        assert opened[id(mask)] == n, (blk.layer, opened[id(mask)], n)
        st.mask_zeroed_upto = n
    return writes


def precreate_mask_pieces(pf) -> int:
    """Before any island capture: the one-chunk zero block ``open_score_masks`` writes, for every CSA source whose state
    already has an A3 zero block (``_source_alloc_state`` builds one only with PREFILL_V41_A3 + the sparse path). With the
    production geometry (chunk >= max(window, top-k x ratio)) it IS that block (cached by key), so nothing new is allocated.
    """
    made = 0
    for blk in pf.blocks:
        ind = getattr(blk.attn, "indexer", None)
        cache = getattr(ind, "_cut_masks", None)
        if not isinstance(cache, dict) or not any(isinstance(k, tuple) and k and k[0] == "zeros" for k in cache):
            continue
        ratio = pf.cfg.role(blk.layer).compress_ratio
        key = ("zeros", int(pf.chunk) // int(pf.sp), int(pf.chunk) // ratio)
        if key not in cache:
            made += 1
        ind._zeros_block_tile(key[1], key[2])
    return made


def resume_at(pf, pos: int) -> int:
    """The CURRENT slot's model at ``pos`` (a chunk start): counters 0 and every score mask -inf (``pf.reset``), then every
    state's counters at ``pos`` (entries ``pos / ratio``) and the masks opened to them. Returns the mask writes."""
    pf.reset()
    pos = int(pos)
    if pos == 0:
        return 0
    assert pos % int(pf.chunk) == 0, (pos, pf.chunk)
    for blk in pf.blocks:
        ratio = pf.cfg.role(blk.layer).compress_ratio
        blk.state.kv_actual = pos
        blk.state.entry_count = pos // ratio if ratio else 0
    if pf.kv_attn is not None:
        ratio = pf.cfg.role(pf.kv_only_layer).compress_ratio
        pf.kv_state.kv_actual = pos
        pf.kv_state.entry_count = pos // ratio
    pf.kv_actual = pos
    return open_score_masks(pf, pos)


class V41PrefixReuse:
    """The runtime's H1 state: per user slot a ``PrefixRetention``, its checkpoint buffers and the export's high-water mark
    (positions at or past it are zero in the export). Built by ``V41PrefillRuntime.compile`` BEFORE its warm chunk."""

    def __init__(self, runtime, kv_caches, n_ckpts: int | None = None):
        self.rt, self.pf, self.exp = runtime, runtime.pf, kv_caches
        self.chunk = int(self.pf.chunk)
        self.max_seq = int(self.pf.max_seq_len)
        self.vocab = int(runtime.cfg.vocab_size)
        self.n_ckpts = n_checkpoints() if n_ckpts is None else int(n_ckpts)
        assert self.max_seq % self.chunk == 0, f"prefix reuse needs MAXSEQ % CHUNK == 0 ({self.max_seq}, {self.chunk})"
        before = self._dram()
        self.ckpt = {
            s: [[ttnn.clone(st.sliding_carry) for st in carry_states(self.pf, s)] for _ in range(self.n_ckpts)]
            for s in range(runtime.num_users)
        }
        self.zeros = export_zero_blocks(kv_caches, self.pf)
        made = precreate_mask_pieces(self.pf)
        after = self._dram()
        self.hwm = {s: 0 for s in range(runtime.num_users)}
        self.bypass = {s: False for s in range(runtime.num_users)}
        self.on = True
        self.last_restored = False  # the last begin_chunk restored its chunk from the host store (H2)
        self.replayed = False  # the last begin_chunk ran silent chunks (island replays) before its chunk
        self.stores = self._build_stores(runtime) if store_gb() > 0 else {}  # H2; {} = off
        self.spill_tips = int(os.environ.get(ENV_SPILL_TIPS, "2"))
        h2 = {
            s: (
                dict(
                    store=self.stores[s],
                    restore_rows=partial(self._restore_rows, s),
                    spill=partial(self._spill, s),
                    load_carries=partial(self._load_carries, s),
                    spill_tips=self.spill_tips,
                    chain_seed=f"dsv41-prefill-chain-v1 chunk {self.chunk}".encode(),
                )
                if s in self.stores
                else {}
            )
            for s in range(runtime.num_users)
        }
        self.ret = {
            s: PrefixRetention(
                chunk=self.chunk,
                n_ckpts=self.n_ckpts,
                save=partial(self._save, s),
                restore=partial(self._restore, s),
                export_carries=partial(self._export_carries, s),
                set_next_start=partial(self._set_next_start, s),
                compute_silent=partial(self._compute_silent, s),
                on_resume=partial(self._on_resume, s),
                log=logger.info,
                tag=f"{TAG} slot {s}",
                # a partial last chunk is computed, never skipped: see the module docstring (device check 11-11, S.17)
                skip_partial_last=False,
                **h2[s],
            )
            for s in range(runtime.num_users)
        }
        n_states = len(carry_states(self.pf, 0))
        dram = (
            f", DRAM {(after - before) / 2**20:.2f} MiB per bank (allocated per bank now {after / 2**20:.0f} MiB)"
            if before is not None and after is not None
            else ""
        )
        logger.info(
            f"[{TAG}] ON: {self.n_ckpts} checkpoint(s) x {n_states} window carries per slot x {runtime.num_users} slot(s), "
            f"export zero blocks for ratios {sorted(self.zeros)}, {made} new mask piece(s){dram}"
        )
        if self.stores:
            st = self.stores[0]
            logger.info(
                f"[{TAG}] H2 ON: host chunk store {st.budget / 2**30:.2f} GiB per slot x {len(self.stores)} slot(s) (tip "
                f"carries {st.tip_budget / 2**30:.2f} GiB of it; ~{st.chunk_budget // self._chunk_record_bytes()} chunk(s) of "
                f"{self.chunk} tokens at {self._chunk_record_bytes() / 2**20:.1f} MiB each, INFERRED from the state shapes), "
                f"spill {self.spill_tips} tip(s) per prompt left, host floor {st.reserve / 2**30:g} GiB MemAvailable"
            )

    def _build_stores(self, runtime) -> dict:
        """H2: one ChunkStore per user slot (``ENV_STORE_GB`` split evenly). The restore writes the encoder + layer-20 KV-only
        layout's bf16 states (the production layer set, V41_PREFILL_DECODER=0); anything else is refused at startup."""
        pf, cfg = self.pf, runtime.cfg
        assert pf.kv_attn is not None and len(pf.blocks) == cfg.first_decoder_layer, (
            f"{ENV_STORE_GB} needs the encoder + layer-20 KV-only prefill (V41_PREFILL_DECODER=0): {len(pf.blocks)} blocks, "
            f"kv-only {pf.kv_attn is not None}"
        )
        for L, _attn, st, ratio, _blk in entry_sources(pf):
            assert st.compressed_kv.dtype == ttnn.bfloat16 and st.index_k.dtype == ttnn.bfloat16, (
                L,
                st.compressed_kv.dtype,
                st.index_k.dtype,
            )
            # the whole capacity (no DS4F-0295 compact dense cache): the export and the restore read / write every entry row
            assert int(st.compressed_kv.shape[2]) >= self.max_seq // ratio, (L, tuple(st.compressed_kv.shape), ratio)
        users = int(runtime.num_users)
        per = int(store_gb() * 2**30) // users
        tip_env = os.environ.get(ENV_STORE_TIP_GB, "")
        tip = None if tip_env == "" else int(float(tip_env) * 2**30) // users
        reserve = int(float(os.environ.get(ENV_RESERVE_GB, "32")) * 2**30)
        return {
            s: ChunkStore(
                per, tip_budget_bytes=tip, log=logger.info, reserve_bytes=reserve, tag=f"{TAG} store slot {s}"
            )
            for s in range(users)
        }

    def _chunk_record_bytes(self) -> int:
        """INFERRED bytes of one stored chunk (bf16 entries + keys of every source; the first capture fixes the real size)."""
        return sum(
            (self.chunk // ratio) * (int(st.compressed_kv.shape[3]) + int(st.index_k.shape[3])) * 2
            for _L, _a, st, ratio, _b in entry_sources(self.pf)
        )

    # ---- runtime hooks ---------------------------------------------------------------------------------------------------
    def reset_all(self, why: str) -> None:
        """The device states / export changed outside the retention's view (warm-ups): forget everything. The export is
        zero after the warm-ups (compile's zero_all, _warm_ragged's zero_slot)."""
        for s, pr in self.ret.items():
            pr.reset(why)
            self.hwm[s] = 0
            self.bypass[s] = False

    def begin_chunk(self, slot: int, ids, start: int, end: int, last, request_id: int = 0) -> bool:
        """True = the runtime only acknowledges this chunk (the device holds it); False = compute it (resumed first if the
        prompt matched so far)."""
        pr = self.ret[slot]
        self.last_restored = False
        self.replayed = False  # the resume below ran chunks through the model (silent rebuild): runtime._ids_stale
        if start == 0:
            self.bypass[slot] = False
            rss = f"; runner RSS {_rss_gib():.1f} GiB" if pr.store is not None else ""
            logger.info(f"[{TAG}] slot {slot} new prompt at chunk {request_id}: {pr.summary()}{rss}")
        if not self.on or self.bypass[slot]:
            return False
        image = any(int(t) >= self.vocab for t in ids)
        t0 = time.perf_counter()
        try:
            # an image chunk too: its virtual ids are never retained nor stored, so it is always computed, and the retention
            # first resumes the model at its start like any computed chunk -- a skipped (H1) or restored (H2) prefix before it
            # leaves the counters, the mask and the carries elsewhere (a7830ff1d74 skipped that: A, then an image prompt sharing
            # A's chunk 0, tripped the runtime's kv_actual assert -- host repro in test_v41_prefix_store.py)
            silent0 = pr.stats.silent_chunks
            skipped = pr.begin_chunk(torch.tensor([int(t) for t in ids], dtype=torch.int32), start, end, last)
            self.replayed = pr.stats.silent_chunks > silent0
            assert not (image and skipped), "an image-span chunk was skipped / restored"
        except BaseException:
            pr.invalidate("begin_chunk failed")
            raise
        if image:
            pr.flush_spills()  # the tips of the prompt the device left (H2; nothing without a store)
            pr.invalidate(
                f"image-span ids in chunk [{start}, {end}): computed, nothing of the prompt retained from here"
            )
            self.bypass[slot] = True
        self.last_restored = bool(skipped and pr.last_restored)
        if not skipped and pr.last_resume is not None:
            ttnn.synchronize_device(self.rt.mesh_device)
            logger.info(
                f"[{TAG}] slot {slot} chunk {request_id}: resumed at {start} ({pr.last_resume['how']}) in "
                f"{time.perf_counter() - t0:.3f} s after skipping {start // self.chunk} chunk(s)"
            )
        return skipped

    def end_compute(self, slot: int, ids, start: int, end: int) -> None:
        """After the chunk's compute + export were issued (before the runtime's sync): retain it (checkpoint at a boundary)."""
        if self.on and not self.bypass[slot]:
            self.ret[slot].end_compute(torch.tensor([int(t) for t in ids], dtype=torch.int32), start, end)

    def capture(self, slot: int, ids, start: int, end: int) -> float:
        """H2, after a computed chunk's sync (its state rows are final) and before its acks: a full chunk of a hashed chain
        the store does not hold yet -> its rows read from chip 0 into the store. Returns the seconds spent (0: nothing read;
        never a device call with the store off)."""
        if not self.stores or not self.on or self.bypass[slot]:
            return 0.0
        pr = self.ret[slot]
        if not pr.wants_rows(start, end):
            return 0.0
        t0 = time.perf_counter()
        parts = self._read_rows(int(start))
        t1 = time.perf_counter()
        ok = pr.store_rows(torch.tensor([int(t) for t in ids], dtype=torch.int32), int(start), int(end), parts)
        logger.info(
            f"[{TAG}] slot {slot} chunk [{start}, {end}) {'stored' if ok else 'NOT stored (no slot free)'}: read "
            f"{t1 - t0:.3f} s, copy {time.perf_counter() - t1:.3f} s"
        )
        return time.perf_counter() - t0

    def after_acks(self, slot: int, last) -> None:
        """H2, after a chunk's acks: when a burst ended (a marked last chunk, an empty chunk), read the checkpoints pinned
        for the prompts the device left (their tips) into the store."""
        pr = self.ret[slot]
        if pr.store is None or not pr.pending_spills or not last:
            return
        t0 = time.perf_counter()
        n = pr.flush_spills()
        logger.info(
            f"[{TAG}] slot {slot} spilled {n} checkpoint(s) to the host store in {time.perf_counter() - t0:.3f} s"
        )

    def note_zeroed(self, slot: int) -> None:
        """The runtime zeroed the slot's export (zero_slot at a cold chunk 0)."""
        self.hwm[slot] = 0

    def note_export(self, slot: int, start: int) -> None:
        """export_chunk wrote a chunk at ``start``: always a FULL chunk width of entry rows (kv_export._export_entries)."""
        self.hwm[slot] = max(self.hwm[slot], min(int(start) + self.chunk, self.max_seq))

    def invalidate(self, slot: int, why: str) -> None:
        self.ret[slot].invalidate(why)

    def note_empty(self, slot: int) -> None:
        """An EMPTY chunk (tt-d-gen prefill_wait_for_arm_us: a request declined before compute) ended a request: nothing on
        the device changed, so the retained prefix and checkpoints stay; only the in-flight prompt's match ends."""
        self.ret[slot].note_empty()

    # ---- PrefixRetention callbacks (all act on the slot the runtime selected with use_slot) ----------------------------
    def _save(self, slot: int, k: int) -> None:
        for st, dst in zip(carry_states(self.pf, slot), self.ckpt[slot][k]):
            ttnn.copy(st.sliding_carry, dst)

    def _restore(self, slot: int, k: int) -> None:
        for st, src in zip(carry_states(self.pf, slot), self.ckpt[slot][k]):
            ttnn.copy(src, st.sliding_carry)

    def _set_next_start(self, slot: int, pos: int) -> None:
        assert self.pf.slot == slot, (self.pf.slot, slot)
        resume_at(self.pf, pos)

    def _compute_silent(self, slot: int, ids: torch.Tensor, start: int) -> None:
        """A chunk the export already holds, recomputed only to rebuild the carries (no export, no acks): the host
        embedding path (``_ids_dev`` None: bit-identical rows to the device one, DS41F-0119)."""
        assert self.pf.slot == slot, (self.pf.slot, slot)
        self.pf._ids_dev = None
        self.pf._chunk([int(v) for v in ids.tolist()], int(start), self.rt.engram, None)

    def _on_resume(self, slot: int, pos: int) -> None:
        """The live carries are at the resume point ``pos`` and the chunk at ``pos`` is computed next: the export as a cold
        prefill has it after its chunk ending at ``pos`` -- window rings zeroed + the carries at ``pos``; rows past the next
        chunk's full-width write zeroed up to the high-water mark."""
        self.exp.zero_windows(slot)
        export_windows(self.exp, self.pf, slot, pos)
        z0 = min(int(pos) + self.chunk, self.max_seq)
        n = zero_export_rows(self.exp, self.pf, slot, z0, self.hwm[slot], self.zeros) if z0 < self.hwm[slot] else 0
        self.hwm[slot] = min(self.hwm[slot], z0)
        if n:
            logger.info(f"[{TAG}] slot {slot} resume at {pos}: zeroed export rows [{z0}, ...) ({n} block writes)")

    def _export_carries(self, slot: int, k, end: int, final: bool) -> bool:
        """A skipped chunk that is (or may be) the burst's last: the window rings from the carries at ``end`` (checkpoint
        ``k`` or the live ones); if it IS the last, the rows past its chunk are zeroed too (True: the retention then forgets
        them). Synchronized: the burst's acks follow at once and the KV Manager reads the export after the last one."""
        self.exp.zero_windows(slot)
        export_windows(self.exp, self.pf, slot, end, None if k is None else self.ckpt[slot][k])
        if final:
            tail = min(-(-int(end) // self.chunk) * self.chunk, self.max_seq)
            if tail < self.hwm[slot]:
                zero_export_rows(self.exp, self.pf, slot, tail, self.hwm[slot], self.zeros)
                self.hwm[slot] = tail
        ttnn.synchronize_device(self.rt.mesh_device)
        return bool(final)

    # ---- H2 callbacks (the host chunk store) ----------------------------------------------------------------------------
    def _read_rows(self, start: int) -> list:
        """The current slot's rows of the full chunk at ``start``, chip 0 (replicated states): per source in ``entry_sources``
        order its entries [C / ratio, 512] and index keys [C / ratio, 128], bf16 host tensors."""
        parts = []
        for _L, _attn, st, ratio, _blk in entry_sources(self.pf):
            e0, n = int(start) // ratio, self.chunk // ratio
            for src in (st.compressed_kv, st.index_k):
                W = int(src.shape[3])
                part = ttnn.slice(src, [0, 0, e0, 0], [1, 1, e0 + n, W])
                parts.append(ttnn.to_torch(ttnn.get_device_tensors(part)[0]).reshape(n, W))
                ttnn.deallocate(part)
        return parts

    def _upload(self, t: torch.Tensor, dtype):
        """A host [n, W] block -> [1, 1, n, W] TILE on every chip (replicated, as the states are)."""
        mesh = self.rt.mesh_device
        return ttnn.from_torch(
            t.reshape(1, 1, int(t.shape[-2]), int(t.shape[-1])),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def _restore_rows(self, slot: int, parts: list, ids: torch.Tensor, start: int) -> None:
        """A stored chunk [start, start + C) back into the device through the writers a computed chunk uses (module
        docstring): the model's entry / key rows (+ the path-A slab), the export's rows from them, the Engram history.
        """
        assert self.pf.slot == slot, (self.pf.slot, slot)
        start = int(start)
        t0 = time.perf_counter()
        if start == 0:  # what a cold chunk 0 does first: the export holds another prompt
            self.pf.reset()
            self.exp.zero_slot(slot)
            self.hwm[slot] = 0
        srcs = entry_sources(self.pf)
        assert len(parts) == 2 * len(srcs), (len(parts), len(srcs))
        for i, (_L, attn, st, ratio, blk) in enumerate(srcs):
            e0 = start // ratio
            rows = self._upload(parts[2 * i], st.compressed_kv.dtype)
            keys = self._upload(parts[2 * i + 1], st.index_k.dtype)
            ttnn.kv_cache.fill_cache_for_user_(st.compressed_kv, rows, 0, update_idx=e0)
            ttnn.kv_cache.fill_cache_for_user_(st.index_k, keys, 0, update_idx=e0)
            if blk and getattr(attn, "sparse_path", False):
                # path A's slab [chunk S | entries | carry]: entry w at row S + w (TtCSA.forward / glue_chunk); layer 20's
                # kv_only writes no slab
                attn._slab_write(st, rows, self.chunk + e0)
            ttnn.deallocate(rows)
            ttnn.deallocate(keys)
        t1 = time.perf_counter()
        export_entry_rows(self.exp, self.pf, slot, start, start + self.chunk)
        self.note_export(slot, start)
        if getattr(self.pf, "engrams", None) and self.rt.engram is not None:
            self.rt.engram.hash_ids(
                slot, [int(v) for v in ids.tolist()], start
            )  # the n-gram history (host, per position)
        logger.info(
            f"[{TAG}] slot {slot} restored [{start}, {start + self.chunk}) from the host store: model rows {t1 - t0:.3f} s, "
            f"export rows + Engram {time.perf_counter() - t1:.3f} s (dispatch)"
        )

    def _spill(self, slot: int, k: int) -> list:
        """Checkpoint ``k`` -> host: the window carries of every attention state (``carry_states`` order), chip 0."""
        return [ttnn.to_torch(ttnn.get_device_tensors(buf)[0]) for buf in self.ckpt[slot][k]]

    def _load_carries(self, slot: int, parts: list) -> None:
        """Host carries -> the live carry buffers (copied INTO them: the islands captured their addresses)."""
        states = carry_states(self.pf, slot)
        assert len(parts) == len(states), (len(parts), len(states))
        for st, host in zip(states, parts):
            tmp = self._upload(host, st.sliding_carry.dtype)
            ttnn.copy(tmp, st.sliding_carry)
            ttnn.deallocate(tmp)

    def _dram(self):
        """Allocated DRAM bytes per bank (the checkpoint pool's cost in the startup line); None where unavailable."""
        try:
            return int(ttnn.get_memory_view(self.rt.mesh_device, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank)
        except Exception:  # noqa: BLE001 -- a report must not fail the build
            return None


def _rss_gib() -> float:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    except OSError:
        return float("nan")
