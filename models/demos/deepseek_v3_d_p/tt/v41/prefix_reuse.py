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

Image prompts: a chunk with a virtual image-span id (>= vocab) is never matched (the same span layout gives the same ids for
different images); its prompt is computed whole and nothing of it is retained.
"""

from __future__ import annotations

import os
import time
from functools import partial

import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.prefix_reuse import PrefixRetention

from .kv_export import export_windows, export_zero_blocks, zero_export_rows

ENV_FLAG = "V41_PREFILL_PREFIX_REUSE"
ENV_CKPTS = "V41_PREFILL_PREFIX_CKPTS"
TAG = "v41-prefix"


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0") == "1"


def n_checkpoints() -> int:
    return int(os.environ.get(ENV_CKPTS, "16"))


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
        if start == 0:
            self.bypass[slot] = False
            logger.info(f"[{TAG}] slot {slot} new prompt at chunk {request_id}: {pr.summary()}")
        if not self.on or self.bypass[slot]:
            return False
        if any(int(t) >= self.vocab for t in ids):
            pr.invalidate(f"image-span ids in chunk [{start}, {end}): the prompt is computed whole, nothing retained")
            self.bypass[slot] = True
            return False
        t0 = time.perf_counter()
        try:
            skipped = pr.begin_chunk(torch.tensor([int(t) for t in ids], dtype=torch.int32), start, end, last)
        except BaseException:
            pr.invalidate("begin_chunk failed")
            raise
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

    def _dram(self):
        """Allocated DRAM bytes per bank (the checkpoint pool's cost in the startup line); None where unavailable."""
        try:
            return int(ttnn.get_memory_view(self.rt.mesh_device, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank)
        except Exception:  # noqa: BLE001 -- a report must not fail the build
            return None
