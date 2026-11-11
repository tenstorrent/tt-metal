# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill-side prefix reuse, the model-independent half (H1: one retained prompt per slot, on the device). Ported from the
GLM-5.3 Flash runner (tt-blaze G53F-0311, ``models/demos/glm53_flash/tt/runners/prefix_reuse.py`` on
``sdawle/glm53f_fix5_runner``) for DeepSeek-V4.1-Flash (``tt/v41/prefix_reuse.py`` holds the device half). torch only: no
ttnn import, every device action is a callback, so the decisions are tested on the host
(``models/demos/deepseek_v3_d_p/tests/v41/test_prefix_reuse.py``).

What the device already keeps. The runner prefills one prompt at a time per slot, in the engine's chunks ``[kC, (k+1)C)``
(``C`` = PREFILL_CHUNK_SIZE, every chunk full but the prompt's last), and a later prompt overwrites only the rows it
computes. So after a prompt of ``L`` tokens the model's position-indexed caches hold ``[0, L)`` and the export holds the
rows the KV Manager ships for ``[0, L)``. What is NOT position-indexed (GLM: the KDA carries; V4.1: the 128-row window
carries of every attention layer) is held only at ``L``; retention adds copies of it at chunk boundaries (a device-DRAM pool
of ``n_ckpts`` checkpoint slots) and the prompt's token ids.

A later prompt still arrives from position 0 (the engine is unchanged). Its chunk ``[s, e)`` is SKIPPED -- no compute, every
layer acknowledged at once -- while every earlier chunk of the prompt was skipped, its ids equal the retained ids
``[s, e)`` and ``e <= L``. A skipped chunk that is (or may be) the burst's last must leave the export as a cold prefill
of ``[0, e)`` would: ``export_carries(slot, e, final)`` writes the non-positional state at ``e`` into the export (from a
checkpoint at ``e``, or the live state when ``e == L``); so such a chunk is skipped only when that state is held.
``final`` = known to be the last (a marked last header or a ragged chunk); ``export_carries`` returns True when it also
CLEARED the export past ``e`` (V4.1: the zero a cold prefill's chunk 0 does), and the retention then forgets ``[e, L)``.

Which chunk is last: the prefill worker marks it (tt-d-gen ``device.prefill.mark_last_chunk``): every chunk header's slot
word carries bit 29 ("this header is marked") and, on a burst's last chunk, bit 30 (``split_slot_word``). Per header, so
nothing is learned across requests (an aborted request ends without a last chunk; a restarted engine may not mark). An
unmarked header may be the last chunk.

The first chunk that is not skipped RESUMES the model at its start ``s``: its non-positional state is restored from the
checkpoint at ``s`` (or is live), or rebuilt from the nearest checkpoint ``b < s`` by recomputing ``[b, s)`` in full chunks
without export or acks (``compute_silent``: the export already holds those rows, the model's rows are rewritten with the
same values); ``set_next_start(pos)`` puts the model's counters at ``pos``; ``on_resume(s)`` then fixes the export for a
resume at ``s`` (V4.1: window rings at ``s``, rows past ``s`` cleared). Resumed chunks start where a cold prefill's
start, so every computed chunk is the same program on the same inputs: meant to be bit-identical to a cold prefill.

Checkpoints: one per chunk boundary while the pool has room; when it is full, the oldest boundary that is not a multiple
of the smallest power-of-two stride (in chunks) that leaves a candidate is evicted, and the two newest are never evicted.
Nothing here is fitted to a workload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

MARKED_FLAG = 1 << 29  # tt-d-gen device.prefill.mark_last_chunk: set in EVERY chunk header's slot word
LAST_CHUNK_FLAG = 1 << 30  # ... and this one on a burst's last chunk
_SENTINEL_BIT = 1 << 31  # the runner's shutdown / warm-up words (0xFFFFFFFF-style): never split


def split_slot_word(word: int) -> tuple[int, Optional[bool]]:
    """(slot, last) from a chunk header's slot word: last is True / False on a marked header, None on an unmarked one (an
    engine without mark_last_chunk: any chunk may be the burst's last). Negative words and words with bit 31 set (the
    runner's sentinels) pass through untouched; bit 30 means nothing without bit 29."""
    word = int(word)
    if word < 0 or word & _SENTINEL_BIT or not word & MARKED_FLAG:
        return word, None
    return word & ~(MARKED_FLAG | LAST_CHUNK_FLAG), bool(word & LAST_CHUNK_FLAG)


class CheckpointPool:
    """Boundary position -> checkpoint slot, at most ``n_slots`` (see the module docstring for the eviction rule)."""

    def __init__(self, n_slots: int, chunk: int):
        self.n_slots = int(n_slots)
        self.chunk = int(chunk)
        self.by_pos: dict[int, int] = {}
        self.free = list(range(self.n_slots))
        self.evictions = 0

    def held(self, pos: int) -> Optional[int]:
        return self.by_pos.get(int(pos))

    def nearest_at_or_below(self, pos: int) -> tuple[int, Optional[int]]:
        """(b, slot) of the highest checkpoint at or below ``pos``; (0, None) when there is none (b = 0 needs none)."""
        below = [p for p in self.by_pos if p <= pos]
        if not below:
            return 0, None
        b = max(below)
        return b, self.by_pos[b]

    def drop_above(self, pos: int) -> None:
        for p in [p for p in self.by_pos if p > pos]:
            self.free.append(self.by_pos.pop(p))

    def clear(self) -> None:
        self.drop_above(-1)

    def admit(self, pos: int) -> Optional[int]:
        """A slot for a new boundary ``pos`` (above every held one), evicting one if the pool is full."""
        pos = int(pos)
        if self.n_slots <= 0:
            return None
        if pos in self.by_pos:
            return self.by_pos[pos]
        if not self.free:
            held = sorted(self.by_pos)
            protected = {held[-1]} if self.n_slots >= 3 and len(held) > 1 else set()  # the newest, while another can go
            stride = self.chunk
            while True:
                cands = [p for p in held if p not in protected and p % stride]
                if cands:
                    victim = min(cands)
                    break
                if (
                    stride > held[-1]
                ):  # every candidate is a multiple of a stride past them all: impossible unless empty
                    victim = min(p for p in held if p not in protected)
                    break
                stride *= 2
            self.free.append(self.by_pos.pop(victim))
            self.evictions += 1
        slot = self.free.pop()
        self.by_pos[pos] = slot
        return slot


@dataclass
class ReuseStats:
    prompts: int = 0
    resumed_prompts: int = 0
    skipped_chunks: int = 0
    skipped_tokens: int = 0
    silent_chunks: int = 0
    computed_chunks: int = 0
    checkpoints_saved: int = 0
    carry_exports: int = 0
    marked_chunks: int = 0
    invalidations: int = 0


class PrefixRetention:
    """The retained prompt of ONE slot and the skip / resume decisions. Device actions are callbacks:

    ``save(slot)``                     live non-positional state -> checkpoint slot
    ``restore(slot)``                  checkpoint slot -> live state
    ``export_carries(slot, end, final) -> bool``  the state of checkpoint ``slot`` (None = the live one) into the export
                                       for a burst ending at ``end``; True = it also cleared the export past ``end``
    ``set_next_start(pos)``            the model's counters (and whatever is derived from them) at ``pos``
    ``compute_silent(ids, start)``     one chunk through the model, no export, no acks (the live state then holds its end)
    ``on_resume(start)``               (optional) after the live state is at a resume point ``start``: fix the export
    """

    def __init__(
        self,
        *,
        chunk: int,
        n_ckpts: int,
        save: Callable[[int], None],
        restore: Callable[[int], None],
        export_carries: Callable[[Optional[int], int, bool], bool],
        set_next_start: Callable[[int], None],
        compute_silent: Callable[[torch.Tensor, int], None],
        on_resume: Optional[Callable[[int], None]] = None,
        log: Callable[[str], None] = print,
        tag: str = "prefix",
    ):
        self.chunk = int(chunk)
        self.pool = CheckpointPool(n_ckpts, chunk)
        self._save, self._restore, self._export = save, restore, export_carries
        self._set_next_start, self._compute_silent, self._on_resume = set_next_start, compute_silent, on_resume
        self.log, self.tag = log, tag
        self.tokens = torch.zeros(0, dtype=torch.int32)
        self.length = 0  # positions [0, length) of the device caches / export hold tokens[:length]
        self.matching = False  # the in-flight prompt: every chunk so far skipped (matching)
        self.cursor = 0  # ... and where its next chunk must start
        self.prompt_resumed = False
        self.last_resume: Optional[dict] = None  # set by a chunk that resumed a matched prompt (for the caller's log)
        self.live_at: Optional[int] = None  # the live state is the device chain's state at this position
        self.stats = ReuseStats()

    # ------------------------------------------------------------------ state
    def reset(self, why: str) -> None:
        self.tokens = torch.zeros(0, dtype=torch.int32)
        self.length = 0
        self.live_at = None
        self.pool.clear()
        self.matching = False
        self.cursor = 0
        self.log(f"[{self.tag}] retention reset ({why})")

    def carries_at(self, pos: int):
        """("live", None) when the live state is at ``pos``, ("ckpt", slot) for a checkpoint, else None."""
        if self.live_at is not None and pos == self.live_at:
            return ("live", None)
        slot = self.pool.held(pos)
        return None if slot is None else ("ckpt", slot)

    # ------------------------------------------------------------------ per chunk
    def begin_chunk(self, ids: torch.Tensor, start: int, end: int, last: Optional[bool]) -> bool:
        """Called before a non-empty chunk ``[start, end)``; ``last`` from ``split_slot_word`` (None = unmarked header).
        True = skipped (the caller only acknowledges it); False = compute it (the model is resumed at ``start`` first if
        the prompt matched so far)."""
        start, end = int(start), int(end)
        if last is not None:
            self.stats.marked_chunks += 1
        if start == 0:
            self._new_prompt()
        ids = ids.to(torch.int32).flatten()
        if self.matching and start == self.cursor and end <= self.length and torch.equal(ids, self.tokens[start:end]):
            final = bool(last) or (end - start) < self.chunk  # a ragged chunk is always the burst's last
            may_be_last = final or last is None
            if not may_be_last:
                self._skipped(start, end)
                return True
            at = self.carries_at(end)
            if at is not None:
                cleared = self._export(at[1], end, final)
                self.stats.carry_exports += 1
                self._skipped(start, end)
                if cleared and end < self.length:
                    self._truncate(end)  # the export past `end` is gone: nothing past it can be shipped from here
                return True
            # a possible last chunk whose end state is not held: compute it (resumed at start below)
        self._prepare_compute(start)
        return False

    def note_empty(self) -> None:
        """An EMPTY chunk (a burst declined before compute): nothing computed or skipped, but it ends a request."""
        self.matching = False

    def end_compute(self, ids: torch.Tensor, start: int, end: int) -> None:
        """After the model computed ``[start, end)`` (the live state is at ``end``): retain it."""
        self._record(ids, int(start), int(end))
        self.stats.computed_chunks += 1

    def invalidate(self, why: str) -> None:
        """A chunk failed part-way, or the prompt cannot be vouched for (an image span): nothing on the device is retained."""
        self.stats.invalidations += 1
        self.reset(why)

    # ------------------------------------------------------------------ internals
    def _new_prompt(self) -> None:
        self.stats.prompts += 1
        self.matching = self.length > 0
        self.cursor = 0
        self.prompt_resumed = False

    def _skipped(self, start: int, end: int) -> None:
        self.cursor = end
        self.stats.skipped_chunks += 1
        self.stats.skipped_tokens += end - start
        if not self.prompt_resumed:
            self.prompt_resumed = True
            self.stats.resumed_prompts += 1

    def _prepare_compute(self, start: int) -> None:
        resume = self.matching and start > 0 and start == self.cursor
        self.matching = False
        self.last_resume = None
        if not resume:
            if start == 0:
                self._truncate(0)  # a new prompt computed from scratch: only [0, 0) stays
            elif start != self.length:
                # not the continuation of the prompt the device holds (tt-d-gen #430's re-send): the model's own contiguity
                # check decides; nothing retained can be vouched for past this point
                self.reset(f"chunk start {start} is neither 0 nor the retained end {self.length}")
            return
        at = self.carries_at(start)
        if at is not None and at[0] == "live":
            self._set_next_start(start)  # the live state is already at start (the retained prompt ended there)
            self.last_resume = {"start": start, "how": "live"}
        elif at is not None:
            self._restore(at[1])
            self.live_at = start
            self._set_next_start(start)
            self.length = start
            self.last_resume = {"start": start, "how": "checkpoint"}
        else:
            b, slot = self.pool.nearest_at_or_below(start)
            if slot is not None:
                self._restore(slot)
            self.live_at = b if b else None
            self._set_next_start(b)
            self.length = b
            n = 0
            for x in range(b, start, self.chunk):
                y = min(start, x + self.chunk)
                self._compute_silent(self.tokens[x:y].to(torch.int64), x)
                self._record(self.tokens[x:y], x, y)
                n += 1
            self.stats.silent_chunks += n
            self.last_resume = {"start": start, "how": f"rebuilt from {b}", "silent": n}
            self.log(f"[{self.tag}] rebuilt the carries at {start} from {b} ({n} chunk(s), no export)")
        self._truncate(start)
        if self._on_resume is not None:
            self._on_resume(start)

    def _truncate(self, pos: int) -> None:
        """Everything from ``pos`` on is about to be overwritten (or was cleared)."""
        self.length = min(self.length, pos)
        if self.live_at is not None and self.live_at > pos:
            self.live_at = None
        self.pool.drop_above(pos)

    def _record(self, ids: torch.Tensor, start: int, end: int) -> None:
        ids = ids.to(torch.int32).flatten()
        assert int(ids.numel()) == end - start, (ids.numel(), start, end)
        if int(self.tokens.numel()) < end:
            grown = torch.zeros(max(end, 2 * int(self.tokens.numel())), dtype=torch.int32)
            grown[: int(self.tokens.numel())] = self.tokens
            self.tokens = grown
        self.tokens[start:end] = ids
        self.length = end
        self.pool.drop_above(start)
        self.live_at = end
        if end % self.chunk == 0:
            slot = self.pool.admit(end)
            if slot is not None:
                self._save(slot)
                self.stats.checkpoints_saved += 1

    def summary(self) -> str:
        s = self.stats
        return (
            f"retained {self.length} tokens, {len(self.pool.by_pos)} checkpoint(s) at "
            f"{sorted(p // self.chunk for p in self.pool.by_pos)} chunk(s); prompts {s.prompts}, resumed {s.resumed_prompts}, "
            f"skipped {s.skipped_chunks} chunk(s) / {s.skipped_tokens} tokens, silent {s.silent_chunks}, computed "
            f"{s.computed_chunks}, checkpoints saved {s.checkpoints_saved} (evicted {self.pool.evictions}), carry exports "
            f"{s.carry_exports}, marked headers {s.marked_chunks}, invalidations {s.invalidations}"
        )
