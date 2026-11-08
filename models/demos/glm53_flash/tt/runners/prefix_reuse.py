# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill-side prefix reuse for the GLM-5.3 Flash runner (tt-blaze G53F-0311). Opt-in: ``GLM53F_PREFIX_REUSE=1``.

What the device already keeps. The runner prefills one prompt at a time, in the engine's chunks ``[kC, (k+1)C)``
(``C`` = PREFILL_CHUNK_SIZE, every chunk full but the prompt's last). After a prompt of ``L`` tokens:

* every DSA layer's latent cache rows and pooled-key index hold ``[0, L)`` (the model's own caches; a later prompt
  overwrites only the rows it computes),
* the export slabs hold the ring-format pages of ``[0, L)`` (what the KV Manager copies to the decode ring),
* the KDA carries (``kda_batched``) hold the state at ``L``.

What retention adds: the prompt's token ids and copies of the KDA carries at chunk boundaries (a device-DRAM pool of
``GLM53F_PREFIX_CKPTS`` slots, allocated before the warm-up).

A later prompt still arrives from position 0 (the engine is unchanged; it does not know about the reuse). Its chunk
``[s, e)`` is SKIPPED -- no compute, no DSA export, every layer acknowledged at once -- while every earlier chunk of the
prompt was skipped, its token ids equal the retained ids ``[s, e)`` and ``e <= L``: the slabs already hold ``[s, e)``
byte for byte (the same tokens through the same chunk partition), so the KV Manager ships the right pages. The only
state the KV Manager reads that a skip does not leave right is the KDA window, which it ships ONCE, after the burst's
last chunk. So a skipped chunk that is (or may be) the burst's last must write the KDA carries at ``e`` into the
export slabs, which needs those carries: a checkpoint at ``e``, or the live carries when ``e == L``.

Which chunk is last: the prefill worker marks it (tt-d-gen ``device.prefill.mark_last_chunk``): every chunk header's
slot word carries bit 29 ("this header is marked") and, on a burst's last chunk, bit 30. Per chunk, so nothing has to be
learned or trusted across requests (an aborted request ends without a last chunk; a restarted engine may not mark).
An unmarked header may be the last chunk, so such a chunk is skipped only when its end carries are held.

The first chunk that is not skipped resumes the model at its start ``s``: the KDA carries are restored from the
checkpoint at ``s``, or rebuilt from the nearest checkpoint ``b < s`` by recomputing ``[b, s)`` in full chunks without
export or acks (the slabs already hold those pages; the DSA rows are rewritten with the same values), and every
decoder's ``next_start`` is set to ``s``. Resumed chunks start where a cold prefill's start, so every computed chunk is
the same program on the same inputs: the result is meant to be bit-identical to a cold prefill of the whole prompt
(MEASURED on device, see the device check in ``tests/prefix_reuse``).

Checkpoints: one per chunk boundary while the pool has room; when it is full, the oldest boundary that is not a
multiple of the smallest power-of-two stride (in chunks) that leaves a candidate is evicted, and the two newest are
never evicted. That keeps a roughly even spread over the retained prompt plus its tail (the resume point of a next
turn that extends it). Nothing here is fitted to a workload.

Everything device-side is a callback, so this module has no ttnn import and is tested on the host
(``tests/test_prefix_reuse.py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

MARKED_FLAG = 1 << 29  # tt-d-gen device.prefill.mark_last_chunk: set in EVERY chunk header's slot word
LAST_CHUNK_FLAG = 1 << 30  # ... and this one on a burst's last chunk


def split_slot_word(word: int) -> tuple[int, Optional[bool]]:
    """(slot, last) from the header's slot word: last is True / False on a marked header, None on an unmarked one
    (an engine without mark_last_chunk: any chunk may be the burst's last). Negative words (sentinels) pass through."""
    word = int(word)
    if word < 0 or not word & MARKED_FLAG:
        return word, None
    return word & ~(MARKED_FLAG | LAST_CHUNK_FLAG), bool(word & LAST_CHUNK_FLAG)


def enabled() -> bool:
    return os.environ.get("GLM53F_PREFIX_REUSE", "0") == "1"


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
            protected = {held[-1]} if self.n_slots >= 3 else set()
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
    kda_slab_writes: int = 0
    marked_chunks: int = 0
    extra: dict = field(default_factory=dict)


class PrefixRetention:
    """The retained prompt and the skip / resume decisions. Device actions are callbacks:

    ``save(slot)``             live KDA carries -> checkpoint slot
    ``restore(slot)``          checkpoint slot -> live KDA carries
    ``export_carries(slot)``   carries of checkpoint ``slot`` (None = the live ones) -> the KDA export slabs (+ sync)
    ``set_next_start(pos)``    every decoder's chunked-prefill cursor
    ``compute_silent(ids, start)``  one chunk through the model, no export, no acks (the carries then hold its end)
    """

    def __init__(
        self,
        *,
        chunk: int,
        n_ckpts: int,
        save: Callable[[int], None],
        restore: Callable[[int], None],
        export_carries: Callable[[Optional[int]], None],
        set_next_start: Callable[[int], None],
        compute_silent: Callable[[torch.Tensor, int], None],
        log: Callable[[str], None] = print,
    ):
        self.chunk = int(chunk)
        self.pool = CheckpointPool(n_ckpts, chunk)
        self._save, self._restore, self._export = save, restore, export_carries
        self._set_next_start, self._compute_silent = set_next_start, compute_silent
        self.log = log
        self.tokens = torch.zeros(0, dtype=torch.int32)
        self.length = 0  # rows [0, length) of the caches / slabs hold tokens[:length]; the live carries are at length
        # the in-flight prompt: every chunk so far skipped (matching), and where its next chunk must start (cursor)
        self.matching = False
        self.cursor = 0
        self.prompt_resumed = False
        self.stats = ReuseStats()

    # ------------------------------------------------------------------ state
    def reset(self, why: str) -> None:
        self.tokens = torch.zeros(0, dtype=torch.int32)
        self.length = 0
        self.pool.clear()
        self.matching = False
        self.cursor = 0
        self.log(f"[glm53f-prefix] retention reset ({why})")

    def carries_at(self, pos: int):
        """("live", None) when the live carries are at ``pos``, ("ckpt", slot) for a checkpoint, else None."""
        if pos == self.length:
            return ("live", None)
        slot = self.pool.held(pos)
        return None if slot is None else ("ckpt", slot)

    # ------------------------------------------------------------------ per chunk
    def begin_chunk(self, ids: torch.Tensor, start: int, end: int, last: Optional[bool]) -> bool:
        """Called before a non-empty chunk ``[start, end)``; ``last`` from ``split_slot_word`` (None = unmarked header).
        True = skipped (the caller only acknowledges it); False = compute it (the model is resumed at ``start`` if the
        prompt matched so far)."""
        if last is not None:
            self.stats.marked_chunks += 1
        if start == 0:
            self._new_prompt()
        ids = ids.to(torch.int32)
        if self.matching and start == self.cursor and end <= self.length and torch.equal(ids, self.tokens[start:end]):
            # a ragged chunk is always the burst's last; a full one is when marked so, or may be when unmarked
            may_be_last = last is None or last or (end - start) < self.chunk
            if not may_be_last:
                self._skipped(start, end)
                return True
            at = self.carries_at(end)
            if at is not None:
                self._export(at[1])
                self.stats.kda_slab_writes += 1
                self._skipped(start, end)
                return True
            # a possible last chunk whose end carries are not held: compute it (resumed at start below)
        self._prepare_compute(start)
        return False

    def note_empty(self) -> None:
        """An EMPTY chunk (a burst declined before compute): nothing computed or skipped, but it ends a request."""
        self.matching = False

    def end_compute(self, ids: torch.Tensor, start: int, end: int) -> None:
        """After the model computed ``[start, end)`` (the live carries are at ``end``): retain it."""
        self._record(ids, start, end)
        self.stats.computed_chunks += 1

    def invalidate(self, why: str) -> None:
        """A chunk failed part-way: nothing on the device can be vouched for."""
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
        if not resume:
            if start == 0:
                # a new prompt computed from scratch (the model zeroes the KDA carries at start 0): only [0, 0) stays
                self._truncate(0)
            elif start != self.length:
                # not the continuation of the prompt the device holds (the engine's tt-d-gen #430 re-send): the model's
                # own contiguity check decides; nothing retained can be vouched for past this point
                self.reset(f"chunk start {start} is neither 0 nor the retained end {self.length}")
            return
        at = self.carries_at(start)
        if at is not None and at[0] == "live":
            self._set_next_start(start)  # the live carries are already at start (the retained prompt ended there)
        elif at is not None:
            self._restore(at[1])
            self._set_next_start(start)
            self.length = start
        else:
            b, slot = self.pool.nearest_at_or_below(start)
            if slot is not None:
                self._restore(slot)
            self._set_next_start(b)
            self.length = b
            n = 0
            for x in range(b, start, self.chunk):
                y = min(start, x + self.chunk)
                self._compute_silent(self.tokens[x:y].to(torch.int64), x)
                self._record(self.tokens[x:y], x, y)
                n += 1
            self.stats.silent_chunks += n
            self.log(f"[glm53f-prefix] rebuilt the KDA carries at {start} from {b} ({n} chunk(s), no export)")
        self._truncate(start)

    def _truncate(self, pos: int) -> None:
        """The chunk about to be computed overwrites everything from ``pos`` on."""
        self.length = min(self.length, pos)
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
            f"{s.computed_chunks}, checkpoints saved {s.checkpoints_saved} (evicted {self.pool.evictions}), KDA slab "
            f"writes {s.kda_slab_writes}, marked headers {s.marked_chunks}"
        )
