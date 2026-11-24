# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Owner-aware physical blocks for the sequence-sharded DSA pool (GLM53_DSA_SHARD; doc/ttnn/14-production-tp8-megakernel/
dsa/shard_tp8_design.md §6.1).

vLLM allocates LOGICAL blocks and knows nothing about owner chips. The sharded pool (tt/dsa_shard.py ShardMap) needs
every page-table entry to be a PHYSICAL global block g that obeys `g % C == owner`, with local block `g // C`, where
column j of a slot's row is unit `j // bpu`, owned by chip `(j // bpu) % C`.

OwnerBlockMap translates:
- A logical block b listed at a covered column j gets, the first time, a physical block of chip owner(j) from that
  chip's free list. The mapping PERSISTS after the last live row lets go of b: vLLM keeps a finished request's blocks
  cached for prefix hits, and a hit lists b at the same column (the same positions), so it finds the same owner and the
  same physical block, content intact.
- Live rows refcount their blocks. An unreferenced mapped block (cached only) is evictable: when an owner chip has no
  free block, its least recently released one is reclaimed. Its content is gone, so the block no longer has a
  mapping, and a later row that claims its content raises.
- `keep` columns of a row (positions the request already computed: a prefix hit's cached tokens, a live request's
  length) must find their block mapped on the column's owner; otherwise the write is refused (never silently fresh).
- A block at a fresh column (>= keep) that is mapped on ANOTHER owner moves when no other live row lists it (vLLM
  reused the id for new content); if another live row lists it, that is a contract violation and the write raises.
- Columns past a row's coverage (and -1 entries) point at the column owner's NULL block: local block 0 of every chip,
  global id r. The null block is never handed out.
- Every call is atomic: a refused row leaves the map exactly as it was (an undo log).

Balance: every request starts at unit 0, which is chip 0, so a chip's live demand exceeds the even share by at most one
unit per live slot. Size each chip's pool at ceil(global / C) + bpu x max_batch blocks (`local_blocks_for`); cached
blocks beyond that are evicted per chip, least recently released first.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence


class OwnerBlockMap:
    EXTEND = (
        True  # set_row's fast path for a row that extends the slot's live row (tests compare it with the full walk)
    )

    def __init__(self, C: int, nb_local: int, bpu: int = 2):
        """C owner chips, nb_local physical blocks per chip (local block 0 = the chip's null block), bpu blocks per
        unit (128-token unit / 64-token block = 2)"""
        assert C >= 1 and nb_local >= 2 and bpu >= 1, (C, nb_local, bpu)
        self.C, self.nb_local, self.bpu = int(C), int(nb_local), int(bpu)
        self.free: List[List[int]] = [list(range(self.nb_local - 1, 0, -1)) for _ in range(self.C)]  # pop() -> 1, 2, ..
        self.phys: Dict[int, int] = {}  # logical block -> physical global block (persistent while cached)
        self.refs: Dict[int, int] = {}  # logical block -> live row occurrences
        self.idle: List["OrderedDict[int, None]"] = [OrderedDict() for _ in range(self.C)]  # unreferenced, LRU first
        self.rows: Dict[int, List[int]] = {}  # slot -> the logical blocks its live (covered) columns list
        self.evictions = 0
        self._undo: Optional[list] = None

    @staticmethod
    def local_blocks_for(C: int, nb_global: int, max_batch: int, bpu: int = 2) -> int:
        """physical blocks per chip for a global pool of nb_global blocks: the even share, the null block, and one unit
        of headroom per live slot (every request's first unit is chip 0's)"""
        return -(-int(nb_global) // int(C)) + 1 + int(bpu) * int(max_batch)

    def owner(self, j: int) -> int:
        return (int(j) // self.bpu) % self.C

    def null(self, j: int) -> int:
        """the physical null block of column j's owner (local block 0 of that chip = global id owner)"""
        return self.owner(j)

    def _nulls(self, n: int) -> List[int]:
        """[null(j) for j < n], built once per row length (set_row's tail past the coverage: 16,392 columns at 1M)"""
        cache = self.__dict__.setdefault("_null_rows", {})
        if n not in cache:
            cache[n] = [self.null(j) for j in range(n)]
        return cache[n]

    # ---- primitive mutations (logged for rollback while a call is open)
    def _log(self, *op) -> None:
        if self._undo is not None:
            self._undo.append(op)

    def _set_phys(self, b: int, g: Optional[int]) -> None:
        self._log("phys", b, self.phys.get(b))
        if g is None:
            self.phys.pop(b, None)
        else:
            self.phys[b] = g

    def _take(self, r: int, avoid: set) -> int:
        """a free physical block of chip r, evicting the least recently released cached block if the chip is dry"""
        if not self.free[r]:
            victim = next((b for b in self.idle[r] if b not in avoid), None)
            if victim is None:
                raise RuntimeError(f"owner chip {r} has no free physical block (nb_local {self.nb_local}, all live)")
            self._idle_drop(victim)
            self._release_phys(victim)
            self.evictions += 1
            self._log("evict")
        lb = self.free[r].pop()
        self._log("pop", r, lb)
        return lb * self.C + r

    def _release_phys(self, b: int) -> None:
        g = self.phys[b]
        self._set_phys(b, None)
        self.free[g % self.C].append(g // self.C)
        self._log("push", g % self.C)

    def _idle_add(self, b: int) -> None:
        r = self.phys[b] % self.C
        self.idle[r][b] = None
        self._log("idle_add", r, b)

    def _idle_drop(self, b: int) -> None:
        for r, od in enumerate(self.idle):
            if b in od:
                keys = list(od)
                del od[b]
                self._log("idle_drop", r, b, keys.index(b))
                return

    def _set_refs(self, b: int, n: int) -> None:
        self._log("refs", b, self.refs.get(b))
        if n:
            self.refs[b] = n
        else:
            self.refs.pop(b, None)

    def _rollback(self, log: list) -> None:
        for op in reversed(log):
            kind = op[0]
            if kind == "phys":
                _, b, old = op
                if old is None:
                    self.phys.pop(b, None)
                else:
                    self.phys[b] = old
            elif kind == "pop":
                self.free[op[1]].append(op[2])
            elif kind == "push":
                self.free[op[1]].pop()
            elif kind == "idle_add":
                del self.idle[op[1]][op[2]]
            elif kind == "idle_drop":
                _, r, b, at = op
                items = list(self.idle[r])
                items.insert(at, b)
                self.idle[r] = OrderedDict((k, None) for k in items)
            elif kind == "refs":
                _, b, old = op
                if old is None:
                    self.refs.pop(b, None)
                else:
                    self.refs[b] = old
            elif kind == "evict":
                self.evictions -= 1

    # ---- the row API
    def set_row(self, slot: int, row: Sequence[int], covered: int, keep: int = 0) -> List[int]:
        """the slot's logical row (columns < covered are live; -1 = no block) -> its physical row (same length).
        keep: columns < keep hold content the request already computed (their blocks must be mapped on the column's
        owner). Releases the blocks the slot's previous row listed that this row no longer lists (they stay mapped,
        evictable). Atomic: raises with the map unchanged."""
        assert self._undo is None, "set_row is not reentrant"
        self._undo = []
        try:
            return self._set_row(slot, row, covered, keep)
        except (RuntimeError, ValueError):
            self._rollback(self._undo)
            raise
        finally:
            self._undo = None

    def _set_row(self, slot: int, row: Sequence[int], covered: int, keep: int) -> List[int]:
        old = self.rows.get(slot, [])
        mine: Dict[int, int] = {}
        for b in old:
            mine[b] = mine.get(b, 0) + 1
        # only the covered columns are walked; the tail past the coverage is the cached null row (cf3h-d lever A: the
        # whole 16,392-column walk was most of a 1M row publish)
        head = [int(b) for b in row[: max(int(covered), 0)]]
        n_old = len(old)
        if self.EXTEND and n_old and len(head) >= n_old and head[:n_old] == old:
            # the row EXTENDS the slot's live row (a growing prompt / decode: the same blocks at the same columns):
            # those columns keep their physical blocks (a live block is never moved, evicted or re-owned while this
            # row lists it) and their references (+1 -1), so only the new columns are walked
            return self._extend_row(slot, row, head, old, keep)
        listed = {b for b in head if b >= 0}
        new: List[int] = []
        out: List[int] = []
        seen: set = set()
        for j, b in enumerate(head):
            if b < 0:
                out.append(self.null(j))
                continue
            if b in seen:
                raise ValueError(f"logical block {b} listed twice in slot {slot}'s row (column {j})")
            seen.add(b)
            r = self.owner(j)
            g = self.phys.get(b)
            if j < keep:
                if g is None:
                    raise ValueError(f"logical block {b} at computed column {j} of slot {slot} has no device content "
                                     f"(never written or evicted)")  # fmt: skip
                if g % self.C != r:
                    raise ValueError(f"logical block {b} is live on chip {g % self.C} but listed at column {j} of "
                                     f"chip {r}")  # fmt: skip
            elif g is not None and g % self.C != r:
                if self.refs.get(b, 0) != mine.get(b, 0):  # another live row lists it at its own column
                    raise ValueError(f"logical block {b} is live on chip {g % self.C} but listed at column {j} of "
                                     f"chip {r}")  # fmt: skip
                self._idle_drop(b)
                self._release_phys(b)  # new content for b: moves to owner r
                g = None
            if g is None:
                g = self._take(r, listed)
                self._set_phys(b, g)
            new.append(b)
            out.append(g)
        out.extend(self._nulls(len(row))[len(head) :])
        for b in new:
            if not self.refs.get(b):
                self._idle_drop(b)
            self._set_refs(b, self.refs.get(b, 0) + 1)
        for b in old:
            self._unref(b)
        self.rows[slot] = new
        return out

    def _extend_row(self, slot: int, row: Sequence[int], head: List[int], old: List[int], keep: int) -> List[int]:
        """_set_row for a head whose first len(old) columns are exactly the slot's live row: the same result, map and
        undo-safe mutations as the full walk, with only the columns past the old row walked"""
        n_old = len(old)
        listed = {b for b in head[n_old:] if b >= 0}  # the old blocks are live, so never eviction victims anyway
        seen: set = set(old)
        out: List[int] = [self.phys[b] for b in old]
        new: List[int] = []
        for j in range(n_old, len(head)):
            b = head[j]
            if b < 0:
                out.append(self.null(j))
                continue
            if b in seen:
                raise ValueError(f"logical block {b} listed twice in slot {slot}'s row (column {j})")
            seen.add(b)
            r = self.owner(j)
            g = self.phys.get(b)
            if j < keep:
                if g is None:
                    raise ValueError(f"logical block {b} at computed column {j} of slot {slot} has no device content "
                                     f"(never written or evicted)")  # fmt: skip
                if g % self.C != r:
                    raise ValueError(f"logical block {b} is live on chip {g % self.C} but listed at column {j} of "
                                     f"chip {r}")  # fmt: skip
            elif g is not None and g % self.C != r:
                if self.refs.get(b, 0):  # another live row lists it at its own column (this row's old blocks: seen)
                    raise ValueError(f"logical block {b} is live on chip {g % self.C} but listed at column {j} of "
                                     f"chip {r}")  # fmt: skip
                self._idle_drop(b)
                self._release_phys(b)  # new content for b: moves to owner r
                g = None
            if g is None:
                g = self._take(r, listed)
                self._set_phys(b, g)
            new.append(b)
            out.append(g)
        out.extend(self._nulls(len(row))[len(head) :])
        for b in new:
            if not self.refs.get(b):
                self._idle_drop(b)
            self._set_refs(b, self.refs.get(b, 0) + 1)
        self.rows[slot] = old + new
        return out

    def _unref(self, b: int) -> None:
        n = self.refs[b] - 1
        self._set_refs(b, n)
        if not n and b in self.phys:
            self._idle_add(b)  # cached: mapped, evictable

    def admit(self, pairs: Sequence) -> List[int]:
        """A KV-connector LOAD (an LMCache hit) writes logical blocks' pages at known columns BEFORE any row lists them:
        map every (logical block b, column j) on column j's owner WITHOUT a live reference (idle = cached, evictable),
        so the request's row (set_row with keep over the hit) finds them mapped. A block already mapped on that owner is
        reused (the load overwrites its content); one cached on another owner moves; one live on another owner raises.
        Blocks of this call are never this call's eviction victims. Atomic. Returns the physical blocks."""
        assert self._undo is None, "admit is not reentrant"
        self._undo = []
        try:
            return self._admit(pairs)
        except (RuntimeError, ValueError):
            self._rollback(self._undo)
            raise
        finally:
            self._undo = None

    def _admit(self, pairs: Sequence) -> List[int]:
        listed = {int(b) for b, _j in pairs}
        out: List[int] = []
        seen: set = set()
        for b, j in pairs:
            b, r = int(b), self.owner(j)
            if b in seen:
                raise ValueError(f"logical block {b} admitted twice")
            seen.add(b)
            g = self.phys.get(b)
            if g is not None and g % self.C != r:
                if self.refs.get(b):
                    raise ValueError(f"logical block {b} is live on chip {g % self.C} but loaded at column {j} of "
                                     f"chip {r}")  # fmt: skip
                self._idle_drop(b)
                self._release_phys(b)  # new content for b: moves to owner r
                g = None
            if g is None:
                g = self._take(r, listed)
                self._set_phys(b, g)
                self._idle_add(b)  # mapped, no live row yet: cached, evictable
            elif not self.refs.get(b):
                self._idle_drop(b)
                self._idle_add(b)  # most recently used
            out.append(g)
        return out

    def release_slot(self, slot: int) -> None:
        """the slot's request finished / was preempted: its blocks lose a live reference (they stay mapped and become
        evictable once no live row lists them: vLLM may serve them as a cached prefix)"""
        for b in self.rows.pop(slot, []):
            self._unref(b)

    def forget(self, b: int) -> None:
        """vLLM dropped logical block b's cached content (optional hint): its physical block is free again"""
        if b in self.phys and not self.refs.get(b):
            self._idle_drop(b)
            self._release_phys(b)

    def remap(self, mapping: Dict[int, int]) -> None:
        """slot compaction (Glm53Generator.remap_slots): the row of slot s moves to slot mapping[s], same blocks (no
        refcount change). A destination may not hold a row of its own unless it is itself a moved source."""
        moved = {s: self.rows.pop(s, []) for s in mapping}
        for s, d in mapping.items():
            if d in self.rows:
                for s2, r in moved.items():  # undo before raising
                    if r:
                        self.rows[s2] = r
                raise ValueError(f"remap destination slot {d} holds a live row")
        for s, d in mapping.items():
            if moved[s]:
                self.rows[d] = moved[s]

    def free_counts(self) -> List[int]:
        """free physical blocks per chip (cached-only blocks are not free: see evictable_counts)"""
        return [len(f) for f in self.free]

    def evictable_counts(self) -> List[int]:
        return [len(od) for od in self.idle]

    def physical(self, b: int) -> Optional[int]:
        return self.phys.get(int(b))
