# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Packed prefill (PACKED=on, owner 2026-11-19; bringup/PACKED_DESIGN.md): the chunk descriptor and the per-segment
index math, host only (torch, no ttnn: the CPU rung imports it without a device).

One prefill chunk of ``chunk`` rows (8192) carries up to ``lanes`` segments (4), each a request continuing its own
slot's cached history. v1 geometry is fixed: lane i = natural chunk rows [i L, (i + 1) L), L = chunk / lanes (2048); a
segment shorter than its lane is padded inside it; a lane may be empty. One segment over the whole chunk (lanes=1,
L = chunk) is today's chunk: ``ChunkDescriptor.single`` maps today's header {slot_id, actual_start, actual_end,
global_write_start} onto it.

Everything here depends only on the descriptor (a few integers a segment, the chunk header) and on load-time
geometry. The device path (PB.3) turns each function into a load-time table sliced per lane, or a tiny descriptor
tensor the engine writes with the ids (PACKED_DESIGN.md "Descriptor on the device"); nothing here is per-prompt
precompute and nothing needs a device read back.

Row conventions (as today's chunk): a lane's rows are natural order, row ``off + d`` = position ``start + d``; pad
rows continue the lane's positions (start + d for d >= len) and are masked exactly as today's pad rows past
actual_end (never attended, never written, engram rows zero); an empty lane's rows are pad at positions 0.. (any
value: nothing reads them). Starts are 32-aligned (the engine's rule, adapter.prefill_chunk).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

LANES = 4  # v1: up to 4 segments a chunk (owner)
LANE_ROWS = 2048  # v1: <= 2048 new tokens a segment (owner)
WINDOW = 128  # the sliding window (tt/attention.WINDOW, cfg.window)
KV_BLOCK = 64  # decode's KV block (settings KV_BLOCK_SIZE; attention.snapshot_cap)
NGRAM_TAIL = 3  # engram: ids before a segment's start its first n-grams read (cfg.engram_ngram - 1)
PAD_ID = 0xFFFFFFFF  # the engine's pad id (K-72)


def cap_of(end: int, block: int = KV_BLOCK) -> int:
    """attention.snapshot_cap without the settings import: decode's cap of an actual_end (PREFILL_END=cap)."""
    return int(end) // block * block if end > 0 else 0


@dataclass(frozen=True)
class Segment:
    """One request's part of a chunk: the engine's per-request header fields.

    slot         the request's slot (its served caches: compressed KV, index keys, window snapshot, ids)
    start        actual_start, 32-aligned: the position of the segment's first row
    end          actual_end (engine): start < end <= start + lane rows
    write_start  global_write_start (P.51; None = start): no history row below it is written; > start = encoder replay
    """

    slot: int
    start: int
    end: int
    write_start: int | None = None

    def __post_init__(self):
        assert self.start >= 0 and self.start % 32 == 0, f"segment start {self.start} must be 32-aligned"
        assert self.end > self.start, (self.start, self.end)
        if self.write_start is not None:
            assert self.write_start % 32 == 0, f"global_write_start {self.write_start} must be 32-aligned"

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def gws(self) -> int:
        """The effective global_write_start (at least start)."""
        return self.start if self.write_start is None else max(self.start, int(self.write_start))

    @property
    def replay(self) -> bool:
        """An encoder replay (P.51): the segment starts below its write frontier; windows truncated at start."""
        return self.gws > self.start

    def run_end(self, prefill_end: str = "cap", block: int = KV_BLOCK) -> int:
        """Where the segment's computation stops (PREFILL_END, P.49): cap(end) or end."""
        return cap_of(self.end, block) if prefill_end == "cap" else int(self.end)


@dataclass(frozen=True)
class ChunkDescriptor:
    """The packed chunk header: ``segments[i]`` occupies lane i (None = empty lane)."""

    chunk: int
    segments: tuple = field(default_factory=tuple)

    def __post_init__(self):
        n = len(self.segments)
        assert n >= 1 and self.chunk % n == 0, (self.chunk, n)
        lr = self.chunk // n
        assert lr % 32 == 0, f"lane rows {lr} must be whole tiles"
        for s in self.segments:
            if s is not None:
                assert s.length <= lr, f"segment of {s.length} rows > lane of {lr}"
        slots = [s.slot for s in self.segments if s is not None]
        assert len(slots) == len(set(slots)), f"two segments of one slot in a chunk: {slots}"

    # ---------------------------------------------------------------- construction
    @classmethod
    def single(cls, chunk: int, slot: int, start: int, end: int, write_start: int | None = None) -> "ChunkDescriptor":
        """Today's single-request header {slot_id, actual_start, actual_end, global_write_start} as a descriptor:
        one segment over the whole chunk (lanes = 1, lane rows = chunk)."""
        return cls(chunk, (Segment(slot, start, end, write_start),))

    @classmethod
    def packed(cls, chunk: int, segments, lanes: int = LANES) -> "ChunkDescriptor":
        """Up to ``lanes`` segments in lanes 0.. in order (None = an empty lane), the rest empty."""
        segs = list(segments) + [None] * (lanes - len(segments))
        assert len(segs) == lanes, (len(segs), lanes)
        return cls(chunk, tuple(segs))

    # ---------------------------------------------------------------- geometry
    @property
    def lanes(self) -> int:
        return len(self.segments)

    @property
    def lane_rows(self) -> int:
        return self.chunk // self.lanes

    def lane_off(self, lane: int) -> int:
        return lane * self.lane_rows

    def live(self) -> list[tuple[int, Segment]]:
        """(lane, segment) of every non-empty lane."""
        return [(i, s) for i, s in enumerate(self.segments) if s is not None]

    def new_tokens(self) -> int:
        return sum(s.length for _, s in self.live())

    def as_glued(self, prefill_end: str = "cap", block: int = KV_BLOCK, align: int = 0) -> "GluedChunk | None":
        """PB.9X: the glued chunk tt-d-gen sends for this header: the live segments in lane order (an empty lane takes
        no rows), back to back (``align`` 0) or each on whole chips (PB.9G, ``align`` = rows a chip). None when no lane
        is live."""
        live = [s for _, s in self.live()]
        return GluedChunk(self.chunk, tuple(live), prefill_end, block, align) if live else None

    def lane_seg_ids(self, tokens: torch.Tensor) -> list:
        """Lane-natural ids [chunk] (pack_ids' layout) -> each live segment's [length] ids, in lane order."""
        t = tokens.reshape(-1)
        return [t[self.lane_off(lane) : self.lane_off(lane) + s.length] for lane, s in self.live()]

    # ---------------------------------------------------------------- per row
    def row_positions(self) -> torch.Tensor:
        """[chunk] int64: each row's position (RoPE, compressor groups, causal masks). Pad rows continue their lane."""
        out = torch.empty(self.chunk, dtype=torch.int64)
        d = torch.arange(self.lane_rows, dtype=torch.int64)
        for i, s in enumerate(self.segments):
            out[self.lane_off(i) : self.lane_off(i) + self.lane_rows] = d + (0 if s is None else s.start)
        return out

    def row_lane(self) -> torch.Tensor:
        """[chunk] int64: each row's lane (also for pad rows)."""
        return torch.arange(self.chunk, dtype=torch.int64) // self.lane_rows

    def row_valid(self, prefill_end: str = "cap", block: int = KV_BLOCK) -> torch.Tensor:
        """[chunk] bool: rows the model computes for real ([start, run_end) of each segment); every other row is pad."""
        out = torch.zeros(self.chunk, dtype=torch.bool)
        for i, s in self.live():
            n = max(0, s.run_end(prefill_end, block) - s.start)
            out[self.lane_off(i) : self.lane_off(i) + n] = True
        return out

    def row_writes(self, prefill_end: str = "cap", block: int = KV_BLOCK) -> torch.Tensor:
        """[chunk] bool: rows whose history-cache rows are written ([max(start, gws), run_end) of each segment)."""
        out = torch.zeros(self.chunk, dtype=torch.bool)
        for i, s in self.live():
            lo, hi = write_range(s, prefill_end, block)
            if hi > lo:
                out[self.lane_off(i) + lo - s.start : self.lane_off(i) + hi - s.start] = True
        return out

    def row_slot(self) -> torch.Tensor:
        """[chunk] int64: the slot of each row's segment, -1 for an empty lane."""
        out = torch.full((self.chunk,), -1, dtype=torch.int64)
        for i, s in self.live():
            out[self.lane_off(i) : self.lane_off(i) + self.lane_rows] = s.slot
        return out

    def same_segment(self) -> torch.Tensor:
        """[chunk, chunk] bool: rows i and j belong to the same lane (the block-diagonal part of every mask)."""
        lane = self.row_lane()
        return lane.unsqueeze(1) == lane.unsqueeze(0)

    def causal_mask(self, prefill_end: str = "cap", block: int = KV_BLOCK) -> torch.Tensor:
        """[chunk, chunk] bool: query row i may read the NEW key row j: same segment, pos j <= pos i, j a real row.
        (History keys come from the slot's own caches; never from another lane.)"""
        pos = self.row_positions()
        valid = self.row_valid(prefill_end, block)
        return self.same_segment() & (pos.unsqueeze(0) <= pos.unsqueeze(1)) & valid.unsqueeze(0)

    # ---------------------------------------------------------------- ids
    def pack_ids(self, ids_per_lane: list, pad: int = PAD_ID) -> torch.Tensor:
        """[chunk] int64 natural-order ids: lane i's segment's ids at its rows, ``pad`` elsewhere. ``ids_per_lane[i]``
        is the segment's [length] ids (None for an empty lane)."""
        out = torch.full((self.chunk,), pad, dtype=torch.int64)
        for i, s in enumerate(self.segments):
            ids = ids_per_lane[i] if i < len(ids_per_lane) else None
            if s is None:
                assert ids is None or len(ids) == 0, f"ids for empty lane {i}"
                continue
            ids = torch.as_tensor(ids, dtype=torch.int64).reshape(-1)
            assert ids.numel() == s.length, (i, ids.numel(), s.length)
            out[self.lane_off(i) : self.lane_off(i) + s.length] = ids
        return out

    def unpack_rows(self, t: torch.Tensor, prefill_end: str | None = None, block: int = KV_BLOCK) -> list:
        """Rows [chunk, ...] -> per lane the segment's rows [length (or run length), ...] (None for an empty lane)."""
        assert t.shape[0] == self.chunk, (tuple(t.shape), self.chunk)
        out = []
        for i, s in enumerate(self.segments):
            if s is None:
                out.append(None)
                continue
            n = s.length if prefill_end is None else max(0, s.run_end(prefill_end, block) - s.start)
            out.append(t[self.lane_off(i) : self.lane_off(i) + n])
        return out

    # ---------------------------------------------------------------- mesh
    def lane_chips(self, sp: int, tp: int) -> list[list[int]]:
        """Chip-linear ids (q = r tp + c) holding each lane's rows. The model's rows are natural blocks: chip q holds
        rows [q R, (q + 1) R), R = chunk / (sp tp) (WindowGeometry.query_offsets). A lane is whole chips whenever
        R divides the lane (every box at 8192: R = 256 / 512 / 1024 / 2048 on 8x4 / 4x4 / 2x4 / 2x2)."""
        n = sp * tp
        assert self.chunk % (32 * n) == 0, f"chunk {self.chunk} does not split into tiles over {n} chips (DS41-297)"
        per = self.chunk // n
        assert self.lane_rows % per == 0, f"lane of {self.lane_rows} rows is not whole chips of {per} rows"
        k = self.lane_rows // per
        return [list(range(i * k, (i + 1) * k)) for i in range(self.lanes)]

    def carry_sources(self, sp: int, tp: int) -> list[tuple[str, int]]:
        """The window carry (the 128 rows before a chip's first row) of every chip q, chip-linear: ("prev", q - 1)
        when the previous chip holds the same lane (today's neighbour send), ("slot", lane) when q holds a lane's
        first row (the segment's own slot: its ring / served snapshot at the segment's start; start 0 or a replay:
        none), never another lane's rows."""
        out = []
        for lane, chips in enumerate(self.lane_chips(sp, tp)):
            for j, q in enumerate(chips):
                out.append(("slot", lane) if j == 0 else ("prev", q - 1))
        return out

    def moe_real_prefix(self, sp: int, valid: torch.Tensor | None = None) -> tuple[list[int], bool]:
        """PB.4 (PACKED_MOE=on): the MoE head's real rows of every SP row as a prefix count (the head's padded chunk
        describes an SP row's real rows as a prefix of its rows: moe_block_ttnn padded_rows.hpp). SP row r holds the
        natural chunk rows [r W, (r + 1) W), W = chunk / sp (WindowGeometry: chip q = r tp + c, rows [q R, ..)).
        ``valid``: [chunk] bool, the rows computed for real (default row_valid()). Returns (n_r a row, exact): n_r =
        1 + the SP row's last real row (0: none); exact = every SP row's real rows ARE that prefix (lanes of whole SP
        rows: 8x4 / 4x4); else (two lanes in one SP row, 2x4 / 2x2) the pad rows between two real runs count as real
        (computed as today: a superset, correct)."""
        v = self.row_valid() if valid is None else valid.reshape(-1).bool()
        assert v.numel() == self.chunk and self.chunk % sp == 0, (v.numel(), self.chunk, sp)
        w = self.chunk // sp
        out, exact = [], True
        for r in range(sp):
            row = v[r * w : (r + 1) * w]
            idx = torch.nonzero(row).reshape(-1)
            n = int(idx[-1]) + 1 if idx.numel() else 0
            exact = exact and bool(row[:n].all())
            out.append(n)
        return out, exact

    def history_slots(self, sp: int, tp: int) -> list[int]:
        """[sp tp] the slot whose history each chip's queries read (-1: an empty lane's chips read none)."""
        out = [-1] * (sp * tp)
        for lane, chips in enumerate(self.lane_chips(sp, tp)):
            s = self.segments[lane]
            for q in chips:
                out[q] = -1 if s is None else s.slot
        return out


# ============================================================================================= per segment
def write_range(s: Segment, prefill_end: str = "cap", block: int = KV_BLOCK) -> tuple[int, int]:
    """Positions [lo, hi) whose history-cache rows (compressed KV, index keys, window ring / snapshot) the segment
    writes into its slot: lo = global_write_start (P.51), hi = the run end (PREFILL_END=cap: decode's cap)."""
    hi = s.run_end(prefill_end, block)
    lo = min(s.gws, hi)
    return lo, hi


def compress_groups(s: Segment, ratio: int, prefill_end: str = "cap", block: int = KV_BLOCK) -> dict:
    """The compressor's groups of ratio ``ratio`` for the segment (reference _compressor: a group of r tokens sits at
    its first token's position; group j holds positions [r j, r j + r)):
        carry_in   tokens of the group straddling the segment's start, carried from the slot's state (start % r)
        g0         the first group written (the one completed by this segment, = start // r)
        groups     [g_lo, g_hi): complete groups whose rows are written (inside write_range)
        carry_out  tokens of the group straddling the run end, carried to the slot's state (run_end % r)
    Ratio 0 / None: no compressor. 32-aligned starts make carry_in 0 for every ratio dividing 32 (this checkpoint:
    1 and 2); a ratio 4 / 128 compressor (other DeepSeek-V4 configs) carries start % r tokens per slot."""
    if not ratio:
        return {"carry_in": 0, "g0": None, "groups": (0, 0), "carry_out": 0}
    lo, hi = write_range(s, prefill_end, block)
    run = s.run_end(prefill_end, block)
    return {
        "carry_in": s.start % ratio,
        "g0": s.start // ratio,
        "groups": (-(-lo // ratio), hi // ratio),  # (a group is written when complete and at or above lo)
        "carry_out": run % ratio,
    }


def window_carry(s: Segment, window: int = WINDOW) -> dict:
    """The segment's window carry: positions [lo, start) its first rows' windows read from the slot's ring /
    served snapshot; ``cls`` = the index table's start class (WindowGeometry.start_class: min(start, 128)); a replay
    truncates at start (class 0, no carry rows read)."""
    if s.replay:
        return {"lo": s.start, "hi": s.start, "cls": 0}
    lo = max(0, s.start - window)
    return {"lo": lo, "hi": s.start, "cls": min(s.start, window)}


def engram_tail(history: torch.Tensor, start: int, n: int = NGRAM_TAIL) -> torch.Tensor:
    """The ``n`` ids before ``start`` from the slot's id history ([>= start] int, positions 0..), -1 where the
    position is < 0 (reference State.tail: a new sequence's tail is -1). The hasher maps them (token_map)."""
    out = torch.full((n,), -1, dtype=torch.int64)
    lo = max(0, start - n)
    if start > lo:
        out[n - (start - lo) :] = torch.as_tensor(history[lo:start], dtype=torch.int64)
    return out


def snapshot_range(s: Segment, prefill_end: str = "cap", block: int = KV_BLOCK, window: int = WINDOW) -> tuple:
    """The served window snapshot the segment's chunk rewrites for its slot (P.42 / P.49): rows [cap - 128, cap) at
    cap = decode's cap of its actual_end."""
    cap = cap_of(s.end, block)
    return max(0, cap - window), cap


# ============================================================================================= DRAM budget
def history_bytes_per_chip(
    histories: list[int],
    sources: int = 3,
    ratio: int = 2,
    row_bytes: int = 320,
    index_key_bytes: int = 136,
    sp: int = 8,
    per_lane: bool = False,
) -> dict:
    """Per chip bytes of the gathered history buffers a packed chunk holds at once (STATIC_TABLES=off, HISTORY_GATHER=
    rows, HISTORY_GATHER_HIDE=on: one buffer per gathered source, P.75; HISTORY_FORMAT=bfp4: 320 B row records, P.77),
    plus the decoded index keys (INDEX_K_SOURCE=served: bfloat8_b [rows / sp, 128] a chip and source: 128 x 1 B + 2
    exponent bytes per 16 = 136 B a row). ``histories``: each live segment's history (positions). ``per_lane``: each
    chip holds only its own lane's slot (PACKED_DESIGN.md v2 gather); else every chip holds every segment's table (v1:
    one stock gather per segment into its own buffer)."""
    rows = [-(-h // ratio) for h in histories]
    tab = [r * row_bytes * sources for r in rows]
    keys = [-(-r // sp) * index_key_bytes * sources for r in rows]
    table = max(tab) if (per_lane and tab) else sum(tab)
    return {"table": table, "index_keys": sum(keys), "total": table + sum(keys)}


# ============================================================================================= glued (PB.7)
GLUED_MAX = 6  # PB.7 (owner 2026-11-20): up to 6 segments a glued chunk


def real_prefix(chunk: int, sp: int, v: torch.Tensor) -> tuple[list[int], bool]:
    """ChunkDescriptor.moe_real_prefix's rule for any [chunk] bool of real rows (a function of the rows only)."""
    v = v.reshape(-1).bool()
    assert v.numel() == chunk and chunk % sp == 0, (v.numel(), chunk, sp)
    w = chunk // sp
    out, exact = [], True
    for r in range(sp):
        row = v[r * w : (r + 1) * w]
        idx = torch.nonzero(row).reshape(-1)
        n = int(idx[-1]) + 1 if idx.numel() else 0
        exact = exact and bool(row[:n].all())
        out.append(n)
    return out, exact


@dataclass(frozen=True)
class GluedChunk:
    """PB.7 (OWNER 2026-11-20, PACKED_GLUED=on): a chunk of ``chunk`` rows holding up to GLUED_MAX segments BACK TO
    BACK, any lengths: no lane alignment, no padding between segments (only the chunk's tail is pad).

    Two row layouts:
        engine rows    what tt-d-gen sends (drafts/tt_d_gen_packed_ask.md): segment k's ids [start, end) at rows
                       [in_off(k), in_off(k) + length), in_off(k) = the lengths before it; pad (PAD_ID) after the last
        model rows     what the model computes ("internal"): segment k's COMPUTED rows [start, run_end) (PREFILL_END:
                       cap = decode's cap of end, P.49; end = every row) at rows [lane_off(k), lane_off(k) + rows(k)),
                       lane_off(k) = the computed rows before it; the chunk's tail pad. The rows [run_end, end) of a
                       segment are pad in both layouts (today's chunk computes nothing past cap either), so dropping
                       them keeps every computed row and no pad between segments. Under cap with block-aligned starts
                       every segment's model rows are whole tiles (32 x k), so a segment never shares a tile row.
    ``lane_off`` / ``live`` / ``pack_ids`` / ``row_positions`` keep ChunkDescriptor's names (a "lane" is a segment's
    index here) so the CPU reference's packed forward (reference/packed_ref.forward_packed) runs a glued chunk as is.

    A request that does not fit the rows left is SPLIT (``split``): its first part ends on the last kv block that fits
    (a non-last chunk of a request ends on a block, as today's chunked prefill: its whole first part is computed) and
    the rest is a segment of a later chunk starting there (its own slot, its own write_start); the model does not know
    the next chunk."""

    chunk: int
    segments: tuple = field(default_factory=tuple)
    prefill_end: str = "cap"
    block: int = KV_BLOCK
    align: int = 0  # PB.9G (OWNER 2026-11-21, chip-aligned lanes): rows a chip; 0 = back to back (PB.7)

    glued = True

    def __post_init__(self):
        n = len(self.segments)
        assert 1 <= n <= GLUED_MAX, f"a glued chunk holds 1..{GLUED_MAX} segments, got {n}"
        assert all(s is not None for s in self.segments), "a glued chunk has no empty lanes"
        if self.align:
            a = int(self.align)
            assert a % 32 == 0 and self.chunk % a == 0, (self.chunk, a)
            for s in self.segments:
                assert s.start % 32 == 0, f"chip-aligned segments start 32-aligned (tt-d-gen), got {s}"
            need = sum(self.ext(k) for k in range(n))
            assert need <= self.chunk, (
                f"chip-aligned segments need {need // a} chips of {a} rows > {self.chunk // a} "
                f"({[s.length for s in self.segments]})"
            )
        else:
            assert (
                sum(s.length for s in self.segments) <= self.chunk
            ), f"glued segments of {sum(s.length for s in self.segments)} rows > chunk {self.chunk}"
        slots = [s.slot for s in self.segments]
        assert len(slots) == len(set(slots)), f"two segments of one slot in a chunk: {slots}"

    # ---------------------------------------------------------------- geometry
    @property
    def lanes(self) -> int:
        return len(self.segments)

    def rows(self, k: int) -> int:
        """Model rows of segment k: its computed rows (run_end - start, >= 0)."""
        s = self.segments[k]
        return max(0, s.run_end(self.prefill_end, self.block) - s.start)

    def ext(self, k: int) -> int:
        """PB.9G (``align`` > 0): segment k's rows in the chunk, engine and model alike: its whole chips (its length
        rounded up to ``align``; a segment that computes nothing still owns one chip: its snapshot and acks)."""
        a = int(self.align)
        return -(-max(1, self.segments[k].length) // a) * a

    def in_off(self, k: int) -> int:
        """Engine row of segment k's first id."""
        if self.align:
            return sum(self.ext(j) for j in range(k))
        return sum(s.length for s in self.segments[:k])

    def lane_off(self, k: int) -> int:
        """Model row of segment k's first computed row (PB.9G: its first chip's first row; engine row = model row)."""
        if self.align:
            return sum(self.ext(j) for j in range(k))
        return sum(self.rows(j) for j in range(k))

    def used_rows(self) -> int:
        return self.lane_off(self.lanes)

    def live(self) -> list[tuple[int, Segment]]:
        return list(enumerate(self.segments))

    def new_tokens(self) -> int:
        return sum(s.length for s in self.segments)

    def tile_aligned(self) -> bool:
        """Every segment's model rows start on a tile row (the device path's condition)."""
        return all(self.lane_off(k) % 32 == 0 for k in range(self.lanes))

    # ---------------------------------------------------------------- per row (model rows)
    def row_seg(self) -> torch.Tensor:
        """[chunk] int64: each model row's segment, -1 for the tail pad."""
        out = torch.full((self.chunk,), -1, dtype=torch.int64)
        for k in range(self.lanes):
            out[self.lane_off(k) : self.lane_off(k) + self.rows(k)] = k
        return out

    def row_positions(self) -> torch.Tensor:
        """[chunk] int64: each model row's position; the tail pad continues the last segment (never read)."""
        out = torch.empty(self.chunk, dtype=torch.int64)
        out[:] = torch.arange(self.chunk, dtype=torch.int64)
        for k, s in self.live():
            a = self.lane_off(k)
            n = self.ext(k) if self.align else self.rows(k)  # (PB.9G: a chip's pad rows continue its segment)
            out[a : a + n] = s.start + torch.arange(n, dtype=torch.int64)
        u = self.used_rows()
        if self.lanes and u < self.chunk:
            k = self.lanes - 1
            n = self.ext(k) if self.align else self.rows(k)
            out[u:] = self.segments[k].start + n + torch.arange(self.chunk - u, dtype=torch.int64)
        return out

    def row_valid(self) -> torch.Tensor:
        return self.row_seg() >= 0

    def row_writes(self) -> torch.Tensor:
        out = torch.zeros(self.chunk, dtype=torch.bool)
        for k, s in self.live():
            lo, hi = write_range(s, self.prefill_end, self.block)
            if hi > lo:
                out[self.lane_off(k) + lo - s.start : self.lane_off(k) + hi - s.start] = True
        return out

    def row_slot(self) -> torch.Tensor:
        seg = self.row_seg()
        slots = torch.tensor([s.slot for s in self.segments] + [-1], dtype=torch.int64)
        return slots[seg]

    def same_segment(self) -> torch.Tensor:
        seg = self.row_seg()
        return (seg.unsqueeze(1) == seg.unsqueeze(0)) & (seg.unsqueeze(1) >= 0)

    def causal_mask(self) -> torch.Tensor:
        pos = self.row_positions()
        return self.same_segment() & (pos.unsqueeze(0) <= pos.unsqueeze(1))

    # ---------------------------------------------------------------- ids
    def engine_ids(self, ids_per_seg: list, pad: int = PAD_ID) -> torch.Tensor:
        """[chunk] engine-order ids: segment k's [length] ids at rows in_off(k).., ``pad`` after the last."""
        out = torch.full((self.chunk,), pad, dtype=torch.int64)
        for k, s in self.live():
            ids = torch.as_tensor(ids_per_seg[k], dtype=torch.int64).reshape(-1)
            assert ids.numel() == s.length, (k, ids.numel(), s.length)
            out[self.in_off(k) : self.in_off(k) + s.length] = ids
        return out

    def model_ids(self, engine: torch.Tensor, pad: int = PAD_ID) -> torch.Tensor:
        """Engine-order ids [chunk] -> model-order ids [chunk] (each segment's computed rows; ``pad`` elsewhere)."""
        engine = engine.reshape(-1)
        out = torch.full((self.chunk,), pad, dtype=torch.int64)
        for k in range(self.lanes):
            a, n = self.lane_off(k), self.rows(k)
            out[a : a + n] = engine[self.in_off(k) : self.in_off(k) + n]
        return out

    def pack_ids(self, ids_per_lane: list, pad: int = PAD_ID) -> torch.Tensor:
        """[chunk] model-order ids from each segment's [length] ids (ChunkDescriptor.pack_ids' name)."""
        return self.model_ids(self.engine_ids(ids_per_lane, pad), pad)

    def unpack_rows(self, t: torch.Tensor) -> list:
        assert t.shape[0] == self.chunk, (tuple(t.shape), self.chunk)
        return [t[self.lane_off(k) : self.lane_off(k) + self.rows(k)] for k in range(self.lanes)]

    # ---------------------------------------------------------------- mesh
    def chip_segments(self, sp: int, tp: int) -> list[list[int]]:
        """Per chip (chip-linear q = r tp + c: model rows [q R, (q + 1) R)) the segments whose rows it holds."""
        n = sp * tp
        per = self.chunk // n
        seg = self.row_seg()
        return [sorted({int(x) for x in seg[q * per : (q + 1) * per].tolist() if x >= 0}) for q in range(n)]

    def moe_real_prefix(self, sp: int, valid: torch.Tensor | None = None) -> tuple[list[int], bool]:
        return real_prefix(self.chunk, sp, self.row_valid() if valid is None else valid)

    def aligned(self, align: int) -> "GluedChunk":
        """PB.9G: the same segments on whole chips of ``align`` rows (self when it already is)."""
        if int(self.align) == int(align):
            return self
        return GluedChunk(self.chunk, tuple(self.segments), self.prefill_end, self.block, int(align))

    def as_lanes(self, lanes: int = LANES) -> "ChunkDescriptor | None":
        """The same chunk as fixed lanes (ChunkDescriptor.packed), when it is one: every segment at model row k L
        (L = chunk / lanes) and at most L engine rows long. Then the glued chunk IS that lane chunk (same model rows,
        same per-chip segments) and runs the accepted lane path (PACKED_ATTN_ONECALL etc.), bit for bit."""
        L = self.chunk // lanes
        if self.lanes > lanes:
            return None
        for k, s in self.live():
            if self.lane_off(k) != k * L or s.length > L:
                return None
        return ChunkDescriptor.packed(self.chunk, list(self.segments), lanes=lanes)

    # ---------------------------------------------------------------- packing (the harness / the draft's packer)
    @classmethod
    def fill(
        cls,
        chunk: int,
        requests: list,
        prefill_end: str = "cap",
        block: int = KV_BLOCK,
        max_segs: int = GLUED_MAX,
        align: int = 0,
    ):
        """Pack ``requests`` (Segment, in order) into one glued chunk: back to back while rows and segment slots are
        left; the first that does not fit is SPLIT (``split``). Returns (GluedChunk, the rest: [the split's
        continuation] + the requests not taken)."""
        segs, left, rest = [], chunk, list(requests)
        while rest and len(segs) < max_segs and left > 0:
            s = rest[0]
            need = -(-max(1, s.length) // align) * align if align else s.length
            if need <= left:
                segs.append(rest.pop(0))
                left -= need
                continue
            first, cont = split(s, left, block)  # (PB.9G: the first part ends inside the chips left, on a kv block)
            if first is not None:
                segs.append(first)
                rest[0] = cont
            break
        return cls(chunk, tuple(segs), prefill_end, block, align), rest


def split(s: Segment, rows: int, block: int = KV_BLOCK) -> tuple:
    """Split request ``s`` to fit ``rows`` engine rows: (first part, continuation). The first part ends on the last kv
    block boundary that fits (position e1 = (start + rows) // block x block, > start; a non-last chunk ends on a block,
    so its cap is its end and every row of it is computed); the continuation starts at e1 with the same slot and
    write_start (a replay's frontier carries over). (None, s) when no block boundary fits."""
    e1 = (s.start + rows) // block * block
    if e1 <= s.start:
        return None, s
    assert e1 < s.end, (s, rows)
    return Segment(s.slot, s.start, e1, s.write_start), Segment(s.slot, e1, s.end, s.write_start)
