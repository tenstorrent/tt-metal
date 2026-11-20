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
