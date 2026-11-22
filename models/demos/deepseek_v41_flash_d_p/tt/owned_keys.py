# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PACKED_KEYS_OWNED=on (PB.KR, OWNER 2026-11-21, option 1): in a chip-aligned packed chunk each segment's decoded index
keys go ONLY to the chips that segment owns, and each chip scores its query rows on its own keys.

Off (PB.7C): the one-call ring (ttnn.bringup.ring_indexer_score_dsa lane_ranges) gathers EVERY segment's key buffer
(each chip's SP row's decoded rows, tt/index_keys) around the SP ring to every chip, inside the score op (at 1M: 4 x
~62 MB received a chip a layer; its all-gather column takes one of the 11 compute columns).

On (this file):
    history    after the first engram wait's decode region (tt/gather_hide.Hide.decodes: every segment's history
               [0, c wr) decoded into its lane's buffer, as today), ``Keys.make_plan`` cuts each index source's rows
               [0, c wr) (local rows, every owner) into pieces of ~PIECE_MS over the attention phases between that
               region and the source's indexer (layers dec_layer .. s - 1, evenly spaced; the source's tails region
               too when they are too few); each piece is ONE tt/owned_gather.OwnedSend program (page = a
               bfloat8_b tile) on the gather column: every chip sends its rows of every segment to that segment's
               chips, into the source's ONE gathered key buffer [1, 1, owners x local rows, 128] (owner q at rows
               q x local rows: the ring's gathered layout).
    tail       at the indexer (``finish``, after ix._keys decoded the chunk's own W blocks [c wr, hi)): the pieces a
               region did not run, then rows [h, hi) of every segment (h: what its history held; 0 when nothing was
               prefetched) in one program, serially.
    score      ring_indexer_score_dsa keys_gathered=True: the same lane_ranges call with every range reading buffer 0
               (the gathered buffer: this chip's own segment's keys only), no all-gather, every column of the grid.

The bytes each chip scores are the ring's (each row decodes on its own; the owned send copies pages): the same
scores, the same top-k. An idle chip (no segment) receives segment 0's keys, as its filler range read buffer 0 before.
Per-chunk values (rule 20): the destination map, the row ranges and the lane buffers' addresses are OwnedSend's
common value slots (one program a (gathered buffer, lane buffer, pages, axis, page) geometry); the ranges table is the
ring's runtime table as before; the gathered buffers are load-time-sized per (capacity, source)."""

from __future__ import annotations

import ttnn
from models.demos.common.bringup.testing import profiler

from . import index_keys
from .common import DRAM
from .settings import settings

PIECE_MS = 1.0  # a history piece's planned send time (an attention phase hides ~1.6 ms of gathering, gather_hide)
FIXED_MS = 0.15  # planning estimate of one send's fixed cost (gather_hide.GATHER_FIXED_MS)
STATS = {"chunks": 0, "pieces": 0, "flushed": 0, "tails": 0}  # since load (diagnostic, tests)


def on() -> bool:
    """PACKED_KEYS_OWNED=on (call-time, bringup/hooks.py batch_call_settings) and the fork's keys_gathered is built."""
    return settings.get("PACKED_KEYS_OWNED") == "on" and _has_fork()


def _has_fork() -> bool:
    return "keys_gathered" in (getattr(ttnn.bringup.ring_indexer_score_dsa, "__doc__", "") or "")


_GATHERED: dict = {}  # (mesh id, rows, width, source) -> (mesh, tensor)


def gathered(mesh, rows: int, width: int, s) -> ttnn.Tensor:
    """Source s's gathered key buffer [1, 1, rows, width] bfloat8_b TILE DRAM interleaved (made once a process)."""
    k = (mesh.id(), int(rows), int(width), s)
    hit = _GATHERED.get(k)
    if hit is None or hit[0] is not mesh or not hit[1].is_allocated():
        # (zeros once, as the ring's gathered buffer is made: rows no send reached hold zeros, never stale bytes)
        t = ttnn.zeros(
            ttnn.Shape([1, 1, int(rows), int(width)]),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
        )
        hit = _GATHERED[k] = (mesh, t)
    return hit[1]


def _page(width: int) -> tuple[int, int]:
    """(bytes a page, pages a 32-row band) of a bfloat8_b TILE key buffer of ``width`` columns."""
    try:
        tb = int(ttnn.tile_size(ttnn.bfloat8_b))
    except Exception:  # noqa: BLE001
        tb = 1088
    return tb, int(width) // 32


class Keys:
    """The owned key sends of one chip-aligned packed chunk (tt/packed_glued.GluedCall.keys). ``segs``: the computing
    segments (SegRuns, segment order), ``chip_seg``: per chip (chip-linear) the index of its segment (idle: 0)."""

    def __init__(self, mesh, segs: list, chip_seg: list):
        self.mesh, self.segs, self.dest = mesh, list(segs), list(chip_seg)
        self.plan: dict = {}  # region key -> [(s, a, b)] (local rows of every owner)
        self.held: dict = {}  # s -> [history rows each entry held after the decodes' region]
        self.done: dict = {}  # s -> [rows of each entry in the gathered buffer]
        STATS["chunks"] += 1
        if STATS["chunks"] == 1:
            from loguru import logger

            logger.info(
                f"PACKED_KEYS_OWNED=on: each segment's index keys to its own chips, scored locally "
                f"({len(self.segs)} segments, chip -> segment {self.dest})"
            )

    # ------------------------------------------------------------------ geometry
    def _entries(self, s) -> list:
        """[(index-key cache, its lane KeyBuffer)] of source s, segment order (None: no served cache)."""
        out = []
        for r in self.segs:
            c = r.state.compress.get(s)
            ik = getattr(c, "index_k", None) if c is not None else None
            if ik is None or not getattr(ik, "k_served", False):
                return []
            out.append((ik, index_keys._BUFFERS[index_keys._key(ik)]))
        return out

    @staticmethod
    def _holds(ik, buf) -> int:
        """Rows [0, n) of ``ik``'s served keys decoded in ``buf`` in this chunk (0: none)."""
        h = buf.holds
        if h is None or h[0] != ik.serial or h[3] != index_keys._EPOCH[0]:
            return 0
        return int(h[4])

    def _out(self, ents: list) -> ttnn.Tensor:
        """The source's gathered buffer: owners (SP rows) x the caches' local rows (the ring's k layout)."""
        ik0 = ents[0][0]
        layer = ik0.key_layer[0] if isinstance(ik0.key_layer, tuple) else ik0.key_layer  # (a lane's: its source's)
        return gathered(self.mesh, int(self.mesh.shape[0]) * int(ik0.crows), int(ik0.width), layer)

    def _send(self, s, ents: list, ranges: list) -> None:
        """ONE owned send: entry e's local rows ranges[e] = (a, b) of every owner to its segment's chips."""
        from .owned_gather import sender

        ik0 = ents[0][0]
        width, crows = int(ik0.width), int(ik0.crows)
        tb, band = _page(width)
        out = self._out(ents)
        R = crows // 32 * band
        entries = [(buf.tensor, a // 32 * band, b // 32 * band) for (_, buf), (a, b) in zip(ents, ranges)]
        sender(self.mesh)(out, R, 1, entries, self.dest, page=tb)

    # ------------------------------------------------------------------ history pieces
    def make_plan(self, order: list, dec_layer, sources) -> None:
        """After the decodes' region: each source's decoded history rows cut into pieces over the regions between it and
        the source's indexer (module docstring)."""
        self.plan.clear()
        for s in sorted(sources):
            ents = self._entries(s)
            if not ents:
                continue
            held = [self._holds(ik, buf) for ik, buf in ents]
            self.held[s] = held
            top = max(held)
            if top <= 0 or dec_layer is None:
                continue
            # as few pieces as cover the send at ~PIECE_MS each (each piece is a launch: ~0.1-0.3 ms of host on a chunk
            # that is host-bound at short histories), on attention phases spread evenly over the ones before the
            # reader (the tails region beside the reader's gates and query too when they are too few)
            ms = FIXED_MS + (int(self.mesh.shape[0]) - 1) * top * int(ents[0][0].width) * 17 / 16 / 24e6
            want = max(1, -(-int(ms * 1000) // int(PIECE_MS * 1000)))
            slots = [("attn", i) for i in order if dec_layer <= i < s]
            if len(slots) < want:  # (too few phases: the tails region too)
                slots.append(("tails", s))
            n = min(len(slots), want)
            picks = [slots[(k * len(slots)) // n] for k in range(n)]
            a = 0
            for k, key in enumerate(picks):
                b = top if k == n - 1 else min(top, -(-((k + 1) * top // n) // 32) * 32)
                if b > a:
                    self.plan.setdefault(key, []).append((s, a, b))
                a = b

    def gathers(self, key) -> list:
        """The planned pieces of region ``key`` as region gathers [(lane, fn(sub), est ms)]."""
        out = []
        for s, a, b in self.plan.pop(key, []):
            ents = self._entries(s)
            held = self.held.get(s)
            if not ents or held is None:
                continue
            ik0 = ents[0][0]
            nbytes = (int(self.mesh.shape[0]) - 1) * (b - a) * int(ik0.width) * 17 // 16  # (bfp8: 1088 B a 1024)

            def g(sub, s=s, a=a, b=b, ents=ents, held=held):
                profiler.signpost("indexer_keys")
                self._send(s, ents, [(min(a, h), min(b, h)) for h in held])
                STATS["pieces"] += 1

            out.append((-1, g, FIXED_MS + nbytes / 24e6))
        return out

    # ------------------------------------------------------------------ at the indexer
    def finish(self, s, ix=None) -> ttnn.Tensor:
        """Before source s's score (after ix._keys decoded every segment's rows [0, hi)): the pieces no region ran, then
        rows [h, hi) of every segment, serially; returns the gathered buffer."""
        ents = self._entries(s)
        assert ents, f"PACKED_KEYS_OWNED: source {s} has no served index keys"
        his = [self._holds(ik, buf) for ik, buf in ents]
        out = self._out(ents)
        if self.done.get(s) == his:  # (a reader of the same source again: the buffer holds it)
            return out
        profiler.signpost("indexer_keys")
        held = self.held.get(s)
        left = [p for key in list(self.plan) for p in self.plan[key] if p[0] == s]
        if left:  # (a region of the plan did not run: its pieces here)
            for key in list(self.plan):
                self.plan[key] = [p for p in self.plan[key] if p[0] != s]
            for _, a, b in left:
                self._send(s, ents, [(min(a, h), min(b, h)) for h in held])
            STATS["flushed"] += len(left)
        lo = held if held is not None else [0] * len(ents)
        if any(h > l for h, l in zip(his, lo)):
            self._send(s, ents, [(min(l, h), h) for l, h in zip(lo, his)])
            STATS["tails"] += 1
        self.done[s] = his
        if STATS["chunks"] <= 2:
            from loguru import logger

            logger.info(f"PB.KR keys: source {s} rows held {held} -> {his}; {dict(STATS)}")
        profiler.signpost("indexer")
        return out
