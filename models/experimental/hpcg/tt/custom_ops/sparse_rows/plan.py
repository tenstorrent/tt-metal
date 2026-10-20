# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Work plan of one sparse-row launch: row ownership per core, x window pieces, column offsets, neighbour exchange."""

import numpy as np

from models.experimental.hpcg.host import native
from models.experimental.hpcg.tt.layout import TILE

ALIGN = 32  # rows per ownership unit (128 B of fp32 values)
MAX_NB = 14  # 16 semaphores per core: 1 ready flag + 1 own progress slot + up to 14 neighbour slots
GRAN = 16  # window granule (words) = 64 B: every window piece is NoC-aligned in DRAM and in L1
WGAP = 8  # column runs at most this many granules apart share one window piece


class SparseRowPlan:
    """Ownership, x windows, offsets and exchange lists of one persistent sparse-row launch.
    A core's x window is a list of pieces (DRAM q range -> L1 byte offset): the granules its rows' columns touch
    (plus, GS, its own rows), merged into runs. With y-bands (reference.problem.Level bands > 1) each band of a colour
    block is split over nc / B cores, so a core's window holds its rows, one band-plane above / below, and short
    ghost lines from the neighbouring bands. A GS launch whose cores would have more than MAX_NB neighbours is
    planned on fewer cores."""

    def __new__(cls, COL, out_blocks, src_regions, ncores, noc_xy, exchange, bands=None, pout=None):
        """COL: [nslot, P] column indices (q order). out_blocks: (start, end) q ranges of the output rows, one per
        phase block (GS: one colour each). src_regions: q ranges of the input vector the columns fall in.
        exchange: True for Gauss-Seidel launches (cores push updated rows into each other's windows).
        bands: per out block, the q cuts of its y-bands (reference.problem.Level.bstarts), or None.
        pout: columns of COL in use (default all); COL's rows may be longer (row padding), and off then has
        COL's row length too, so it can be uploaded as is."""
        nc = ncores
        while True:
            p = object.__new__(cls)
            p._init(COL, out_blocks, src_regions, nc, noc_xy, exchange, bands=bands, pout=pout)
            if not exchange or max((len(x) for x in p.nbr), default=0) <= MAX_NB or nc == 1:
                return p
            nc = max(1, int(nc * 0.75))

    def __init__(self, *a, **k):
        pass

    def _init(self, COL, out_blocks, src_regions, ncores, noc_xy, exchange, bands=None, pout=None):
        nslot, stride = COL.shape
        pout = stride if pout is None else pout
        self.nslot, self.pout = nslot, pout
        NB = len(out_blocks)
        lens = [e - s for s, e in out_blocks]
        nc = max(1, min(ncores, max(-(-l // ALIGN) for l in lens)))
        if bands is not None and (len(bands) != NB or nc % (len(bands[0]) - 1) or nc < len(bands[0]) - 1):
            bands = None
        own = np.zeros((nc, NB, 2), np.int64)
        for b, (s, e) in enumerate(out_blocks):
            if bands is not None:  # band j of the block -> cores j*pb .. (j+1)*pb - 1, even ALIGN-row cuts
                cuts = list(bands[b])
                cuts[0], cuts[-1] = s, e
                B = len(cuts) - 1
                pb = nc // B
                k = 0
                for j in range(B):
                    bs, be = cuts[j], cuts[j + 1]
                    c = [bs] + [min(be, -(-(bs + i * (be - bs) // pb) // ALIGN) * ALIGN) for i in range(1, pb)] + [be]
                    for i in range(pb):
                        own[k, b] = (c[i], max(c[i], c[i + 1]))
                        k += 1
            else:  # equal contiguous ALIGN-row shares of the block
                per = -(-(e - s) // (nc * ALIGN)) * ALIGN
                for k in range(nc):
                    a = min(e, s + k * per)
                    own[k, b] = (a, min(e, a + per))
        ng = max(e for _, e in src_regions) // GRAN + 2  # granules of the input vector
        lib = native.lib()
        if lib is not None:
            pieces, off, own_l1, ents = self._plan_native(lib, COL, own, exchange, ng, src_regions[0][0])
        else:
            pieces, off, own_l1, ents = self._plan_numpy(COL, pout, own, exchange, ng, src_regions[0][0])
        # window CB bytes: a multiple of 512 (ZEROX clears it in 512-B pieces from the hardware zero region)
        self.win_b = -(-int(max(int(p[1].sum()) for p in pieces)) * 4 // 512) * 512
        self.win_words = [int(p[1].sum()) for p in pieces]
        self.R = max(len(p[0]) for p in pieces)
        self.off = off
        # exchange lists (GS): rows [s, f) of block b owned by k that sit in a window piece of m -> (m's NoC x, y,
        # s, f, L1 byte offset of s in m's window)
        exch = [[[] for _ in out_blocks] for _ in range(nc)]
        nbr = [set() for _ in range(nc)]
        for k, b, m, s, f, dst in ents:
            exch[k][b].append((noc_xy[m][0], noc_xy[m][1], s, f, dst))
            nbr[k].add(m)
            nbr[m].add(k)
        self.nc, self.own, self.own_l1, self.pieces, self.exch, self.nbr = nc, own, own_l1, pieces, exch, nbr
        self.out_blocks = out_blocks
        self._tables = {}  # _core_tables cache

    @staticmethod
    def _plan_native(lib, COL, own, exchange, ng, empty_lo):
        """Window pieces, offsets and exchange entries from host/native_setup.cpp (same results as _plan_numpy)."""
        nslot, stride = COL.shape
        nc, NB = own.shape[:2]
        COL = np.ascontiguousarray(COL, np.int32)
        own_f = np.ascontiguousarray(own.reshape(-1), np.int64)
        maxp = 4096
        plo, pln, pl1 = (np.zeros(nc * maxp, np.int64) for _ in range(3))
        npc = np.zeros(nc, np.int64)
        if (
            lib.hpcg_plan_windows(
                nslot, stride, COL, nc, NB, own_f, int(exchange), GRAN, WGAP, ng, empty_lo, maxp, plo, pln, pl1, npc
            )
            < 0
        ):
            raise ValueError(f"a core needs more than {maxp} window pieces")
        off = native.zeros((nslot, stride), np.int32)  # huge pages: written once, then uploaded
        own_l1 = np.zeros(nc * NB, np.int64)
        bad = lib.hpcg_plan_offsets(
            nslot, stride, COL, nc, NB, own_f, int(exchange), GRAN, ng, maxp, plo, pln, pl1, npc, stride, off, own_l1
        )
        assert bad == 0, "column outside its window piece, or own rows split over pieces"
        pieces = [
            (
                plo[k * maxp : k * maxp + npc[k]].copy(),
                pln[k * maxp : k * maxp + npc[k]].copy(),
                pl1[k * maxp : k * maxp + npc[k]].copy(),
            )
            for k in range(nc)
        ]
        ents = np.zeros((0, 6), np.int64)
        if exchange:
            n = lib.hpcg_plan_exchange(nc, NB, own_f, maxp, plo, pln, pl1, npc, None)
            ents = np.zeros((n, 6), np.int64)
            lib.hpcg_plan_exchange(nc, NB, own_f, maxp, plo, pln, pl1, npc, ents.ctypes.data)
        return pieces, off, own_l1.reshape(nc, NB), [tuple(int(v) for v in e) for e in ents]

    @staticmethod
    def _plan_numpy(COL, pout, own, exchange, ng, empty_lo):
        """The same plan with numpy: the readable definition, and the fallback without a C++ compiler."""
        nslot, stride = COL.shape
        nc, NB = own.shape[:2]
        mask = np.zeros(ng, bool)
        pieces = []
        for k in range(nc):
            mask[:] = False
            used = False
            for a, e in own[k]:
                if e > a:
                    mask[COL[:, a:e] // GRAN] = True
                    if exchange:  # GS: the window holds this core's own rows too
                        mask[a // GRAN : (e - 1) // GRAN + 1] = True
                    used = True
            if not used:  # cores without rows still get one (unused) piece
                mask[empty_lo // GRAN] = True
            g = np.flatnonzero(mask)
            brk = np.flatnonzero(np.diff(g) > WGAP)
            st, en = g[np.r_[0, brk + 1]], g[np.r_[brk, g.size - 1]] + 1
            lo, ln = st * GRAN, (en - st) * GRAN
            l1 = np.concatenate([[0], np.cumsum(ln)[:-1]]) * 4
            pieces.append((lo, ln, l1))
        off = np.zeros((nslot, stride), np.int32)
        own_l1 = np.zeros((nc, NB), np.int64)
        for k in range(nc):
            lo, ln, l1 = pieces[k]
            for b, (a, e) in enumerate(own[k]):
                if e <= a:
                    continue
                c = COL[:, a:e]
                pi = np.searchsorted(lo, c, side="right") - 1
                assert np.all(c < lo[pi] + ln[pi]), "column outside its window piece"
                off[:, a:e] = l1[pi] + (c - lo[pi]) * 4
                if exchange:
                    p0 = int(np.searchsorted(lo, a, side="right") - 1)
                    assert e <= lo[p0] + ln[p0], "own rows split over window pieces"
                    own_l1[k, b] = l1[p0] + (a - lo[p0]) * 4
        ents = []
        if exchange:
            for b in range(NB):
                A, E = own[:, b, 0], own[:, b, 1]
                for m in range(nc):
                    if not np.any(own[m, :, 1] > own[m, :, 0]):
                        continue  # a core without rows reads no window: its placeholder piece gets no updates
                    lo, ln, l1 = pieces[m]
                    for p in range(len(lo)):
                        plo, phi = int(lo[p]), int(lo[p] + ln[p])
                        for k in np.flatnonzero((A < phi) & (E > plo) & (E > A)):
                            if k == m:
                                continue
                            s, f = max(int(A[k]), plo), min(int(E[k]), phi)
                            ents.append((int(k), b, m, s, f, int(l1[p]) + (s - plo) * 4))
        return pieces, off, own_l1, ents

    def _core_tables(self, noc_xy, tpw):
        """The parts of each core's metadata row that do not depend on the launch's phases, computed once per tpw
        (launches sharing this plan reuse them): the head without its phase list, the body (segments, then exchange
        entries), each block's segment and exchange offsets relative to the body, and segments per block."""
        key = (tpw, tuple(map(tuple, noc_xy)))
        if key in self._tables:
            return self._tables[key]
        NB = len(self.out_blocks)
        slot = {k: sorted(self.nbr[k]) for k in range(self.nc)}
        out = []
        for k in range(self.nc):
            nbrs = slot[k]
            lo, ln, l1 = self.pieces[k]
            head = np.stack([lo, ln, l1], axis=1).reshape(-1).tolist()
            nb_words = []
            for m in nbrs:
                nb_words += [noc_xy[m][0], noc_xy[m][1], 1 + slot[m].index(k)]  # my slot on m
            body, blk, nsegs = [], [], []
            for b in range(NB):
                a, e = (int(v) for v in self.own[k, b])
                # segments never cross an L1 page (tpw words): cuts at the page boundaries inside [a, e)
                cuts = [a] + list(range((a // tpw + 1) * tpw, e, tpw)) + [e] if e > a else []
                blk.append([len(body), max(0, len(cuts) - 1)])
                nsegs.append(max(0, len(cuts) - 1))
                for s, f in zip(cuts[:-1], cuts[1:]):
                    body += [s, f]
            for b in range(NB):
                ex = self.exch[k][b]
                a, e = (int(v) for v in self.own[k, b])
                blk[b] += [len(body), len(ex), a, e, int(self.own_l1[k, b])]
                for x in ex:
                    body += list(x)
            out.append((len(lo), len(nbrs), head, nb_words, body, blk, nsegs))
        self._tables[key] = out
        return out

    def meta(self, phases, noc_xy, tpw=TILE):
        """Per-core metadata rows (layout in kernels/common.h) and segment counts for the compute kernel.
        phases: out block ids; 8 + b = an SpMV pass over block b (RESID launches)."""
        rows = []
        NB = len(self.out_blocks)
        phases = list(phases)
        nseg_core, nseg_gs_core, nseg_r_core = [], [], []
        for npieces, nnb, pieces, nb_words, body, blk, nsegs in self._core_tables(noc_xy, tpw):
            pos = 5 + len(pieces) + 7 * NB + len(nb_words) + len(phases) + 2  # body start in the row
            blocks = []
            for seg_at, nseg, ex_at, nex, a, e, own_l1 in blk:
                blocks += [pos + seg_at, nseg, pos + ex_at, nex, a, e, own_l1]
            head = [npieces, len(phases), nnb + 1, nnb, NB] + pieces + blocks + nb_words + phases + [0, 0]  # reserved
            rows.append(head + body)
            nseg_core.append(sum(nsegs[b & 7] for b in phases))
            nseg_gs_core.append(sum(nsegs[b] for b in phases if b < 8))
            nseg_r_core.append(sum(nsegs[0] for b in phases if b == 8))
        w = -(-max(len(r) for r in rows) // 16) * 16
        m = np.zeros((self.nc, w), np.int64)
        for k, r in enumerate(rows):
            m[k, : len(r)] = r
        return m, nseg_core, nseg_gs_core, nseg_r_core
