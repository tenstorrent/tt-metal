# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Work plan of one sparse-row launch: row ownership per core, x window pieces, column offsets, neighbour exchange."""

import numpy as np

from models.experimental.hpcg.host import native
from models.experimental.hpcg.tt.layout import TILE

ALIGN = 32  # rows per ownership unit (128 B of fp32 values)
MAX_PIECES = (256, 4096)  # window pieces per core the planner first makes room for, then retries with
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

    def __new__(cls, COL, out_blocks, src_regions, ncores, noc_xy, exchange, bands=None, pout=None, stencil=None):
        """COL: [nslot, P] column indices (q order). out_blocks: (start, end) q ranges of the output rows, one per
        phase block (GS: one colour each). src_regions: q ranges of the input vector the columns fall in.
        exchange: True for Gauss-Seidel launches (cores push updated rows into each other's windows).
        bands: per out block, the q cuts of its y-bands (reference.problem.Level.bstarts), or None.
        pout: columns of COL in use (default all); COL's rows may be longer (row padding), and off then has
        COL's row length too, so it can be uploaded as is.
        stencil: a reference.problem.Level whose 27-point matrix the launch uses, with COL = None: the windows are
        planned from the level's storage-order formula (no column table) and off is None, because the device
        generates the offsets (tt/custom_ops/stencil_matrix). Gauss-Seidel plans only (exchange=True)."""
        nc = ncores
        while True:
            p = object.__new__(cls)
            p._init(COL, out_blocks, src_regions, nc, noc_xy, exchange, bands=bands, pout=pout, stencil=stencil)
            if not exchange or max((len(x) for x in p.nbr), default=0) <= MAX_NB or nc == 1:
                return p
            nc = max(1, int(nc * 0.75))

    def __init__(self, *a, **k):
        pass

    def _init(self, COL, out_blocks, src_regions, ncores, noc_xy, exchange, bands=None, pout=None, stencil=None):
        nslot, stride = COL.shape if stencil is None else (27, stencil.stride)
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
        if stencil is not None:
            assert exchange and lib is not None
            pieces, off, own_l1, ents = self._plan_stencil(lib, stencil, own, ng, src_regions[0][0])
        elif lib is not None:
            pieces, off, own_l1, ents = self._plan_native(lib, COL, own, exchange, ng, src_regions[0][0])
        else:
            pieces, off, own_l1, ents = self._plan_numpy(COL, pout, own, exchange, ng, src_regions[0][0])
        # window CB bytes: a multiple of 512 (ZEROX clears it in 512-B pieces from the hardware zero region)
        self.win_b = -(-int(max(int(p[1].sum()) for p in pieces)) * 4 // 512) * 512
        self.win_words = [int(p[1].sum()) for p in pieces]
        self.R = max(len(p[0]) for p in pieces)
        self.off = off
        # exchange lists (GS): rows [s, f) of block b owned by k that sit in a window piece of m -> (m's NoC x, y,
        # s, f, L1 byte offset of s in m's window), grouped by (k, b) in their original order
        E = np.asarray(ents, np.int64).reshape(-1, 6)
        key = E[:, 0] * NB + E[:, 1]
        order = np.argsort(key, kind="stable")
        E, key = E[order], key[order]
        nxy = np.asarray(noc_xy, np.int64).reshape(-1, 2)
        self._exch_rows = np.column_stack([nxy[E[:, 2], 0], nxy[E[:, 2], 1], E[:, 3], E[:, 4], E[:, 5]])
        self._exch_at = np.searchsorted(key, np.arange(nc * NB + 1))
        nbr = [set() for _ in range(nc)]
        for p in np.unique(np.concatenate([E[:, 0] * nc + E[:, 2], E[:, 2] * nc + E[:, 0]])).tolist():
            nbr[p // nc].add(p % nc)
        self.nc, self.own, self.own_l1, self.pieces, self.nbr = nc, own, own_l1, pieces, nbr
        self.out_blocks = out_blocks
        self._tables = {}  # _core_tables cache

    def exch_rows(self, k, b):
        """Exchange entries of core k, block b: an [n, 5] array of (NoC x, NoC y, s, f, L1 byte offset)."""
        NB = len(self.out_blocks)
        return self._exch_rows[self._exch_at[k * NB + b] : self._exch_at[k * NB + b + 1]]

    @property
    def exch(self):
        """The exchange entries as exch[k][b] = [(NoC x, NoC y, s, f, L1 byte offset), ...]."""
        return [
            [[tuple(r) for r in self.exch_rows(k, b).tolist()] for b in range(len(self.out_blocks))]
            for k in range(self.nc)
        ]

    @staticmethod
    def _plan_stencil(lib, L, own, ng, empty_lo):
        """Window pieces from the storage-order formula of level L (host/native_setup.cpp), the own rows' window
        offsets and the exchange entries; no offsets table (the device generates it)."""
        nc, NB = own.shape[:2]
        own_f = np.ascontiguousarray(own.reshape(-1), np.int64)
        for maxp in MAX_PIECES:
            plo, pln, pl1 = (np.zeros(nc * maxp, np.int64) for _ in range(3))
            npc = np.zeros(nc, np.int64)
            args = (*L.geo(), nc, NB, own_f, GRAN, WGAP, ng, empty_lo, maxp, plo, pln, pl1, npc)
            if lib.hpcg_plan_windows_stencil(*args) >= 0:
                break
        else:
            raise ValueError(f"a core needs more than {maxp} window pieces")
        pieces = [
            (
                plo[k * maxp : k * maxp + npc[k]].copy(),
                pln[k * maxp : k * maxp + npc[k]].copy(),
                pl1[k * maxp : k * maxp + npc[k]].copy(),
            )
            for k in range(nc)
        ]
        own_l1 = np.zeros((nc, NB), np.int64)
        for k in range(nc):  # the window piece holding each block's own rows
            lo, ln, l1 = pieces[k]
            a, e = own[k, :, 0], own[k, :, 1]
            p0 = np.searchsorted(lo, a, side="right") - 1
            has = e > a
            assert np.all(e[has] <= lo[p0[has]] + ln[p0[has]]), "own rows split over window pieces"
            own_l1[k, has] = l1[p0[has]] + (a[has] - lo[p0[has]]) * 4
        n = lib.hpcg_plan_exchange(nc, NB, own_f, maxp, plo, pln, pl1, npc, None)
        ents = np.zeros((n, 6), np.int64)
        lib.hpcg_plan_exchange(nc, NB, own_f, maxp, plo, pln, pl1, npc, ents.ctypes.data)
        return pieces, None, own_l1, ents

    @staticmethod
    def _plan_native(lib, COL, own, exchange, ng, empty_lo):
        """Window pieces, offsets and exchange entries from host/native_setup.cpp (same results as _plan_numpy)."""
        nslot, stride = COL.shape
        nc, NB = own.shape[:2]
        COL = np.ascontiguousarray(COL, np.int32)
        own_f = np.ascontiguousarray(own.reshape(-1), np.int64)
        for maxp in MAX_PIECES:
            plo, pln, pl1 = (np.zeros(nc * maxp, np.int64) for _ in range(3))
            npc = np.zeros(nc, np.int64)
            args = (nslot, stride, COL, nc, NB, own_f, int(exchange), GRAN, WGAP, ng, empty_lo, maxp, plo, pln, pl1, npc)
            if lib.hpcg_plan_windows(*args) >= 0:
                break
        else:
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
        return pieces, off, own_l1.reshape(nc, NB), ents

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
        (launches sharing this plan reuse them): window pieces, neighbour words, the body (segments, then exchange
        entries), each block's [segment offset, segments, exchange offset, entries, own a, own e, own L1 offset] with
        offsets relative to the body, and segments per block. Plain lists: the rows are short, so per-row numpy calls
        cost more than they save."""
        key = (tpw, tuple(map(tuple, noc_xy)))
        if key in self._tables:
            return self._tables[key]
        NB = len(self.out_blocks)
        slot = {k: sorted(self.nbr[k]) for k in range(self.nc)}
        own, own_l1 = self.own.tolist(), self.own_l1.tolist()
        out = []
        for k in range(self.nc):
            lo, ln, l1 = self.pieces[k]
            pieces = np.stack([lo, ln, l1], axis=1).reshape(-1).tolist()
            nb_words = [w for m in slot[k] for w in (noc_xy[m][0], noc_xy[m][1], 1 + slot[m].index(k))]  # my slot on m
            body, blk, nsegs = [], [], []
            for b in range(NB):
                a, e = own[k][b]
                # segments never cross an L1 page (tpw words): cuts at the page boundaries inside [a, e)
                cuts = [a] + list(range((a // tpw + 1) * tpw, e, tpw)) + [e] if e > a else [a]
                blk.append([len(body), len(cuts) - 1])
                nsegs.append(len(cuts) - 1)
                for i in range(len(cuts) - 1):
                    body += (cuts[i], cuts[i + 1])
            for b in range(NB):
                ex = self.exch_rows(k, b)
                blk[b] += [len(body), len(ex), own[k][b][0], own[k][b][1], own_l1[k][b]]
                body += ex.reshape(-1).tolist()
            out.append((len(lo), len(slot[k]), pieces, nb_words, body, blk, nsegs))
        self._tables[key] = out
        return out

    def meta(self, phases, noc_xy, tpw=TILE):
        """Per-core metadata rows (layout in kernels/common.h) and segment counts for the compute kernel.
        phases: out block ids; 8 + b = an SpMV pass over block b (RESID launches). Built by the compiled helper
        (host/native_setup.cpp's hpcg_plan_meta); _meta_python is the readable definition and the fallback."""
        lib = native.lib()
        if lib is None:
            return self._meta_python(phases, noc_xy, tpw)
        key = ("native", tuple(map(tuple, noc_xy)))
        if key not in self._tables:
            NB = len(self.out_blocks)
            pat = np.cumsum([0] + [len(p[0]) for p in self.pieces]).astype(np.int64)
            plo, pln, pl1 = (np.concatenate([p[i] for p in self.pieces]).astype(np.int64) for i in range(3))
            slot = [sorted(self.nbr[k]) for k in range(self.nc)]
            nat = np.cumsum([0] + [len(s) for s in slot]).astype(np.int64)
            nbr = np.asarray([m for s in slot for m in s], np.int64)
            own = np.ascontiguousarray(self.own.reshape(-1), np.int64)
            own_l1 = np.ascontiguousarray(self.own_l1.reshape(-1), np.int64)
            xat = np.ascontiguousarray(self._exch_at, np.int64)
            xrows = np.ascontiguousarray(self._exch_rows.reshape(-1), np.int64)
            noc = np.ascontiguousarray(np.asarray(noc_xy, np.int64).reshape(-1))
            self._tables[key] = (NB, [pat, plo, pln, pl1, nat, nbr, noc, own, own_l1, xat, xrows])
        NB, args = self._tables[key]
        ph = np.asarray(list(phases), np.int64)
        counts = [np.zeros(self.nc, np.int64) for _ in range(3)]
        longest = lib.hpcg_plan_meta(self.nc, NB, ph.size, ph, *args, tpw, 0, None, *counts)
        m = np.zeros((self.nc, -(-longest // 16) * 16), np.int64)
        lib.hpcg_plan_meta(self.nc, NB, ph.size, ph, *args, tpw, m.shape[1], m.ctypes.data, *counts)
        return (m,) + tuple(c.tolist() for c in counts)

    def _meta_python(self, phases, noc_xy, tpw=TILE):
        """meta() in Python: the readable definition, and the fallback without a C++ compiler."""
        NB = len(self.out_blocks)
        phases = [int(p) for p in phases]
        tables = self._core_tables(noc_xy, tpw)
        rows = []
        for npieces, nnb, pieces, nb_words, body, blk, _ in tables:
            pos = 5 + len(pieces) + 7 * NB + len(nb_words) + len(phases) + 2  # body start in the row
            blocks = [v for sa, ns, xa, nx_, a, e, o in blk for v in (pos + sa, ns, pos + xa, nx_, a, e, o)]
            head = [npieces, len(phases), nnb + 1, nnb, NB] + pieces + blocks + nb_words + phases + [0, 0]  # reserved
            rows.append(head + body)
        nsegs = np.asarray([t[6] for t in tables], np.int64)  # [core, block]
        ph = np.asarray(phases, np.int64)
        nseg_core = nsegs[:, ph & 7].sum(axis=1).tolist()
        nseg_gs_core = nsegs[:, ph[ph < 8]].sum(axis=1).tolist()
        nseg_r_core = (nsegs[:, 0] * int((ph == 8).sum())).tolist()
        w = -(-max(len(r) for r in rows) // 16) * 16
        m = np.zeros((self.nc, w), np.int64)
        for k, r in enumerate(rows):
            m[k, : len(r)] = r
        return m, nseg_core, nseg_gs_core, nseg_r_core
