# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Work plan of one sparse-row launch: row ownership per core, x window pieces, column offsets, neighbour exchange."""

import numpy as np

from models.experimental.hpcg.host import native
from models.experimental.hpcg.tt import constants as C
from models.experimental.hpcg.tt.layout import TILE, u32_rows

# shared with xhpcg/tt/plan.cpp (tt/shared_constants.h, where each is described)
ALIGN, MAX_PIECES, MAX_NB, GRAN, WGAP = C.ALIGN, C.MAX_PIECES, C.MAX_NB, C.GRAN, C.WGAP


_NOC = {}


def _noc_array(noc_xy):
    """noc_xy (the device's cached list of NoC (x, y)) as a flat int64 array, made once per list."""
    if id(noc_xy) not in _NOC:
        _NOC[id(noc_xy)] = (noc_xy, np.ascontiguousarray(np.asarray(noc_xy, np.int64).reshape(-1)))
    return _NOC[id(noc_xy)][1]


_SCRATCH = [np.zeros(0, np.int32)]


def _scratch(n):
    """At least n int32 words of a host buffer reused by every meta_tensor call (grown as needed, huge pages when
    large): TT-NN copies the rows when it uploads them, so the buffer is free again once u32_rows returns. n = 0:
    the buffer as it is (starting at 1 MB)."""
    if _SCRATCH[0].size < max(n, 1 << 18):
        _SCRATCH[0] = native.zeros(max(n, 2 * _SCRATCH[0].size, 1 << 18), np.int32)
    return _SCRATCH[0] if n == 0 else _SCRATCH[0][:n]


def _no_neighbours(nc):
    """Neighbour lists (sorted index array, per-core starts) of a launch without exchange: none."""
    return np.zeros(0, np.int64), np.zeros(nc + 1, np.int64)


_PIECES = [0, ()]


def _piece_buffers(n):
    """Three int64 buffers of n words for a compiled planner's window pieces (lo, words, L1 offset; max_pieces per
    core), reused by every plan: _compact copies out the used part, so no fresh memory is faulted in per plan."""
    if _PIECES[0] < n:
        _PIECES[:] = [n, tuple(np.empty(n, np.int64) for _ in range(3))]
    return tuple(b[:n] for b in _PIECES[1])


def _compact(nc, maxp, bufs, npc):
    """[nc, maxp] piece buffers -> [nc, R] copies (R = the most pieces of a core), zero past each core's count."""
    R = max(1, int(npc.max()))
    out = [b.reshape(nc, maxp)[:, :R].copy() for b in bufs]
    unused = np.arange(R)[None, :] >= npc[:, None]
    for o in out:
        o[unused] = 0
    return out


class SparseRowPlan:
    """Ownership, x windows, offsets and exchange lists of one persistent sparse-row launch.
    A core's x window is a list of pieces (DRAM q range -> L1 byte offset): the granules its rows' columns touch
    (plus, GS, its own rows), merged into runs. With y-bands (reference.problem.Level bands > 1) each band of a colour
    block is split over nc / B cores, so a core's window holds its rows, one band-plane above / below, and short
    ghost lines from the neighbouring bands. A GS launch whose cores would have more than MAX_NB neighbours is
    planned on fewer cores."""

    def __new__(
        cls,
        COL,
        out_blocks,
        src_regions,
        ncores,
        noc_xy,
        exchange,
        bands=None,
        pout=None,
        stencil=None,
        transfer=None,
        host_tables=True,
        defer_exchange=False,
    ):
        """COL: [nslot, P] column indices (q order). out_blocks: (start, end) q ranges of the output rows, one per
        phase block (GS: one colour each). src_regions: q ranges of the input vector the columns fall in.
        exchange: True for Gauss-Seidel launches (cores push updated rows into each other's windows).
        bands: per out block, the q cuts of its y-bands (reference.problem.Level.bstarts), or None.
        pout: columns of COL in use (default all); COL's rows may be longer (row padding), and off then has
        COL's row length too, so it can be uploaded as is.
        stencil: a reference.problem.Level whose 27-point matrix the launch uses, with COL = None: the windows are
        planned from the level's storage-order formula (no column table) and off is None, because the device
        generates the offsets (tt/custom_ops/stencil_matrix). Gauss-Seidel plans only (exchange=True).
        transfer: (kind, F, C) with COL = None: a grid transfer launch between fine level F and coarse level C (kind
        "restrict" or "prolong"), planned from their storage-order formulas without Level.link's tables; off holds the
        1-slot window offsets and weights the 1-slot values (exchange=False, pout = the out rows' vector length).
        host_tables=False: off and weights are None (the device generates them, tt/custom_ops/stencil_matrix).
        defer_exchange (stencil plans): leave the exchange lists for complete_exchange(), so the windows are ready
        sooner (matrix generation needs only them); the neighbour limit is then checked there."""
        nc = ncores
        while True:
            p = object.__new__(cls)
            p._init(
                COL,
                out_blocks,
                src_regions,
                nc,
                noc_xy,
                exchange,
                bands,
                pout,
                stencil,
                transfer,
                host_tables,
                defer_exchange,
            )
            if not exchange or defer_exchange or int(np.diff(p._nbr_csr[1]).max(initial=0)) <= MAX_NB or p.nc == 1:
                return p
            nc = max(1, int(p.nc * 0.75))  # from the cores used: nc may be above _init's block-size clamp

    def __init__(self, *a, **k):
        pass

    def _init(
        self,
        COL,
        out_blocks,
        src_regions,
        ncores,
        noc_xy,
        exchange,
        bands=None,
        pout=None,
        stencil=None,
        transfer=None,
        host_tables=True,
        defer_exchange=False,
    ):
        if stencil is not None:
            nslot, stride = 27, stencil.stride
        elif transfer is not None:
            nslot, stride = 1, pout
        else:
            nslot, stride = COL.shape
        pout = stride if pout is None else pout
        self.nslot, self.pout = nslot, pout
        NB = len(out_blocks)
        lens = [e - s for s, e in out_blocks]
        nc = max(1, min(ncores, max(-(-l // ALIGN) for l in lens)))
        # bands only when the cores divide evenly over each block's bands (blocks can differ: on an odd nx the even-y
        # colours have one band more than the odd-y ones)
        if bands is not None and (len(bands) != NB or any(nc % (len(b) - 1) or nc < len(b) - 1 for b in bands)):
            bands = None
        lib = native.lib()
        own = self._ownership(lib, out_blocks, bands, nc)
        ng = max(e for _, e in src_regions) // GRAN + 2  # granules of the input vector
        self.weights, self.transfer = None, transfer
        if transfer is not None:
            assert not exchange and lib is not None
            pieces, off, own_l1, ents, self.weights = self._plan_transfer(
                lib, transfer, own, ng, src_regions[0][0], stride, host_tables
            )
        elif stencil is not None:
            assert exchange and lib is not None
            pieces, off, own_l1, ents = self._plan_stencil(
                lib, stencil, own, ng, src_regions[0][0], None if defer_exchange else noc_xy
            )
            self._noc_xy = noc_xy
        elif lib is not None:
            pieces, off, own_l1, ents = self._plan_native(lib, COL, own, exchange, ng, src_regions[0][0], noc_xy)
        else:
            pieces, off, own_l1, ents = self._plan_numpy(COL, pout, own, exchange, ng, src_regions[0][0])
            ents = self._group_exchange(ents, nc, NB, noc_xy)
        plo, pln, pl1, npc = pieces  # [nc, maxp] each (zero past npc[k]), and pieces per core
        self.piece_arrays = (plo, pln, pl1, npc)
        words = pln.sum(axis=1)
        # window CB bytes: a multiple of 512 (ZEROX clears it in 512-B pieces from the hardware zero region)
        self.win_b = -(-int(words.max()) * 4 // 512) * 512
        self.win_words = words.tolist()
        self.R = int(npc.max())
        self.off = off
        # exchange lists (GS): rows [s, f) of block b owned by k that sit in a window piece of m -> (m's NoC x, y,
        # s, f, L1 byte offset of s in m's window), grouped by (k, b) in their original order, and each core's
        # neighbours (sorted)
        self._exch_rows, self._exch_at, self._nbr_csr = ents  # neighbours of core k: nbr[nat[k] .. nat[k + 1])
        self.nc, self.own, self.own_l1 = nc, own, own_l1
        self.out_blocks = out_blocks
        self._tables = {}  # _core_tables cache

    @staticmethod
    def _ownership(lib, out_blocks, bands, nc):
        """[nc, NB, 2] row ranges (a, e) of each core in each out block (_ownership_numpy is the definition; the
        compiled helper computes the same in one call instead of about ten small numpy calls per block)."""
        NB = len(out_blocks)
        if lib is None:
            return SparseRowPlan._ownership_numpy(out_blocks, bands, nc)
        s = np.array([a for a, _ in out_blocks], np.int64)
        e = np.array([b for _, b in out_blocks], np.int64)
        cuts = np.array([v for b in bands for v in b] if bands is not None else [], np.int64)
        cat = np.cumsum([0] + [len(b) for b in bands] if bands is not None else [0] * (NB + 1)).astype(np.int64)
        own = np.empty((nc, NB, 2), np.int64)
        lib.hpcg_ownership(NB, s, e, cuts, cat, nc, ALIGN, own)
        return own

    @staticmethod
    def _ownership_numpy(out_blocks, bands, nc):
        NB = len(out_blocks)
        own = np.zeros((nc, NB, 2), np.int64)
        for b, (s, e) in enumerate(out_blocks):
            if bands is not None:  # band j of the block -> cores j*pb .. (j+1)*pb - 1, even ALIGN-row cuts
                cuts = np.array(bands[b], np.int64)
                cuts[0], cuts[-1] = s, e
                B = len(cuts) - 1
                pb = nc // B
                bs, be = cuts[:-1, None], cuts[1:, None]
                inner = np.minimum(be, -(-(bs + np.arange(1, pb)[None, :] * (be - bs) // pb) // ALIGN) * ALIGN)
                c = np.concatenate([bs, inner, be], axis=1)  # band j's cuts: c[j, i] .. c[j, i + 1] -> core j*pb + i
                own[:, b, 0] = c[:, :-1].reshape(-1)
                own[:, b, 1] = np.maximum(c[:, :-1], c[:, 1:]).reshape(-1)
            else:  # equal contiguous ALIGN-row shares of the block
                per = -(-(e - s) // (nc * ALIGN)) * ALIGN
                a = np.minimum(e, s + np.arange(nc) * per)
                own[:, b, 0], own[:, b, 1] = a, np.minimum(e, a + per)
        return own

    @property
    def nbr(self):
        """Each core's neighbours (the cores it pushes rows to or receives rows from), sorted, as lists."""
        nbr, nat = self._nbr_csr
        return [nbr[nat[k] : nat[k + 1]].tolist() for k in range(self.nc)]

    @property
    def pieces(self):
        """Each core's window pieces as (lo, words, L1 byte offset) arrays (views of piece_arrays)."""
        plo, pln, pl1, npc = self.piece_arrays
        return [(plo[k, : npc[k]], pln[k, : npc[k]], pl1[k, : npc[k]]) for k in range(self.nc)]

    @staticmethod
    def _group_exchange(ents, nc, NB, noc_xy):
        """Plain (k, b, m, s, f, dst) exchange entries -> the grouped form the compiled planner returns: rows of
        (m's NoC x, y, s, f, dst) grouped by (k, b) in their original order, the group starts, and each core's
        neighbours (the cores it pushes rows to or receives rows from), sorted."""
        E = np.asarray(ents, np.int64).reshape(-1, 6)
        key = E[:, 0] * NB + E[:, 1]
        order = np.argsort(key, kind="stable")
        E, key = E[order], key[order]
        nxy = np.asarray(noc_xy, np.int64).reshape(-1, 2)
        rows = np.column_stack([nxy[E[:, 2], 0], nxy[E[:, 2], 1], E[:, 3], E[:, 4], E[:, 5]])
        at = np.searchsorted(key, np.arange(nc * NB + 1))
        pairs = np.unique(np.concatenate([E[:, 0] * nc + E[:, 2], E[:, 2] * nc + E[:, 0]]))
        return rows, at, (pairs % nc, np.searchsorted(pairs // nc, np.arange(nc + 1)))

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
    def _plan_stencil(lib, L, own, ng, empty_lo, noc_xy):
        """Window pieces from the storage-order formula of level L (host/native_setup.cpp), the own rows' window
        offsets and the exchange entries; no offsets table (the device generates it)."""
        nc, NB = own.shape[:2]
        own_f = np.ascontiguousarray(own.reshape(-1), np.int64)
        own_l1 = np.zeros((nc, NB), np.int64)  # L1 byte offset of each block's own rows, from the compiled planner
        for maxp in MAX_PIECES:
            plo, pln, pl1 = _piece_buffers(nc * maxp)
            npc = np.zeros(nc, np.int64)
            args = (*L.geo(), nc, NB, own_f, GRAN, WGAP, ng, empty_lo, maxp, plo, pln, pl1, npc, own_l1)
            if lib.hpcg_plan_windows_stencil(*args) >= 0:
                break
        else:
            raise ValueError(f"a core needs more than {maxp} window pieces")
        ents = (np.zeros((0, 5), np.int64), np.zeros(nc * NB + 1, np.int64), _no_neighbours(nc))
        if noc_xy is not None:  # None: deferred (complete_exchange)
            ents = SparseRowPlan._exchange_native(lib, nc, NB, own_f, maxp, plo, pln, pl1, npc, noc_xy)
        return (*_compact(nc, maxp, (plo, pln, pl1), npc), npc), None, own_l1, ents

    def complete_exchange(self):
        """The exchange lists of a plan made with defer_exchange=True; False when a core then has more than MAX_NB
        neighbours (the caller re-plans without deferring, which retries on fewer cores)."""
        nc, NB = self.own.shape[:2]
        own_f = np.ascontiguousarray(self.own.reshape(-1), np.int64)
        plo, pln, pl1, npc = (np.ascontiguousarray(x, np.int64) for x in self.piece_arrays)
        lib = native.lib()
        ents = SparseRowPlan._exchange_native(lib, nc, NB, own_f, plo.shape[1], plo, pln, pl1, npc, self._noc_xy)
        self._exch_rows, self._exch_at, self._nbr_csr = ents
        self._tables = {}
        return int(np.diff(self._nbr_csr[1]).max(initial=0)) <= MAX_NB

    @staticmethod
    def _plan_transfer(lib, transfer, own, ng, empty_lo, pout, host_tables=True):
        """Window pieces, 1-slot offsets and weights of a grid transfer launch, from the storage-order formulas of its
        two levels (host/native_setup.cpp's hpcg_plan_windows_transfer / hpcg_transfer_tables). host_tables=False:
        no offsets or weights (None), the device generates them."""
        kind, F, C = transfer
        k = {"restrict": 1, "prolong": 2}[kind]
        nc, NB = own.shape[:2]
        own_f = np.ascontiguousarray(own.reshape(-1), np.int64)
        for maxp in MAX_PIECES:
            plo, pln, pl1 = _piece_buffers(nc * maxp)
            npc = np.zeros(nc, np.int64)
            args = (k, *F.geo(), *C.geo(), nc, NB, own_f, GRAN, WGAP, ng, empty_lo, maxp, plo, pln, pl1, npc)
            if lib.hpcg_plan_windows_transfer(*args) >= 0:
                break
        else:
            raise ValueError(f"a core needs more than {maxp} window pieces")
        off = w = None
        if host_tables:
            off = native.zeros((1, pout), np.int32)
            w = native.zeros((1, pout), np.float32)
            bad = lib.hpcg_transfer_tables(k, *F.geo(), *C.geo(), nc, NB, own_f, maxp, plo, pln, pl1, npc, off, w)
            assert bad == 0, "column outside its window piece"
        pieces = (*_compact(nc, maxp, (plo, pln, pl1), npc), npc)
        ents = (np.zeros((0, 5), np.int64), np.zeros(nc * NB + 1, np.int64), _no_neighbours(nc))
        return pieces, off, np.zeros((nc, NB), np.int64), ents, w

    @staticmethod
    def _exchange_native(lib, nc, NB, own_f, maxp, plo, pln, pl1, npc, noc_xy):
        """Exchange entries grouped by (core, block), and each core's sorted neighbours (host/native_setup.cpp)."""
        noc = _noc_array(noc_xy)
        cap = 64 * nc * NB
        while True:
            rows, at = np.empty((cap, 5), np.int64), np.empty(nc * NB + 1, np.int64)
            nbr, nat = np.empty(nc * nc, np.int64), np.empty(nc + 1, np.int64)
            n = lib.hpcg_plan_exchange_grouped(nc, NB, own_f, maxp, plo, pln, pl1, npc, noc, cap, rows, at, nbr, nat)
            if n >= 0:
                return rows[:n], at, (nbr[: nat[nc]], nat)
            cap = -n

    @staticmethod
    def _plan_native(lib, COL, own, exchange, ng, empty_lo, noc_xy=()):
        """Window pieces, offsets and exchange entries from host/native_setup.cpp (same results as _plan_numpy)."""
        nslot, stride = COL.shape
        nc, NB = own.shape[:2]
        COL = np.ascontiguousarray(COL, np.int32)
        own_f = np.ascontiguousarray(own.reshape(-1), np.int64)
        for maxp in MAX_PIECES:
            plo, pln, pl1 = (np.zeros(nc * maxp, np.int64) for _ in range(3))
            npc = np.zeros(nc, np.int64)
            args = (
                nslot,
                stride,
                COL,
                nc,
                NB,
                own_f,
                int(exchange),
                GRAN,
                WGAP,
                ng,
                empty_lo,
                maxp,
                plo,
                pln,
                pl1,
                npc,
            )
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
        pieces = tuple(x.reshape(nc, maxp) for x in (plo, pln, pl1)) + (npc,)
        ents = (np.zeros((0, 5), np.int64), np.zeros(nc * NB + 1, np.int64), _no_neighbours(nc))
        if exchange:
            ents = SparseRowPlan._exchange_native(lib, nc, NB, own_f, maxp, plo, pln, pl1, npc, noc_xy)
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
        npc = np.asarray([len(p[0]) for p in pieces], np.int64)
        flat = [np.zeros((nc, int(npc.max())), np.int64) for _ in range(3)]
        for k, p in enumerate(pieces):
            for i in range(3):
                flat[i][k, : npc[k]] = p[i]
        return (*flat, npc), off, own_l1, ents

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
        out, views = [], self.pieces
        for k in range(self.nc):
            lo, ln, l1 = views[k]
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

    def meta(self, phases, noc_xy, tpw=TILE, align=16):
        """Per-core metadata rows (layout in kernels/common.h), segment counts for the compute kernel, and the
        longest row's length in words. phases: out block ids; 8 + b = an SpMV pass over block b (RESID launches).
        align: the rows' width is a multiple of it (TILE: whole 4 KB pages). Built by the compiled helper
        (host/native_setup.cpp's hpcg_plan_meta) as the 32-bit table that is uploaded; _meta_python is the readable
        definition and the fallback."""
        lib = native.lib()
        if lib is None:
            return self._meta_python(phases, noc_xy, tpw, align)
        done = ("rows", id(noc_xy), tuple(phases), tpw, align)
        if done not in self._tables:  # launches with the same phases (CG start and CG SpMV) share the rows
            self._tables[done] = self._meta_native(lib, phases, noc_xy, tpw, align, None)
        return self._tables[done]

    def _native_args(self, key, noc_xy):
        """The plan's arrays for hpcg_plan_meta, as raw pointers (kept alive with them in self._tables[key])."""
        NB = len(self.out_blocks)
        plo, pln, pl1, npc = (np.ascontiguousarray(x, np.int64) for x in self.piece_arrays)  # [nc, maxp], counts
        nbr, nat = (np.ascontiguousarray(x, np.int64) for x in self._nbr_csr)  # sorted per core
        own = np.ascontiguousarray(self.own.reshape(-1), np.int64)
        own_l1 = np.ascontiguousarray(self.own_l1.reshape(-1), np.int64)
        xat = np.ascontiguousarray(self._exch_at, np.int64)
        xrows = np.ascontiguousarray(self._exch_rows.reshape(-1), np.int64)
        arrays = [plo, pln, pl1, npc, nat, nbr, _noc_array(noc_xy), own, own_l1, xat, xrows]
        # arrays kept alive with their pointers; max_pieces: the piece arrays' row stride
        self._tables[key] = (NB, arrays, [plo.shape[1]] + [a.ctypes.data for a in arrays])

    def segment_counts(self, phases, tpw=TILE):
        """Per-core segment counts of a launch running these phases (what meta() returns with its rows): all
        segments, Gauss-Seidel ones (phases < 8), and RESTF ones (phase 8, block 0). Segments never cross an L1
        page of tpw words."""
        a, e = self.own[:, :, 0], self.own[:, :, 1]
        segs = np.where(e > a, (e - 1) // tpw - a // tpw + 1, 0)  # [core, block]
        ph = np.asarray(list(phases), np.int64)
        nseg = segs[:, ph & 7].sum(axis=1)
        nseg_gs = segs[:, ph[ph < 8]].sum(axis=1)
        nseg_r = segs[:, 0] * int((ph == 8).sum())
        return nseg.tolist(), nseg_gs.tolist(), nseg_r.tolist()

    def meta_tensor(self, device, phases, noc_xy, tpw=TILE):
        """meta(..., align=16) uploaded as a ROW_MAJOR [nc, w] device tensor (u32_rows): one DRAM page per core, of
        the longest row rounded up to 64 bytes (rows of whole 4 KB pages uploaded 1.1 to 24 times the bytes: 0.47 ms
        of uploads per HPCG() at 104^3, 43 to 433 words used of 1024). Returns the rest of meta's result too:
        (tensor, nseg, nseg_gs, nseg_r, longest). The rows are built in a reused host buffer (TT-NN copies them on
        upload), so no fresh memory is faulted in per launch; launches with the same phases (CG start and CG SpMV)
        share one upload."""
        lib = native.lib()
        key = ("tensor", id(device), id(noc_xy), tuple(phases), tpw)
        if key not in self._tables:
            if lib is None:
                m, *rest = self._meta_python(phases, noc_xy, tpw, 16)
                self._tables[key] = (u32_rows(device, m), *rest)
            else:
                m, *rest = self._meta_native(lib, phases, noc_xy, tpw, 16, _scratch)
                self._tables[key] = (u32_rows(device, m), *rest)
        return self._tables[key]

    def _meta_native(self, lib, phases, noc_xy, tpw, align, scratch):
        """meta() by the compiled helper; scratch(n) returns n reusable int32 words for the rows (None: new memory)."""
        key = ("native", id(noc_xy))  # noc_xy: the device's cached list (sparse_rows.op.noc_coords)
        if key not in self._tables:
            self._native_args(key, noc_xy)
        NB, _, args = self._tables[key]
        ph = np.asarray(list(phases), np.int64)
        counts = np.zeros((3, self.nc), np.int64)
        cp = [counts.ctypes.data + 8 * self.nc * i for i in range(3)]
        if scratch is None:
            longest = lib.hpcg_plan_meta(self.nc, NB, ph.size, ph.ctypes.data, *args, tpw, 0, None, *cp)
            m = np.empty((self.nc, -(-longest // align) * align), np.int32)
            lib.hpcg_plan_meta(self.nc, NB, ph.size, ph.ctypes.data, *args, tpw, m.shape[1], m.ctypes.data, *cp)
            return (m,) + tuple(counts.tolist()) + (int(longest),)
        buf = scratch(0)  # the reused buffer as it is; one call sizes and writes the rows when they fit
        while True:
            longest = lib.hpcg_plan_meta_into(
                self.nc, NB, ph.size, ph.ctypes.data, *args, tpw, align, buf.size, buf.ctypes.data, *cp
            )
            if longest >= 0:
                break
            buf = scratch(-longest)
        w = -(-longest // align) * align
        return (buf[: self.nc * w].reshape(self.nc, w),) + tuple(counts.tolist()) + (int(longest),)

    def _meta_python(self, phases, noc_xy, tpw=TILE, align=16):
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
        longest = max(len(r) for r in rows)
        m = np.zeros((self.nc, -(-longest // align) * align), np.int64)
        for k, r in enumerate(rows):
            m[k, : len(r)] = r
        return m, nseg_core, nseg_gs_core, nseg_r_core, longest
