# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""HPCG 3.1's synthetic problem in the device storage order, and a numpy reference of the device algorithm.

Storage order. A level vector has length P (a multiple of 1024). Position q is the order of the 4-byte words in
the [P / 32, 32] fp32 TILE tensor's DRAM pages: q = page * 1024 + word. So any contiguous q range is contiguous
bytes inside a page, and the device never needs to know about tile faces (tt/layout.py does the host-side
permutation).

Rows are grouped into 8 colours, c = (ix & 1) + 2 (iy & 1) + 4 (iz & 1); no two neighbours share a colour, so a
Gauss-Seidel phase updates a whole colour at once. Each colour block starts on a tile boundary. Inside a block,
rows are stored band by band in y (Level(nx, bands)) and in natural order inside a band, so a core owning part
of one band touches one band-plane of neighbours above and below rather than a whole plane.

Matrix rows have 27 slots, j = (sz + 1) * 9 + (sy + 1) * 3 + (sx + 1) (HPCG's column order); slot 13 is the
diagonal. A missing neighbour is a zero-valued slot pointing at the row itself (HPCG FAQ 3 allows zero padding).

Coarse-grid transfer (HPCG's f2c keeps the all-even fine points, which are fine colour 0):
  restriction   rc[qc] = t[R[qc]] with t = r - A z on block 0 (a 1-term indexed read)
  prolongation  z[block 0][q] += zc[Pm[q]] (a 1-term indexed read of zc, then an in-place add)

usage: python -m models.experimental.hpcg.reference.problem nx {32|64} [colour order] [iterations]
"""

import numpy as np

TILE = 1024
NLEV = 4
COLOUR_ORDER = (7, 6, 5, 4, 3, 2, 1, 0)  # forward-sweep colour order (fewest CG iterations of 5 orders tested)


def _ceil(a, b):
    return -(-a // b) * b


class Level:
    def __init__(self, nx, bands=1, slot_stride=None):
        """bands: y-bands per colour block. Inside a colour block rows are stored band by band (natural order inside
        a band), so a core that owns part of one band touches one band-plane above / below, not a whole plane.
        slot_stride(P): row length of the V / COL storage (default P); the device pads each slot row so that the 27
        rows of one segment start in different DRAM banks. V and COL are [27, P] views of that storage."""
        self.nx, self.bands = nx, bands
        n = nx**3
        self.n = n
        from models.experimental.hpcg.host import native

        lib = native.lib()
        if lib is not None:
            self._build_native(lib, slot_stride)
        else:
            self._build_numpy(slot_stride)
        self.real = self.nat_at >= 0
        self.coarse = None

    def _layout(self):
        """Colour-block and band starts (q), from the per-colour counts alone (no per-point work)."""
        nx, bands = self.nx, self.bands
        h = max(1, nx // 2)
        starts, bstarts, p = [], [], 0
        for c in range(8):
            px, py, pz = c & 1, (c >> 1) & 1, c >> 2
            cx, cy, cz = (nx - px + 1) // 2, (nx - py + 1) // 2, (nx - pz + 1) // 2
            cnt = np.bincount(np.arange(cy) * bands // h, minlength=bands) * cx * cz
            starts.append(p)
            bstarts.append([int(v) for v in p + np.concatenate([[0], np.cumsum(cnt)])])
            p = _ceil(p + cx * cy * cz, TILE)
        starts.append(p)
        return p, starts, bstarts

    def _build_native(self, lib, slot_stride):
        nx, bands = self.nx, self.bands
        P, starts, bstarts = self._layout()
        h = max(1, nx // 2)
        nl = (nx + 1) // 2
        nb = max(len(b) - 1 for b in bstarts)
        line_band = np.zeros(2 * nl, np.int32)
        band_y0 = np.zeros(8 * nb, np.int32)
        band_ny = np.zeros(8 * nb, np.int32)
        band_start = np.zeros(8 * nb, np.int64)
        for py in (0, 1):
            cy = (nx - py + 1) // 2
            line_band[py * nl : py * nl + cy] = np.arange(cy) * bands // h
        for c in range(8):
            py = (c >> 1) & 1
            cy = (nx - py + 1) // 2
            lb = np.arange(cy) * bands // h
            for j in range(len(bstarts[c]) - 1):
                sel = np.flatnonzero(lb == j)
                band_y0[c * nb + j] = sel[0] if sel.size else 0
                band_ny[c * nb + j] = sel.size
                band_start[c * nb + j] = bstarts[c][j]
        stride = slot_stride(P) if slot_stride else P
        self.pos = np.empty(self.n, np.int64)
        self.nat_at = np.empty(P, np.int64)
        self.V_rows = np.zeros((27, stride), np.float32)
        self.COL_rows = np.zeros((27, stride), np.int32)
        self.B = np.empty(P)
        self.nnz = int(
            lib.hpcg_build_level(
                nx,
                nb,
                nl,
                line_band,
                band_y0,
                band_ny,
                band_start,
                P,
                stride,
                self.pos,
                self.nat_at,
                self.V_rows,
                self.COL_rows,
                self.B,
            )
        )
        self.P, self.starts, self.bstarts = P, starts, bstarts
        self.V, self.COL = self.V_rows[:, :P], self.COL_rows[:, :P]

    def _build_numpy(self, slot_stride):
        """The same matrix with numpy array operations: the readable definition, and the fallback."""
        nx, bands, n = self.nx, self.bands, self.n
        nat = np.arange(n)
        ix, iy, iz = nat % nx, (nat // nx) % nx, nat // (nx * nx)
        col = (ix & 1) + 2 * (iy & 1) + 4 * (iz & 1)
        band = (iy // 2) * bands // max(1, nx // 2)
        pos = np.full(n, -1, np.int64)
        starts, bstarts, p = [], [], 0
        for c in range(8):
            starts.append(p)
            m = nat[col == c]
            bm = band[m]
            m = m[np.argsort(bm, kind="stable")]
            pos[m] = p + np.arange(m.size)
            cnt = np.bincount(bm, minlength=bands)
            bstarts.append([int(v) for v in p + np.concatenate([[0], np.cumsum(cnt)])])
            p = _ceil(p + m.size, TILE)
        starts.append(p)
        self.P, self.starts, self.pos, self.bstarts = p, starts, pos, bstarts
        self.nat_at = np.full(p, -1, np.int64)
        self.nat_at[pos] = nat
        stride = slot_stride(p) if slot_stride else p
        self.V_rows = np.zeros((27, stride), np.float32)  # values are exact in fp32 (26, -1, 0, 1)
        self.COL_rows = np.zeros((27, stride), np.int32)
        V, COL = self.V_rows[:, :p], self.COL_rows[:, :p]
        COL[:] = np.arange(p, dtype=np.int32)
        cnt = np.zeros(n, np.int64)
        for sz in (-1, 0, 1):
            for sy in (-1, 0, 1):
                for sx in (-1, 0, 1):
                    j = (sz + 1) * 9 + (sy + 1) * 3 + (sx + 1)
                    ok = (
                        (ix + sx >= 0)
                        & (ix + sx < nx)
                        & (iy + sy >= 0)
                        & (iy + sy < nx)
                        & (iz + sz >= 0)
                        & (iz + sz < nx)
                    )
                    rows = pos[ok]
                    V[j, rows] = 26.0 if j == 13 else -1.0
                    COL[j, rows] = pos[(nat + sz * nx * nx + sy * nx + sx)[ok]]
                    cnt += ok
        # padding rows: diagonal 1 so the GS update (b - s + d*x)/d keeps them at 0
        V[13, self.nat_at < 0] = 1.0
        self.V, self.COL = V, COL
        self.nnz = int(cnt.sum())
        self.B = np.zeros(p)
        self.B[pos] = 26.0 - (cnt - 1)

    def link(self, C):
        """C = next coarser level. R[qc]: block-0 position (relative) of coarse point qc's fine point.
        Pm[q]: coarse position of fine block-0 position q (pads -> any, with weight 0)."""
        self.coarse = C
        cn = C.nat_at
        real = cn >= 0
        cx, cy, cz = cn % C.nx, (cn // C.nx) % C.nx, cn // (C.nx * C.nx)
        fine_nat = 2 * cx + 2 * cy * self.nx + 2 * cz * self.nx * self.nx
        R = np.zeros(C.P, np.int64)
        R[real] = self.pos[fine_nat[real]]
        # padding rows (weight 0) point at the previous real row's column, so they never widen a core's window
        idx = np.where(real, np.arange(C.P), 0)
        R = R[np.maximum.accumulate(idx)]
        assert R[real].max() < self.starts[1]
        self.R, self.Rw = R, real.astype(np.float64)
        nb0 = self.starts[1]
        Pm = np.zeros(nb0, np.int64)
        Pw = np.zeros(nb0)
        Pm[R[real]] = np.nonzero(real)[0]
        Pw[R[real]] = 1.0
        idx = np.where(Pw > 0, np.arange(nb0), 0)  # zero-weight entries: previous real entry's column
        Pm = Pm[np.maximum.accumulate(idx)]
        self.Pm, self.Pw = Pm, Pw


def build(nx, bands=None, slot_stride=None):
    """The 4 multigrid levels nx, nx/2, nx/4, nx/8 (bands: y-bands per level, default 1), linked fine to coarse.
    slot_stride: see Level."""
    bands = list(bands or []) + [1] * NLEV
    levels = [Level(nx >> l, bands[l], slot_stride) for l in range(NLEV)]
    for l in range(NLEV - 1):
        levels[l].link(levels[l + 1])
    return levels


# numpy reference of the device algorithm (same colour ordering, same multigrid, same rounding order per row)
def spmv(L, x, dt):
    V = L.V.astype(dt)
    s = V[0] * x[L.COL[0]]
    for j in range(1, 27):
        s = s + V[j] * x[L.COL[j]]
    return s


def gs_phase(L, x, r, c, dt):
    a, e = L.starts[c], L.starts[c + 1]
    V = L.V[:, a:e].astype(dt)
    C = L.COL[:, a:e]
    s = V[0] * x[C[0]]
    for j in range(1, 27):
        s = s + V[j] * x[C[j]]
    d = V[13]
    x[a:e] = (r[a:e] - s + d * x[a:e]) / d


def symgs(L, x, r, dt, order):
    for c in order:
        gs_phase(L, x, r, c, dt)
    for c in reversed(order):
        gs_phase(L, x, r, c, dt)


def mg(levels, l, r, dt, order):
    L = levels[l]
    z = np.zeros(L.P, dt)
    symgs(L, z, r, dt, order)
    if l + 1 < len(levels):
        Az = spmv(L, z, dt)
        nb0 = L.starts[1]
        t = r[:nb0] - Az[:nb0]
        rc = (t[L.R] * L.Rw.astype(dt)).astype(dt)
        zc = mg(levels, l + 1, rc, dt, order)
        z[:nb0] = z[:nb0] + (zc[L.Pm] * L.Pw.astype(dt)).astype(dt)
        symgs(L, z, r, dt, order)
    return z


def dot(a, b, dt):
    return dt(np.dot(a.astype(np.float64), b.astype(np.float64))) if dt == np.float64 else dt(np.dot(a, b))


def cg(levels, maxit, dt, order, log=None):
    L = levels[0]
    b = L.B.astype(dt)
    x = np.zeros(L.P, dt)
    p = x.copy()
    r = b - spmv(L, p, dt)
    normr0 = np.sqrt(dot(r, r, dt))
    hist, rtz = [], None
    for k in range(1, maxit + 1):
        z = mg(levels, 0, r, dt, order)
        if k == 1:
            p = z + dt(0) * z
            rtz = dot(r, z, dt)
        else:
            oldrtz, rtz = rtz, dot(r, z, dt)
            p = z + dt(rtz / oldrtz) * p
        Ap = spmv(L, p, dt)
        alpha = dt(rtz / dot(p, Ap, dt))
        x = x + alpha * p
        r = r - alpha * Ap
        hist.append(float(np.sqrt(dot(r, r, dt)) / normr0))
        if log:
            log(k, hist[-1])
    return x, hist


if __name__ == "__main__":
    import sys

    nx = int(sys.argv[1])
    dt = {"64": np.float64, "32": np.float32}[sys.argv[2]]
    order = [int(c) for c in sys.argv[3]] if len(sys.argv) > 3 else list(COLOUR_ORDER)
    maxit = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    levels = build(nx)
    for L in levels:
        assert np.allclose(spmv(L, L.real.astype(np.float64), np.float64), L.B)
    x, hist = cg(levels, maxit, dt, order, log=lambda k, h: print(f"iter {k:2d} scaled residual {h:.6e}", flush=True))
    print(f"final {hist[-1]:.6e}  max|x-1| {np.max(np.abs(x[levels[0].real] - 1)):.3e}")
