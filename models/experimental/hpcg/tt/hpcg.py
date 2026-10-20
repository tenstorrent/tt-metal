# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""HPCG's preconditioned CG solve on one Blackhole P150, fp32 (not an official HPCG run: the rules require fp64).

One CG iteration, as HPCG 3.1's CG.cpp with a 4-level multigrid preconditioner (MG.cpp):
  multigrid, per level l = 0..2: pre-smoother + residual SpMV + r - A z (one sparse-row launch), restriction,
      the coarser level, prolongation (gather and add in one launch), post-smoother; level 3: one smoother
  rtz = r.z, p = z + (rtz / rtz_old) p, Ap = A p, pAp = p.Ap, x += alpha p, r -= alpha Ap, |r|
Every matrix and vector pass is a custom op (tt/custom_ops), including the final sums of the dot products. The CG
start (start()) and every iteration run on the device only and are trace-capturable; the CG scalars stay on device.
"""

import ttnn

from models.experimental.hpcg.reference.problem import COLOUR_ORDER, build
from models.experimental.hpcg.tt.custom_ops.cg_vector import CgVectorOp, DotFinishOp
from models.experimental.hpcg.tt.custom_ops.sparse_rows import SparseRowOp, SparseRowPlan, slot_pages
from models.experimental.hpcg.tt.custom_ops.sparse_rows.op import debug_buffer, noc_coords
from models.experimental.hpcg.tt.layout import TILE, rows_to_device, to_device, zeros_device


def default_bands(nx_level):
    """y-bands per colour block for a level of nx_level^3 points, measured fastest: 10 from 160^3 (192^3: 24.02 ->
    23.76 ms per iteration against 1 band), 5 from 112^3 (128^3: 7.85 -> 7.76 ms), none below (104^3: 4.94 ms with
    1, 5.05 with 5)."""
    return 10 if nx_level >= 160 else 5 if nx_level >= 112 else 1


def slot_stride(P):
    """Row length of the 27-slot matrix storage: whole pages, an odd number of them (see slot_pages)."""
    return slot_pages(P // TILE) * TILE


class DeviceLevel:
    """One multigrid level on device: matrix values and window offsets, and the level's work vectors."""

    def __init__(self, device, L, ncores, noc_xy, r=None):
        self.L, self.P = L, L.P
        blocks = [(L.starts[c], L.starts[c + 1]) for c in range(8)]
        self.plan = SparseRowPlan(
            L.COL_rows,
            blocks,
            blocks,
            ncores,
            noc_xy,
            exchange=True,
            bands=L.bstarts if L.bands > 1 else None,
            pout=L.P,
        )
        # the storage rows are already padded to slot_stride, so both tables upload as they are
        self.values = rows_to_device(device, L.V_rows)
        self.offsets = rows_to_device(device, self.plan.off)  # 32-bit window offsets
        self.z, self.Az = (zeros_device(device, L.P) for _ in range(2))
        self.r = r if r is not None else zeros_device(device, L.P)  # the fine level shares CG's r


class HPCG:
    """Device state and ops of the HPCG solve on an nx^3 grid (nx a multiple of 8, as in HPCG).

    ops: ("gs_pre", l), ("gs", l): pre- and post-smoother of level l;
    ("restrict", l), ("prolong", l): level l -> l + 1 and back; "spmv_cg": Ap = A p.
    vops: "rr", "rz", "pAp" dot partials and their "*_fin" sums; "p": p update; "xr": x and r update with the
    partial r.r. A CG solve is reset(), start(), then iteration() repeated.
    """

    def __init__(self, device, nx, bands=None, order=COLOUR_ORDER, kernel_debug=False, level_cores=None):
        if nx % 8:
            raise ValueError(f"nx must be a multiple of 8 (4 multigrid levels), got {nx}")
        self.device = device
        g = device.compute_with_storage_grid_size()
        nc = g.x * g.y
        bands = list(bands) if bands is not None else [default_bands(nx >> l) for l in range(4)]
        self.levels = build(nx, bands, slot_stride)
        self.order = list(order)
        noc_xy = noc_coords(device)
        lc = list(level_cores or []) + [nc] * 4  # cores per level for the smoothers (default: all)
        L0 = self.levels[0]
        self.r = zeros_device(device, L0.P)
        self.lv = [
            DeviceLevel(device, L, min(nc, lc[l]), noc_xy, r=self.r if l == 0 else None)
            for l, L in enumerate(self.levels)
        ]
        self.b = to_device(device, L0.B)
        self.x, self.p, self.Ap, self.zero = (zeros_device(device, L0.P) for _ in range(4))
        self.debug = debug_buffer(device)  # per-core cycle counters, written by kernel_debug builds
        dbg = dict(kernel_debug=kernel_debug, debug=self.debug)
        self._dbg = dbg
        phases = self.order + list(reversed(self.order))  # forward then backward sweep
        self.ops = {}
        for l, d in enumerate(self.lv):
            L = d.L
            # L1 page: one 16x16 face when each core owns <= 256 rows of a colour (the unpacker and packer then
            # move one face instead of four; slower on the finer levels)
            tpw = d.tpw = 256 if (L.starts[1] - L.starts[0]) / d.plan.nc <= 256 else TILE
            # post-smoother; on the fine grid the writer loads half the window (6.86 -> 6.73 ms at 192^3; slower on
            # the coarser levels)
            self.ops[("gs", l)] = SparseRowOp(
                device, d.plan, d.values, d.offsets, "gs", phases, d.z, d.z, b=d.r, tpw=tpw, wsplit=(l == 0), **dbg
            )
            # pre-smoother: z = 0 on entry, then (above the coarsest level) Az = A z with block 0 = r - A z
            self.ops[("gs_pre", l)] = SparseRowOp(
                device,
                d.plan,
                d.values,
                d.offsets,
                "gs",
                phases + ([8 + b for b in range(8)] if l < 3 else []),
                d.z,
                d.z,
                b=d.r,
                out2=d.Az if l < 3 else None,
                zerox=True,
                restf=True,
                tpw=tpw,
                **dbg,
            )
            if l + 1 < len(self.lv):
                C, dc = self.levels[l + 1], self.lv[l + 1]
                nb0 = L.starts[1]
                cblocks = [(C.starts[c], C.starts[c + 1]) for c in range(8)]
                # restriction: rc[qc] = Rw * t[R[qc]], t = r - A z = block 0 of Az (written by the pre-smoother)
                rp = SparseRowPlan(
                    L.R[None, :],
                    cblocks,
                    [(0, nb0)],
                    nc,
                    noc_xy,
                    exchange=False,
                    bands=C.bstarts if C.bands > 1 else None,
                )
                d.Rt = rows_to_device(device, L.Rw)
                d.Roff = rows_to_device(device, rp.off)
                self.ops[("restrict", l)] = SparseRowOp(
                    device, rp, d.Rt, d.Roff, "spmv", list(range(8)), d.Az, dc.r, **dbg
                )
                # prolongation: z[block 0] = z[block 0] + Pw * zc[Pm], gather and add in one launch
                pp = SparseRowPlan(
                    L.Pm[None, :],
                    [(0, nb0)],
                    cblocks,
                    nc,
                    noc_xy,
                    exchange=False,
                    bands=[L.bstarts[0]] if L.bands > 1 else None,
                )
                d.Pt = rows_to_device(device, L.Pw)
                d.Poff = rows_to_device(device, pp.off)
                self.ops[("prolong", l)] = SparseRowOp(
                    device, pp, d.Pt, d.Poff, "spmv", [0], dc.z, d.z, b=d.z, addb=True, **dbg
                )
        # CG SpMV: an SpMV-only gs launch on the smoother plan, so cores move through the matrix in lockstep
        d0 = self.lv[0]
        self.ops["spmv_cg"] = SparseRowOp(
            device,
            d0.plan,
            d0.values,
            d0.offsets,
            "gs",
            [8 + b for b in range(8)],
            self.p,
            self.p,
            b=self.p,
            out2=self.Ap,
            **dbg,
        )
        ncv = min(nc, L0.P // TILE)
        self.part = zeros_device(device, ncv * TILE)  # per-core dot partials, one tile each
        # CG scalars, one fp32 tile each (value at element (0, 0)); they never leave the device
        scalar = ttnn.Shape([1, 1])
        self._one = ttnn.full(scalar, 1.0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        self.rtz, self.rtz_old, self.pAp, self.normr, self.normr0 = (
            ttnn.full(scalar, 0.0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device) for _ in range(5)
        )
        z = self.lv[0].z
        rz = CgVectorOp(device, "dot", self.r, z, part=self.part)
        n = -(-rz.nc * 64 // (4 * TILE))  # pages holding one 64-byte slot per core
        self.vops = dict(
            rr=CgVectorOp(device, "dot", self.r, self.r, part=self.part),
            nr0_fin=DotFinishOp(device, self.part, n, self.normr0, sqrt=True),
            rz=rz,
            rz_fin=DotFinishOp(device, self.part, n, self.rtz, old=self.rtz_old, keep_old=True),
            pAp=CgVectorOp(device, "dot", self.p, self.Ap, part=self.part),
            pAp_fin=DotFinishOp(device, self.part, n, self.pAp),
            p=CgVectorOp(device, "axpy", z, self.p, s1=self.rtz, s2=self.rtz_old),
            xr=CgVectorOp(device, "xr", self.x, self.p, self.r, self.Ap, s1=self.rtz, s2=self.pAp, part=self.part),
            nr_fin=DotFinishOp(device, self.part, n, self.normr, sqrt=True),
        )

    def spmv(self, l):
        """A plain z -> Az launch of level l (used by tests and diagnostics, built on first use)."""
        key = ("spmv", l)
        if key not in self.ops:
            d = self.lv[l]
            self.ops[key] = SparseRowOp(
                self.device, d.plan, d.values, d.offsets, "spmv", list(range(8)), d.z, d.Az, tpw=d.tpw, **self._dbg
            )
        return self.ops[key]

    def mg(self, l):
        """z = M^-1 r on level l (HPCG's ComputeMG): one pre- and one post-smoothing SYMGS per level."""
        self.ops[("gs_pre", l)]()  # z = 0, pre-smoother, then Az = A z (block 0: r - A z)
        if l + 1 < len(self.lv):
            self.ops[("restrict", l)]()
            self.mg(l + 1)
            self.ops[("prolong", l)]()
            self.ops[("gs", l)]()

    def reset(self):
        """x = 0. HPCG zeroes x before each CG call (main.cpp), outside the CG timer."""
        ttnn.copy(self.zero, self.x)

    def start(self):
        """CG start, the part of HPCG's CG.cpp that runs inside its timer before iteration 1: p = x, Ap = A p,
        r = b - Ap, normr0 = |r|. Device only and trace-capturable; returns normr0 as a 1x1 device tensor.
        r.z starts at 1, so iteration 1 keeps 1 as the previous r.z and its p = z + (r.z / 1) p = z (p = x = 0)."""
        ttnn.copy(self.x, self.p)
        self.ops["spmv_cg"]()
        ttnn.subtract(self.b, self.Ap, output_tensor=self.r)
        self.vops["rr"]()
        self.vops["nr0_fin"]()
        ttnn.copy(self._one, self.rtz)
        return self.normr0

    def iteration(self):
        """One CG iteration; returns |r| as a 1x1 device tensor (the same tensor every call)."""
        self.mg(0)
        v = self.vops
        v["rz"]()  # per-core partials of r.z
        v["rz_fin"]()  # previous r.z kept, then r.z
        v["p"]()  # p = z + (r.z / previous r.z) p
        self.ops["spmv_cg"]()  # Ap = A p
        v["pAp"]()
        v["pAp_fin"]()  # p.Ap
        v["xr"]()  # x += alpha p, r -= alpha Ap (alpha = r.z / p.Ap), partials of r.r
        v["nr_fin"]()  # |r| = sqrt(r.r)
        return self.normr
