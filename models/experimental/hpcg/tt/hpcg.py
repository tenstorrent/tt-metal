# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""HPCG's preconditioned CG solve on one Blackhole P150, fp32 (not an official HPCG run: the rules require fp64).

One CG iteration, as HPCG 3.1's CG.cpp with a 4-level multigrid preconditioner (MG.cpp):
  multigrid, per level l = 0..2: pre-smoother + residual SpMV + r - A z (one sparse-row launch), restriction,
      the coarser level, prolongation (gather and add in one launch), post-smoother; level 3: one smoother
  rtz = r.z, p = z + (rtz / rtz_old) p, Ap = A p, pAp = p.Ap, x += alpha p, r -= alpha Ap, |r|
Every matrix and vector pass is a custom op (tt/custom_ops); TT-NN reduces the per-core dot partials and takes
the square root. The whole iteration is trace-capturable.
"""

import ttnn

from models.experimental.hpcg.reference.problem import COLOUR_ORDER, build
from models.experimental.hpcg.tt.custom_ops.cg_vector import CgVectorOp
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

    def __init__(self, device, L, ncores, noc_xy):
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
        self.z, self.Az, self.r = (zeros_device(device, L.P) for _ in range(3))


class HPCG:
    """Device state and ops of the HPCG solve on an nx^3 grid (nx a multiple of 8, as in HPCG).

    ops: ("gs_pre", l), ("gs", l): pre- and post-smoother of level l;
    ("restrict", l), ("prolong", l): level l -> l + 1 and back; "spmv_cg": Ap = A p.
    vops: "rz", "pAp" dot partials; "p": p update; "xr": x and r update with the partial r.r.
    """

    def __init__(self, device, nx, bands=None, order=COLOUR_ORDER, kernel_debug=False):
        if nx % 8:
            raise ValueError(f"nx must be a multiple of 8 (4 multigrid levels), got {nx}")
        self.device = device
        g = device.compute_with_storage_grid_size()
        nc = g.x * g.y
        bands = list(bands) if bands is not None else [default_bands(nx >> l) for l in range(4)]
        self.levels = build(nx, bands, slot_stride)
        self.order = list(order)
        noc_xy = noc_coords(device)
        self.lv = [DeviceLevel(device, L, nc, noc_xy) for L in self.levels]
        L0 = self.levels[0]
        self.b = to_device(device, L0.B)
        self.x, self.r, self.p, self.Ap, self.zero = (zeros_device(device, L0.P) for _ in range(5))
        self.lv[0].r = self.r
        self.acc = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )
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
        self.vops = None

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

    def psum(self):
        """Sum of the per-core dot partials (a 1x1 tensor)."""
        s = ttnn.sum(self.part, dim=-2, keepdim=True, compute_kernel_config=self.acc)
        return ttnn.sum(s, dim=-1, keepdim=True, compute_kernel_config=self.acc)

    def dot(self, a, b):
        s = ttnn.sum(ttnn.multiply(a, b), dim=-2, keepdim=True, compute_kernel_config=self.acc)
        return ttnn.sum(s, dim=-1, keepdim=True, compute_kernel_config=self.acc)

    def setup_cg(self):
        """CG start (HPCG CG.cpp): x = 0, p = 0, r = b - A p, normr0 = |r|; rtz_old = 1 so iteration 1's
        p = z + rtz * 0 = z. Untraced; writes only persistent buffers."""
        ttnn.copy(self.zero, self.x)
        ttnn.copy(self.zero, self.p)
        self.ops["spmv_cg"]()
        ttnn.subtract(self.b, self.Ap, output_tensor=self.r)
        nr = ttnn.sqrt(self.dot(self.r, self.r))
        self.normr0 = float(ttnn.to_torch(nr).reshape(-1)[0])
        one = ttnn.add(ttnn.multiply(nr, 0.0), 1.0)
        if self.vops is None:
            self.rtz_old = one
            self.rtz = ttnn.add(ttnn.multiply(nr, 0.0), 0.0)
            self.pAp = ttnn.add(ttnn.multiply(nr, 0.0), 0.0)
            z = self.lv[0].z
            self.vops = dict(
                rz=CgVectorOp(self.device, "dot", self.r, z, part=self.part),
                pAp=CgVectorOp(self.device, "dot", self.p, self.Ap, part=self.part),
                p=CgVectorOp(self.device, "axpy", z, self.p, s1=self.rtz, s2=self.rtz_old),
                xr=CgVectorOp(
                    self.device, "xr", self.x, self.p, self.r, self.Ap, s1=self.rtz, s2=self.pAp, part=self.part
                ),
            )
        else:
            ttnn.copy(one, self.rtz_old)

    def iteration(self):
        """One CG iteration; returns |r| as a 1x1 device tensor."""
        self.mg(0)
        v = self.vops
        v["rz"]()
        ttnn.copy(self.psum(), self.rtz)  # rtz = r.z
        v["p"]()  # p = z + (rtz / rtz_old) p
        ttnn.copy(self.rtz, self.rtz_old)
        self.ops["spmv_cg"]()  # Ap = A p
        v["pAp"]()
        ttnn.copy(self.psum(), self.pAp)  # pAp = p.Ap
        v["xr"]()  # x += alpha p, r -= alpha Ap (alpha = rtz / pAp), r.r
        return ttnn.sqrt(self.psum())
