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

from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import ttnn

from models.experimental.hpcg.host import native
from models.experimental.hpcg.reference.problem import COLOUR_ORDER, Level, build
from models.experimental.hpcg.tt.custom_ops.cg_vector import CgVectorOp, DotFinishOp
from models.experimental.hpcg.tt.custom_ops.sparse_rows import SparseRowOp, SparseRowPlan, slot_pages
from models.experimental.hpcg.tt.custom_ops.sparse_rows.op import debug_buffer, noc_coords
from models.experimental.hpcg.tt.custom_ops.stencil_matrix import StencilMatrixOp
from models.experimental.hpcg.tt.custom_ops.stencil_matrix.op import StencilJob, TransferJob, warm_up
from models.experimental.hpcg.tt.layout import (
    TILE,
    empty_device,
    empty_page,
    rows_to_device,
    scalar_device,
    to_device,
    upload_zeros,
)

# The compiled setup helper (host/native_setup.cpp) is loaded with this module, as TT-NN is with `import ttnn`:
# loading a library is program start-up (in HPCG's C++ code this code is part of the executable), not setup
native.lib()
# The thread that hands the matrix-generation launches to TT-NN (see HPCG.__init__), started here too: starting a thread
# waits for it to run (76 to 206 us on the P150's host), program start-up as well
_LAUNCHER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hpcg-launch")
_LAUNCHER.submit(int).result()


def default_bands(nx_level):
    """y-bands per colour block for a level of nx_level^3 points, measured fastest: 10 from 160^3 (192^3: 24.02 ->
    23.76 ms per iteration against 1 band), 5 from 112^3 (128^3: 7.85 -> 7.76 ms), none below (104^3: 4.94 ms with
    1, 5.05 with 5)."""
    return 10 if nx_level >= 160 else 5 if nx_level >= 112 else 1


def slot_stride(P):
    """Row length of the 27-slot matrix storage: whole pages, an odd number of them (see slot_pages)."""
    return slot_pages(P // TILE) * TILE


class DeviceLevel:
    """One multigrid level on device: matrix values and window offsets, and the level's work vectors.
    b: for the fine level, CG's right-hand side, which the device fills when it generates the matrix.
    zeros: vectors of the level to set to zero alongside (the fine level's x and p).
    job: on the device-generated path, the StencilJob that writes values, offsets, b and zeros (the caller launches
    it, StencilMatrixOp); None on the host-tables path.
    Work vectors are allocated without a fill: every launch writes them before any launch reads them."""

    def __init__(self, device, L, ncores, noc_xy, r=None, b=None, zeros=(), plan=None, tables=None):
        self.L, self.P = L, L.P
        blocks = [(L.starts[c], L.starts[c + 1]) for c in range(8)]
        plan_args = (blocks, blocks, ncores, noc_xy, True)
        bands = L.bstarts if L.bands > 1 else None
        if L.tables:  # matrix built on the host (no compiled helper): plan from its columns, upload both tables
            self.plan = SparseRowPlan(L.COL_rows, *plan_args, bands=bands, pout=L.P)
            # the storage rows are already padded to slot_stride, so both tables upload as they are
            self.values = rows_to_device(device, L.V_rows)
            self.offsets = rows_to_device(device, self.plan.off)  # 32-bit window offsets
            if b is not None:
                ttnn.copy(to_device(device, L.B), b)
            for t in zeros:
                ttnn.copy(to_device(device, np.zeros(L.P, np.float32)), t)
            self.job = None
        else:  # the device writes the values, the offsets, b and the zero vectors; the host plans the windows only
            self.plan = plan if plan is not None else DeviceLevel.stencil_plan(L, ncores, noc_xy)
            self.job = StencilJob(device, L, self.plan, b=b, zeros=zeros, tables=tables)
            self.values, self.offsets = self.job.values, self.job.offsets
        self.z, self.Az = (empty_device(device, L.P) for _ in range(2))
        self.r = r if r is not None else empty_device(device, L.P)  # the fine level shares CG's r

    @staticmethod
    def stencil_plan(L, ncores, noc_xy, defer_exchange=False):
        """The Gauss-Seidel plan of level L from its storage-order formula (host work only: no device calls)."""
        blocks = [(L.starts[c], L.starts[c + 1]) for c in range(8)]
        bands = L.bstarts if L.bands > 1 else None
        return SparseRowPlan(
            None, blocks, blocks, ncores, noc_xy, True, bands=bands, pout=L.P, stencil=L, defer_exchange=defer_exchange
        )


class HPCG:
    """Device state and ops of the HPCG solve on an nx^3 grid (nx a multiple of 8, as in HPCG).

    ops: ("gs_pre", l), ("gs", l): pre- and post-smoother of level l;
    ("restrict", l), ("prolong", l): level l -> l + 1 and back; "spmv_cg": Ap = A p.
    vops: "rr", "rz", "pAp" dot partials and their "*_fin" sums; "p": p update; "xr": x and r update with the
    partial r.r. A CG solve is reset(), start(), then iteration() repeated.
    """

    def __init__(
        self, device, nx, bands=None, order=COLOUR_ORDER, kernel_debug=False, level_cores=None, host_tables=False
    ):
        """host_tables: build the matrix and its offset tables on the host and upload them (the reference path)
        instead of generating them on the device (tt/custom_ops/stencil_matrix). Without the compiled host helper
        the host tables are used whatever host_tables says: device generation is planned by the helper."""
        if nx % 8:
            raise ValueError(f"nx must be a multiple of 8 (4 multigrid levels), got {nx}")
        host_tables = host_tables or native.lib() is None
        launched = []  # generation launches handed to _LAUNCHER
        try:
            self._setup(device, nx, bands, order, kernel_debug, level_cores, host_tables, launched)
        finally:
            # a launch still running on the worker thread must not outlive a failed setup (the caller then closes
            # the device)
            wait(launched)
        for f in launched:  # every generation launch is enqueued (and its errors raised) before any other launch
            f.result()

    def _setup(self, device, nx, bands, order, kernel_debug, level_cores, host_tables, launched):
        self.device = device
        g = device.compute_with_storage_grid_size()
        nc = g.x * g.y
        bands = list(bands) if bands is not None else [default_bands(nx >> l) for l in range(4)]
        noc_xy = noc_coords(device)
        self.order = list(order)
        lc = list(level_cores or []) + [nc] * 4  # cores per level for the smoothers (default: all)
        # generation launches go through one worker thread, in order (TT-NN releases the GIL while it builds and
        # enqueues a program): first a launch with no jobs, shaped like the fine level's, that loads the kernel
        # binaries and builds that program while this thread builds and plans the fine level; then each real launch
        # (the fine level's reuses the program), while this thread goes on. A launch is handed over only after its
        # table upload returned, so the upload is ahead of it in the queue
        gen = None if host_tables else _LAUNCHER
        if gen is not None:
            launched.append(gen.submit(warm_up, device, min(nc, lc[0])))
        tables0 = None
        if host_tables:
            self.levels = build(nx, bands, slot_stride, tables=True)
            L0, plan0 = self.levels[0], None
        else:  # the fine level first: the coarser ones are built after its generation launch
            L0 = Level(nx, bands[0], slot_stride, tables=False)
            tables0 = StencilJob.allocate(device, L0)  # allocations need no plan: done before planning
        self.r, self.b, self.x, self.p, self.Ap = (empty_device(device, L0.P) for _ in range(5))
        if not host_tables:
            # exchange lists after the fine level's generation launch (it needs only the windows)
            plan0 = DeviceLevel.stencil_plan(L0, min(nc, lc[0]), noc_xy, defer_exchange=True)
        fine = dict(r=self.r, b=self.b, zeros=(self.x, self.p))  # x = 0 (initial guess) and p = 0 on the fine level
        # matrix generation in two launches: the fine level's as soon as it is planned (the device writes it while
        # the host plans the rest), then the coarser levels' and the grid transfers' together
        self.lv = [DeviceLevel(device, L0, min(nc, lc[0]), noc_xy, plan=plan0, tables=tables0, **fine)]
        if self.lv[0].job is not None:
            launched.append(gen.submit(StencilMatrixOp(device, [self.lv[0].job])))
            if not plan0.complete_exchange():  # too many neighbours: re-plan (fewer cores) and generate again
                tables = (self.lv[0].values, self.lv[0].offsets)
                self.lv[0] = DeviceLevel(device, L0, min(nc, lc[0]), noc_xy, tables=tables, **fine)
                launched.append(gen.submit(StencilMatrixOp(device, [self.lv[0].job])))
            self.levels = build(nx, bands, slot_stride, tables=False, finest=L0)
        self.lv += [DeviceLevel(device, L, min(nc, lc[l]), noc_xy) for l, L in enumerate(self.levels) if l > 0]
        jobs = [d.job for d in self.lv[1:] if d.job is not None]
        # grid transfers l -> l + 1: restriction rc[qc] = Rw * t[R[qc]] (t = r - A z = block 0 of Az, written by the
        # pre-smoother); prolongation z[block 0] += Pw * zc[Pm]. Planned from the two levels' storage-order formulas
        # (no R / Pm tables) with the device writing weights and offsets, or from Level.link's tables on the
        # host-tables path
        for l in range(len(self.lv) - 1):
            L, C, d = self.levels[l], self.levels[l + 1], self.lv[l]
            nb0 = L.starts[1]
            cblocks = [(C.starts[c], C.starts[c + 1]) for c in range(8)]
            cb = C.bstarts if C.bands > 1 else None
            fb = [L.bstarts[0]] if L.bands > 1 else None
            if L.tables:
                d.rp = SparseRowPlan(L.R[None, :], cblocks, [(0, nb0)], nc, noc_xy, False, bands=cb)
                d.pp = SparseRowPlan(L.Pm[None, :], [(0, nb0)], cblocks, nc, noc_xy, False, bands=fb)
                d.Rt, d.Roff = rows_to_device(device, L.Rw), rows_to_device(device, d.rp.off)
                d.Pt, d.Poff = rows_to_device(device, L.Pw), rows_to_device(device, d.pp.off)
                continue
            tr = dict(transfer=("restrict", L, C), host_tables=False)
            d.rp = SparseRowPlan(None, cblocks, [(0, nb0)], nc, noc_xy, False, bands=cb, pout=C.P, **tr)
            tr = dict(transfer=("prolong", L, C), host_tables=False)
            d.pp = SparseRowPlan(None, [(0, nb0)], cblocks, nc, noc_xy, False, bands=fb, pout=nb0, **tr)
            jr, jp = TransferJob(device, d.rp), TransferJob(device, d.pp)
            d.Rt, d.Roff, d.Pt, d.Poff = jr.values, jr.offsets, jp.values, jp.offsets
            jobs += [jr, jp]
        if jobs:
            launched.append(gen.submit(StencilMatrixOp(device, jobs)))
        self.debug = debug_buffer(device, zeros=kernel_debug)  # per-core cycle counters (kernel_debug builds)
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
            # the coarser levels). It runs the first 16 phases of the pre-smoother's table (one table and upload less)
            pre = phases + ([8 + b for b in range(8)] if l < 3 else [])
            self.ops[("gs", l)] = SparseRowOp(
                device,
                d.plan,
                d.values,
                d.offsets,
                "gs",
                phases,
                d.z,
                d.z,
                b=d.r,
                tpw=tpw,
                wsplit=(l == 0),
                table_phases=pre,
                **dbg,
            )
            # pre-smoother: z = 0 on entry, then (above the coarsest level) Az = A z with block 0 = r - A z
            self.ops[("gs_pre", l)] = SparseRowOp(
                device,
                d.plan,
                d.values,
                d.offsets,
                "gs",
                pre,
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
                dc = self.lv[l + 1]
                self.ops[("restrict", l)] = SparseRowOp(
                    device, d.rp, d.Rt, d.Roff, "spmv", list(range(8)), d.Az, dc.r, **dbg
                )
                # prolongation: gather and add in one launch
                self.ops[("prolong", l)] = SparseRowOp(
                    device, d.pp, d.Pt, d.Poff, "spmv", [0], dc.z, d.z, b=d.z, addb=True, **dbg
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
        # CG start: A x written straight into r (then r = b - A x): the CG SpMV's binary, on other buffers. HPCG
        # copies x to p and multiplies p; the product of x is the same and saves the copy
        self.ops["spmv_start"] = SparseRowOp(
            device,
            d0.plan,
            d0.values,
            d0.offsets,
            "gs",
            [8 + b for b in range(8)],
            self.x,
            self.x,
            b=self.x,
            out2=self.r,
            **dbg,
        )
        ncv = min(nc, L0.P // TILE)
        # per-core dot partials, one 64-byte slot per core; zero beyond the slots (the finish adds whole pages).
        # Small constants are uploaded: a device fill would load one more kernel binary per process
        self.part = upload_zeros(device, ncv * TILE)
        # CG scalars, one 4 KB page each (the value in word 0); they never leave the device
        self._one, self._minus_one = (scalar_device(device, v) for v in (1.0, -1.0))
        self._inf_slot = scalar_device(device, float("inf"), only_first=True)  # inf in word 0, zeros elsewhere
        # written on the device before they are read (start() writes r.z = inf and |r0|; an iteration writes the
        # rest before using them): allocated, not uploaded
        self.rtz, self.rtz_old, self.pAp, self.normr, self.normr0 = (empty_page(device) for _ in range(5))
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
            # reset and CG start with the same binaries (no TT-NN copy / subtract / fill programs to load):
            x0=CgVectorOp(device, "zero", self.x, self.p),  # x = 0, p = 0 (zeros written, nothing read)
            r_b=CgVectorOp(device, "axpy", self.b, self.r, s1=self._minus_one, s2=self._one),  # r = b - A x
            # r.z = inf + 0 + ... = inf whatever r.z held before (so start() can run again before an iteration)
            rtz_inf=DotFinishOp(device, self._inf_slot, 1, self.rtz),
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
        """x = 0 and p = 0. HPCG zeroes x before each CG call (main.cpp), outside the CG timer; p is zeroed with it
        for iteration 1 (start()). Zeros are written, not computed, so a NaN or inf left by a diverged solve does not
        carry over to the next one."""
        self.vops["x0"]()

    def start(self):
        """CG start, the part of HPCG's CG.cpp that runs inside its timer before iteration 1: r = b - A x (HPCG
        copies x to p and multiplies p; multiplying x gives the same r) and normr0 = |r|. Device only and
        trace-capturable; returns normr0 as a 1x1 device tensor. The update is r = b + (-1) A x: the same bits as
        b - A x. r.z starts at +inf, so iteration 1 keeps inf as the previous r.z and its p = z + (r.z / inf) p =
        z + 0 p = z exactly (p = 0: zeroed at setup and by reset())."""
        v = self.vops
        self.ops["spmv_start"]()  # A x, into r
        v["r_b"]()
        v["rr"]()
        v["nr0_fin"]()
        v["rtz_inf"]()
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
