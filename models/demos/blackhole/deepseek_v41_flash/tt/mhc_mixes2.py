# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused mHC mixes (two generic_ops): projection x_flat @ fn^T (+ sum of squares) split over cores, then one post kernel
(sum of partials, RMS scale, Sinkhorn parametrisation, token-major pre/post/comb).

v2 layout: the projection cores split the hidden axis in COLUMN TILES (core r owns tiles [r*TPJ, (r+1)*TPJ) of all four streams) and the
tokens in groups of TG <= 8; each core gathers one tile per column tile holding rows (token, stream) (two 256 B reads per
(token, tile)) and contracts it with the per-stream fn chunks. The partials of PPT cores are packed into one 32-row page (TG rows each)
so the post kernel reads a handful of pages and sums them with one selection matmul per page."""

import struct

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash, _kernel

f32 = ttnn.float32


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _grid_cores(mesh, n):
    grid = mesh.compute_with_storage_grid_size()
    return [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)][:n]


class ProjPlan:
    """Work split of the projection for T token rows per device."""

    def __init__(self, T, D=5120, max_cores=100):
        assert T % 4 == 0 and T <= 32
        self.T, self.D = T, D
        self.TG = min(T, 8)  # tokens per core group (4 rows per token must fit one 32-row tile)
        self.G = T // self.TG
        self.NJ = D // 32
        self.S = next(
            s for s in (40, 32, 20) if self.G * s <= max_cores and (32 // self.TG) and s % (32 // self.TG) == 0
        )  # column ranges
        self.TPJ = self.NJ // self.S
        assert self.TPJ * self.S == self.NJ
        self.PPT = 32 // self.TG  # partials per packed page
        assert self.S % self.PPT == 0
        self.PGPG = self.S // self.PPT  # pages per group
        self.NPG = self.G * self.PGPG
        self.cores = self.G * self.S


_PLANS = {}


def proj_plan(T, D=5120, mesh=None):
    key = (T, D)
    p = _PLANS.get(key)
    if p is None:
        mc = 100
        if mesh is not None:
            g = mesh.compute_with_storage_grid_size()
            mc = int(g.x) * int(g.y)
        p = _PLANS[key] = ProjPlan(T, D, mc)
    return p


def mhc_proj2(x, wt, ncol, plan):
    """x [T,1,4,D] fp32 TILE; wt [S,1,4*TPJ*32,32] fp32 TILE (fn chunks, see DSV41MHC) -> packed partial pages [NPG,1,32,32] fp32:
    page g*PGPG + r//PPT, row (r%PPT)*TG + tl = column-range-r partial of token g*TG+tl (columns 0..mix_hc-1 projection, column ncol
    its sum of squares)."""
    T, _, n, D = (int(v) for v in x.shape)
    assert n == 4 and T == plan.T and D == plan.D
    TG, G, S, TPJ, PPT, PGPG = plan.TG, plan.G, plan.S, plan.TPJ, plan.PPT, plan.PGPG
    assert int(wt.shape[0]) == S and int(wt.shape[2]) == 4 * TPJ * 32
    mesh = x.device()
    cores = _grid_cores(mesh, plan.cores)
    assert len(cores) == plan.cores
    part = ttnn.allocate_tensor_on_device(
        ttnn.Shape([plan.NPG, 1, 32, 32]), f32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (cx, cy) in enumerate(cores):
        g, r = divmod(c, S)
        rrt[cx][cy] = [r, g * TG]
        wrt[cx][cy] = [g * PGPG + r // PPT, (r % PPT) * TG]
        crt[cx][cy] = []
    X, W, ONES, SEL, SQ, Z, Q, P = range(8)
    cbs = [
        _cb(core_set, X, TPJ, 4096, f32),
        _cb(core_set, W, 4 * TPJ, 4096, f32),
        _cb(core_set, ONES, 1, 4096, f32),
        _cb(core_set, SEL, 5, 4096, f32),
        _cb(core_set, SQ, TPJ, 4096, f32),
        _cb(core_set, Z, 4, 4096, f32),
        _cb(core_set, Q, 1, 4096, f32),
        _cb(core_set, P, 1, 4096, f32),
    ]
    reader = _kernel(
        "mhc_proj2_reader.cpp",
        core_set,
        [X, W, TG, plan.NJ, TPJ] + _acc(x) + _acc(wt),
        rrt,
        None,
        ttnn.ReaderConfigDescriptor(),
        common=[x.buffer_address(), wt.buffer_address()],
    )
    writer = _kernel(
        "mhc_proj2_writer.cpp",
        core_set,
        [ONES, SEL, P, ncol, TG] + _acc(part),
        wrt,
        None,
        ttnn.WriterConfigDescriptor(),
        common=[part.buffer_address()],
    )
    compute = _kernel(
        "mhc_proj2_compute.cpp",
        core_set,
        [X, W, ONES, SEL, SQ, Z, Q, P, TPJ],
        crt,
        None,
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(0x5B1, T, D, S, TPJ, G, ncol, tuple(_acc(x)), tuple(_acc(wt)), tuple(_acc(part)))
    ttnn.generic_op([x, wt, part], prog)
    return part


def mhc_post2(part, consts, plan, iters, eps, ss_eps, fidelity=ttnn.MathFidelity.HiFi4):
    """part [NPG,1,32,32] fp32 (from mhc_proj), consts [9,32,32] fp32 (see DSV41MHC) -> pre [T,1,1,4], post [T,1,4,1], comb [T,1,4,4]."""
    T = plan.T
    mesh = part.device()
    NPG, G, PGPG = plan.NPG, plan.G, plan.PGPG
    NCONST = int(consts.shape[0])
    mk = lambda shape: ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), f32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    pre, post, comb = mk([T, 1, 1, 4]), mk([T, 1, 4, 1]), mk([T, 1, 4, 4])
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    CONSTS, Q, SSEL, RAW, MIXES, PP, COMB, OUT, LOG, SKIN = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    TB = min(T, 8)
    cbs = [
        _cb(core_set, CONSTS, NCONST, 4096, f32),
        _cb(core_set, Q, NPG, 4096, f32),
        _cb(core_set, SSEL, G, 4096, f32),
        _cb(core_set, RAW, 1, 4096, f32),
        _cb(core_set, MIXES, 1, 4096, f32),
        _cb(core_set, PP, 1, 4096, f32),
        _cb(core_set, COMB, 1, 4096, f32),
        _cb(core_set, OUT, 3 * TB, 4096, f32),
    ]
    cbs += [_cb(core_set, LOG, 1, 4096, f32), _cb(core_set, SKIN, 1, 4096, f32)]
    empty = ttnn.RuntimeArgs()
    reader = _kernel(
        "mhc_post2_reader.cpp",
        core_set,
        [CONSTS, Q, SSEL, NPG, NCONST, G, plan.TG, plan.PPT, LOG, SKIN, T] + _acc(consts) + _acc(part),
        empty,
        None,
        ttnn.ReaderConfigDescriptor(),
        common=[consts.buffer_address(), part.buffer_address()],
    )
    writer = _kernel(
        "mhc_post2_writer.cpp",
        core_set,
        [PP, COMB, OUT, T] + _acc(pre) + _acc(post) + _acc(comb),
        empty,
        None,
        ttnn.WriterConfigDescriptor(),
        common=[pre.buffer_address(), post.buffer_address(), comb.buffer_address()],
    )
    compute = _kernel(
        "mhc_post2_compute.cpp",
        core_set,
        [iters, _bits(eps), NPG, _bits(ss_eps), PGPG],
        empty,
        None,
        ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x5B2,
        str(fidelity),
        T,
        NPG,
        iters,
        _bits(eps),
        _bits(ss_eps),
        tuple(_acc(consts)),
        tuple(_acc(part)),
        tuple(_acc(pre)),
        tuple(_acc(post)),
        tuple(_acc(comb)),
    )
    ttnn.generic_op([part, consts, pre, post, comb], prog)
    return pre, post, comb
