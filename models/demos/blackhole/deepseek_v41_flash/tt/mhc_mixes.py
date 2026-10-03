# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused mHC mixes (two generic_ops): projection x_flat @ fn^T (+ sum of squares) split over cores, then one post kernel
(sum of partials, RMS scale, Sinkhorn parametrisation, token-major pre/post/comb)."""

import struct

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash, _kernel

f32 = ttnn.float32


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _grid_cores(mesh, n):
    grid = mesh.compute_with_storage_grid_size()
    return [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)][:n]


def mhc_proj(x, wt, ncol):
    """x [T,1,4,D] fp32 TILE; wt [S,1,Kc,32] fp32 TILE (fn chunks, S = 4*D/Kc <= #cores) -> partials [S,1,32,32] fp32:
    row t of partial k = chunk-k projection of token t (columns 0..mix_hc-1) and its sum of squares (column ncol)."""
    T, _, n, D = (int(v) for v in x.shape)
    S, Kc = int(wt.shape[0]), int(wt.shape[2])
    assert n == 4 and T <= 32 and S * Kc == n * D
    mesh = x.device()
    nt, tpc = D // 32, Kc // 32
    cps = D // Kc
    cores = _grid_cores(mesh, S)
    assert len(cores) == S
    part = ttnn.allocate_tensor_on_device(
        ttnn.Shape([S, 1, 32, 32]), f32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    rt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for k, (cx, cy) in enumerate(cores):
        rt[cx][cy] = [k]
        crt[cx][cy] = []
    A, W, ONES, SQ, P = range(5)
    cbs = [
        _cb(core_set, A, tpc, 4096, f32),
        _cb(core_set, W, tpc, 4096, f32),
        _cb(core_set, ONES, 1, 4096, f32),
        _cb(core_set, SQ, tpc, 4096, f32),
        _cb(core_set, P, 1, 4096, f32),
    ]
    reader = _kernel(
        "mhc_proj_reader.cpp",
        core_set,
        [A, W, ONES, T, nt, tpc, cps, ncol] + _acc(x) + _acc(wt),
        rt,
        None,
        ttnn.ReaderConfigDescriptor(),
        common=[x.buffer_address(), wt.buffer_address()],
    )
    writer = _kernel(
        "mhc_proj_writer.cpp",
        core_set,
        [P] + _acc(part),
        rt,
        None,
        ttnn.WriterConfigDescriptor(),
        common=[part.buffer_address()],
    )
    compute = _kernel(
        "mhc_proj_compute.cpp",
        core_set,
        [A, W, ONES, SQ, P, tpc],
        crt,
        None,
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(0x5A1, T, D, S, Kc, ncol, tuple(_acc(x)), tuple(_acc(wt)), tuple(_acc(part)))
    ttnn.generic_op([x, wt, part], prog)
    return part


def mhc_post(part, consts, T, iters, eps, ss_eps, fidelity=ttnn.MathFidelity.HiFi4):
    """part [S,1,32,32] fp32 (from mhc_proj), consts [10,32,32] fp32 (8 sinkhorn tiles, SEL_ss, identity)
    -> pre [T,1,1,4], post [T,1,4,1], comb [T,1,4,4] fp32."""
    mesh = part.device()
    S = int(part.shape[0])
    mk = lambda shape: ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), f32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    pre, post, comb = mk([T, 1, 1, 4]), mk([T, 1, 4, 1]), mk([T, 1, 4, 4])
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    RAW, CONSTS, PRE, POST, COMB, P, MIXES, OUT = 0, 1, 2, 3, 4, 5, 6, 7
    MA, MB, RS, RECIP, TMP = 16, 17, 18, 24, 25
    cbs = [
        _cb(core_set, RAW, 1, 4096, f32),
        _cb(core_set, CONSTS, 10, 4096, f32),
        _cb(core_set, PRE, 1, 4096, f32),
        _cb(core_set, POST, 1, 4096, f32),
        _cb(core_set, COMB, 1, 4096, f32),
        _cb(core_set, P, S, 4096, f32),
        _cb(core_set, MIXES, 1, 4096, f32),
        _cb(core_set, OUT, 3 * min(T, 8), 4096, f32),
    ]
    cbs += [_cb(core_set, i, 2, 4096, f32) for i in (MA, MB, RS, RECIP, TMP)]
    empty = ttnn.RuntimeArgs()
    reader = _kernel(
        "mhc_post_reader.cpp",
        core_set,
        [CONSTS, P, S, 10, T] + _acc(consts) + _acc(part),
        empty,
        None,
        ttnn.ReaderConfigDescriptor(),
        common=[consts.buffer_address(), part.buffer_address()],
    )
    writer = _kernel(
        "mhc_post_writer.cpp",
        core_set,
        [PRE, POST, COMB, OUT, T] + _acc(pre) + _acc(post) + _acc(comb),
        empty,
        None,
        ttnn.WriterConfigDescriptor(),
        common=[pre.buffer_address(), post.buffer_address(), comb.buffer_address()],
    )
    compute = _kernel(
        "mhc_post_compute.cpp",
        core_set,
        [iters, _bits(eps), S, _bits(ss_eps)],
        empty,
        None,
        ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x5A2,
        str(fidelity),
        T,
        S,
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
