# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""L02: the whole front of an mHC sub-block as ONE program: mixes (projection cores -> L1 -> one post core: Sinkhorn) next to the collapse + RMSNorm cores.
Both only need the stream x and the previous coefficient `pre_in` (V4.1 staggers pre), so they run concurrently; the program takes max(mixes chain, collapse+norm)
instead of their sum. The arithmetic is the one of mhc_proj2 / mhc_post2 / mhc_collapse_norm unchanged (same tiles, same summation order, bit-identical results); only the
data movement differs: the projection partials go straight into the post core's L1 (no DRAM round trip) signalled by a semaphore, the post core's constants are prefetched
while the projections run.

Core map: collapse+norm = the 4x8 rectangle (cx 0..3); projection cores = columns >= 4 (plan.cores of them); post core = the next one."""

import struct

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash, _kernel

f32, bf16 = ttnn.float32, ttnn.bfloat16


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def mhc_mixes_cn(
    x, pre_in, w4, wt, consts, plan, ncol, iters, eps, ss_eps, norm_eps, emit_rm=False, fidelity=ttnn.MathFidelity.HiFi4
):
    """-> (pre [T,1,1,4], post [T,1,4,1], comb [T,1,4,4], h bf16 [1,1,T,D] (norm applied), rm bf16 [T,1,1,D] or None)"""
    T, _, n, D = (int(v) for v in x.shape)
    assert n == 4 and T == plan.tv and D == plan.D and T <= 8
    TG, G, S, TPJ, PPT, PGPG, NPG = plan.TG, plan.G, plan.S, plan.TPJ, plan.PPT, plan.PGPG, plan.NPG
    mesh = x.device()
    grid = mesh.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    avail = [(cx, cy) for cy in range(gy) for cx in range(4, gx)]
    pcores = avail[:S]  # one core per column range handles ALL token groups (weights read once)
    assert gy >= 9, gy
    post_core = (
        0,
        8,
    )  # below the collapse+norm rectangle (rows 0..7 of columns 0..3): always free, the projection cores need columns >= 4
    assert S <= len(avail), (S, len(avail))
    NPROJ = S * G
    TILE = 4096
    cs = lambda cores: ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores]
    )
    p_set, q_set = cs(pcores), cs([post_core])
    u_set = cs(pcores + [post_core])
    mk = lambda shape, dt=f32, lay=ttnn.TILE_LAYOUT: ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dt, lay, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    pre, post, comb = mk([T, 1, 1, 4]), mk([T, 1, 4, 1]), mk([T, 1, 4, 4])
    h = mk([1, 1, T, D], bf16)
    rm = mk([T, 1, 1, D], bf16, ttnn.ROW_MAJOR_LAYOUT) if emit_rm else None
    rm_t = rm if emit_rm else h
    pn = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*post_core))
    SEMID = 0
    sems = [ttnn.SemaphoreDescriptor(id=SEMID, core_ranges=u_set, initial_value=0)]
    kernels, cbs = [], []

    # ---------------- projection cores (proj2 reader / compute, proj3 writer) ----------------
    rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for r, (cx, cy) in enumerate(pcores):
        nvs = [max(0, min(TG, plan.tv - g * TG)) for g in range(2)]
        rrt[cx][cy] = [r, nvs[0], nvs[1]]
        wrt[cx][cy] = [r // PPT, (r % PPT) * TG, int(pn.x), int(pn.y)]
        crt[cx][cy] = []
    # projection CB indices 10..17 (index 1 is the shared packed-partial pages: post2_compute hard-codes Q = 1)
    X, W, ONES, SEL, SQ, Z, Q1, P = range(10, 18)
    cbs += [
        _cb(p_set, X, G * TPJ, TILE, f32),
        _cb(p_set, W, 4 * TPJ, TILE, f32),
        _cb(p_set, ONES, 1, TILE, f32),
        _cb(p_set, SEL, 5, TILE, f32),
        _cb(p_set, SQ, TPJ, TILE, f32),
        _cb(p_set, Z, 4, TILE, f32),
        _cb(p_set, Q1, 1, TILE, f32),
        _cb(p_set, P, 1, TILE, f32),
    ]
    NCONST = int(consts.shape[0])
    kernels += [
        _kernel(
            "mhc_proj3_reader.cpp",
            p_set,
            [X, W, TG, plan.NJ, TPJ, G] + _acc(x) + _acc(wt),
            rrt,
            None,
            ttnn.ReaderConfigDescriptor(),
            common=[x.buffer_address(), wt.buffer_address()],
        ),
        _kernel(
            "mhc_proj3_writer.cpp",
            p_set,
            [ONES, SEL, P, ncol, TG, 1, SEMID, G, PGPG],
            wrt,
            None,
            ttnn.WriterConfigDescriptor(),
        ),
        _kernel(
            "mhc_proj3_compute.cpp",
            p_set,
            [X, W, ONES, SEL, SQ, Z, Q1, P, TPJ, G],
            crt,
            None,
            ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        ),
    ]

    # ---------------- post core (post2 compute / writer, post3 reader) ----------------
    TB = min(T, 8)
    CONSTS, Q, SSEL, RAW, MIXES, PP, COMB, OUT, LOG, SKIN = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    cbs += [
        _cb(u_set, Q, NPG, TILE, f32),
        _cb(q_set, CONSTS, NCONST, TILE, f32),
        _cb(q_set, SSEL, G, TILE, f32),
        _cb(q_set, RAW, 1, TILE, f32),
        _cb(q_set, MIXES, 1, TILE, f32),
        _cb(q_set, PP, 1, TILE, f32),
        _cb(q_set, COMB, 1, TILE, f32),
        _cb(q_set, OUT, 3 * TB, TILE, f32),
        _cb(q_set, LOG, 1, TILE, f32),
        _cb(q_set, SKIN, 1, TILE, f32),
    ]
    empty = ttnn.RuntimeArgs()
    kernels += [
        _kernel(
            "mhc_post3_reader.cpp",
            q_set,
            [CONSTS, Q, SSEL, NPG, NCONST, G, plan.TG, plan.PPT, LOG, SKIN, plan.T, SEMID, NPROJ] + _acc(consts),
            empty,
            None,
            ttnn.ReaderConfigDescriptor(),
            common=[consts.buffer_address()],
        ),
        _kernel(
            "mhc_post2_writer.cpp",
            q_set,
            [PP, COMB, OUT, T] + _acc(pre) + _acc(post) + _acc(comb),
            empty,
            None,
            ttnn.WriterConfigDescriptor(),
            common=[pre.buffer_address(), post.buffer_address(), comb.buffer_address()],
        ),
        _kernel(
            "mhc_post2_compute.cpp",
            q_set,
            [iters, _bits(eps), NPG, _bits(ss_eps), PGPG],
            empty,
            None,
            ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True),
        ),
    ]

    # ---------------- collapse + norm cores: the unchanged mhc_cn kernels on the 4x8 rectangle ----------------
    NC, GX, GY = 32, 4, 8
    nt = D // 32
    gpc = nt // NC
    GG = (T + 7) // 8
    c_cores = [(cx, cy) for cy in range(GY) for cx in range(GX)]
    c_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
    p0 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
    p1 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(GX - 1, GY - 1))
    assert p1.x - p0.x == GX - 1 and p1.y - p0.y == GY - 1
    rt = ttnn.RuntimeArgs()
    for k, (cx, cy) in enumerate(c_cores):
        rt[cx][cy] = [k]
    A, X_, ONES_, HS, SQ_, P_, TOT, W_, RS, TMP, O, RMB, PRE, SLOT = range(14)
    # the cn kernels use CB indices 0..13: they live on the disjoint rectangle (indices may repeat across disjoint core sets)
    cbs += [
        _cb(c_set, A, GG, TILE, f32),
        _cb(c_set, X_, gpc * GG, TILE, f32),
        _cb(c_set, ONES_, 1, TILE, f32),
        _cb(c_set, HS, gpc, 2048, bf16),
        _cb(c_set, SQ_, gpc, TILE, f32),
        _cb(c_set, P_, 1, TILE, f32),
        _cb(c_set, TOT, 1, TILE, f32),
        _cb(c_set, W_, gpc, TILE, f32),
        _cb(c_set, RS, 1, TILE, f32),
        _cb(c_set, TMP, gpc, TILE, f32),
        _cb(c_set, O, gpc, 2048, bf16),
        _cb(c_set, RMB, max(1, (gpc * T * 64 + 4095) // 4096), TILE, bf16),
        _cb(c_set, PRE, 1, TILE, f32),
        _cb(c_set, SLOT, 2, TILE, f32),
    ]
    eps_bits = _bits(D * norm_eps)
    kernels += [
        _kernel(
            "mhc_cn_reader.cpp",
            c_set,
            [A, X_, W_, PRE, P_, SLOT, TOT, T, nt, GG, gpc, NC, p0.x, p0.y, GX] + _acc(x) + _acc(pre_in) + _acc(w4),
            rt,
            None,
            ttnn.ReaderConfigDescriptor(),
            common=[x.buffer_address(), pre_in.buffer_address(), w4.buffer_address()],
        ),
        _kernel(
            "mhc_cn_writer.cpp",
            c_set,
            [O, RMB, int(emit_rm), T, D * 2, ONES_, gpc] + _acc(h) + _acc(rm_t),
            rt,
            None,
            ttnn.WriterConfigDescriptor(),
            common=[h.buffer_address(), rm_t.buffer_address()],
        ),
        _kernel(
            "mhc_cn_compute.cpp",
            c_set,
            [A, X_, ONES_, HS, SQ_, P_, TOT, W_, RS, TMP, O, T, GG, gpc, eps_bits],
            ttnn.RuntimeArgs(),
            None,
            ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        ),
    ]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    prog.custom_program_hash = _hash(
        0x5D2,
        T,
        plan.T,
        D,
        S,
        TPJ,
        G,
        ncol,
        iters,
        _bits(eps),
        _bits(ss_eps),
        eps_bits,
        emit_rm,
        str(fidelity),
        tuple(_acc(x)),
        tuple(_acc(wt)),
        tuple(_acc(consts)),
        tuple(_acc(pre_in)),
        tuple(_acc(w4)),
        tuple(_acc(pre)),
        tuple(_acc(post)),
        tuple(_acc(comb)),
        tuple(_acc(h)),
        tuple(_acc(rm_t)),
    )
    ttnn.generic_op([x, pre_in, w4, wt, consts, pre, post, comb, h, rm_t], prog)
    return pre, post, comb, h, rm
