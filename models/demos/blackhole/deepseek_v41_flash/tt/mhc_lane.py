# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LANEPAR: the attention-side mHC expand fused INTO the front of the ffn-side sub-block (mhc_pre.mhc_mixes_cn with an expand prologue), ONE program.
The projection cores compute the new stream tiles x2 = post * y + comb^T x (the mhc_expand5 math, one fp32 tile matmul per token group and column tile) for exactly the
column tiles they project; the tiles go straight into the projection / sum-of-squares matmuls, to the collapse+norm cores (FRONT_FAST forward) and, as the two valid
256 B pieces per token and tile, into the pooled stream buffer (the residual of the ffn-side expand). No expand program, no DRAM round trip of x2, no re-read of x2.
The arithmetic of every tile is unchanged (bit-identical to mhc_expand3 + mhc_mixes_cn); only the data movement differs.

Core map as mhc_pre: collapse+norm = the 4x8 rectangle (cx 0..3); projection cores = columns >= 4; post core(s) below the rectangle."""

import os
import struct

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash, _kernel, compact_extra

f32, bf16 = ttnn.float32, ttnn.bfloat16
_PRE_MAX = int(
    os.environ.get("DSV41_MHC_PRE_LIMIT", "32")
)  # largest T the fused front is built for (callers gate with pre_fused_ok; DSV41_MHC_PRE16 / drafter pre_max)


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _merged_set(cores):
    """CoreRangeSet of the logical cores, merged into as few rectangles as possible (consecutive cores of a row -> one range, equal row spans of consecutive rows -> one rectangle):
    the dispatcher writes the program's CB / semaphore / kernel configuration once per range"""
    rows = {}
    for cx, cy in sorted(set(cores), key=lambda c: (c[1], c[0])):
        spans = rows.setdefault(cy, [])
        if spans and spans[-1][1] == cx - 1:
            spans[-1][1] = cx
        else:
            spans.append([cx, cx])
    rects = []  # (x0, x1, y0, y1)
    for cy in sorted(rows):
        for x0, x1 in rows[cy]:
            for r in rects:
                if r[0] == x0 and r[1] == x1 and r[3] == cy - 1:
                    r[3] = cy
                    break
            else:
                rects.append([x0, x1, cy, cy])
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, x1, y0, y1 in rects]
    )


def lane_ok(T, plan, y):
    """the fused expand + front is built for: bf16 token-row y [1,1,T,D], T <= 16, token groups of 4 (T <= 4: the one 8-token group holds one valid quad)"""
    return (
        y.dtype == bf16
        and y.layout == ttnn.TILE_LAYOUT
        and int(y.shape[0]) == 1
        and int(y.shape[2]) == T
        and T <= 16
        and (plan.TG == 4 or (plan.TG == 8 and T <= 4))
    )


def mhc_expand_front(
    y,
    x,
    post_a,
    comb_a,
    out,
    pre_in,
    w4,
    wt,
    consts,
    plan,
    ncol,
    iters,
    eps,
    ss_eps,
    norm_eps,
    emit_rm=False,
    fidelity=ttnn.MathFidelity.HiFi4,
):
    """x2 = expand(y, x, post_a, comb_a) written to `out` (pooled stream buffer, valid rows only) and the front of the ffn sub-block on x2:
    -> (out, pre [T,1,1,4], post [T,1,4,1], comb [T,1,4,4], h bf16 [1,1,T,D] (norm applied), rm bf16 [T,1,1,D] or None)
    """
    T, _, n, D = (int(v) for v in x.shape)
    assert n == 4 and T == plan.tv and D == plan.D and lane_ok(T, plan, y)
    assert tuple(int(v) for v in out.shape) == (T, 1, 4, D) and out.dtype == f32
    TG, G, S, TPJ, PPT, PGPG, NPG = plan.TG, plan.G, plan.S, plan.TPJ, plan.PPT, plan.PGPG, plan.NPG
    mesh = x.device()
    grid = mesh.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    avail = [(cx, cy) for cy in range(gy) for cx in range(4, gx)]
    pcores = avail[:S]  # one core per column range handles ALL token groups (weights read once)
    assert gy >= 9, gy
    # DSV41_MHC_PAR_SINKHORN=1: one post core per token group of 4 (each runs exactly the T=4 post program on its group: rows are independent, so the
    # result is bit-identical); the post-core work (partial sums, mixes, permutes, SFPU Sinkhorn, output writes) no longer grows with T
    par = G >= 2 and os.environ.get("DSV41_MHC_PAR_SINKHORN", "0") == "1"
    if par:
        assert gy >= 8 + (G + 3) // 4, (gy, G)
        post_cores = [
            (g % 4, 8 + g // 4) for g in range(G)
        ]  # below the collapse+norm rectangle (rows 0..7 of columns 0..3)
    else:
        post_cores = [
            (0, 8)
        ]  # below the collapse+norm rectangle (rows 0..7 of columns 0..3): always free, the projection cores need columns >= 4
    post_core = post_cores[0]
    assert S <= len(avail), (S, len(avail))
    NPROJ = S if par else S * G  # partial-row deliveries per post core
    # DSV41_MHC_FRONT_FAST=1: the projection cores forward the x rows they gathered to the collapse+norm cores (one DRAM read of x instead of two), the collapse+norm writer
    # emits the row-major copy as one chunk per token (identical data, fewer / no small DRAM transactions)
    ff = True  # the forward of the new x tiles to the collapse+norm cores is part of this program
    TILE = 4096
    merged = True
    cs = (
        _merged_set
        if merged
        else (
            lambda cores: ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores]
            )
        )
    )
    p_set, q_set = cs(pcores), cs(post_cores)
    u_set = cs(pcores + post_cores)
    mk = lambda shape, dt=f32, lay=ttnn.TILE_LAYOUT: ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dt, lay, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    pre, post, comb = mk([T, 1, 1, 4]), mk([T, 1, 4, 1]), mk([T, 1, 4, 4])
    h = mk([1, 1, T, D], bf16)
    rm = mk([T, 1, 1, D], bf16, ttnn.ROW_MAJOR_LAYOUT) if emit_rm else None
    rm_t = rm if emit_rm else h
    _cx = (
        compact_extra(mesh, T, h) if ff else ([0, 160, 14] + _acc(h), [h.buffer_address()])
    )  # compact h for the skinny chains (DSV41_SKINNY_HC)
    pns = [mesh.worker_core_from_logical_core(ttnn.CoreCoord(*pc)) for pc in post_cores]
    pn = pns[0]
    SEMID = 0
    sems = [ttnn.SemaphoreDescriptor(id=SEMID, core_ranges=u_set, initial_value=0)]
    SEM2 = 1
    kernels, cbs = [], []

    NC, GX, GY = 32, 4, 8
    nt = D // 32
    gpc = nt // NC
    GG = (T + 7) // 8
    c_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
    p0 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
    p1 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(GX - 1, GY - 1))
    assert p1.x - p0.x == GX - 1 and p1.y - p0.y == GY - 1
    FX = 20  # x tiles of the collapse+norm cores: ONE CB over the collapse+norm rectangle and the projection cores (same L1 address everywhere), written by the projection cores
    if ff:
        sems.append(ttnn.SemaphoreDescriptor(id=SEM2, core_ranges=c_set, initial_value=0))

    # ---------------- projection cores (proj2 reader / compute, proj3 writer) ----------------
    rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for r, (cx, cy) in enumerate(pcores):
        nvs = [max(0, min(TG, plan.tv - g * TG)) for g in range(max(G, 2))]
        rrt[cx][cy] = [r] + nvs
        if par:
            wrt[cx][cy] = [r // PPT, (r % PPT) * TG] + [int(v) for q in pns for v in (q.x, q.y)]
        else:
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
    # fused expand: B tiles (old x rows + y rows), A tiles (block-diagonal comb / post), raw bf16 y rows, comb / post pieces (CB indices 18, 19, 21, 22; 20 = FX)
    EB, EA, EYS, ECS = 18, 19, 21, 22
    cbs += [
        _cb(p_set, EB, G * TPJ, TILE, f32),
        _cb(p_set, EA, G, TILE, f32),
        _cb(p_set, EYS, max(1, (G * TPJ * 256 + TILE - 1) // TILE), TILE, f32),
        _cb(p_set, ECS, max(1, (G * 4 * 512 + TILE - 1) // TILE), TILE, f32),
    ]
    NCONST = int(consts.shape[0])
    kernels += [
        _kernel(
            "mhc_lf_reader.cpp",
            p_set,
            [X, W, TG, plan.NJ, TPJ, G, FX, SEM2, gpc, GG, int(p0.x), int(p0.y), GX, EB, EA, EYS, ECS]
            + _acc(x)
            + _acc(wt)
            + _acc(y)
            + _acc(comb_a)
            + _acc(post_a)
            + _acc(out),
            rrt,
            None,
            ttnn.ReaderConfigDescriptor(),
            common=[
                x.buffer_address(),
                wt.buffer_address(),
                y.buffer_address(),
                comb_a.buffer_address(),
                post_a.buffer_address(),
                out.buffer_address(),
            ],
        ),
        _kernel(
            "mhc_proj4_writer.cpp" if par else "mhc_proj3_writer.cpp",
            p_set,
            [ONES, SEL, P, ncol, TG, 1, SEMID, G, PGPG],
            wrt,
            None,
            ttnn.WriterConfigDescriptor(),
        ),
        _kernel(
            "mhc_lf_compute.cpp",
            p_set,
            [X, W, ONES, SEL, SQ, Z, Q1, P, TPJ, G, EA, EB],
            crt,
            None,
            ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        ),
    ]

    # ---------------- post core (post2 compute / writer, post3 reader) ----------------
    # per post core: the pages / groups / tokens it handles (par: its own group: PGPG pages, 1 group, TG tokens)
    NPGL, GL, TL = (PGPG, 1, TG) if par else (NPG, G, plan.T)
    TB = min(TL if par else T, 8)
    CONSTS, Q, SSEL, RAW, MIXES, PP, COMB, OUT, LOG, SKIN = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    cbs += [
        _cb(u_set, Q, NPGL, TILE, f32),
        _cb(q_set, CONSTS, NCONST, TILE, f32),
        _cb(q_set, SSEL, GL, TILE, f32),
        _cb(q_set, RAW, 1, TILE, f32),
        _cb(q_set, MIXES, 1, TILE, f32),
        _cb(q_set, PP, 1, TILE, f32),
        _cb(q_set, COMB, 1, TILE, f32),
        _cb(q_set, OUT, 3 * TB, TILE, f32),
        _cb(q_set, LOG, 1, TILE, f32),
        _cb(q_set, SKIN, 1, TILE, f32),
    ]
    empty = ttnn.RuntimeArgs()
    qrt = ttnn.RuntimeArgs()
    for g, (cx, cy) in enumerate(post_cores):
        qrt[cx][cy] = [g * TG, max(0, min(TG, plan.tv - g * TG))]  # first token and valid tokens of the group
    if not par:
        qrt = empty
    kernels += [
        _kernel(
            "mhc_post3_reader.cpp",
            q_set,
            [CONSTS, Q, SSEL, NPGL, NCONST, GL, plan.TG, plan.PPT, LOG, SKIN, TL, SEMID, NPROJ] + _acc(consts),
            empty,
            None,
            ttnn.ReaderConfigDescriptor(),
            common=[consts.buffer_address()],
        ),
        _kernel(
            "mhc_post4_writer.cpp" if par else "mhc_post2_writer.cpp",
            q_set,
            [PP, COMB, OUT, TL if par else T] + _acc(pre) + _acc(post) + _acc(comb),
            qrt,
            None,
            ttnn.WriterConfigDescriptor(),
            common=[pre.buffer_address(), post.buffer_address(), comb.buffer_address()],
        ),
        _kernel(
            "mhc_post2_compute.cpp",
            q_set,
            [iters, _bits(eps), NPGL, _bits(ss_eps), PGPG],
            empty,
            None,
            ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True),
        ),
    ]

    # ---------------- collapse + norm cores: the unchanged mhc_cn kernels on the 4x8 rectangle ----------------
    c_cores = [(cx, cy) for cy in range(GY) for cx in range(GX)]
    rt = ttnn.RuntimeArgs()
    for k, (cx, cy) in enumerate(c_cores):
        rt[cx][cy] = [k]
    A, X_, ONES_, HS, SQ_, P_, TOT, W_, RS, TMP, O, RMB, PRE, SLOT = range(14)
    if ff:
        X_ = FX
    # the cn kernels use CB indices 0..13: they live on the disjoint rectangle (indices may repeat across disjoint core sets)
    cbs += [
        _cb(c_set, A, GG, TILE, f32),
        _cb(cs(pcores + c_cores) if ff else c_set, X_, gpc * GG, TILE, f32),
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
            "mhc_cn2_reader.cpp" if ff else "mhc_cn_reader.cpp",
            c_set,
            [A, X_, W_, PRE, P_, SLOT, TOT, T, nt, GG, gpc, NC, p0.x, p0.y, GX]
            + ([SEM2, (T + 3) // 4] + _acc(pre_in) + _acc(w4) if ff else _acc(x) + _acc(pre_in) + _acc(w4)),
            rt,
            None,
            ttnn.ReaderConfigDescriptor(),
            common=[pre_in.buffer_address(), w4.buffer_address()]
            if ff
            else [x.buffer_address(), pre_in.buffer_address(), w4.buffer_address()],
        ),
        _kernel(
            "mhc_cn2_writer.cpp" if ff else "mhc_cn_writer.cpp",
            c_set,
            [O, RMB, int(emit_rm), T, D * 2, ONES_, gpc] + _acc(h) + _acc(rm_t) + _cx[0],
            rt,
            None,
            ttnn.WriterConfigDescriptor(),
            common=[h.buffer_address(), rm_t.buffer_address()] + _cx[1],
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
        0x5D3,
        tuple(_cx[0]),
        int(par),
        int(ff),
        int(merged),
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
        tuple(_acc(y)),
        tuple(_acc(comb_a)),
        tuple(_acc(post_a)),
        tuple(_acc(out)),
    )
    ttnn.generic_op([x, y, comb_a, post_a, pre_in, w4, wt, consts, out, pre, post, comb, h, rm_t], prog)
    return out, pre, post, comb, h, rm
