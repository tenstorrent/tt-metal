# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""mHC expand fused with the NEXT projection (one generic_op):  x_new = post * (y [+ y2]) + comb^T x  and the packed projection
partials of x_new (mhc_mixes2.mhc_proj2 layout), so the following mhc_post2 needs no separate projection program and no re-read of
x_new from DRAM.  Core (g, r) = projection plan core: token group g (TG tokens), column tiles [r*TPJ, (r+1)*TPJ) of all four streams.
y: token-row layout [1,1,T,D] bf16 or fp32 TILE; y2 (optional): [1,1,T,D] fp32 TILE; x, post, comb as in mhc_expand."""

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash, _kernel
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_mixes2 import _grid_cores

f32 = ttnn.float32


def mhc_expand_proj(y, x, post, comb, y2, wt, ncol, plan):
    T, _, n, D = (int(v) for v in x.shape)
    assert n == 4 and T == plan.T and D == plan.D
    TG, G, S, TPJ, PPT, PGPG, NT = plan.TG, plan.G, plan.S, plan.TPJ, plan.PPT, plan.PGPG, plan.NJ
    for t in (x, post, comb) + ((y2,) if y2 is not None else ()):
        assert t.dtype == f32 and t.layout == ttnn.TILE_LAYOUT
    assert (
        y.layout == ttnn.TILE_LAYOUT
        and y.dtype in (f32, ttnn.bfloat16)
        and int(y.shape[2]) == T
        and int(y.shape[0]) == 1
    )
    y_bf16 = y.dtype == ttnn.bfloat16
    has_y2 = y2 is not None
    assert not has_y2 or (int(y2.shape[2]) == T and int(y2.shape[0]) == 1)
    y2t = y2 if has_y2 else y
    mesh = x.device()
    cores = _grid_cores(mesh, plan.cores)
    assert len(cores) == plan.cores
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([T, 1, n, D]), f32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
    part = ttnn.allocate_tensor_on_device(
        ttnn.Shape([plan.NPG, 1, 32, 32]), f32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (cx, cy) in enumerate(cores):
        g, r = divmod(c, S)
        rrt[cx][cy] = [r, g * TG]
        wrt[cx][cy] = [r, g * TG, g * PGPG + r // PPT, (r % PPT) * TG]
        crt[cx][cy] = []
    B1, B2, A, W, SC, XN, XW, ONES, SEL, SQ, Z, Q, P, STG = range(14)
    NSTG = min(8, TG * TPJ)
    scr_bytes = TG * 512 + (2 * TPJ * TG * 32 if y_bf16 else 0)
    cbs = [
        _cb(core_set, B1, TPJ, 4096, f32),
        _cb(core_set, B2, TPJ, 4096, f32),
        _cb(core_set, A, 2, 4096, f32),
        _cb(core_set, W, 4 * TPJ, 4096, f32),
        _cb(core_set, SC, (scr_bytes + 4095) // 4096, 4096, f32),
        _cb(core_set, XN, TPJ, 4096, f32),
        _cb(core_set, XW, TPJ, 4096, f32),
        _cb(core_set, ONES, 1, 4096, f32),
        _cb(core_set, SEL, 5, 4096, f32),
        _cb(core_set, SQ, TPJ, 4096, f32),
        _cb(core_set, Z, 4, 4096, f32),
        _cb(core_set, Q, 1, 4096, f32),
        _cb(core_set, P, 1, 4096, f32),
        _cb(core_set, STG, NSTG, 4096, f32),
    ]
    reader = _kernel(
        "mhc_ep_reader.cpp",
        core_set,
        [B1, B2, A, W, SC, T, TG, NT, TPJ, int(y_bf16), int(has_y2)]
        + _acc(x)
        + _acc(y)
        + _acc(y2t)
        + _acc(comb)
        + _acc(post)
        + _acc(wt),
        rrt,
        None,
        ttnn.ReaderConfigDescriptor(),
        common=[
            x.buffer_address(),
            y.buffer_address(),
            y2t.buffer_address(),
            comb.buffer_address(),
            post.buffer_address(),
            wt.buffer_address(),
        ],
    )
    writer = _kernel(
        "mhc_ep_writer.cpp",
        core_set,
        [ONES, SEL, XW, STG, P, ncol, TG, TPJ, NT, NSTG] + _acc(out) + _acc(part),
        wrt,
        None,
        ttnn.WriterConfigDescriptor(),
        common=[out.buffer_address(), part.buffer_address()],
    )
    compute = _kernel(
        "mhc_ep_compute.cpp",
        core_set,
        [B1, B2, A, W, XN, ONES, SEL, SQ, Z, Q, P, TPJ, XW],
        crt,
        None,
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x5D1,
        T,
        D,
        S,
        TPJ,
        G,
        ncol,
        int(y_bf16),
        int(has_y2),
        tuple(_acc(x)),
        tuple(_acc(y)),
        tuple(_acc(y2t)),
        tuple(_acc(comb)),
        tuple(_acc(post)),
        tuple(_acc(wt)),
        tuple(_acc(out)),
        tuple(_acc(part)),
    )
    ttnn.generic_op([x, y, y2t, comb, post, wt, out, part], prog)
    return out, part
