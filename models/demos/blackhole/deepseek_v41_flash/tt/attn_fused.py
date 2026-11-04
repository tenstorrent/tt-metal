# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""JIT fused attention helpers (``ttnn.generic_op``), see tt/attn_kernels/*.cpp.  Env flags (each default off until gated):
DSV41_ATTN_FUSED_ROPE: partial RoPE as one program (rotation-matrix matmul per 32-col tile) in place of mul + 512x512 matmul + addcmul."""

import os

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_kernels")


def flag(name, default="0"):
    return os.environ.get(name, default) == "1"


def _kernel(name, core_set, ct, rt, cfg, common):
    return ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/{name}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=common,
        config=cfg,
    )


def rope_inplace(x, c, s, users, rows_layout=False):
    """Rotate dims 448..511 of x IN PLACE (adjacent pairs).  x bf16 TILE DRAM: heads layout [1,T,32,512] (page = t*16+n; tables
    [1,T,1,512] row 0 of page t*16+n) or rows layout [1,1,T,512] (page = n; every user shares table row 0 of tables [1,1,B,512]).
    ``users`` = T (heads) or 1 (rows).  c, s: bf16 TILE tables (s may be the negated table for the inverse rotation)."""
    assert x.dtype == ttnn.bfloat16 and c.dtype == ttnn.bfloat16 and s.dtype == ttnn.bfloat16
    mesh = x.device()
    XS = 0 if rows_layout else 16
    n = users * 2
    grid = mesh.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)][:n]
    assert len(cores) == n
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    rt = ttnn.RuntimeArgs()
    for w, (cx, cy) in enumerate(cores):
        rt[cx][cy] = [w]
    bf = ttnn.bfloat16
    X, R, SC, O = range(4)
    cbs = [
        _cb(core_set, X, 1, 2048, bf),
        _cb(core_set, R, 1, 2048, bf),
        _cb(core_set, SC, 1, 256, bf),
        _cb(core_set, O, 1, 2048, bf),
    ]
    TS = XS
    reader = _kernel(
        "rope_reader.cpp",
        core_set,
        [X, R, SC, XS, TS] + _acc(x) + _acc(c) + _acc(s),
        rt,
        ttnn.ReaderConfigDescriptor(),
        [x.buffer_address(), c.buffer_address(), s.buffer_address()],
    )
    writer = _kernel(
        "rope_writer.cpp", core_set, [O, XS] + _acc(x), rt, ttnn.WriterConfigDescriptor(), [x.buffer_address()]
    )
    compute = _kernel(
        "rope_compute.cpp",
        core_set,
        [X, R, O],
        ttnn.RuntimeArgs(),
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        [],
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(0xA70, users, XS, tuple(_acc(x)), tuple(_acc(c)), tuple(_acc(s)))
    ttnn.generic_op([c, s, x], prog)
    return x


_MR_LOGGED = False


def _crs(cores):
    if (
        os.environ.get("DSV41_ROPE_MERGED_RANGES", "1") == "1"
    ):  # overlap2: one CoreRange per row span instead of per core (the dispatcher writes the program config once per range)
        from models.demos.blackhole.deepseek_v41_flash.tt.mhc_pre import _merged_set

        r = _merged_set(cores)
        global _MR_LOGGED
        if not _MR_LOGGED:
            _MR_LOGGED = True
            print(f"OVERLAP2 rope_concat merged ranges: {len(cores)} cores", flush=True)
        return r
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])


COMPACT_KSTRIDE = 160  # tiles per row tile of the compact layout (5 pages of 16 KiB, a multiple of the page; the 144 real tiles + 16 pad)


def compact_pb():
    """page bytes of the compact buffer (env DSV41_SKINNY_XPAGE, power of two, default 16384)"""
    return int(os.environ.get("DSV41_SKINNY_XPAGE", "16384"))


def rope_concat(o, c, s, users, nh, compact=False):
    """Post-SDPA glue (DSV41_ATTN_POSTFUSE): o = SDPA output [1,32,T,512] bf16 ROW_MAJOR (page = h*T + t, ``users`` = T <= 32), c / s = the
    [1,T,1,512] bf16 TILE cos / negated-sin tables -> [1,1,32,nh*512] bf16 TILE DRAM (row t = user t, columns h*512 + d), the dims 448..511 of
    every head rotated exactly like ``rope_inplace`` (x @ R).  Replaces permute + tilize + rope_inplace + to_memory_config + nlp_concat_heads_decode
    + to_memory_config.  Rows >= T of the output tiles are not written (stale memory, like the stock chain; ignored by wo_a).
    """
    assert (
        o.dtype == ttnn.bfloat16
        and o.layout == ttnn.ROW_MAJOR_LAYOUT
        and c.dtype == ttnn.bfloat16
        and s.dtype == ttnn.bfloat16
    )
    T = users
    assert 1 <= T <= 32
    mesh = o.device()
    ROW = int(o.shape[-1]) * 2
    assert int(o.shape[-1]) == 512 and int(o.shape[-2]) == T
    grid = mesh.compute_with_storage_grid_size()
    allc = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)]
    rc = allc[: 2 * T]
    npass = min(nh * T, len(allc) - 2 * T)
    pc = allc[2 * T : 2 * T + npass]
    if compact:
        # skinny chains (tt/skinny.py): tiny 8 x 32 tiles, row tile m = t >> 3 at tile m * COMPACT_KSTRIDE, as a ROW_MAJOR buffer of 16 KiB pages (one contiguous 80 KiB read per row tile)
        assert nh * 16 <= COMPACT_KSTRIDE and T <= 16
        mt = 1 if T <= 8 else 2
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([81920 * mt // compact_pb(), compact_pb() // 2]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh,
            ttnn.L1_MEMORY_CONFIG if os.environ.get("DSV41_SKINNY_XL1") == "1" else ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 32, nh * 512]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            mesh,
            ttnn.L1_MEMORY_CONFIG if os.environ.get("DSV41_RC_L1") == "1" else ttnn.DRAM_MEMORY_CONFIG,
        )
    cmp_args = [int(compact), COMPACT_KSTRIDE, compact_pb().bit_length() - 1]
    bf = ttnn.bfloat16
    X, R, SC, XS, O, P, R2, RI, M, PP = range(10)
    unf = T > 8 and T % 8 != 0  # the stock spec _finish rotates these T with the unfused addcmul formula
    _pbc = (
        unf and os.environ.get("DSV41_OL_POSTBCAST", "0") == "1" and o.memory_config().buffer_type == ttnn.BufferType.L1
    )  # otherlayers: raw cos / sin tiles broadcast by the unpacker (ol_kernels/post2_reader.cpp, qpost2_compute.cpp)
    rset, pset = _crs(rc), _crs(pc)
    cbs = [
        _cb(rset, X, 1, 2048, bf),
        _cb(rset, R, 1, 2048, bf),
        _cb(rset, SC, 1, 256, bf),
        _cb(rset, XS, 1, 1024, bf),
        _cb(rset, O, 1, 2048, bf),
        _cb(pset, P, 1, 1024, bf),
    ] + [
        _cb(rset, R2, 1, 2048, bf),
        _cb(rset, RI, 1, 2048, bf),
        _cb(rset, M, 1, 2048, bf),
        _cb(rset, PP, 1, 2048, bf),
    ]
    rt = ttnn.RuntimeArgs()
    for w, (cx, cy) in enumerate(rc):
        rt[cx][cy] = [w]
    prt = ttnn.RuntimeArgs()
    units, base, rem, u0 = nh * T, (nh * T) // npass, (nh * T) % npass, 0
    for i, (cx, cy) in enumerate(pc):
        cnt = base + (1 if i < rem else 0)
        prt[cx][cy] = [u0, cnt]
        u0 += cnt
    reader = _kernel(
        "../ol_kernels/post2_reader.cpp" if _pbc else "post_reader.cpp",
        rset,
        [
            X,
            R,
            (int(o.memory_config().buffer_type == ttnn.BufferType.L1) if _pbc else SC),
            XS,
            T,
            nh,
            16,
            ROW,
            int(unf),
            R2,
            RI,
        ]
        + _acc(o)
        + _acc(c)
        + _acc(s),
        rt,
        ttnn.ReaderConfigDescriptor(),
        [o.buffer_address(), c.buffer_address(), s.buffer_address()],
    )
    writer = _kernel(
        "post_writer.cpp",
        rset,
        [O, nh] + _acc(out) + cmp_args,
        rt,
        ttnn.WriterConfigDescriptor(),
        [out.buffer_address()],
    )
    compute = _kernel(
        "../ol_kernels/qpost2_compute.cpp" if _pbc else "post_compute.cpp",
        rset,
        [X, R, O, int(unf), R2, RI, M, PP],
        ttnn.RuntimeArgs(),
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=not unf),
        [],
    )
    passk = _kernel(
        "post_pass.cpp",
        pset,
        [P, T, ROW] + _acc(o) + _acc(out) + cmp_args,
        prt,
        ttnn.ReaderConfigDescriptor(),
        [o.buffer_address(), out.buffer_address()],
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute, passk], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0xA71, _pbc, T, nh, unf, compact, tuple(_acc(o)), tuple(_acc(c)), tuple(_acc(s)), tuple(_acc(out))
    )
    ttnn.generic_op([o, c, s, out], prog)
    return out
