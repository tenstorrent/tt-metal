# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused prefill attention glue (``ttnn.generic_op`` programs, JIT kernels in tt/fa_kernels/), all default OFF, switches read when used:

DSV41_FA_QPRE=1   q heads + partial RoPE (+ the 24 zero heads pad and the row-major layout of the sparse_sdpa input) in ONE program in place of
                  nlp_create_qkv_heads + rope_pe (slice/multiply/matmul/multiply/add/slice/concat) + per-user slice/to_layout/concat(zero heads).
DSV41_FA_KV=1     the RoPE of the chunk's kv latent (rope_pe on [U,1,C,512]) in one program.
DSV41_FA_POST=1   sparse_sdpa output (row-major [1,32,C,512] per user) -> inverse RoPE + head concat + the zero head slot -> the [1,1,R,4608] input of wo_a in one program
                  in place of slice/to_layout/concat + rope_pe + nlp_concat_heads + concat(zero).

One block = one tile row (32 token rows) x one head: reader (tt/fa_kernels/fa_rope_reader.cpp), compute (tilize / untilize + RoPE with the SAME bf16 stages as the
unfused ops: x*c, x@P, *s, +), writer.  Persistent buffers (padded q per user, post output) are allocated once and referenced from module caches.
"""

import os

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_kernels")
HEAD_DIM = 512
PAD_HEADS = 32
_BUF = {}


def on(name):
    return os.environ.get(name, "0") == "1"


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


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])


def rope_blocks(
    xs,
    c,
    s,
    P,
    outs,
    *,
    U,
    C,
    nh,
    in_rm,
    out_rm,
    in_str,
    out_str,
    tab_per_user,
    rope=True,
    max_cores=None,
    nt=16,
    mode=None,
    fp32=None,
):
    """xs: tile tensor (in_rm False) or list of U row-major [1,32,C,512] tensors; outs: tile tensor (out_rm False) or list of U row-major tensors.
    in_str / out_str = (user stride, head stride, tile-row stride, first column tile) in tile pages. c, s: [Ut,1,C,512] bf16 TILE tables, P: [1,1,64,64] pair swap.
    """
    if mode is None:
        mode = int(os.environ.get("DSV41_FA_ROPE_MODE", "0"))
    if fp32 is None:
        fp32 = os.environ.get("DSV41_FA_FP32", "1") == "1"
    xs = list(xs) if in_rm else [xs]
    outs = list(outs) if out_rm else [outs]
    mesh = c.device()
    R = U * C
    cb32 = C // 32
    nb = (R // 32) * nh
    grid = mesh.compute_with_storage_grid_size()
    allc = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)]
    ncores = min(nb, max_cores or len(allc))
    cores = allc[:ncores]
    core_set = _crs(cores)
    bf = ttnn.bfloat16
    IN, X, CC, SS, PP, T1, T2, T3, Y, RM = range(10)
    cbs = [
        _cb(core_set, IN, 2 * nt, 2048, bf),
        _cb(core_set, CC, 4, 2048, bf),
        _cb(core_set, SS, 4, 2048, bf),
        _cb(core_set, PP, 1, 2048, bf),
        _cb(core_set, T1, 2, 2048, bf),
        _cb(core_set, T2, 2, 2048, bf),
        _cb(core_set, T3, 2, 2048, bf),
        _cb(core_set, Y, 2 * nt, 2048, bf),
    ]
    if in_rm:
        cbs.append(_cb(core_set, X, nt, 2048, bf))
    if out_rm:
        cbs.append(_cb(core_set, RM, 2 * nt, 2048, bf))
    cb_x = X if in_rm else IN
    cb_out = RM if out_rm else Y
    base, rem = nb // ncores, nb % ncores
    rt_r, rt_w, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    b0 = 0
    for i, (cx, cy) in enumerate(cores):
        n = base + (1 if i < rem else 0)
        rt_r[cx][cy] = [b0, n]
        rt_w[cx][cy] = [b0, n]
        rt_c[cx][cy] = [n]
        b0 += n
    ct_r = [IN, CC, SS, PP, int(in_rm), nh, cb32, C, int(tab_per_user), *in_str, len(xs), nt]
    ct_r += _acc(xs[0]) + _acc(c) + _acc(s) + _acc(P)
    reader = _kernel(
        "fa_rope_reader.cpp",
        core_set,
        ct_r,
        rt_r,
        ttnn.ReaderConfigDescriptor(),
        [c.buffer_address(), s.buffer_address(), P.buffer_address()] + [x.buffer_address() for x in xs],
    )
    ct_w = [cb_out, int(out_rm), nh, cb32, C, *out_str, nt] + _acc(outs[0])
    writer = _kernel(
        "fa_rope_writer.cpp", core_set, ct_w, rt_w, ttnn.WriterConfigDescriptor(), [o.buffer_address() for o in outs]
    )
    ct_c = [IN, cb_x, CC, SS, PP, T1, T2, T3, Y, RM, int(in_rm), int(out_rm), int(rope), nt, mode]
    compute = _kernel(
        "fa_rope_compute.cpp",
        core_set,
        ct_c,
        rt_c,
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=bool(fp32)),
        [],
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0xFA1,
        U,
        C,
        nh,
        int(in_rm),
        int(out_rm),
        int(rope),
        int(tab_per_user),
        tuple(in_str),
        tuple(out_str),
        len(xs),
        mode,
        bool(fp32),
        nt,
        ncores,
        tuple(_acc(xs[0])),
        tuple(_acc(c)),
        tuple(_acc(s)),
        tuple(_acc(P)),
        tuple(_acc(outs[0])),
    )
    ttnn.generic_op([c, s, P] + xs + outs, prog)
    return outs[0] if not out_rm else outs


def _zeros(md, shape, layout):
    return ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=layout, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def q_pad_bufs(md, U, C):
    """U persistent row-major [1,32,C,512] bf16 buffers: heads 0..7 are rewritten by every qprep, heads 8..31 stay zero."""
    key = ("qpad", id(md), U, C)
    if key not in _BUF:
        _BUF[key] = [_zeros(md, [1, PAD_HEADS, C, HEAD_DIM], ttnn.ROW_MAJOR_LAYOUT) for _ in range(U)]
    return _BUF[key]


def post_buf(md, R, nh):
    """persistent [1,1,R,(nh+1)*512] tile buffer: column tiles 0..15 (the zero head slot of the wo_a input) stay zero, the heads go to 16.."""
    key = ("post", id(md), R, nh)
    if key not in _BUF:
        _BUF[key] = _zeros(md, [1, 1, R, (nh + 1) * HEAD_DIM], ttnn.TILE_LAYOUT)
    return _BUF[key]


def qprep(q, tabs, P, U, C, nh, sparse):
    """q [1,1,U*C,nh*512] (output of wq_b) -> RoPE'd heads. sparse: the U padded row-major buffers of ``q_pad_bufs`` (returned, only heads < nh written) else a new
    tile tensor [U,nh,C,512] (the output of nlp_create_qkv_heads + rope)."""
    c, s, _ = tabs
    cb32 = C // 32
    tpu = int(c.shape[0]) > 1
    in_str = (cb32 * nh * 16, 16, nh * 16, 0)
    if sparse:
        bufs = q_pad_bufs(q.device(), U, C)
        return rope_blocks(
            q,
            c,
            s,
            P,
            bufs,
            U=U,
            C=C,
            nh=nh,
            in_rm=False,
            out_rm=True,
            in_str=in_str,
            out_str=(0, 0, 0, 0),
            tab_per_user=tpu,
        )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([U, nh, C, HEAD_DIM]), ttnn.bfloat16, ttnn.TILE_LAYOUT, q.device(), ttnn.DRAM_MEMORY_CONFIG
    )
    return rope_blocks(
        q,
        c,
        s,
        P,
        out,
        U=U,
        C=C,
        nh=nh,
        in_rm=False,
        out_rm=False,
        in_str=in_str,
        out_str=(nh * cb32 * 16, cb32 * 16, 16, 0),
        tab_per_user=tpu,
    )


def kv_rope(kv, tabs, P, U, C):
    """kv [1,1,U*C,512] (or [U,1,C,512]) tile -> [U,1,C,512] tile with the RoPE on dims 448..511."""
    c, s, _ = tabs
    cb32 = C // 32
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([U, 1, C, HEAD_DIM]), ttnn.bfloat16, ttnn.TILE_LAYOUT, kv.device(), ttnn.DRAM_MEMORY_CONFIG
    )
    return rope_blocks(
        kv,
        c,
        s,
        P,
        out,
        U=U,
        C=C,
        nh=1,
        in_rm=False,
        out_rm=False,
        in_str=(cb32 * 16, 0, 16, 0),
        out_str=(cb32 * 16, 0, 16, 0),
        tab_per_user=int(c.shape[0]) > 1,
    )


def post(o_list, tabs, P, U, C, nh):
    """sparse_sdpa outputs (U row-major [1,32,C,512], heads < nh real) -> [1,1,U*C,(nh+1)*512] tile, head h at column tiles 16 (h + 1) .., inverse RoPE on dims 448..511."""
    c, _, ns = tabs
    cb32 = C // 32
    out = post_buf(c.device(), U * C, nh)
    tr = (nh + 1) * 16
    return rope_blocks(
        o_list,
        c,
        ns,
        P,
        out,
        U=U,
        C=C,
        nh=nh,
        in_rm=True,
        out_rm=False,
        in_str=(0, 0, 0, 0),
        out_str=(cb32 * tr, 16, tr, 16),
        tab_per_user=int(c.shape[0]) > 1,
    )


def idx_bufs(md, U, C):
    """U persistent [1,1,C,640] uint32 row-major index buffers (the sparse_sdpa ``indices`` input)."""
    key = ("idx", id(md), U, C)
    if key not in _BUF:
        _BUF[key] = [
            ttnn.zeros(
                [1, 1, C, 640],
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=md,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(U)
        ]
    return _BUF[key]


def ids_build(win, ids, perm, U, C):
    """Per user u: idx[u] = compact_rows(concat([win[u], ids[u]], dim=3), perm[u]) in ONE program (tt/fa_kernels/fa_ids.cpp).
    win [U,1,C,128] u32 RM, ids [U*C rows, 512] u32 RM (any shape with that row count), perm [Up,1,128,640] u32 RM (Up = U or 1). Returns the U persistent buffers.
    """
    mesh = win.device()
    outs = idx_bufs(mesh, U, C)
    rows = U * C
    grid = mesh.compute_with_storage_grid_size()
    allc = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)]
    ncores = min(rows, len(allc))
    maxr = (rows + ncores - 1) // ncores
    cores = allc[:ncores]
    core_set = _crs(cores)
    RB = 640 * 4
    size = (maxr * 2 + 1) * RB
    cbs = [_cb(core_set, 0, 1, size, ttnn.uint32)]
    rt = ttnn.RuntimeArgs()
    for i, (cx, cy) in enumerate(cores):
        n = (rows - i + ncores - 1) // ncores
        rt[cx][cy] = [i, ncores, n]
    pu = int(perm.shape[0]) > 1
    ct = [0, C, U, int(pu), maxr] + _acc(win) + _acc(ids) + _acc(perm) + _acc(outs[0])
    k = _kernel(
        "fa_ids.cpp",
        core_set,
        ct,
        rt,
        ttnn.ReaderConfigDescriptor(),
        [win.buffer_address(), ids.buffer_address(), perm.buffer_address()] + [o.buffer_address() for o in outs],
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0xFA2, U, C, int(pu), ncores, tuple(_acc(win)), tuple(_acc(ids)), tuple(_acc(perm)), tuple(_acc(outs[0]))
    )
    ttnn.generic_op([win, ids, perm] + outs, prog)
    return outs


IHEADS, IDIM = 32, 128


def idx_q(q, tabs, P, U, C):
    """indexer queries: wq_b output [1,1,U*C,32*128] tile -> head layout [U,32,C,128] tile with the addcmul RoPE on dims 64..127 (mode 2: the arithmetic of
    ``DSV41PrefillIndexer._rope``), in place of to_layout RM + reshape + permute + to_layout TILE + the 3-op rope.  tabs: the attention tables [U or 1,1,C,512] (only
    columns 448..511 are read)."""
    c, s = tabs[0], tabs[1]
    cb32 = C // 32
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([U, IHEADS, C, IDIM]), ttnn.bfloat16, ttnn.TILE_LAYOUT, q.device(), ttnn.DRAM_MEMORY_CONFIG
    )
    return rope_blocks(
        q,
        c,
        s,
        P,
        out,
        U=U,
        C=C,
        nh=IHEADS,
        in_rm=False,
        out_rm=False,
        in_str=(cb32 * IHEADS * 4, 4, IHEADS * 4, 0),
        out_str=(IHEADS * cb32 * 4, cb32 * 4, 4, 0),
        tab_per_user=int(c.shape[0]) > 1,
        nt=4,
        mode=2,
    )


def idx_k(k, tabs, P, U, Cc):
    """indexer keys of the chunk: [1,1,U*Cc,128] (or [U,1,Cc,128]) tile -> [U,1,Cc,128] tile with the addcmul RoPE on dims 64..127. tabs: the latent-position tables
    [U or 1,1,Cc,512]."""
    c, s = tabs[0], tabs[1]
    cb32 = Cc // 32
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([U, 1, Cc, IDIM]), ttnn.bfloat16, ttnn.TILE_LAYOUT, k.device(), ttnn.DRAM_MEMORY_CONFIG
    )
    return rope_blocks(
        k,
        c,
        s,
        P,
        out,
        U=U,
        C=Cc,
        nh=1,
        in_rm=False,
        out_rm=False,
        in_str=(cb32 * 4, 0, 4, 0),
        out_str=(cb32 * 4, 0, 4, 0),
        tab_per_user=int(c.shape[0]) > 1,
        nt=4,
        mode=2,
    )


def _row_kernel_setup(mesh, rows):
    grid = mesh.compute_with_storage_grid_size()
    allc = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)]
    ncores = min(rows, len(allc))
    cores = allc[:ncores]
    rt = ttnn.RuntimeArgs()
    for i, (cx, cy) in enumerate(cores):
        rt[cx][cy] = [i, ncores, (rows - i + ncores - 1) // ncores]
    return cores, ncores, (rows + ncores - 1) // ncores, rt


def idx_local(sc, ids, cid, lc, cols, C):
    """indexer local top-512 post-processing (tt/fa_kernels/fa_idxloc.cpp): sc [1,1,C,W] bf16 RM scores, ids [1,1,C,512] u32 RM local slots (0xFFFFFFFF: fewer visible),
    cid [1,1,1,1] fp32 tile (mesh column id) -> (vals [1,1,C,512] bf16 RM = score at the id or -inf, pg [1,1,C,512] u32 RM = id * cols + cid or 0xFFFFFFFF).
    """
    mesh = sc.device()
    W = int(sc.shape[-1])
    vals = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, C, 512]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    pg = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, C, 512]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    cores, ncores, maxr, rt = _row_kernel_setup(mesh, C)
    core_set = _crs(cores)
    rowb = ((W * 2 + 63) // 64) * 64
    size = 2 * (rowb + 2048) + 4096 + 2 * (1024 + 2048)
    cbs = [_cb(core_set, 0, 1, size, ttnn.uint32)]
    ct = [0, W, int(lc), int(cols)] + _acc(sc) + _acc(ids) + _acc(vals) + _acc(pg) + _acc(cid)
    k = _kernel(
        "fa_idxloc.cpp",
        core_set,
        ct,
        rt,
        ttnn.ReaderConfigDescriptor(),
        [sc.buffer_address(), ids.buffer_address(), vals.buffer_address(), pg.buffer_address(), cid.buffer_address()],
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0xFA3,
        W,
        int(lc),
        int(cols),
        C,
        ncores,
        tuple(_acc(sc)),
        tuple(_acc(ids)),
        tuple(_acc(vals)),
        tuple(_acc(pg)),
        tuple(_acc(cid)),
    )
    ttnn.generic_op([sc, ids, cid, vals, pg], prog)
    return vals, pg


def idx_final(pos, cand, C):
    """indexer merge post-processing (tt/fa_kernels/fa_idxfin.cpp): pos [1,1,C,512] u32 RM positions into cand [1,1,C,NCAND] u32 RM -> [1,1,C,512] u32 RM,
    cand[pos] or 0xFFFFFFFF where pos >= NCAND."""
    mesh = pos.device()
    ncand = int(cand.shape[-1])
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, C, 512]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    cores, ncores, maxr, rt = _row_kernel_setup(mesh, C)
    core_set = _crs(cores)
    size = maxr * (ncand * 4 + 2 * 2048)
    cbs = [_cb(core_set, 0, 1, size, ttnn.uint32)]
    ct = [0, ncand, maxr] + _acc(pos) + _acc(cand) + _acc(out)
    k = _kernel(
        "fa_idxfin.cpp",
        core_set,
        ct,
        rt,
        ttnn.ReaderConfigDescriptor(),
        [pos.buffer_address(), cand.buffer_address(), out.buffer_address()],
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(0xFA4, ncand, C, ncores, tuple(_acc(pos)), tuple(_acc(cand)), tuple(_acc(out)))
    ttnn.generic_op([pos, cand, out], prog)
    return out


_RD = [0]


def real_dump(tag, t):
    """diagnostics (FA_REAL_DUMP=<dir>): save device 0's copy of ``t`` as <dir>/<n>_<tag>.pt in eager passes only (real activations of the first window for the unit tests)."""
    d = os.environ.get("FA_REAL_DUMP")
    if not d:
        return
    from models.demos.blackhole.deepseek_v41_flash.tt import prefill_single as PS

    if PS.CAPTURING[0]:
        return
    import torch

    _RD[0] += 1
    torch.save(ttnn.to_torch(ttnn.get_device_tensors(t)[0]), f"{d}/{_RD[0]:04d}_{tag}.pt")
