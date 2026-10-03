# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused mHC collapse + RMSNorm (two generic_ops).

op1 (mhc_collapse):  h = sum_i pre_i x_i  -> bf16 [1,1,T,D] (token rows) plus one partial sum-of-squares tile per core.
op2 (mhc_norm_apply): out = bf16(h) * rsqrt(sum_k partial_k + C*eps) * (w*sqrt(C)) -> bf16 [1,1,T,D]; same maths as the
layer's  rms_norm(bf16(collapse), weight=w)."""


import os

import ttnn

KDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mhc_kernels"
)  # absolute: the overlay package must use ITS kernels


def _cores(mesh, nt, n_cores):
    grid = mesh.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)][:n_cores]
    while nt % len(cores):
        cores.pop()
    return cores


def _cb(core_set, idx, pages, page, dt):
    return ttnn.CBDescriptor(
        total_size=pages * page,
        core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)],
    )


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _kernel(name, core_set, ct, rt, crt, cfg, common=None):
    return ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/{name}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=common or [],
        config=cfg,
    )


def _hash(tag, *parts):
    return (tag << 40) | (hash(parts) & ((1 << 40) - 1))


def mhc_collapse(x, pre, n_cores=32):
    """x [T,1,4,D] fp32 TILE, pre [T,1,1,4] fp32 TILE -> (h bf16 [1,1,T,D], partials fp32 [ncores,1,32,32])."""
    T, _, n, D = (int(v) for v in x.shape)
    assert n == 4 and T <= 32 and x.dtype == ttnn.float32 and pre.dtype == ttnn.float32
    nt = D // 32
    mesh = x.device()
    cores = _cores(mesh, nt, n_cores)
    nc = len(cores)
    assert nc <= 32
    gpc = nt // nc
    jb = max(1, min(gpc, 64 // T))
    h = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, T, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    part = ttnn.allocate_tensor_on_device(
        ttnn.Shape([nc, 1, 32, 32]), ttnn.float32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    rt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for k, (cx, cy) in enumerate(cores):
        rt[cx][cy] = [k * gpc, (k + 1) * gpc, k]
        crt[cx][cy] = [gpc]
    A, X, ONES, S, HW, HS, SQ, P = range(8)
    f32, bf = ttnn.float32, ttnn.bfloat16
    cbs = [
        _cb(core_set, A, T, 4096, f32),
        _cb(core_set, X, jb * T, 4096, f32),
        _cb(core_set, ONES, 1, 4096, f32),
        _cb(core_set, S, 1, 4096, f32),
        _cb(core_set, HW, gpc, 2048, bf),
        _cb(core_set, HS, gpc, 2048, bf),
        _cb(core_set, SQ, gpc, 4096, f32),
        _cb(core_set, P, 1, 4096, f32),
    ]
    reader = _kernel(
        "mhc_collapse_reader.cpp",
        core_set,
        [A, X, ONES, S, T, nt, jb] + _acc(x) + _acc(pre),
        rt,
        None,
        ttnn.ReaderConfigDescriptor(),
        common=[x.buffer_address(), pre.buffer_address()],
    )
    writer = _kernel(
        "mhc_collapse_writer.cpp",
        core_set,
        [HW, P] + _acc(h) + _acc(part),
        rt,
        None,
        ttnn.WriterConfigDescriptor(),
        common=[h.buffer_address(), part.buffer_address()],
    )
    compute = _kernel(
        "mhc_collapse_compute.cpp",
        core_set,
        [A, X, ONES, HW, HS, SQ, P, T, jb],
        crt,
        None,
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x5C1, T, D, nc, jb, tuple(_acc(x)), tuple(_acc(pre)), tuple(_acc(h)), tuple(_acc(part))
    )
    ttnn.generic_op([x, pre, h, part], prog)
    return h, part


def mhc_norm_apply(h, part, w4, eps, n_cores=32, emit_rm=False):
    """h bf16 [1,1,T,D], part fp32 [nc,1,32,32] (from mhc_collapse), w4 fp32 [1,1,T,D] = weight*sqrt(D) (rows replicated)."""
    _, _, T, D = (int(v) for v in h.shape)
    nt = D // 32
    mesh = h.device()
    nc = int(part.shape[0])
    cores = _cores(mesh, nt, n_cores)
    gpc = nt // len(cores)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, T, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    rt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for k, (cx, cy) in enumerate(cores):
        rt[cx][cy] = [k * gpc, (k + 1) * gpc]
        crt[cx][cy] = [gpc]
    rm = None
    if emit_rm:
        rm = ttnn.allocate_tensor_on_device(
            ttnn.Shape([T, 1, 1, D]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
    R, ONES, S, H, W, RS, TMP, O, RMB = range(9)
    f32, bf = ttnn.float32, ttnn.bfloat16
    cbs = [
        _cb(core_set, R, 1, 4096, f32),
        _cb(core_set, ONES, 1, 4096, f32),
        _cb(core_set, S, max(2, nc * T // 64), 4096, f32),
        _cb(core_set, H, gpc, 2048, bf),
        _cb(core_set, W, gpc, 4096, f32),
        _cb(core_set, RS, 1, 4096, f32),
        _cb(core_set, TMP, gpc, 4096, f32),
        _cb(core_set, O, gpc, 2048, bf),
        _cb(core_set, RMB, max(1, (gpc * T * 64 + 4095) // 4096), 4096, bf),
    ]
    import struct

    eps_bits = struct.unpack("<I", struct.pack("<f", D * eps))[0]
    reader = _kernel(
        "mhc_norm_reader.cpp",
        core_set,
        [R, ONES, S, H, W, nc, T, max(2, nc * T // 64)] + _acc(h) + _acc(part) + _acc(w4),
        rt,
        None,
        ttnn.ReaderConfigDescriptor(),
        common=[h.buffer_address(), part.buffer_address(), w4.buffer_address()],
    )
    rm_t = rm if emit_rm else out
    writer = _kernel(
        "mhc_norm_writer.cpp",
        core_set,
        [O, RMB, int(emit_rm), T, D * 2] + _acc(out) + _acc(rm_t),
        rt,
        None,
        ttnn.WriterConfigDescriptor(),
        common=[out.buffer_address(), rm_t.buffer_address()],
    )
    compute = _kernel(
        "mhc_norm_compute.cpp",
        core_set,
        [R, ONES, H, W, RS, TMP, O, eps_bits],
        crt,
        None,
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x5C2,
        T,
        D,
        nc,
        eps_bits,
        emit_rm,
        tuple(_acc(rm_t)),
        tuple(_acc(h)),
        tuple(_acc(part)),
        tuple(_acc(w4)),
        tuple(_acc(out)),
    )
    ttnn.generic_op([h, part, w4, out, rm_t], prog)
    return (out, rm) if emit_rm else out


def mhc_collapse_norm(x, pre, w4, eps, emit_rm=False):
    """ONE generic_op: out = bf16(bf16(sum_i pre_i x_i) * rsqrt(sum h^2 + D*eps) * w4) (w4 = weight*sqrt(D), rows replicated
    [1,1,T,D] fp32). x [T,1,4,D] fp32 TILE, pre [T,1,1,4] fp32 TILE -> out bf16 [1,1,T,D] (and, if emit_rm, the bf16 row-major
    [T,1,1,D] copy).  32 cores (4x8 worker rectangle) exchange their partial sums of squares through L1 writes. T in {4, 8, 16, 32}.
    """
    import struct

    T, _, n, D = (int(v) for v in x.shape)
    assert n == 4 and T in (4, 8, 16, 32) and x.dtype == ttnn.float32 and pre.dtype == ttnn.float32
    nt = D // 32
    # worker columns 0..3 are contiguous in physical NOC coordinates (logical x = 7 jumps to physical 10); NC <= 32 (one R tile)
    NC, GX, GY = 32, 4, 8
    assert nt % NC == 0
    gpc = nt // NC
    G = (T + 7) // 8
    mesh = x.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, T, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    rm = None
    if emit_rm:
        rm = ttnn.allocate_tensor_on_device(
            ttnn.Shape([T, 1, 1, D]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
    rm_t = rm if emit_rm else out
    cores = [(cx, cy) for cy in range(GY) for cx in range(GX)]  # core k = cy * GX + cx
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
    p0 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
    p1 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(GX - 1, GY - 1))
    assert p1.x - p0.x == GX - 1 and p1.y - p0.y == GY - 1
    rt = ttnn.RuntimeArgs()
    for k, (cx, cy) in enumerate(cores):
        rt[cx][cy] = [k]
    f32, bf = ttnn.float32, ttnn.bfloat16
    A, X, ONES, HS, SQ, P, TOT, W, RS, TMP, O, RMB, PRE, SLOT = range(14)
    cbs = [
        _cb(core_set, A, G, 4096, f32),
        _cb(core_set, X, gpc * G, 4096, f32),
        _cb(core_set, ONES, 1, 4096, f32),
        _cb(core_set, HS, gpc, 2048, bf),
        _cb(core_set, SQ, gpc, 4096, f32),
        _cb(core_set, P, 1, 4096, f32),
        _cb(core_set, TOT, 1, 4096, f32),
        _cb(core_set, W, gpc, 4096, f32),
        _cb(core_set, RS, 1, 4096, f32),
        _cb(core_set, TMP, gpc, 4096, f32),
        _cb(core_set, O, gpc, 2048, bf),
        _cb(core_set, RMB, max(1, (gpc * T * 64 + 4095) // 4096), 4096, bf),
        _cb(core_set, PRE, 1, 4096, f32),
        _cb(core_set, SLOT, 2, 4096, f32),
    ]
    eps_bits = struct.unpack("<I", struct.pack("<f", D * eps))[0]
    reader = _kernel(
        "mhc_cn_reader.cpp",
        core_set,
        [A, X, W, PRE, P, SLOT, TOT, T, nt, G, gpc, NC, p0.x, p0.y, GX] + _acc(x) + _acc(pre) + _acc(w4),
        rt,
        None,
        ttnn.ReaderConfigDescriptor(),
        common=[x.buffer_address(), pre.buffer_address(), w4.buffer_address()],
    )
    writer = _kernel(
        "mhc_cn_writer.cpp",
        core_set,
        [O, RMB, int(emit_rm), T, D * 2, ONES, gpc] + _acc(out) + _acc(rm_t),
        rt,
        None,
        ttnn.WriterConfigDescriptor(),
        common=[out.buffer_address(), rm_t.buffer_address()],
    )
    compute = _kernel(
        "mhc_cn_compute.cpp",
        core_set,
        [A, X, ONES, HS, SQ, P, TOT, W, RS, TMP, O, T, G, gpc, eps_bits],
        ttnn.RuntimeArgs(),
        None,
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x5C3,
        T,
        D,
        eps_bits,
        emit_rm,
        tuple(_acc(x)),
        tuple(_acc(pre)),
        tuple(_acc(w4)),
        tuple(_acc(out)),
        tuple(_acc(rm_t)),
    )
    ttnn.generic_op([x, pre, w4, out, rm_t], prog)
    return (out, rm) if emit_rm else out
