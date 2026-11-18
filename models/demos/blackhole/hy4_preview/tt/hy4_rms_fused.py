# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""B47c (HY4_B47C_RMS=1, default OFF): fused [residual add +] RMSNorm of the Hy4 MTP block as ONE 32-core program (replaces the 6-8 op fp32 chain
typecast / add / multiply / sum / add / rsqrt / multiply / multiply of ``Hy4MTP._rms6``, ~100 us per call, 5 calls per MTP pass). Adapted from the DSV4 mhc_cn program
(partial sums of squares exchanged through L1 slots on a 4x8 worker rectangle). All elementwise maths in exact fp32 SFPU ops (same as the stock ops); the row sum of
squares is a ones-matmul over fp32 (summation order differs from ttnn.sum: draft-only numerics, the verify rows decide).
  rms_fused(x, w, eps, res=None, out=None, out_off=0, out_x=False)
    x   [1,1,T,D] fp32 or bf16 TILE; res bf16 TILE (optional, x += res in fp32, the sum is the normalised input); w fp32 [1,1,T,D] = weight*sqrt(D), rows replicated
    out bf16 [1,1,T,D'] TILE (allocated if None, D' = D); the result goes to tile columns out_off*32.. (cat writes: out_off = D/32 for the second half)
    returns out (and the fp32 sum if out_x)."""
import os
import struct

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_rms_kernels")
NC, GX, GY = 32, 4, 8


def enabled():
    return os.environ.get("HY4_B47C_RMS", "0") == "1"


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _cb(core_set, idx, pages, page, dt):
    return ttnn.CBDescriptor(
        total_size=pages * page,
        core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)],
    )


def _kernel(name, core_set, ct, rt, cfg, common=None):
    return ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/{name}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=common or [],
        config=cfg,
    )


def rms_fused(x, w, eps, res=None, out=None, out_off=0, out_x=False):
    _, _, T, D = (int(v) for v in x.shape)
    assert 1 <= T <= 16 and x.dtype in (ttnn.float32, ttnn.bfloat16) and w.dtype == ttnn.float32
    nt = D // 32
    assert nt % NC == 0
    gpc = nt // NC
    xf32 = x.dtype == ttnn.float32
    has_r = res is not None
    mesh = x.device()
    if out is None:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, T, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
    xo = None
    if out_x:
        assert has_r
        xo = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, T, D]), ttnn.float32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
    r_t = res if has_r else x
    xo_t = xo if out_x else out
    cores = [(cx, cy) for cy in range(GY) for cx in range(GX)]  # core k = cy * GX + cx
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
    p0 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
    p1 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(GX - 1, GY - 1))
    assert p1.x - p0.x == GX - 1 and p1.y - p0.y == GY - 1
    rt = ttnn.RuntimeArgs()
    for k, (cx, cy) in enumerate(cores):
        rt[cx][cy] = [k]
    f32, bf = ttnn.float32, ttnn.bfloat16
    X, R, W, ONES, XN, XO, SQ, P, TOT, RS, O, SLOT = range(12)
    xp = 4096 if xf32 else 2048
    cbs = [
        _cb(core_set, X, gpc, xp, f32 if xf32 else bf),
        _cb(core_set, R, gpc if has_r else 1, 2048, bf),
        _cb(core_set, W, gpc, 4096, f32),
        _cb(core_set, ONES, 1, 4096, f32),
        _cb(core_set, XN, gpc if has_r else 1, 4096, f32),
        _cb(core_set, XO, gpc if out_x else 1, 4096, f32),
        _cb(core_set, SQ, gpc, 4096, f32),
        _cb(core_set, P, 1, 4096, f32),
        _cb(core_set, TOT, 1, 4096, f32),
        _cb(core_set, RS, 1, 4096, f32),
        _cb(core_set, O, gpc, 2048, bf),
        _cb(core_set, SLOT, 2, 4096, f32),
    ]
    eps_bits = struct.unpack("<I", struct.pack("<f", D * eps))[0]
    reader = _kernel(
        "rms_reader.cpp",
        core_set,
        [X, R, W, P, SLOT, TOT, T, gpc, NC, p0.x, p0.y, GX, int(xf32), int(has_r)] + _acc(x) + _acc(r_t) + _acc(w),
        rt,
        ttnn.ReaderConfigDescriptor(),
        common=[x.buffer_address(), r_t.buffer_address(), w.buffer_address()],
    )
    writer = _kernel(
        "rms_writer.cpp",
        core_set,
        [O, XO, ONES, gpc, int(out_x), int(out_off)] + _acc(out) + _acc(xo_t),
        rt,
        ttnn.WriterConfigDescriptor(),
        common=[out.buffer_address(), xo_t.buffer_address()],
    )
    compute = _kernel(
        "rms_compute.cpp",
        core_set,
        [X, R, W, ONES, XN, XO, SQ, P, TOT, RS, O, gpc, eps_bits, int(has_r), int(out_x)],
        ttnn.RuntimeArgs(),
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = (0x4B47 << 40) | (
        hash(
            (
                T,
                D,
                eps_bits,
                int(xf32),
                int(has_r),
                int(out_x),
                int(out_off),
                tuple(_acc(x)),
                tuple(_acc(r_t)),
                tuple(_acc(w)),
                tuple(_acc(out)),
                tuple(_acc(xo_t)),
            )
        )
        & ((1 << 40) - 1)
    )
    ttnn.generic_op([x, r_t, w, out, xo_t], prog)
    return (out, xo) if out_x else out
