# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""T19 prefill mHC / glue fusions (ttnn.generic_op, JIT kernels in tt/glue_kernels/; no ttnn C++ rebuild). Every switch is read when used, default OFF:

DSV41_PF_GLUE_MIX=1    packed mHC ``mixes``: the 8-op chain after pk_proj (sum over the K-split partials, sum-of-squares select matmul, scale, rsqrt, multiply)
                       in ONE program (``pk_mix``).
DSV41_PF_GLUE_NORM=1   packed mHC ``collapse_norm``: pk_comb collapse that also emits the per-core sum of squares of its bf16 output tiles + one multi-core
                       ``pk_norm`` program (replaces ttnn.rms_norm on 4 tile rows = 4 cores).
DSV41_PF_GLUE_TC=1     bf16 ``y`` / ``y2`` operands accepted by the expand pk_comb (no bf16 -> fp32 typecast op in front of it).
DSV41_PF_GLUE_RS=1     attention output reduce-scatter (prefill_attention._reduce_scatter_tokens): one reduce_scatter over the batch of 256-row pieces instead of
                       R/256 slice + reduce_scatter pairs and a concat.
DSV41_PF_GLUE_FUSE=1   all of the above.
"""
import os
import struct

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash

GDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glue_kernels")
f32, bf16 = ttnn.float32, ttnn.bfloat16
VER = 2  # bump with every kernel edit: the JIT cache key only sees custom_program_hash


def on(name):
    return os.environ.get(name, "0") == "1" or os.environ.get("DSV41_PF_GLUE_FUSE", "0") == "1"


def _kernel(name, core_set, ct, rt, crt, cfg, common=None):
    return ttnn.KernelDescriptor(
        kernel_source=f"{GDIR}/{name}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=common or [],
        config=cfg,
    )


def _cc(*fp32_cbs):
    """HiFi4 / fp32 DST compute config; the listed CBs unpack straight to DST in full fp32 (copy_tile + SFPU ops, no FPU operand use)."""
    cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for c in fp32_cbs:
        modes[c] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc.unpack_to_dest_mode = modes
    return cc


def _cores(mesh, n):
    g = mesh.compute_with_storage_grid_size()
    return [(cx, cy) for cy in range(int(g.y)) for cx in range(int(g.x))][:n]


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def pk_mix(part, ncol, inv_nc, eps, depth=8):
    """part [R,S,32,32] fp32 TILE (pk_proj partials) -> mixes [1,1,R*32,32] fp32 TILE = mu * rsqrt(mu[:, ncol] * inv_nc + eps), mu = part.sum(dim=1)."""
    R, S = int(part.shape[0]), int(part.shape[1])
    mesh = part.device()
    while S % depth:
        depth -= 1
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, R * 32, 32]), f32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    cores = _cores(mesh, R)
    cs = _crs(cores)
    rt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for r, (cx, cy) in enumerate(cores):
        rt[cx][cy] = [r]
        crt[cx][cy] = []
    PART, EYE, SEL, MU, RS, OUT, MUS = range(7)
    cbs = [
        _cb(cs, PART, 2 * depth, 4096, f32),
        _cb(cs, EYE, 1, 4096, f32),
        _cb(cs, SEL, 1, 4096, f32),
        _cb(cs, MU, 1, 4096, f32),
        _cb(cs, RS, 1, 4096, f32),
        _cb(cs, OUT, 1, 4096, f32),
        _cb(cs, MUS, 1, 4096, f32),
    ]
    reader = _kernel(
        "pkm_reader.cpp",
        cs,
        [PART, EYE, SEL, S, ncol, depth] + _acc(part),
        rt,
        None,
        ttnn.ReaderConfigDescriptor(),
        common=[part.buffer_address()],
    )
    writer = _kernel(
        "pkm_writer.cpp", cs, [OUT] + _acc(out), rt, None, ttnn.WriterConfigDescriptor(), common=[out.buffer_address()]
    )
    compute = _kernel(
        "pkm_compute.cpp",
        cs,
        [PART, EYE, SEL, MU, RS, OUT, S, depth, _bits(inv_nc), _bits(eps), MUS],
        crt,
        None,
        _cc(PART, RS, MUS),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x7A1, VER, R, S, ncol, depth, _bits(inv_nc), _bits(eps), tuple(_acc(part)), tuple(_acc(out))
    )
    ttnn.generic_op([part, out], prog)
    return out


def _divs(n, m):
    return [d for d in range(1, min(n, m) + 1) if n % d == 0]


def pk_collapse_sq(x, cta, max_cores=None):
    """pk_comb collapse (nout=1, bf16 out) + per-core partial sum of squares of the bf16 output: x [1,1,N,4C] fp32, cta [N,128] pre coefficient tiles
    -> (h [1,1,N,C] bf16, part [R,NRC,32,32] fp32 (column 0 = partial sum of squares), NRC)."""
    N, W = int(x.shape[-2]), int(x.shape[-1])
    CT = W // 128
    R = N // 32
    NCA = int(cta.shape[-1]) // 32
    mesh = x.device()
    g = mesh.compute_with_storage_grid_size()
    max_cores = max_cores or int(g.x) * int(g.y)
    per_row = max(1, max_cores // R)
    cs = min(d for d in _divs(CT, CT) if CT // d <= per_row)
    nrc = CT // cs
    BLK = max(d for d in _divs(cs, 4))
    cores = _cores(mesh, R * nrc)
    assert len(cores) == R * nrc
    core_set = _crs(cores)
    h = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, N, CT * 32]), bf16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    part = ttnn.allocate_tensor_on_device(
        ttnn.Shape([R, nrc, 32, 32]), f32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    rt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (cx, cy) in enumerate(cores):
        r, p = divmod(c, nrc)
        rt[cx][cy] = [r, p * cs, cs]
        crt[cx][cy] = [cs]
        wrt[cx][cy] = [r, p * cs, cs, p]
    CB_I, CB_CT, CB_X, CB_D, CB_O, CB_H2, CB_SQ, CB_ONES, CB_P = range(9)
    cbs = [
        _cb(core_set, CB_I, 1, 4096, f32),
        _cb(core_set, CB_CT, NCA, 4096, f32),
        _cb(core_set, CB_X, 8 * BLK, 4096, f32),
        _cb(core_set, CB_D, NCA, 4096, f32),
        _cb(core_set, CB_O, 2, 2048, bf16),
        _cb(core_set, CB_H2, cs, 2048, bf16),
        _cb(core_set, CB_SQ, cs, 4096, f32),
        _cb(core_set, CB_ONES, 1, 4096, f32),
        _cb(core_set, CB_P, 1, 4096, f32),
    ]
    reader = _kernel(
        "pkc_reader.cpp",
        core_set,
        [CB_I, CB_CT, CB_X, CB_ONES, NCA, CT, BLK] + _acc(x) + _acc(cta),
        rt,
        None,
        ttnn.ReaderConfigDescriptor(),
        common=[x.buffer_address(), cta.buffer_address()],
    )
    writer = _kernel(
        "pkc_writer.cpp",
        core_set,
        [CB_O, CB_P, CT, nrc, BLK] + _acc(h) + _acc(part),
        wrt,
        None,
        ttnn.WriterConfigDescriptor(),
        common=[h.buffer_address(), part.buffer_address()],
    )
    compute = _kernel(
        "pkc_compute.cpp",
        core_set,
        [CB_I, CB_CT, CB_X, CB_D, CB_O, CB_H2, CB_SQ, CB_ONES, CB_P, NCA, BLK],
        crt,
        None,
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x7A2, VER, R, CT, NCA, cs, BLK, tuple(_acc(x)), tuple(_acc(cta)), tuple(_acc(h)), tuple(_acc(part))
    )
    ttnn.generic_op([x, cta, h, part], prog)
    return h, part, nrc


def pk_norm(h, part, w, eps, k_split=8):
    """h [1,1,N,C] bf16, part [R,NRC,32,32] fp32 (pk_collapse_sq), w [1,1,1,C] fp32 TILE -> bf16 [1,1,N,C] = rms_norm(h, weight=w, epsilon=eps)."""
    N, C = int(h.shape[-2]), int(h.shape[-1])
    CT = C // 32
    R, nrc = int(part.shape[0]), int(part.shape[1])
    mesh = h.device()
    g = mesh.compute_with_storage_grid_size()
    kmax = max(1, int(g.x) * int(g.y) // R)
    K = max(d for d in _divs(CT, min(kmax, max(k_split, 1))))
    cn = CT // K
    cores = _cores(mesh, R * K)
    core_set = _crs(cores)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, N, C]), bf16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    rt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (cx, cy) in enumerate(cores):
        r, k = divmod(c, K)
        rt[cx][cy] = [r, k * cn, cn]
        crt[cx][cy] = [cn]
    PART, ONES, H, W, RS, TMP, OUT = range(7)
    cbs = [
        _cb(core_set, PART, nrc, 4096, f32),
        _cb(core_set, ONES, 1, 4096, f32),
        _cb(core_set, H, cn, 2048, bf16),
        _cb(core_set, W, cn, 4096, f32),
        _cb(core_set, RS, 1, 4096, f32),
        _cb(core_set, TMP, 2, 4096, f32),
        _cb(core_set, OUT, 2, 2048, bf16),
    ]
    reader = _kernel(
        "pkn_reader.cpp",
        core_set,
        [PART, ONES, H, W, nrc, CT] + _acc(part) + _acc(h) + _acc(w),
        rt,
        None,
        ttnn.ReaderConfigDescriptor(),
        common=[part.buffer_address(), h.buffer_address(), w.buffer_address()],
    )
    writer = _kernel(
        "pkn_writer.cpp",
        core_set,
        [OUT, CT] + _acc(out),
        rt,
        None,
        ttnn.WriterConfigDescriptor(),
        common=[out.buffer_address()],
    )
    compute = _kernel(
        "pkn_compute.cpp",
        core_set,
        [PART, ONES, H, W, RS, TMP, OUT, nrc, _bits(1.0 / C), _bits(eps)],
        crt,
        None,
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x7A3, VER, R, CT, nrc, K, _bits(eps), tuple(_acc(part)), tuple(_acc(h)), tuple(_acc(w)), tuple(_acc(out))
    )
    ttnn.generic_op([part, h, w, out], prog)
    return out
