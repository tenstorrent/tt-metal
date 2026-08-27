# SPDX-License-Identifier: Apache-2.0
"""Single-core validation descriptor for the fused AttnOut-matmul + residual + LN op.
Correctness-first (bf16, 1 core, full-N-per-core => local LN is the full row).
Not the fast path; validates the compute-kernel LN math before multicore/cross-core."""
from __future__ import annotations

import struct

import ttnn

_K = "models/demos/wormhole/bge_m3/tt/custom_ops/fused_attn_out_ln/kernels"
_TB = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.float32: 4096}


def _cb(idx, tiles, dt, cores):
    tb = _TB[dt]
    return ttnn.CBDescriptor(
        total_size=tiles * tb,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=tb)],
    )


def _acc(args, *tensors):
    for t in tensors:
        args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _bf16_bits(x: float) -> int:
    # pack two bf16(x) into a uint32 for the scaler/eps fill
    h = struct.unpack("<H", struct.pack("<e", x)[:2] if False else b"\x00\x00")[0]
    # derive bf16 = top 16 bits of fp32
    f = struct.unpack("<I", struct.pack("<f", x))[0]
    bf16 = (f >> 16) & 0xFFFF
    return (bf16 << 16) | bf16


def fused_attn_out_ln_singlecore(
    A, W, residual, gamma, beta, *, eps=1e-5, dtype=ttnn.bfloat16, M_block=1, bypass_ln=False
):
    dev = A.device()
    Msh, Ksh = tuple(A.padded_shape), tuple(W.padded_shape)
    M, K = Msh[-2] // 32, Msh[-1] // 32
    N = Ksh[-1] // 32
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape((*Msh[:-1], N * 32)), dtype, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
    )
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    tb = _TB[dtype]
    obn = M_block * N
    cbs = [
        _cb(0, M_block * K * 2, dtype, cores),  # in0
        _cb(1, K * N * 2, dtype, cores),  # in1
        _cb(2, obn * 2, dtype, cores),  # out
        _cb(3, obn, dtype, cores),  # interm
        _cb(4, obn * 2, dtype, cores),  # residual
        _cb(5, N, dtype, cores),  # gamma
        _cb(6, N, dtype, cores),  # beta
        _cb(7, 1, dtype, cores),  # scaler
        _cb(8, 1, dtype, cores),  # eps
        _cb(9, M_block, dtype, cores),  # ex
        _cb(10, obn, dtype, cores),  # xmm
        _cb(11, obn, dtype, cores),  # xmm2
        _cb(12, M_block, dtype, cores),  # var
        _cb(13, M_block, dtype, cores),  # rstd
        _cb(14, obn, dtype, cores),  # x = mm+resid
        _cb(15, obn, dtype, cores),  # norm
        _cb(16, obn, dtype, cores),  # normg
    ]
    scaler_packed = _bf16_bits(1.0 / (N * 32))
    eps_packed = _bf16_bits(eps)

    reader_ct = [M, K, N, M_block, tb, tb, tb, tb]
    _acc(reader_ct, A, W, residual, gamma, beta)
    reader_rt = [
        (0, 0),
        [
            A.buffer_address(),
            W.buffer_address(),
            residual.buffer_address(),
            gamma.buffer_address(),
            beta.buffer_address(),
            scaler_packed,
            eps_packed,
        ],
    ]
    writer_ct = [M, N, M_block, tb]
    _acc(writer_ct, out)
    writer_rt = [(0, 0), [out.buffer_address()]]
    compute_ct = [1, M_block, K, N, M // M_block, 1, M_block, min(N, 2)]  # arg0=K_num_blocks=1
    compute_rt = [(0, 0), []]
    _cdefs = []
    if bypass_ln:
        _cdefs.append(("BYPASS_LN", "1"))

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{_K}/reader_fused_ln.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=reader_ct,
            runtime_args=[reader_rt],
            defines=[],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_K}/writer_fused_ln.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=writer_ct,
            runtime_args=[writer_rt],
            defines=[],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_K}/compute_fused_ln.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=compute_ct,
            runtime_args=[compute_rt],
            defines=_cdefs,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        ),
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
    ttnn.generic_op([A, W, residual, gamma, beta, out], desc)
    return out


def fused_attn_out_ln_multicore(
    A,
    W,
    residual,
    gamma,
    beta,
    out,
    *,
    eps=1e-5,
    dtype=ttnn.bfloat16,
    M_block=1,
    K_block=4,
    grid=(8, 8),
    bypass_ln=False,
):
    """Full-N-per-core multi-core fused matmul+residual+LN. Each core owns an M-slice,
    reads full N, streams K. No multicast (weights read per-core from DRAM)."""
    _K = "models/demos/wormhole/bge_m3/tt/custom_ops/fused_attn_out_ln/kernels"
    dev = A.device()
    Msh, Ksh = tuple(A.padded_shape), tuple(W.padded_shape)
    M, K, N = Msh[-2] // 32, Msh[-1] // 32, Ksh[-1] // 32
    gx, gy = grid
    ncores = gx * gy
    assert M % ncores == 0, f"M tiles {M} must divide {ncores} cores"
    per_core_M = M // ncores
    assert per_core_M % M_block == 0
    per_core_M_blocks = per_core_M // M_block
    tb = _TB[dtype]
    obn = M_block * N
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    cbs = [
        _cb(0, M_block * K_block * 2, dtype, cores),
        _cb(1, K_block * N * 2, dtype, cores),
        _cb(2, obn * 2, dtype, cores),
        _cb(3, obn, dtype, cores),
        _cb(4, obn * 2, dtype, cores),
        _cb(5, N, dtype, cores),
        _cb(6, N, dtype, cores),
        _cb(7, 1, dtype, cores),
        _cb(8, 1, dtype, cores),
        _cb(9, M_block, dtype, cores),
        _cb(10, obn, dtype, cores),
        _cb(11, obn, dtype, cores),
        _cb(12, M_block, dtype, cores),
        _cb(13, M_block, dtype, cores),
        _cb(14, obn, dtype, cores),
        _cb(15, obn, dtype, cores),
        _cb(16, obn, dtype, cores),
    ]
    scaler_packed = _bf16_bits(1.0 / (N * 32))
    eps_packed = _bf16_bits(eps)
    reader_ct = [K, N, M_block, per_core_M_blocks, K_block, tb, tb, tb, tb]
    _acc(reader_ct, A, W, residual, gamma, beta)
    writer_ct = [N, M_block, per_core_M_blocks, tb]
    _acc(writer_ct, out)
    compute_ct = [K // K_block, M_block, K_block, N, per_core_M_blocks, 1, M_block, 2]
    reader_rt, writer_rt, compute_rt = [], [], []
    for x in range(gx):
        for y in range(gy):
            ci = x * gy + y
            m0 = ci * per_core_M
            reader_rt.append(
                (
                    (x, y),
                    [
                        A.buffer_address(),
                        W.buffer_address(),
                        residual.buffer_address(),
                        gamma.buffer_address(),
                        beta.buffer_address(),
                        scaler_packed,
                        eps_packed,
                        m0,
                    ],
                )
            )
            writer_rt.append(((x, y), [out.buffer_address(), m0]))
            compute_rt.append(((x, y), []))
    _cdefs = [("BYPASS_LN", "1")] if bypass_ln else []
    dm0 = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)
    dm1 = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{_K}/reader_fused_ln_mc.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            defines=[],
            config=dm0,
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_K}/writer_fused_ln_mc.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            defines=[],
            config=dm1,
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_K}/compute_fused_ln.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            defines=_cdefs,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        ),
    ]
    ttnn.generic_op([A, W, residual, gamma, beta, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return out


def fused_attn_out_ln_split(
    A,
    W,
    residual,
    gamma,
    beta,
    out,
    *,
    eps=1e-5,
    dtype=ttnn.bfloat16,
    gx=8,
    gy=8,
    M_t=8,
    K_block=8,
    subblock_h=1,
    subblock_w=2,
):
    """Production fast N-split fused matmul+residual+cross-core-LayerNorm.
    Grid (gx, gy): M partitioned across gx columns, N (LN feature dim) split across
    the gy=P cores of each column and reduced cross-core (proven §13-§15 protocol).
    Each core streams its M-slice in MBPC blocks of M_t tiles. LN intermediates are
    bf16 (numerics); DRAM I/O + matmul operands use `dtype`."""
    _KP = "models/demos/wormhole/bge_m3/tt/custom_ops/fused_attn_out_ln/kernels"
    dev = A.device()
    M = tuple(A.padded_shape)[-2] // 32
    K = tuple(A.padded_shape)[-1] // 32
    N = tuple(W.padded_shape)[-1] // 32
    P = gy
    assert N % gy == 0 and M % gx == 0 and K % K_block == 0
    Ns = N // gy
    per_col_M = M // gx
    assert per_col_M % M_t == 0, f"per_col_M {per_col_M} % M_t {M_t}"
    MBPC = per_col_M // M_t
    K_num_blocks = K // K_block
    obn = M_t * Ns
    b16 = ttnn.bfloat16
    tb_io = _TB[dtype]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])

    def cbd(idx, tiles, dt):
        tbb = _TB[dt]
        return ttnn.CBDescriptor(
            total_size=tiles * tbb,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=tbb)],
        )

    cbs = [
        cbd(0, M_t * K_block * 2, dtype),
        cbd(1, K_block * Ns * 2, dtype),
        cbd(2, obn * 2, dtype),
        cbd(3, obn, b16),
        cbd(4, obn, dtype),
        cbd(5, Ns, dtype),
        cbd(6, Ns, dtype),
        cbd(7, 1, b16),
        cbd(8, 1, b16),
        cbd(9, M_t, b16),
        cbd(10, P * M_t, b16),
        cbd(11, M_t, b16),
        cbd(12, obn, b16),
        cbd(13, M_t, b16),
        cbd(14, P * M_t, b16),
        cbd(15, M_t, b16),
        cbd(16, M_t, b16),
        cbd(17, 1, b16),
        cbd(18, obn, b16),
        cbd(21, obn, b16),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in range(3)]
    sp = _bf16_bits(1.0 / (N * 32))
    sgp = _bf16_bits(1.0)
    ep = _bf16_bits(eps)

    reader_ct = [K, N, Ns, M_t, K_block, tb_io, P, 0, 1, 2, MBPC]
    _acc(reader_ct, A, W, residual, gamma, beta)
    writer_ct = [N, Ns, M_t, tb_io, P, 2, MBPC]
    _acc(writer_ct, out)
    compute_ct = [M_t, Ns, P, K_block, K_num_blocks, subblock_h, subblock_w, MBPC]

    phys = {(x, y): dev.worker_core_from_logical_core(ttnn.CoreCoord(x, y)) for x in range(gx) for y in range(gy)}
    reader_rt, writer_rt, compute_rt = [], [], []
    for x in range(gx):
        col_coords = []
        for j in range(P):
            col_coords += [phys[(x, j)].x, phys[(x, j)].y]
        m_base = x * per_col_M
        for y in range(gy):
            n_start = y * Ns
            reader_rt.append(
                (
                    (x, y),
                    [
                        A.buffer_address(),
                        W.buffer_address(),
                        residual.buffer_address(),
                        gamma.buffer_address(),
                        beta.buffer_address(),
                        sp,
                        sgp,
                        ep,
                        n_start,
                        y,
                        m_base,
                    ]
                    + col_coords,
                )
            )
            writer_rt.append(((x, y), [out.buffer_address(), n_start, y, m_base] + col_coords))
            compute_rt.append(((x, y), []))

    dm0 = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)
    dm1 = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{_KP}/reader_xr_mm2.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            defines=[],
            config=dm0,
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KP}/writer_xr_mm2.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            defines=[],
            config=dm1,
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KP}/compute_xr_mm2.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            defines=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        ),
    ]
    ttnn.generic_op(
        [A, W, residual, gamma, beta, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems)
    )
    return out


def fused_attn_out_ln_split_mcast(
    A,
    W,
    residual,
    gamma,
    beta,
    out,
    *,
    eps=1e-5,
    dtype=ttnn.bfloat16,
    gx=8,
    gy=8,
    M_t=8,
    K_block=8,
    subblock_h=1,
    subblock_w=2,
    math_fidelity=ttnn.MathFidelity.LoFi,
):
    """Like fused_attn_out_ln_split but MULTICASTS the shared A operand down each column
    (sender y=0 reads once, mcasts to y=1..gy-1) via reader_xr_mm3.cpp, removing the
    (gy-1)x redundant A DRAM reads. Adds 2 mcast semaphores (ids 3,4)."""
    _KP = "models/demos/wormhole/bge_m3/tt/custom_ops/fused_attn_out_ln/kernels"
    dev = A.device()
    M = tuple(A.padded_shape)[-2] // 32
    K = tuple(A.padded_shape)[-1] // 32
    N = tuple(W.padded_shape)[-1] // 32
    P = gy
    assert N % gy == 0 and M % gx == 0 and K % K_block == 0
    Ns = N // gy
    per_col_M = M // gx
    assert per_col_M % M_t == 0
    MBPC = per_col_M // M_t
    K_num_blocks = K // K_block
    obn = M_t * Ns
    b16 = ttnn.bfloat16
    tb_io = _TB[dtype]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])

    def cbd(idx, tiles, dt):
        tbb = _TB[dt]
        return ttnn.CBDescriptor(
            total_size=tiles * tbb,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=tbb)],
        )

    cbs = [
        cbd(0, M_t * K_block * 2, dtype),
        cbd(1, K_block * Ns * 2, dtype),
        cbd(2, obn * 2, dtype),
        cbd(3, obn, b16),
        cbd(4, obn, dtype),
        cbd(5, Ns, dtype),
        cbd(6, Ns, dtype),
        cbd(7, 1, b16),
        cbd(8, 1, b16),
        cbd(9, M_t, b16),
        cbd(10, P * M_t, b16),
        cbd(11, M_t, b16),
        cbd(12, obn, b16),
        cbd(13, M_t, b16),
        cbd(14, P * M_t, b16),
        cbd(15, M_t, b16),
        cbd(16, M_t, b16),
        cbd(17, 1, b16),
        cbd(18, obn, b16),
        cbd(21, obn, b16),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in range(5)]
    sp = _bf16_bits(1.0 / (N * 32))
    sgp = _bf16_bits(1.0)
    ep = _bf16_bits(eps)

    reader_ct = [K, N, Ns, M_t, K_block, tb_io, P, 0, 1, 2, MBPC, 3, 4]
    _acc(reader_ct, A, W, residual, gamma, beta)
    writer_ct = [N, Ns, M_t, tb_io, P, 2, MBPC]
    _acc(writer_ct, out)
    compute_ct = [M_t, Ns, P, K_block, K_num_blocks, subblock_h, subblock_w, MBPC]

    phys = {(x, y): dev.worker_core_from_logical_core(ttnn.CoreCoord(x, y)) for x in range(gx) for y in range(gy)}
    num_recv = P - 1
    reader_rt, writer_rt, compute_rt = [], [], []
    for x in range(gx):
        col_coords = []
        for j in range(P):
            col_coords += [phys[(x, j)].x, phys[(x, j)].y]
        m_base = x * per_col_M
        snx, sny = phys[(x, 0)].x, phys[(x, 0)].y  # sender = y0
        mdsx, mdsy = phys[(x, 1)].x, phys[(x, 1)].y  # receiver box start
        mdex, mdey = phys[(x, gy - 1)].x, phys[(x, gy - 1)].y  # receiver box end
        for y in range(gy):
            n_start = y * Ns
            is_sender = 1 if y == 0 else 0
            reader_rt.append(
                (
                    (x, y),
                    [
                        A.buffer_address(),
                        W.buffer_address(),
                        residual.buffer_address(),
                        gamma.buffer_address(),
                        beta.buffer_address(),
                        sp,
                        sgp,
                        ep,
                        n_start,
                        y,
                        m_base,
                        is_sender,
                        snx,
                        sny,
                        mdsx,
                        mdsy,
                        mdex,
                        mdey,
                        num_recv,
                    ]
                    + col_coords,
                )
            )
            writer_rt.append(((x, y), [out.buffer_address(), n_start, y, m_base] + col_coords))
            compute_rt.append(((x, y), []))

    dm0 = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)
    dm1 = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{_KP}/reader_xr_mm3.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            defines=[],
            config=dm0,
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KP}/writer_xr_mm2.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            defines=[],
            config=dm1,
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KP}/compute_xr_mm2.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            defines=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=math_fidelity, math_approx_mode=False, fp32_dest_acc_en=False, dst_full_sync_en=False
            ),
        ),
    ]
    ttnn.generic_op(
        [A, W, residual, gamma, beta, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems)
    )
    return out
