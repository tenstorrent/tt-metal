# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct-to-pool chunked prefill attention (task pool-prefill; design: program/pool-prefill/DESIGN.md).

``sparse_sdpa_pf`` = the Hy4 key-split decode kernel (hy4_split, SCALED_FP8 in kernel, per-head (max, sum) stats, empty-column rule) extended with a ROW LOOP so one launch covers a whole
512-row prefill chunk: row task = (token, head group of 32 heads), rows = hg*S + tok, core i takes rows i, i+NC, ... (G = 1, no key split). Kernels: ``hy4_pf_kernels`` (a copy of
``hy4_split_kernels`` with the loop; the decode kernels are untouched).
"""
import os
import struct

import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_pf_kernels")


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _bits(f):
    return struct.unpack("<I", struct.pack("<f", float(f)))[0]


def _core_set(n, gx):
    rngs, r = [], 0
    while n > 0:
        w = min(n, gx)
        rngs.append(ttnn.CoreRange(ttnn.CoreCoord(0, r), ttnn.CoreCoord(w - 1, r)))
        n -= w
        r += 1
    return ttnn.CoreRangeSet(rngs)


def sparse_sdpa_pf(q, kv, idx, scale, k_chunk=128, ckc=None, out=None, ncores=None, v_dim=512, task_heads=None):
    """q bf16 ROW_MAJOR [1,HT,S,576] (HT = 64 heads: head group hg = heads 32hg..32hg+31), kv SCALED_FP8 packed fp8 [1,1,R,656] (this column's pool shard), idx uint32 ROW_MAJOR [1,1,S,TOPK]
    LOCAL row ids of this column (sentinel 0xFFFFFFFF = contiguous tail; a row with no local key emits max=-inf, sum=0). -> out bf16 ROW_MAJOR [1,HT,S,v_dim+32]: per head row
    [512 o (normalised by the LOCAL sum, sink excluded) | m (scaled-logit domain) | l | 1.0 | zeros]."""
    md = q.device()
    HT, S, D = int(q.shape[1]), int(q.shape[2]), int(q.shape[3])
    topk = int(idx.shape[3])
    H = task_heads or int(
        os.environ.get("PF_TASK_HEADS", "64")
    )  # heads per row task (32 or 64: 64 gathers every K row once for all heads)
    assert HT % H == 0 and v_dim == 512 and D == 576
    assert (
        q.dtype == ttnn.bfloat16 and idx.dtype == ttnn.uint32 and kv.dtype == ttnn.fp8_e4m3 and int(kv.shape[3]) == 656
    )
    scaled = True
    HG = HT // H
    rows = HG * S
    grid = md.compute_with_storage_grid_size()
    NC = min(rows, grid.x * grid.y if ncores is None else ncores)
    G = 1
    RING = int(os.environ.get("SS_RING", "8"))
    QMODE = 1
    QROWS = H
    n_chunks_row = -(-topk // k_chunk)
    DHt, vDHt = D // 32, v_dim // 32
    Skt = k_chunk // 32
    Sqt = H // 32
    dst = 4 if bool(getattr(ckc, "fp32_dest_acc_en", True)) else 8
    qsb = max(d for d in range(1, min(Sqt, dst) + 1) if Sqt % d == 0)
    kv_b = 1
    PACKED = 704
    TILE = 2048
    bf = ttnn.bfloat16
    k_in_tile, k_in_df = 1088, ttnn.bfloat8_b
    emit_stats = 2
    sink = _dummy_sink(md, H)
    if out is None:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, HT, S, v_dim + 32]), bf, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
        )
    cores = [(i % grid.x, i // grid.x) for i in range(NC)]
    core_set = _core_set(NC, grid.x)
    (
        Q_RM,
        Q_IN,
        K_RM,
        K_IN,
        NEGINF,
        MASK_PART,
        SCALE,
        QK_IM,
        MAX_A,
        MAX_B,
        SUM_A,
        SUM_B,
        OUT_A,
        OUT_B,
        CORR,
        OUT_IM,
        OUT_RM,
        IDX,
        CTRL,
        COL_ID,
        RECIP,
        KREQ,
        KACK,
        K_ROPE_RM,
        K_SCALE_BC,
        K_LAT_T,
        K_ROPE_T,
        SINK,
        SINK_SCR,
        RECV_O,
        RECV_M,
        RECV_S,
        MA,
        MB,
        CORR_R,
        M_OUT,
        L_OUT,
        STAT_SCR,
    ) = range(38)

    def cb(i, page, n, df):
        return ttnn.CBDescriptor(
            total_size=page * n,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=df, page_size=page)],
        )

    nrecv = 1
    cbs = [
        cb(Q_RM, D * 2, H, bf),
        cb(Q_IN, TILE, Sqt * DHt, bf),
        ttnn.CBDescriptor(
            total_size=PACKED * 32 * 2,
            core_ranges=core_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=K_RM, data_format=ttnn.fp8_e4m3, page_size=PACKED),
                ttnn.CBFormatDescriptor(buffer_index=K_ROPE_RM, data_format=bf, page_size=PACKED),
            ],
        ),
        cb(K_IN, k_in_tile, Skt * vDHt, k_in_df),
        cb(NEGINF, TILE, 1, bf),
        cb(MASK_PART, TILE, 1, bf),
        cb(SCALE, TILE, 1, bf),
        cb(QK_IM, TILE, Sqt * Skt, bf),
        cb(MAX_A, TILE, Sqt, bf),
        cb(MAX_B, TILE, Sqt, bf),
        cb(SUM_A, TILE, Sqt, bf),
        cb(SUM_B, TILE, Sqt, bf),
        cb(OUT_A, TILE, Sqt * vDHt, bf),
        cb(OUT_B, TILE, Sqt * vDHt, bf),
        cb(CORR, TILE, Sqt, bf),
        cb(OUT_IM, TILE, Sqt * vDHt, bf),
        cb(OUT_RM, TILE, Sqt * vDHt, bf),
        cb(IDX, topk * 4, 1, bf),
        cb(CTRL, 32, 2, bf),
        cb(COL_ID, TILE, 1, bf),
        cb(RECIP, TILE, 1, bf),
        cb(KREQ, 32, 2, bf),
        cb(KACK, 16, 2, bf),
        cb(SINK, TILE, Sqt, bf),
        cb(SINK_SCR, H * 2, 1, bf),
        cb(RECV_O, TILE, nrecv * Sqt * vDHt, bf),
        cb(RECV_M, TILE, nrecv * Sqt, bf),
        cb(RECV_S, TILE, nrecv * Sqt, bf),
        cb(MA, TILE, 1, bf),
        cb(MB, TILE, 1, bf),
        cb(CORR_R, TILE, nrecv, bf),
        cb(K_SCALE_BC, 4096, (v_dim // 128) * 2, ttnn.float32),
        cb(K_LAT_T, 1088, vDHt, ttnn.bfloat8_b),
        cb(K_ROPE_T, TILE, Skt * (DHt - vDHt), bf),
        cb(M_OUT, TILE, Sqt, bf),
        cb(L_OUT, TILE, Sqt, bf),
        cb(STAT_SCR, 64 * H, 1, bf),
    ]
    bc = [0, 1, 1, 0, 0]
    reader_ct = [H, S, topk, k_chunk, D, Q_RM, K_RM, MASK_PART, IDX, CTRL, 2, kv_b, 4] + bc
    reader_ct += (
        [1, v_dim, K_SCALE_BC, PACKED, KREQ, KACK, G, RING, QMODE, QROWS, 0, 0, 0] + _acc(q) + _acc(kv) + _acc(idx)
    )
    writer_ct = [H, S, vDHt, OUT_RM, SCALE, COL_ID, NEGINF, 2] + bc
    writer_ct += [
        1,
        D,
        kv_b,
        IDX,
        KREQ,
        KACK,
        PACKED,
        0,
        SINK,
        SINK_SCR,
        G,
        RECV_O,
        RECV_M,
        RECV_S,
        OUT_A,
        OUT_B,
        MAX_A,
        MAX_B,
        SUM_A,
        SUM_B,
        0,
        RING,
        Q_RM,
        QROWS,
        QMODE,
        emit_stats,
        M_OUT,
        L_OUT,
        STAT_SCR,
        0,
        _bits(scale),
    ]
    writer_ct += _acc(out) + _acc(kv) + _acc(sink) + _acc(q) + _acc(out)
    fid = getattr(ckc, "math_fidelity", ttnn.MathFidelity.HiFi4)
    approx = bool(getattr(ckc, "math_approx_mode", False))
    fp32 = bool(getattr(ckc, "fp32_dest_acc_en", True)) and os.environ.get("SS_FP32", "1") == "1"
    compute_ct = [
        H,
        DHt,
        vDHt,
        Skt,
        _bits(scale),
        Q_RM,
        Q_IN,
        K_RM,
        K_IN,
        NEGINF,
        MASK_PART,
        SCALE,
        QK_IM,
        MAX_A,
        MAX_B,
        SUM_A,
        SUM_B,
        OUT_A,
        OUT_B,
        CORR,
        OUT_IM,
        OUT_RM,
        CTRL,
        COL_ID,
        RECIP,
        1,
        K_ROPE_RM,
        K_SCALE_BC,
        K_LAT_T,
        K_ROPE_T,
        int(approx),
        qsb,
        PACKED,
        0,
        SINK,
        G,
        RECV_O,
        RECV_M,
        RECV_S,
        0,
        MA,
        MB,
        CORR_R,
        0,
        emit_stats,
        M_OUT,
        L_OUT,
    ]
    rt_r, rt_w, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, (cx, cy) in enumerate(cores):
        cnt = len(range(i, rows, NC))
        ln = md.worker_core_from_logical_core(ttnn.CoreCoord(cx, cy))
        rt_r[cx][cy] = [i, 0, NC, cnt]
        rt_w[cx][cy] = [i, 0, ln.x, ln.y] + [0] * 10 + [NC, cnt]
        rt_c[cx][cy] = [i, 0, cnt]
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/ssplit_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=reader_ct,
        runtime_args=rt_r,
        common_runtime_args=[q.buffer_address(), kv.buffer_address(), idx.buffer_address(), 0],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/ssplit_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=writer_ct,
        runtime_args=rt_w,
        common_runtime_args=[
            out.buffer_address(),
            kv.buffer_address(),
            0,
            sink.buffer_address(),
            q.buffer_address(),
            out.buffer_address(),
        ],
        config=ttnn.WriterConfigDescriptor(),
    )
    cc = ttnn.ComputeConfigDescriptor(math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=fp32)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[K_RM] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/ssplit_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=compute_ct,
        runtime_args=rt_c,
        common_runtime_args=[],
        config=cc,
    )
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=core_set, initial_value=0) for i in range(2)]
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=sems, cbs=cbs)
    prog.custom_program_hash = (0x5B7 << 40) | (
        hash(
            (
                HT,
                H,
                S,
                NC,
                k_chunk,
                topk,
                str(fid),
                int(approx),
                int(fp32),
                _bits(scale),
                RING,
                tuple(_acc(q)),
                tuple(_acc(kv)),
                tuple(_acc(idx)),
                tuple(_acc(out)),
                tuple(_acc(sink)),
            )
        )
        & ((1 << 40) - 1)
    )
    ttnn.generic_op([q, kv, idx, sink, out], prog)
    return out


_SINK = {}


def _dummy_sink(md, H):
    k = (id(md), H)
    if k not in _SINK:
        _SINK[k] = ttnn.from_torch(
            torch.zeros(1, 1, 1, H, dtype=torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return _SINK[k]
