# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Key-split ``sparse_sdpa`` for DECODE (rows of one verify block: T <= ~16): one core per (query row, key-chunk share) instead of one core per row.

``ttnn.transformer.sparse_sdpa`` gives each query row ONE core that loops over all its top-k (640) keys (8 rows -> 8 of 120 cores, 72 us at 8k).
Here the row's k-chunks are shared by ``split`` cores (the flash-decode layout of ttnn sdpa_decode: ``sdpa_decode_program_factory.cpp:172-216`` core
assignment, ``kernels/compute/sdpa_flash_decode.cpp:583-720`` + ``ring_joint_sdpa.cpp:318-355`` K-split merge): every core runs the stock sparse_sdpa
flash loop on its chunks, the non-leader cores NoC-write their (out, max, sum) state into the leader's L1 (semaphore), the leader merges the states
(SALAD merge, exact softmax) and normalises with the attention sink. Kernels: ``sparse_split_kernels/`` (the stock sparse_sdpa kernels + the split
protocol). Same inputs / outputs as ``ttnn.transformer.sparse_sdpa`` (plain bf16 / fp8_e4m3 kv).
"""
import os
import struct

import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_split_kernels")


_DUMMY_SINK = {}


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


def max_split(rows, grid_x, grid_y):
    return max(1, (grid_x * grid_y) // max(1, rows))


def sparse_sdpa_split(q, kv, idx, v_dim, scale, k_chunk, split, ckc, sink, emit_stats=False):
    """q: bf16 ROW_MAJOR [1,H=32,S,D]; kv: fp8_e4m3 / bf16 ROW_MAJOR [1,1,R,D]; idx: uint32 ROW_MAJOR [1,1,S,TOPK]; sink: bf16 ROW_MAJOR [1,1,1,H].
    ``ckc``: the device compute kernel config (fidelity / approx / fp32 acc are read from it). -> bf16 ROW_MAJOR [1,H,S,v_dim].
    """
    md = q.device()
    qt = (
        q.layout == ttnn.TILE_LAYOUT
    )  # q [1,S,32,512] TILE (qk_heads output): read as tile pages, no RM regroup / tilize
    QLAY = int(os.environ.get("SS_QLAY", "0"))  # 1: q is [1,S,H,D] row-major (token-major pages)
    if qt:
        H, S, D = 32, int(q.shape[1]), int(q.shape[3])
    elif QLAY:
        H, S, D = int(q.shape[2]), int(q.shape[1]), int(q.shape[3])
    else:
        H, S, D = int(q.shape[1]), int(q.shape[2]), int(q.shape[3])
    topk = int(idx.shape[3])
    assert H == 32 and v_dim == 512 and D in (512, 576)
    assert q.dtype == ttnn.bfloat16 and idx.dtype == ttnn.uint32 and kv.dtype in (ttnn.bfloat16, ttnn.fp8_e4m3)
    # SCALED_FP8 KV (the stock sparse_sdpa packed format): fp8 bytes, row = 512 fp8 latent + 4 fp32 scales + 64 bf16 rope = 656 B, stored at the 64 B aligned stride 704
    scaled = kv.dtype == ttnn.fp8_e4m3 and int(kv.shape[3]) == v_dim + 16 + (D - v_dim) * 2
    if emit_stats:
        assert sink is None, "emit_stats: the sink is applied by the cross-column merge, not in the kernel"
    else:
        assert sink is not None
    grid = md.compute_with_storage_grid_size()
    G = max(1, min(int(split), max_split(S, grid.x, grid.y)))
    n_cores = S * G
    RING = int(os.environ.get("SS_RING", "8"))
    QMODE = int(os.environ.get("SS_QMODE", "1"))
    QROWS = int(os.environ.get("SS_QROWS", "32"))  # q head rows loaded (rest zero-filled); Hy4 pads 8 real heads to 32
    if qt:
        QMODE = 2
    assert n_cores <= grid.x * grid.y
    n_chunks_row = -(-topk // k_chunk)
    assert G == 1 or -(-n_chunks_row // G) <= 2, "senders hold at most 2 chunks (ping-pong CB parity)"
    DHt, vDHt = (
        D // 32,
        v_dim // 32,
    )  # D=576, V=512: 18 K tiles, 16 V tiles (the DSV4 original had DHt = vDHt, which made the writer emit 1152 B rows into 1024 B output pages: S>=5 race, the 'S>=12 PCC 0.93' bug)
    Skt = k_chunk // 32
    Sqt = H // 32
    assert topk % k_chunk == 0 or True
    fp8 = kv.dtype == ttnn.fp8_e4m3 and not scaled  # plain fp8_e4m3 rows (no block scales)
    kv_b = 1 if kv.dtype == ttnn.fp8_e4m3 else 2
    PACKED = ((v_dim + 16 + (D - v_dim) * 2 + 63) // 64) * 64 if scaled else 0  # 704
    TILE = 2048
    bf = ttnn.bfloat16
    k_in_tile = 1088 if (fp8 or scaled) else 2048
    k_in_df = ttnn.bfloat8_b if (fp8 or scaled) else ttnn.bfloat16
    if (
        sink is None
    ):  # dummy (the kernel ignores it when emit_stats); created once per device (host writes are illegal during trace capture)
        key = id(md)
        if key not in _DUMMY_SINK:
            _DUMMY_SINK[key] = ttnn.from_torch(
                torch.zeros(1, 1, 1, H, dtype=torch.bfloat16),
                device=md,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        sink = _DUMMY_SINK[key]
    use_sink = 0 if emit_stats else 1
    stats = (
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, S, 64]), bf, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
        )
        if emit_stats
        else None
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, H, S, v_dim]),
        bf,
        ttnn.ROW_MAJOR_LAYOUT,
        md,
        ttnn.L1_MEMORY_CONFIG
        if os.environ.get("DSV41_SDPAOUT_L1", os.environ.get("DSV41_SDPA_L1")) == "1"
        else ttnn.DRAM_MEMORY_CONFIG,
    )
    cores = [(i % grid.x, i // grid.x) for i in range(n_cores)]
    core_set = _core_set(n_cores, grid.x)

    # CB ids (the stock SparseCB enum + the partial-state receive CBs)
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

    nrecv = max(1, G - 1)
    cbs = [
        cb(Q_RM, D * 2, 1 if qt else H, bf),  # unused with tile-page q (QTILE): keep the L1 footprint low
        cb(Q_IN, TILE, Sqt * DHt, bf),
        # scaled: one double-buffered 32-row slab FIFO (owner K_RM fp8 + format-only bf16 alias K_ROPE_RM over the same bytes), as in the stock sparse_sdpa factory
        ttnn.CBDescriptor(
            total_size=PACKED * 32 * 2,
            core_ranges=core_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=K_RM, data_format=ttnn.fp8_e4m3, page_size=PACKED),
                ttnn.CBFormatDescriptor(buffer_index=K_ROPE_RM, data_format=bf, page_size=PACKED),
            ],
        )
        if scaled
        else cb(K_RM, D * kv_b, k_chunk, kv.dtype),
        cb(K_IN, k_in_tile, Skt * (vDHt if scaled else DHt), k_in_df),
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
    ]
    if scaled:
        cbs += [
            cb(K_SCALE_BC, 4096, (v_dim // 128) * 2, ttnn.float32),
            cb(K_LAT_T, 1088, vDHt, ttnn.bfloat8_b),
            cb(K_ROPE_T, TILE, Skt * (DHt - vDHt), bf),
        ]
    if emit_stats:
        cbs += [cb(M_OUT, TILE, 1, bf), cb(L_OUT, TILE, 1, bf), cb(STAT_SCR, 128, 1, bf)]
    bc = [0, 1, 1, 0, 0]
    reader_ct = [H, S, topk, k_chunk, D, Q_RM, K_RM, MASK_PART, IDX, CTRL, 2, kv_b, 4] + bc
    reader_ct += (
        [
            int(scaled),
            v_dim,
            K_SCALE_BC,
            PACKED,
            KREQ,
            KACK,
            G,
            RING,
            QMODE,
            (Q_IN if qt else QROWS),
            QLAY,
            int(os.environ.get("SS_KSPLIT", "0")),
        ]
        + _acc(q)
        + _acc(kv)
        + _acc(idx)
    )
    writer_ct = [H, S, vDHt, OUT_RM, SCALE, COL_ID, NEGINF, 2] + bc
    writer_ct += [
        int(scaled),
        D,
        kv_b,
        IDX,
        KREQ,
        KACK,
        PACKED,
        use_sink,
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
        int(emit_stats),
        M_OUT,
        L_OUT,
        STAT_SCR,
    ]
    writer_ct += _acc(out) + _acc(kv) + _acc(sink) + _acc(q) + (_acc(stats) if emit_stats else _acc(out))
    fid = getattr(ckc, "math_fidelity", ttnn.MathFidelity.HiFi4)
    approx = bool(getattr(ckc, "math_approx_mode", False))
    fp32 = bool(getattr(ckc, "fp32_dest_acc_en", True)) and os.environ.get("SS_FP32", "1") == "1"
    dst_size = 4 if fp32 else 8
    qsb = 1
    for d in range(min(Sqt, dst_size), 0, -1):
        if Sqt % d == 0:
            qsb = d
            break
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
        int(scaled),
        K_ROPE_RM,
        K_SCALE_BC,
        K_LAT_T,
        K_ROPE_T,
        int(approx),
        qsb,
        PACKED,
        use_sink,
        SINK,
        G,
        RECV_O,
        RECV_M,
        RECV_S,
        int(os.environ.get("SS_DBG", "0")),
        MA,
        MB,
        CORR_R,
        int(qt),
        int(emit_stats),
        M_OUT,
        L_OUT,
    ]
    rt_r, rt_w, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, (cx, cy) in enumerate(cores):
        t, g = i // G, i % G
        lx, ly = cores[t * G]
        ln = md.worker_core_from_logical_core(ttnn.CoreCoord(lx, ly))
        rt_r[cx][cy] = [t, g]
        rt_w[cx][cy] = [t, g, ln.x, ln.y]
        rt_c[cx][cy] = [t, g]
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
            (stats if emit_stats else out).buffer_address(),
        ],
        config=ttnn.WriterConfigDescriptor(),
    )
    cc = ttnn.ComputeConfigDescriptor(math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=fp32)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    if fp8 or scaled:
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
    prog = ttnn.ProgramDescriptor(
        kernels=[reader, writer, compute],
        semaphores=[
            ttnn.SemaphoreDescriptor(id=i, core_ranges=core_set, initial_value=0) for i in range(max(1, 2 * (G - 1)))
        ],
        cbs=cbs,
    )
    prog.custom_program_hash = (0x5A1 << 40) | (
        hash(
            (
                S,
                G,
                k_chunk,
                topk,
                int(fp8),
                int(scaled),
                int(emit_stats),
                D,
                v_dim,
                RING,
                QMODE,
                QROWS,
                QLAY,
                int(os.environ.get("SS_KSPLIT", "0")),
                str(fid),
                int(approx),
                int(fp32),
                _bits(scale),
                tuple(_acc(q)),
                tuple(_acc(kv)),
                tuple(_acc(idx)),
                tuple(_acc(out)),
                tuple(_acc(sink)),
            )
        )
        & ((1 << 40) - 1)
    )
    if emit_stats:
        ttnn.generic_op([q, kv, idx, sink, stats, out], prog)
        return out, stats
    return ttnn.generic_op([q, kv, idx, sink, out], prog)
