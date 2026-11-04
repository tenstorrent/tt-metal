# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""sdpa2: experiments on the decode key-split sparse_sdpa (tt/sparse_split.py). Gather-only floor bench + new variants (VARIANTS registry)."""
import os

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sdpa2_kernels")


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _core_set(n, gx):
    rngs, r = [], 0
    while n > 0:
        w = min(n, gx)
        rngs.append(ttnn.CoreRange(ttnn.CoreCoord(0, r), ttnn.CoreCoord(w - 1, r)))
        n -= w
        r += 1
    return ttnn.CoreRangeSet(rngs)


def gather_bench(kv, idx, S, G, depth, nocs, reps, mode, tag=0):
    """Pure gather: S rows x G cores, each core gathers 640/G keys REPS times. nocs: 1 (reader NoC only) | 2 (both)."""
    md = kv.device()
    grid = md.compute_with_storage_grid_size()
    n = S * G
    assert n <= grid.x * grid.y
    per = 640 // G
    cores = [(i % grid.x, i // grid.x) for i in range(n)]
    cs = _core_set(n, grid.x)
    cb = lambda i, page, k: ttnn.CBDescriptor(
        total_size=page * k,
        core_ranges=cs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)],
    )
    cbs = [cb(0, per * 512, 1), cb(1, 640 * 4, 1)]
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        rt[x][y] = [i // G, i % G]
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
    )

    def kd(src, sel, cfg):
        ct = [640, G, depth, reps, mode, sel, 0, 1] + _acc(kv) + _acc(idx)
        return ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/{src}",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cs,
            compile_time_args=ct,
            runtime_args=rt,
            common_runtime_args=[kv.buffer_address(), idx.buffer_address()],
            config=cfg,
        )

    ks = [kd("gather_bench.cpp", 1 if nocs == 2 else 2, ttnn.ReaderConfigDescriptor())]
    if nocs == 2:
        ks.append(kd("gather_bench.cpp", 0, ttnn.WriterConfigDescriptor()))
    prog = ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs)
    prog.custom_program_hash = (0x5B2 << 40) | (hash((S, G, depth, nocs, reps, mode, tag)) & ((1 << 40) - 1))
    return ttnn.generic_op([kv, idx, out], prog)


def _bits(f):
    import struct

    return struct.unpack("<I", struct.pack("<f", float(f)))[0]


SS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sparse_split_kernels")


def sparse_sdpa_v2(
    q,
    kv,
    idx,
    v_dim,
    scale,
    k_chunk,
    split,
    ckc,
    sink,
    qface=1024,
    tag="v2",
    W=4,
    out_l1=False,
    out_rows=9,
    stream=True,
    fuse_m=True,
):
    """Variant 2: same inputs / outputs as sparse_split.sparse_sdpa_split (q must be the [1,S,32,512] TILE layout), kernels sdpa2_kernels/s2_*.cpp."""
    md = q.device()
    H, S, D = 32, int(q.shape[1]), int(q.shape[3])
    topk = int(idx.shape[3])
    assert q.layout == ttnn.TILE_LAYOUT and D == v_dim == 512 and q.dtype == ttnn.bfloat16 and idx.dtype == ttnn.uint32
    assert kv.dtype in (ttnn.bfloat16, ttnn.fp8_e4m3)
    grid = md.compute_with_storage_grid_size()
    G = max(1, min(int(split), (grid.x * grid.y) // S))
    n_cores = S * G
    n_chunks_row = -(-topk // k_chunk)
    assert G == 1 or -(-n_chunks_row // G) <= 2
    DHt = vDHt = D // 32
    Skt = k_chunk // 32
    Sqt = 1
    fp8 = kv.dtype == ttnn.fp8_e4m3
    kv_b = 1 if fp8 else 2
    TILE = 2048
    k_in_tile = 1088 if fp8 else 2048
    k_in_df = ttnn.bfloat8_b if fp8 else ttnn.bfloat16
    bf = ttnn.bfloat16
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, H, S, v_dim]),
        bf,
        ttnn.ROW_MAJOR_LAYOUT,
        md,
        ttnn.L1_MEMORY_CONFIG
        if (out_l1 or os.environ.get("DSV41_SDPAOUT_L1", os.environ.get("DSV41_SDPA_L1")) == "1")
        else ttnn.DRAM_MEMORY_CONFIG,
    )
    cores = [(i % grid.x, i // grid.x) for i in range(n_cores)]
    core_set = _core_set(n_cores, grid.x)
    (Q_RM, Q_IN, K_RM, K_IN, NEGINF, MASK_PART, SCALE, QK_IM, MAX_A, MAX_B, SUM_A, SUM_B, OUT_A, OUT_B, CORR, OUT_IM,
     OUT_RM, IDX, CTRL, COL_ID, RECIP, KREQ, KACK, K_ROPE_RM, K_SCALE_BC, K_LAT_T, K_ROPE_T, SINK, SINK_SCR, RECV_O,
     RECV_M, RECV_S, MA, MB, CORR_R) = range(35)  # fmt: skip

    def cb(i, page, n, df):
        return ttnn.CBDescriptor(
            total_size=page * n,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=df, page_size=page)],
        )

    nrecv = G
    assert G >= 16 // W, "sliced merge: at least vDHt / W cores per row"
    cbs = [
        cb(Q_RM, D * 2, 1, bf),
        cb(Q_IN, TILE, Sqt * DHt, bf),
        cb(K_RM, D * kv_b, k_chunk, kv.dtype),
        cb(K_IN, k_in_tile, Skt * DHt, k_in_df),
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
        cb(RECV_O, TILE, nrecv * W, bf),
        cb(RECV_M, TILE, nrecv * Sqt, bf),
        cb(RECV_S, TILE, nrecv * Sqt, bf),
        cb(MA, TILE, 1, bf),
        cb(MB, TILE, 1, bf),
        cb(CORR_R, TILE, nrecv, bf),
    ]
    reader_ct = (
        [S, topk, k_chunk, D, K_RM, MASK_PART, IDX, CTRL, kv_b, KREQ, KACK, G, Q_IN, qface]
        + _acc(q)
        + _acc(kv)
        + _acc(idx)
    )
    writer_ct = [S, vDHt, OUT_RM, SCALE, COL_ID, NEGINF, 2, kv_b, IDX, KREQ, KACK, SINK, SINK_SCR, G, RECV_O, RECV_M, RECV_S,
                 OUT_A, OUT_B, MAX_A, MAX_B, SUM_A, SUM_B, H, D, W, out_rows, int(stream)] + _acc(out) + _acc(kv) + _acc(sink)  # fmt: skip
    fid = getattr(ckc, "math_fidelity", ttnn.MathFidelity.HiFi4)
    approx = bool(getattr(ckc, "math_approx_mode", False))
    fp32 = bool(getattr(ckc, "fp32_dest_acc_en", True))
    dst_size = 4 if fp32 else 8
    qsb = 1
    compute_ct = [H, DHt, vDHt, Skt, _bits(scale), Q_RM, Q_IN, K_RM, K_IN, NEGINF, MASK_PART, SCALE, QK_IM, MAX_A, MAX_B,
                  SUM_A, SUM_B, OUT_A, OUT_B, CORR, OUT_IM, OUT_RM, CTRL, COL_ID, RECIP, 0, K_ROPE_RM, K_SCALE_BC, K_LAT_T,
                  K_ROPE_T, int(approx), qsb, 0, 1, SINK, G, RECV_O, RECV_M, RECV_S, 0, MA, MB, CORR_R, 1, 1, W, int(stream), int(fuse_m)]  # fmt: skip
    rt_r, rt_w, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, (cx, cy) in enumerate(cores):
        t, g = i // G, i % G
        lx, ly = cores[t * G]
        ln = md.worker_core_from_logical_core(ttnn.CoreCoord(lx, ly))
        rt_r[cx][cy] = [t, g]
        extra = []
        for m in range(1, 16 // W):
            mx_, my_ = cores[t * G + m]
            mn = md.worker_core_from_logical_core(ttnn.CoreCoord(mx_, my_))
            extra += [mn.x, mn.y]
        rt_w[cx][cy] = [t, g, ln.x, ln.y] + extra
        rt_c[cx][cy] = [t, g]
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/s2_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=reader_ct,
        runtime_args=rt_r,
        common_runtime_args=[q.buffer_address(), kv.buffer_address(), idx.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/s2_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=writer_ct,
        runtime_args=rt_w,
        common_runtime_args=[out.buffer_address(), kv.buffer_address(), sink.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    cc = ttnn.ComputeConfigDescriptor(math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=fp32)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    if fp8:
        modes[K_RM] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/s2_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=compute_ct,
        runtime_args=rt_c,
        common_runtime_args=[],
        config=cc,
    )
    prog = ttnn.ProgramDescriptor(
        kernels=[reader, writer, compute],
        semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=core_set, initial_value=0) for i in range(G + 1)],
        cbs=cbs,
    )
    prog.custom_program_hash = (0x5C3 << 40) | (
        hash((tag, W, out_l1, out_rows, stream, fuse_m, S, G, k_chunk, topk, int(fp8), str(fid), int(approx), int(fp32), _bits(scale), qface, tuple(_acc(q)),
              tuple(_acc(kv)), tuple(_acc(idx)), tuple(_acc(out)), tuple(_acc(sink))))
        & ((1 << 40) - 1)
    )  # fmt: skip
    return ttnn.generic_op([q, kv, idx, sink, out], prog)


def _v2(q, kv, idx, scale, ckc, sink, out_l1=False):
    return sparse_sdpa_v2(
        q,
        kv,
        idx,
        512,
        scale,
        int(os.environ.get("SS2_KC", "128")),
        int(os.environ.get("SS2_G", "5")),
        ckc,
        sink,
        qface=int(os.environ.get("SS2_QFACE", "1024")),
        W=int(os.environ.get("SS2_W", "4")),
        out_l1=out_l1,
        out_rows=int(os.environ.get("SS2_OUTROWS", "9")),
        stream=os.environ.get("SS2_STREAM", "1") == "1",
        fuse_m=os.environ.get("SS2_FUSEM", "1") == "1",
    )


VARIANTS = {"v2": _v2, "v2l1": lambda q, kv, idx, scale, ckc, sink: _v2(q, kv, idx, scale, ckc, sink, out_l1=True)}
