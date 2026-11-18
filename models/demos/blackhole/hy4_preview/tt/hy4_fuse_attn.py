# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""B17 attention-chain fusion (flag HY4_ATTN_FUSE=1, default OFF).

fused_back: o_lat head-major [1,8,S,512] (S <= 8 rows) TILE bf16 (the merge writer output, HY4_ATTN_HM=1, S == 1) x w_uv_t [1,8,512,256] TILE -> * sigmoid(gate) -> [1,1,1,2048] TILE
(o-proj input): replaces matmul + nlp_concat_heads + multiply (3 ops, ~24 us + gaps) with ONE program of 64 cores (core = head x output tile column, 16 K tiles each).
"""
import os

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_fuse_attn_kernels")


def enabled(part=None):
    """HY4_ATTN_FUSE=1 (all parts) or a comma list of parts: q (fused q-prep), back (fused value up-projection + gate)."""
    v = os.environ.get("HY4_ATTN_FUSE", "0")
    if v in ("0", ""):
        return False
    return part is None or v == "1" or part in v.split(",")


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


def fused_back(o_lat, w_uv_t, g, gfold, out=None):
    md = o_lat.device()
    H, KT, NT = int(w_uv_t.shape[1]), int(w_uv_t.shape[2]) // 32, int(w_uv_t.shape[3]) // 32
    assert int(o_lat.shape[1]) == H and int(o_lat.shape[3]) == KT * 32 and o_lat.layout == ttnn.TILE_LAYOUT
    assert w_uv_t.layout == ttnn.TILE_LAYOUT and w_uv_t.dtype == ttnn.bfloat16 and g.layout == ttnn.TILE_LAYOUT
    S = int(o_lat.shape[2])
    assert 1 <= S <= 8
    n_cores = H * NT
    grid = md.compute_with_storage_grid_size()
    assert n_cores <= grid.x * grid.y
    if out is None:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, S, H * NT * 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
        )
    core_set = _core_set(n_cores, grid.x)
    CX, CW, CG, CO = range(4)
    bf = ttnn.bfloat16

    def cb(i, n):
        return ttnn.CBDescriptor(
            total_size=2048 * n,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=bf, page_size=2048)],
        )

    cbs = [cb(CX, KT), cb(CW, KT), cb(CG, 1), cb(CO, 1)]
    rt = ttnn.RuntimeArgs()
    for i in range(n_cores):
        rt[i % grid.x][i // grid.x] = [i]
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/hb_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CX, CW, CG, KT, NT, S] + _acc(o_lat) + _acc(w_uv_t) + _acc(g),
        runtime_args=rt,
        common_runtime_args=[o_lat.buffer_address(), w_uv_t.buffer_address(), g.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/hb_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CO] + _acc(out),
        runtime_args=rt,
        common_runtime_args=[out.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/hb_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[
            CX,
            CW,
            CG,
            CO,
            KT,
            int(bool(gfold)),
            int(os.environ.get("HY4_FA_ROUND", "1")),
            int(os.environ.get("HY4_FA_SIGFAST", "0")),
        ],
        runtime_args=ttnn.RuntimeArgs(),
        common_runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        ),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = (0x3B1 << 40) | (
        hash(
            (
                H,
                KT,
                NT,
                S,
                int(bool(gfold)),
                os.environ.get("HY4_FA_ROUND", "1"),
                os.environ.get("HY4_FA_SIGFAST", "0"),
                tuple(_acc(o_lat)),
                tuple(_acc(w_uv_t)),
                tuple(_acc(g)),
                tuple(_acc(out)),
            )
        )
        & ((1 << 40) - 1)
    )
    ttnn.generic_op([o_lat, w_uv_t, g, out], prog)
    return out


def fused_q(q, w_uk, cos, sin, tm, out=None, kv=None, push=None):
    """q [1,1,1,H*256] TILE bf16 (q_b output, head h = cols 256h.. : 192 nope | 64 pe), w_uk [1,H,192,512] TILE bf16, cos/sin [1,1,1,64] TILE, tm [1,1,32,32] TILE rotary trans mat
    -> RM bf16 [1,H,1,576] = per head [q_nope @ w_uk | rope(q_pe)] (the layout the q all-gather / sparse split kernel read): replaces nlp_create_q_heads_split + matmul + rope + concat + to_layout.
    """
    md = q.device()
    H = int(w_uk.shape[1])
    S = int(q.shape[2])
    assert 1 <= S <= 8
    assert int(w_uk.shape[2]) == 192 and int(w_uk.shape[3]) == 512 and int(q.shape[3]) == H * 256
    assert all(t.layout == ttnn.TILE_LAYOUT and t.dtype == ttnn.bfloat16 for t in (q, w_uk, cos, sin, tm))
    grid = md.compute_with_storage_grid_size()
    n_abs = H * 8
    n_cores = n_abs + H
    assert n_cores <= grid.x * grid.y
    if out is None:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, H, S, 576]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
        )
    core_set = _core_set(n_cores, grid.x)
    abs_set = _core_set(n_abs, grid.x)
    rope_cores = [(i % grid.x, i // grid.x) for i in range(n_abs, n_cores)]
    rope_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in rope_cores])
    CX, CW, CC, CS, CT, CO, CSC = range(7)
    bf = ttnn.bfloat16

    def cb(i, n):
        return ttnn.CBDescriptor(
            total_size=2048 * n,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=bf, page_size=2048)],
        )

    cbs = [cb(CX, 6), cb(CW, 12), cb(CC, 2), cb(CS, 2), cb(CT, 1), cb(CO, 2), cb(CSC, 1)]
    rt = ttnn.RuntimeArgs()
    comm_xy = [v for xy in push.comm_phys() for v in xy] if push is not None else []
    for i in range(n_cores):
        cx, cy = i % grid.x, i // grid.x
        rt[cx][cy] = ([0, i // 8, i % 8] if i < n_abs else [1, i - n_abs, 0]) + comm_xy
    rt_c = ttnn.RuntimeArgs()
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/qp_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CX, CW, CC, CS, CT, S] + _acc(q) + _acc(w_uk) + _acc(cos) + _acc(sin) + _acc(tm),
        runtime_args=rt,
        common_runtime_args=[t.buffer_address() for t in (q, w_uk, cos, sin, tm)],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/qp_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CO, CSC, S] + _acc(out),
        runtime_args=rt,
        common_runtime_args=[out.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    if (
        push is not None
    ):  # B47b (HY4_B47B_QFOLD): the writers signal the push comm cores (QF_NC of them) when their q rows are in DRAM
        writer.defines = [("QF_NC", str(len(comm_xy) // 2))]
    cc = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    c_abs = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/qp_compute_mm.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=abs_set,
        compile_time_args=[CX, CW, CO],
        runtime_args=rt_c,
        common_runtime_args=[],
        config=cc,
    )
    c_rope = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/qp_compute_rope.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=rope_set,
        compile_time_args=[CX, CC, CS, CT, CO],
        runtime_args=rt_c,
        common_runtime_args=[],
        config=cc,
    )
    kernels = [reader, writer, c_abs, c_rope]
    extra_t, kvh = [], ()
    if kv is not None:  # B20 (HY4_KV_FUSE=2): the KV append runs concurrently on 4 free cores of this program
        lat, k_pe, ids, pool = kv
        kcores = [(x, grid.y - 2) for x in range(4)]
        assert not any(c in kcores for c in rope_cores) and n_cores <= (grid.y - 2) * grid.x
        kk, kc = _kv_parts(lat, k_pe, ids, pool, S, kcores)
        kernels += kk
        cbs += kc
        extra_t = [lat, k_pe, ids, pool]
        kvh = tuple(tuple(_acc(t)) for t in extra_t)
    tag = (H, S, kvh, tuple(tuple(_acc(t)) for t in (q, w_uk, cos, sin, tm, out)))
    if push is not None:
        push.fold_nsig = n_cores
        mpd, tl = push.fold_mesh(kernels, [], cbs, out, [q, w_uk, cos, sin, tm, out] + extra_t, tag)
        ttnn.generic_op(tl, mpd)
        return out
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    prog.custom_program_hash = (0x3B2 << 40) | (hash(tag) & ((1 << 40) - 1))
    ttnn.generic_op([q, w_uk, cos, sin, tm, out] + extra_t, prog)
    return out


STOCK_CAST_KERNEL = (
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/kernels/compute/"
    "compute_per_token_cast_to_fp8.cpp"
)


def kv_enabled():
    """B20: HY4_KV_FUSE=1 (default OFF): per_token_cast_to_fp8 + to_layout + pack_scaled_fp8_kv_cache + row_write as ONE 4-core program;
    HY4_KV_FUSE=2: the same work as 4 extra cores of the fused_q program (needs HY4_ATTN_FUSE q)."""
    return os.environ.get("HY4_KV_FUSE", "0") in ("1", "2")


def kv_in_q():
    return os.environ.get("HY4_KV_FUSE", "0") == "2"


def _kv_parts(lat, k_pe, ids, pool, S, cores):
    """kernel + CB descriptors of the fused KV append on the 4 given cores (list of (x, y))."""
    import struct

    assert 1 <= S <= 8 and int(lat.shape[3]) == 512 and int(k_pe.shape[3]) == 64
    assert lat.layout == ttnn.TILE_LAYOUT and lat.dtype == ttnn.bfloat16
    assert k_pe.layout == ttnn.TILE_LAYOUT and k_pe.dtype == ttnn.bfloat16
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    fp32, bf, e4 = ttnn.float32, ttnn.bfloat16, ttnn.fp8_e4m3

    def cb(i, dt, page, n):
        return ttnn.CBDescriptor(
            total_size=page * n,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
        )

    cbs = [
        cb(0, bf, 2048, 8),  # in (2 blocks of 4 tiles)
        cb(1, fp32, 4096, 4),  # tile
        cb(3, fp32, 4096, 1),  # scaler
        cb(4, fp32, 4096, 8),  # abs
        cb(5, fp32, 4096, 2),  # scale tile
        cb(7, fp32, 4096, 2),  # 1/scale
        cb(16, e4, 1024, 8),  # e4m3 out
        cb(6, fp32, 4096, 1),  # stage (ids | kpe tile | scale/rope staging)
    ]
    rt = ttnn.RuntimeArgs()
    for b, (x, yy) in enumerate(cores):
        rt[x][yy] = [b]
    defs = [("INPUT_TILE_LAYOUT", "1")]
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/kv_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[0, 3] + _acc(lat),
        runtime_args=rt,
        common_runtime_args=[lat.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/kv_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[16, 5, 6, S, 704] + _acc(k_pe) + _acc(ids) + _acc(pool),
        runtime_args=rt,
        common_runtime_args=[k_pe.buffer_address(), ids.buffer_address(), pool.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    bits = lambda f: struct.unpack("<I", struct.pack("<f", f))[0]
    compute = ttnn.KernelDescriptor(
        kernel_source=STOCK_CAST_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[0, 1, 3, 4, 5, 7, 16, bits(1.0e-4), bits(3.0e38), bits(1.0 / 448.0), 32, 1],
        runtime_args=_rt_all(cores, [1]),
        common_runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        ),
    )
    for k in (reader, compute):
        k.defines = defs
    return [reader, writer, compute], cbs


def fused_kv_append(lat, k_pe, ids, pool, S):
    """lat [1,1,S,512] bf16 TILE, k_pe [1,1,S,64] bf16 TILE, ids uint32 [.,16] per device (0xFFFFFFFF = skip), pool row-major pages of 704 B (SCALED_FP8 656 B rows):
    pool[ids[r]] = pack(e4m3(lat[r]), scales(lat[r]), k_pe[r]) in place. The stock quantiser compute kernel is reused unchanged (block b = 128 columns on core b).
    """
    md = lat.device()
    grid = md.compute_with_storage_grid_size()
    cores = [(x, grid.y - 2) for x in range(4)]
    kernels, cbs = _kv_parts(lat, k_pe, ids, pool, S, cores)
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    prog.custom_program_hash = (0x3B3 << 40) | (
        hash((S, tuple(tuple(_acc(t)) for t in (lat, k_pe, ids, pool)))) & ((1 << 40) - 1)
    )
    ttnn.generic_op([lat, k_pe, ids, pool], prog)


def _rt_all(cores, args):
    rt = ttnn.RuntimeArgs()
    for x, y in cores:
        rt[x][y] = list(args)
    return rt


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# B22 (HY4_NORM_FUSE=1, default OFF): the six small front-norm programs of a decode layer (q_a to_memory_config -> rms_norm -> to_memory_config, kv slice -> rms_norm, k_pe slice)
# as ONE program: q_a rms on 16 cores (4 tiles each), kv latent rms on 4 cores (4 tiles each), k_pe tiles copied by 2 of the kv cores; the per-row sum of squares is exchanged between the
# cores of a group through flag-word slots (iHC fused-pre exchange pattern).
def norm_enabled():
    return os.environ.get("HY4_NORM_FUSE", "0") == "1"


def _bits(v):
    import struct

    return struct.unpack("<I", struct.pack("<f", v))[0]


NF_Q_ORIGIN = (
    0,
    int(os.environ.get("HY4_NORM_Y0", "0")),
)  # logical (x, y) of core 0 of the q group; q: 4 x 4 cores, kv group: the 4 cores of the row below


_QAP = {}


def qa_push(md):
    """B29: one Push object (8 tiles of 2 KiB per source column, 2 tiles per packet) for the q_a shards, per mesh"""
    if id(md) not in _QAP:
        from models.demos.blackhole.hy4_preview.tt import hy4_qpush

        _QAP[id(md)] = hy4_qpush.Push(md, 8, 2048, 2)
    return _QAP[id(md)]


def fused_norms(qa_g, kv, qw, kvw, eps, mc_q=None, push=None, qa_src=None):
    """qa_g [1,1,S,2048] bf16 TILE, kv [1,1,S,576] bf16 TILE, qw [1,1,64,32] / kvw [1,1,16,32] bf16 ROW_MAJOR (the rms weights, one 64 B page per 32 values), S <= 8
    -> (qa [1,1,S,2048], lat [1,1,S,512], k_pe [1,1,S,64]) bf16 TILE: qa = rms(qa_g) * qw, lat = rms(kv[..., :512]) * kvw, k_pe = kv[..., 512:] (replaces 6 ops).
    """
    # B29: push is not None -> qa_src [1,1,S,256] TILE (this column's q_a shard) is pushed to the 8 chips (hy4_qpush.Push) instead of all-gathered; qa_g is then None and the q-group readers wait for the flag of their source column
    qa_g = qa_g if push is None else push.land
    shard = qa_src if push is not None else None
    md = kv.device()
    S = int(kv.shape[2]) if push is not None else int(qa_g.shape[2])
    assert 1 <= S <= 8 and int(kv.shape[3]) == 576 and int(kv.shape[2]) == S
    assert push is not None or int(qa_g.shape[3]) == 2048
    bf, f32 = ttnn.bfloat16, ttnn.float32
    # B22fix: the three outputs (6 KB per L1 bank, rewritten by every layer of a decode trace) default to DRAM. In L1 they sit in the topmost few KB of the L1 buffer region, which is also where persistent L1 buffers
    # created lazily AFTER the trace captures land (GlobalSemaphores of new all_gather / all_broadcast workloads of longer prompts): the next trace replay overwrote one and the following prefill hung on it
    # (RESULT_B22FIX.md). HY4_NORM_OUT=l1 restores the model's L1 choice (the original B22 behaviour; do not use in a server). Numerics identical, cost +0.02 ms at NL=16 (8.77 -> 8.79).
    mq = (
        ttnn.DRAM_MEMORY_CONFIG if os.environ.get("HY4_NORM_OUT", "dram") != "l1" else (mc_q or ttnn.DRAM_MEMORY_CONFIG)
    )
    if push is not None:
        assert (
            int(shard.shape[3]) * 8 == 2048
            and (int(shard.shape[3]) // 32) % 4 == 0
            and push.nr == int(shard.shape[3]) // 32
        )
        push(shard)  # the push program (returns without waiting for the peers)
    qa = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, S, 2048]), bf, ttnn.TILE_LAYOUT, md, mq)
    lat = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, S, 512]), bf, ttnn.TILE_LAYOUT, md, mq)
    kpe = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, S, 64]), bf, ttnn.TILE_LAYOUT, md, mq)
    x0, y0 = NF_Q_ORIGIN
    NW = 4
    q_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x0 + NW - 1, y0 + 3))])
    k_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0 + 4), ttnn.CoreCoord(x0 + NW - 1, y0 + 4))])

    def phys(x, y):
        return md.worker_core_from_logical_core(ttnn.CoreCoord(x, y))

    p0 = phys(x0, y0)
    assert phys(x0 + NW - 1, y0 + 3).x - p0.x == NW - 1 and phys(x0 + NW - 1, y0 + 3).y - p0.y == 3
    pk = phys(x0, y0 + 4)
    assert phys(x0 + NW - 1, y0 + 4).x - pk.x == NW - 1
    cc = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    kernels, cbs = [], []
    for gi, (cset, n_g, xin, wgt, outt, w_el, kpe_on, (px, py), ncy) in enumerate(
        [
            (q_set, 16, qa_g, qw, qa, 2048, 0, (p0.x, p0.y), 4),
            (k_set, 4, kv, kvw, lat, 512, 1, (pk.x, pk.y), 1),
        ]
    ):
        TPC = w_el // 32 // n_g
        cb0 = 10 * gi

        def cb(i, dt, page, n, cset=cset, cb0=cb0):
            return ttnn.CBDescriptor(
                total_size=page * n,
                core_ranges=cset,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb0 + i, data_format=dt, page_size=page)],
            )

        CX, CG, CONES, CP2, CSLOT, CTOT, CKPE, CRS, CTMP, COUT = [cb0 + i for i in range(10)]
        cbs += [
            cb(0, bf, 2048, TPC),
            cb(1, bf, 2048, TPC),
            cb(2, f32, 4096, 1),
            cb(3, f32, 4096, 1),
            cb(4, f32, 4096, 1),
            cb(5, f32, 4096, 1),
            cb(6, bf, 2048, 1),
            cb(7, f32, 4096, 1),
            cb(8, f32, 4096, TPC),
            cb(9, bf, 2048, TPC),
        ]
        rt = ttnn.RuntimeArgs()
        for k in range(n_g):
            rt[x0 + (k % NW)][(y0 if gi == 0 else y0 + 4) + k // NW] = [k] + (
                push.split_args() if (gi == 0 and push is not None) else []
            )
        rd = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/nf_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cset,
            compile_time_args=[
                CX,
                CG,
                CONES,
                CP2,
                CSLOT,
                CTOT,
                CKPE,
                S,
                TPC,
                n_g,
                px,
                py,
                NW,
                _bits(1.0 / w_el),
                kpe_on,
                int(gi == 0 and push is not None),
                (int(shard.shape[3]) // 32) if (gi == 0 and push is not None) else 0,
                push.np if (gi == 0 and push is not None) else 0,
                push.rp if (gi == 0 and push is not None) else 0,
            ]
            + _acc(xin)
            + _acc(wgt)
            + _acc(kpe),
            runtime_args=rt,
            common_runtime_args=[xin.buffer_address(), wgt.buffer_address(), kpe.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        wr = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/nf_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cset,
            compile_time_args=[COUT, TPC] + _acc(outt),
            runtime_args=rt,
            common_runtime_args=[outt.buffer_address()],
            config=ttnn.WriterConfigDescriptor(),
        )
        cp = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/nf_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cset,
            compile_time_args=[CX, CG, CONES, CP2, CTOT, CRS, CTMP, COUT, TPC, _bits(eps)],
            runtime_args=ttnn.RuntimeArgs(),
            common_runtime_args=[],
            config=cc,
        )
        kernels += [rd, wr, cp]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    prog.custom_program_hash = (0x3B4 << 40) | (
        hash((S, x0, y0, push is not None, tuple(tuple(_acc(t)) for t in (qa_g, kv, qw, kvw, qa, lat, kpe))))
        & ((1 << 40) - 1)
    )
    if os.environ.get("HY4_NORM_DBG", "") == "1" and not getattr(fused_norms, "_dbg", 0) > 6:
        fused_norms._dbg = getattr(fused_norms, "_dbg", 0) + 1
        print(
            f"HY4 NF out addrs qa={qa.buffer_address()} lat={lat.buffer_address()} kpe={kpe.buffer_address()} mq={mq}",
            flush=True,
        )
    ttnn.generic_op([qa_g, kv, qw, kvw, qa, lat, kpe], prog)
    return qa, lat, kpe


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# B22 step 4 (HY4_BACK_FUSE=1, needs HY4_ATTN_FUSE back + HY4_SKINNY=oproj; default OFF): fused_back (64 cores) and the skinny o-proj (8 DRAM-bank cores) in ONE program. The o-proj readers start
# streaming their weight shard at program start (hidden behind the back phase); the back cores write og to DRAM and increment the ready semaphore of the bank core of their head; the o-proj
# writer waits for the 8 increments of its head, then runs the unchanged x exchange + matmul.
def back_fuse_enabled():
    return os.environ.get("HY4_BACK_FUSE", "0") == "1"


def fused_back_oproj(o_lat, w_uv_t, g, gfold, sk):
    md = o_lat.device()
    H, KT, NT = int(w_uv_t.shape[1]), int(w_uv_t.shape[2]) // 32, int(w_uv_t.shape[3]) // 32
    assert int(o_lat.shape[1]) == H and int(o_lat.shape[3]) == KT * 32 and o_lat.layout == ttnn.TILE_LAYOUT
    assert w_uv_t.layout == ttnn.TILE_LAYOUT and w_uv_t.dtype == ttnn.bfloat16 and g.layout == ttnn.TILE_LAYOUT
    S = int(o_lat.shape[2])
    assert 1 <= S <= 8 and H == 8 and sk.xsplit and sk.KT == H * NT  # skinny core b loads the x pages of head b
    n_cores = H * NT
    grid = md.compute_with_storage_grid_size()
    og = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, S, H * NT * 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
    )
    SEM = 3
    kernels, sems, cbs_sk, out, key = sk._parts(og, S, None, ready=(SEM, 0 if os.environ.get("HY4_B22_NOWAIT") else NT))
    bank = {(c.x, c.y) for c in sk.cores}
    cl = [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in bank][:n_cores]
    assert len(cl) == n_cores
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cl])
    CX, CW, CG, CO = 16, 17, 18, 19  # CB ids disjoint from the o-proj's 0..2
    bf = ttnn.bfloat16

    def cb(i, n):
        return ttnn.CBDescriptor(
            total_size=2048 * n,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=bf, page_size=2048)],
        )

    cbs = cbs_sk + [cb(CX, KT), cb(CW, KT), cb(CG, 1), cb(CO, 1)]
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cl):
        rt[x][y] = [i]
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/hb_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CX, CW, CG, KT, NT, S] + _acc(o_lat) + _acc(w_uv_t) + _acc(g),
        runtime_args=rt,
        common_runtime_args=[o_lat.buffer_address(), w_uv_t.buffer_address(), g.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/hb_writer_sk.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CO, NT, SEM] + _acc(og),
        runtime_args=rt,
        common_runtime_args=[og.buffer_address()] + [v for q in sk.phys for v in (q.x, q.y)],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/hb_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[
            CX,
            CW,
            CG,
            CO,
            KT,
            int(bool(gfold)),
            int(os.environ.get("HY4_FA_ROUND", "1")),
            int(os.environ.get("HY4_FA_SIGFAST", "0")),
        ],
        runtime_args=ttnn.RuntimeArgs(),
        common_runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        ),
    )
    # the ready semaphore exists on the back cores too (the same L1 address on every core of the program)
    all_rngs = [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in sk.cores] + [
        ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cl
    ]
    sems = [sm for sm in sems if sm.id != SEM] + [
        ttnn.SemaphoreDescriptor(id=SEM, core_ranges=ttnn.CoreRangeSet(all_rngs), initial_value=0)
    ]
    prog = ttnn.ProgramDescriptor(kernels=kernels + [reader, writer, compute], semaphores=sems, cbs=cbs)
    prog.custom_program_hash = (0x3B5 << 40) | (
        hash((H, KT, NT, S, int(bool(gfold)), key, tuple(_acc(o_lat)), tuple(_acc(w_uv_t)), tuple(_acc(g))))
        & ((1 << 40) - 1)
    )
    ttnn.generic_op([o_lat, w_uv_t, g, og, sk.w, out], prog)
    ttnn.deallocate(og)
    return out
