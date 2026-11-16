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


def fused_q(q, w_uk, cos, sin, tm, out=None):
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
    for i in range(n_cores):
        cx, cy = i % grid.x, i // grid.x
        rt[cx][cy] = [0, i // 8, i % 8] if i < n_abs else [1, i - n_abs, 0]
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
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, c_abs, c_rope], semaphores=[], cbs=cbs)
    prog.custom_program_hash = (0x3B2 << 40) | (
        hash((H, S, tuple(tuple(_acc(t)) for t in (q, w_uk, cos, sin, tm, out)))) & ((1 << 40) - 1)
    )
    ttnn.generic_op([q, w_uk, cos, sin, tm, out], prog)
    return out
