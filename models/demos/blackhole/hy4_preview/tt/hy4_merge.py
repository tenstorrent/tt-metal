# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused cross-column merge of the Hy4 decode sparse attention (task fuse-sparse-merge, stage A1): ONE generic_op program replaces the 6 small ttnn ops + 2 slices of ``Block.merge3``.

x    [1, 8*S, 32, 576] TILE bf16 (a2a output, src-major then user): o cols 0..511 (normalised by the source's local sum), tile 16 col 0 = scaled max, tile 17 col 0 = local sum
sink [S, 1, 32, 32] (or [1,S,32,1]) TILE bf16: col 0 = the sink logit (scaled domain) per head row
-> [1, S, 32, 512] TILE bf16 (rows 0..7 = the 8 real heads: the TP8 layout of o_proj)
Exact flash-decode merge with the sink applied once: w_c = l_c exp(ms_c - sink), den = sum w_c + 1, out = sum_c o_c w_c / den (valid while ms - sink < 88).
"""
import os

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_merge_kernels")
NC = 8


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


_HM_OUT = {}


def hm_out(md, hm_users):
    """the PERSISTENT head-major merge output [1,8,hm_users,512] (shared by all layers and by the A1 merge / the B26 compact exchange)"""
    import torch

    key = (id(md), hm_users)
    if key not in _HM_OUT:
        _HM_OUT[key] = ttnn.from_torch(
            torch.zeros(1, 8, hm_users, 512, dtype=torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
    return _HM_OUT[key]


def merge_fused(x, sink, S, nsplit=None, out=None, full=False, hm_users=0, xp=None):
    """hm_users > 0 (HY4_ATTN_HM=1, compact verify rows): S = tile rows TR, the writer scatters into the PERSISTENT head-major tensor [1,8,hm_users,512] (shared by all layers; the caller must not deallocate it)."""
    md = x.device()
    assert xp is not None or (x.layout == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16)
    assert xp is None or S == 1, "B47a XPUSH: TR == 1 (<= 4 users)"
    nsplit = nsplit or int(
        os.environ.get("HY4_MRG_NSPLIT", "16" if S * 16 <= 120 else "8")
    )  # S=8 verify rows: 8 cores per row
    assert 16 % nsplit == 0
    tpc = 16 // nsplit
    grid = md.compute_with_storage_grid_size()
    n_cores = S * nsplit
    assert n_cores <= grid.x * grid.y
    if hm_users:
        import torch

        key = (id(md), hm_users)
        if key not in _HM_OUT:
            _HM_OUT[key] = ttnn.from_torch(
                torch.zeros(1, 8, hm_users, 512, dtype=torch.bfloat16),
                device=md,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            )
        out = _HM_OUT[key]
    if out is None:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, S, 32, 512]), ttnn.bfloat16, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
        )
    cores = [(i % grid.x, i // grid.x) for i in range(n_cores)]
    rngs, n, r = [], n_cores, 0
    while n > 0:
        w = min(n, grid.x)
        rngs.append(ttnn.CoreRange(ttnn.CoreCoord(0, r), ttnn.CoreCoord(w - 1, r)))
        n -= w
        r += 1
    core_set = ttnn.CoreRangeSet(rngs)
    ST, A, W, R, ONE, OUT = range(6)
    bf = ttnn.bfloat16
    T = 2048

    def cb(i, n, df=bf, page=T):
        return ttnn.CBDescriptor(
            total_size=page * n,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=df, page_size=page)],
        )

    cbs = [cb(ST, 1 + 2 * NC), cb(A, NC * tpc), cb(W, NC), cb(R, 1), cb(ONE, 1), cb(OUT, tpc)]
    rt_r, rt_w, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, (cx, cy) in enumerate(cores):
        s, g = i // nsplit, i % nsplit
        rt_r[cx][cy] = [s, g * tpc] + (xp.consumer_args() if xp is not None else [])
        rt_w[cx][cy] = [s, g * tpc]
        rt_c[cx][cy] = []
    reader = ttnn.KernelDescriptor(
        kernel_source=(xp.reader_path if xp is not None else f"{KDIR}/mrg_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[S, tpc, ST, A, ONE] + _acc(x) + _acc(sink),
        runtime_args=rt_r,
        common_runtime_args=[x.buffer_address(), sink.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/{'mrg_writer_hmc' if hm_users else 'mrg_writer'}.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[tpc, OUT] + ([hm_users] if hm_users else []) + _acc(out),
        runtime_args=rt_w,
        common_runtime_args=[out.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    cc = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/mrg_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[tpc, ST, A, W, R, ONE, OUT, int(os.environ.get("HY4_MRG_DBG", "0")), int(full)],
        runtime_args=rt_c,
        common_runtime_args=[],
        config=cc,
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = (0x3E6 << 40) | (
        hash(
            (
                S,
                nsplit,
                int(full),
                hm_users,
                xp is not None,
                os.environ.get("HY4_MRG_DBG", "0"),
                tuple(_acc(x)),
                tuple(_acc(sink)),
                tuple(_acc(out)),
            )
        )
        & ((1 << 40) - 1)
    )
    ttnn.generic_op([x, sink, out], prog)
    return out
