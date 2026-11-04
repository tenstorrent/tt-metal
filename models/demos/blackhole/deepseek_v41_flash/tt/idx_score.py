# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused score + validity mask of the speculative-verify indexer (lever L12): ONE generic_op replaces

    slice(key slab, nb) -> matmul(q, K^T, relu) -> matmul(w, .) -> slice row 0 -> untilize -> [mask chain]

The key slab is read in place (no slice copy), the [R*32, NS] relu intermediate never goes to DRAM, the output is the row-major bf16 score matrix [1,1,R,NS] with
entries >= nvalid[r] already set to -1e30 (what ``indexer.mask_scores`` produces). Numerics: stage 1 is the matmul (HiFi4, fp32 DEST accumulation over the 4 d tiles, relu in the packer,
bf16 rounding of S), stage 2 sums the R gate-weighted S tiles as one matmul with a block-diagonal gate (zero blocks add exact zeros) -- see tests/test_idx_score_fused.py (bit-exact vs the chain).
"""

import os
import struct

import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "idx_kernels")


def _score_mc(ns, R):
    """DSV41_OL_SCORE_L1=1: the score rows (R x ns bf16, <= 400 KB: interleaved over the worker L1s) stay in L1: transient, read by the top-k op"""
    if os.environ.get("DSV41_OL_SCORE_L1", "0") == "1" and ns * R * 2 <= 400000:
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def _zd():
    return [("OL_ZONES", "1")] if os.environ.get("DSV41_OL_ZONES", "0") == "1" else []


def _kdir():
    """DSV41_OL_SCORE_DIAG=1: the copies with profiler zones (tt/ol_kernels/diag), same code"""
    if os.environ.get("DSV41_OL_SCORE_NORTA", "0") == "1":
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ol_kernels", "score2")
    if os.environ.get("DSV41_OL_SCORE_DIAG", "0") == "1":
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ol_kernels", "diag")
    return KDIR


def _bf16_bits(x):
    return int(torch.tensor([x], dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF


NEG_BITS = (
    struct.unpack(">I", struct.pack(">f", -1e30))[0] >> 16
)  # the chain's where(..., -1e30) on bf16 truncates (-9.953e29)


def score_mask_fused3(q, kc, w, nvalid_u32, n_rows, ns, valid):
    """DSV41_OL_SCORE3=1: the same op with the cores split PER USER (ol_kernels/score3): cores per user = grid cores // U, each core scores the key columns of ONE user (q: n_rows x 4 tiles instead of R x 4,
    4 key tiles per column); the compute kernel is the stock one with R = n_rows, U = 1. Bit-identical numerics."""
    U = int(q.shape[0])
    NROW = n_rows
    R = U * NROW
    n_alloc = int(kc.shape[-2])
    mesh = q.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, R, ns]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, _score_mc(ns, R)
    )
    nct_total = ns // 32
    grid = mesh.compute_with_storage_grid_size()
    allc = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)]
    cpu = max(1, min(len(allc) // U, nct_total, int(os.environ.get("DSV41_IDX_FUSED_CORES", "110")) // U))
    cores = allc[: U * cpu]
    from models.demos.blackhole.deepseek_v41_flash.tt.mhc_pre import _merged_set

    core_set = _merged_set(cores)
    CTG = 1
    ktb = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}[kc.dtype]
    CH = max(1, min(64, 65536 // (NROW * 64)))

    def cb(idx, pages, page, dt):
        return ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)],
        )

    CB_Q, CB_K, CB_WT, CB_W, CB_S, CB_OUT, CB_STAGE, CB_NV, CB_MAIL = range(9)
    cbs = [
        cb(CB_Q, NROW * 4, 2048, ttnn.bfloat16),
        cb(CB_K, 2 * CTG * 4, ktb, kc.dtype),
        cb(CB_WT, NROW, 2048, ttnn.bfloat16),
        cb(CB_W, 1, 2048, ttnn.bfloat16),
        cb(CB_S, NROW * CTG, 2048, ttnn.bfloat16),
        cb(CB_OUT, 2 * CTG, 2048, ttnn.bfloat16),
        cb(CB_STAGE, 1, NROW * CH * 64, ttnn.bfloat16),
        cb(CB_NV, 1, 256, ttnn.uint32),
        cb(CB_MAIL, 1, 2048, ttnn.uint32),
    ]
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ol_kernels")
    zd = _zd()
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{base}/score3/score_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CB_Q, CB_K, CB_WT, CB_W, NROW, U, ktb, n_alloc // 32, CB_MAIL, cpu, nct_total]
        + acc(q)
        + acc(kc)
        + acc(w)
        + acc(valid),
        runtime_args=ttnn.RuntimeArgs(),
        common_runtime_args=[
            q.buffer_address(),
            kc.buffer_address(),
            w.buffer_address(),
            valid.buffer_address(),
            grid.x,
        ],
        config=ttnn.ReaderConfigDescriptor(),
        defines=zd,
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{base}/score3/score_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CB_OUT, CB_STAGE, NROW, CH, ns, NEG_BITS, CB_NV, cpu, nct_total]
        + acc(out)
        + acc(nvalid_u32)
        + acc(valid),
        runtime_args=ttnn.RuntimeArgs(),
        common_runtime_args=[out.buffer_address(), nvalid_u32.buffer_address(), valid.buffer_address(), grid.x],
        config=ttnn.WriterConfigDescriptor(),
        defines=zd,
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{base}/score2/score_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CB_Q, CB_K, CB_WT, CB_S, CB_OUT, NROW, 1, CTG, CB_MAIL],
        runtime_args=ttnn.RuntimeArgs(),
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        defines=zd,
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = (0x1D7 << 40) | (
        hash(
            (
                str(_zd()),
                NROW,
                U,
                ns,
                n_alloc,
                cpu,
                str(kc.dtype),
                tuple(acc(q)),
                tuple(acc(kc)),
                tuple(acc(w)),
                tuple(acc(out)),
                tuple(acc(nvalid_u32)),
                tuple(acc(valid)),
            )
        )
        & ((1 << 40) - 1)
    )
    return ttnn.generic_op([q, kc, w, nvalid_u32, valid, out], prog)


def score_mask_fused(q, kc, w, nvalid_u32, n_rows, ns, valid=None, n_cores=None):
    """q [U,1,32*n,128] bf16 TILE (row j of user u: tile row j, heads on the 32 rows); kc [U',1,n_alloc,128] TILE key slab (U' >= U; bf16 / bfp8_b); w [1,1,R,32] bf16 TILE (R = U*n gates);
    nvalid_u32 [1,1,1,32] uint32 RM (valid entries of row r, r < R); ns = scored entries (multiple of 32, <= n_alloc) -> bf16 RM [1,1,R,ns].
    """
    U = int(q.shape[0])
    R = U * n_rows
    if os.environ.get("DSV41_OL_SCORE3", "0") == "1" and valid is not None and n_cores is None:
        return score_mask_fused3(q, kc, w, nvalid_u32, n_rows, ns, valid)
    assert R <= 32 and ns % 32 == 0 and int(q.shape[-2]) == 32 * n_rows and int(q.shape[-1]) == 128
    assert q.dtype == ttnn.bfloat16 and w.dtype == ttnn.bfloat16 and tuple(int(s) for s in w.shape) == (1, 1, R, 32)
    n_alloc = int(kc.shape[-2])
    assert ns <= n_alloc
    mesh = q.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, R, ns]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, _score_mc(ns, R)
    )
    nct_total = ns // 32
    grid = mesh.compute_with_storage_grid_size()
    allc = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)]
    ncores = min(n_cores or int(os.environ.get("DSV41_IDX_FUSED_CORES", "110")), len(allc), nct_total)
    cores = allc[:ncores]
    if os.environ.get("DSV41_OL_MERGED_RANGES", "0") == "1":
        # otherlayers: one CoreRange per row span instead of per core (the dispatcher writes the program config once per range: 110 ranges cost a ~15 us launch gap)
        from models.demos.blackhole.deepseek_v41_flash.tt.mhc_pre import _merged_set

        core_set = _merged_set(cores)
    else:
        core_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores]
        )
    base, rem = divmod(nct_total, ncores)
    norta = (
        os.environ.get("DSV41_OL_SCORE_NORTA", "0") == "1"
    )  # no per-core runtime args: the kernels derive the core index from the logical coordinates
    rt = ttnn.RuntimeArgs()
    if not norta:
        for i, (cx, cy) in enumerate(cores):
            rt[cx][cy] = [i]
    if valid is None:  # static: every entry of the slab prefix ``ns`` is live
        valid = ttnn.from_torch(
            torch.full((1, 1, 1, 1), ns, dtype=torch.int32),
            device=mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
    CTG = int(os.environ.get("DSV41_IDX_FUSED_CT", "1"))  # key tile columns per DEST pass
    ktb = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}[kc.dtype]
    CH = max(1, min(64, 65536 // (R * 64)))

    def cb(idx, pages, page, dt):
        return ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)],
        )

    CB_Q, CB_K, CB_WT, CB_W, CB_S, CB_OUT, CB_STAGE, CB_NV, CB_MAIL = range(9)
    cbs = [
        cb(CB_Q, R * 4, 2048, ttnn.bfloat16),
        cb(CB_K, 2 * CTG * U * 4, ktb, kc.dtype),
        cb(CB_WT, R, 2048, ttnn.bfloat16),
        cb(CB_W, 1, 2048, ttnn.bfloat16),
        cb(CB_S, R * CTG, 2048, ttnn.bfloat16),
        cb(CB_OUT, 2 * CTG, 2048, ttnn.bfloat16),
        cb(CB_STAGE, 1, R * CH * 64, ttnn.bfloat16),
        cb(CB_NV, 1, 256, ttnn.uint32),
        cb(CB_MAIL, 1, 2048, ttnn.uint32),
    ]
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{_kdir()}/score_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CB_Q, CB_K, CB_WT, CB_W, R, U, ktb, n_alloc // 32, CB_MAIL, ncores, nct_total]
        + acc(q)
        + acc(kc)
        + acc(w)
        + acc(valid),
        runtime_args=rt,
        common_runtime_args=[q.buffer_address(), kc.buffer_address(), w.buffer_address(), valid.buffer_address()]
        + ([grid.x] if norta else []),
        config=ttnn.ReaderConfigDescriptor(),
        defines=_zd(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{_kdir()}/score_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CB_OUT, CB_STAGE, R, CH, ns, NEG_BITS, CB_NV, ncores, nct_total]
        + acc(out)
        + acc(nvalid_u32)
        + acc(valid),
        runtime_args=rt,
        common_runtime_args=[out.buffer_address(), nvalid_u32.buffer_address(), valid.buffer_address()]
        + ([grid.x] if norta else []),
        config=ttnn.WriterConfigDescriptor(),
        defines=_zd(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{_kdir()}/score_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CB_Q, CB_K, CB_WT, CB_S, CB_OUT, R, U, CTG, CB_MAIL],
        runtime_args=rt,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        defines=_zd(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = (0x1D5 << 40) | (
        hash(
            (
                os.environ.get("DSV41_OL_MERGED_RANGES", "0"),
                _kdir(),
                str(_zd()),
                R,
                U,
                ns,
                n_alloc,
                ncores,
                CTG,
                str(kc.dtype),
                tuple(acc(q)),
                tuple(acc(kc)),
                tuple(acc(w)),
                tuple(acc(out)),
                tuple(acc(nvalid_u32)),
                tuple(acc(valid)),
            )
        )
        & ((1 << 40) - 1)
    )
    return ttnn.generic_op([q, kc, w, nvalid_u32, valid, out], prog)
