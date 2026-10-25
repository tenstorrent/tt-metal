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


def _bf16_bits(x):
    return int(torch.tensor([x], dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF


NEG_BITS = (
    struct.unpack(">I", struct.pack(">f", -1e30))[0] >> 16
)  # the chain's where(..., -1e30) on bf16 truncates (-9.953e29)


def score_mask_fused(q, kc, w, nvalid_u32, n_rows, ns, valid=None, n_cores=None):
    """q [U,1,32*n,128] bf16 TILE (row j of user u: tile row j, heads on the 32 rows); kc [U',1,n_alloc,128] TILE key slab (U' >= U; bf16 / bfp8_b); w [1,1,R,32] bf16 TILE (R = U*n gates);
    nvalid_u32 [1,1,1,32] uint32 RM (valid entries of row r, r < R); ns = scored entries (multiple of 32, <= n_alloc) -> bf16 RM [1,1,R,ns].
    """
    U = int(q.shape[0])
    R = U * n_rows
    assert R <= 32 and ns % 32 == 0 and int(q.shape[-2]) == 32 * n_rows and int(q.shape[-1]) == 128
    assert q.dtype == ttnn.bfloat16 and w.dtype == ttnn.bfloat16 and tuple(int(s) for s in w.shape) == (1, 1, R, 32)
    n_alloc = int(kc.shape[-2])
    assert ns <= n_alloc
    mesh = q.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, R, ns]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    nct_total = ns // 32
    grid = mesh.compute_with_storage_grid_size()
    allc = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)]
    ncores = min(n_cores or int(os.environ.get("DSV41_IDX_FUSED_CORES", "110")), len(allc), nct_total)
    cores = allc[:ncores]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    base, rem = divmod(nct_total, ncores)
    rt = ttnn.RuntimeArgs()
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
        kernel_source=f"{KDIR}/score_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CB_Q, CB_K, CB_WT, CB_W, R, U, ktb, n_alloc // 32, CB_MAIL, ncores, nct_total]
        + acc(q)
        + acc(kc)
        + acc(w)
        + acc(valid),
        runtime_args=rt,
        common_runtime_args=[q.buffer_address(), kc.buffer_address(), w.buffer_address(), valid.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/score_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CB_OUT, CB_STAGE, R, CH, ns, NEG_BITS, CB_NV, ncores, nct_total]
        + acc(out)
        + acc(nvalid_u32)
        + acc(valid),
        runtime_args=rt,
        common_runtime_args=[out.buffer_address(), nvalid_u32.buffer_address(), valid.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/score_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[CB_Q, CB_K, CB_WT, CB_S, CB_OUT, R, U, CTG, CB_MAIL],
        runtime_args=rt,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = (0x1D5 << 40) | (
        hash(
            (
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
