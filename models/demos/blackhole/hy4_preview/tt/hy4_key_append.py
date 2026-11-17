# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""B32 (HY4_B32_KAPPEND=1, default OFF): the S (<= 4) key-row appends of a verify step as ONE one-core program, bit-identical to S stock ``paged_update_cache`` calls.

Stock (hy4_pool_decode.append_key_rows): S x (slice + to_memory_config + paged_update_cache); the rows of one user hit the same bfp8 tile row so the calls cannot run concurrently. Here one core does the
same read-modify-write per row (read the 4 tiles of the tile row, untilize, patch the 256 B row, tilize, write back) with the stock untilize/tilize helpers, in order, with a done-handshake between the
writer and the reader so a tile row written by row j is re-read by row j+1.
"""
import os

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_key_append_kernels")


def enabled():
    return os.environ.get("HY4_B32_KAPPEND", "0") == "1"


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def key_append(k, key_pool4, upd_rows, pt, S):
    """k [1,1,S,128] bf16 TILE (rows 0..S-1 valid, norm + rope applied), key_pool4 [pages,1,64,128] bfp8 TILE (in place), upd_rows: S int32 RM [1,1]-per-device index tensors (0xFFFFFFFF / -1 = skip), pt [1,pages] int32 page table."""
    assert 1 <= S <= 4 and len(upd_rows) >= S
    assert k.dtype == ttnn.bfloat16 and k.layout == ttnn.TILE_LAYOUT and int(k.shape[3]) == 128
    md = k.device()
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    TB_IN, TB_C = 2048, 1088

    def cb(i, dt, page, n, tile=True):
        return ttnn.CBDescriptor(
            total_size=page * n,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
        )

    bf, b8, u32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint32
    ut = ttnn.CBDescriptor(
        total_size=TB_IN * 4,
        core_ranges=core,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=2, data_format=bf, page_size=TB_IN),
            ttnn.CBFormatDescriptor(buffer_index=3, data_format=bf, page_size=TB_IN),
        ],
    )
    cbs = [
        cb(0, bf, TB_IN, 4),  # kin
        cb(1, b8, TB_C, 4),  # cache tiles
        ut,  # 2 / 3 untilized block (aliased)
        cb(4, bf, TB_IN, 4),  # kin_rm
        cb(5, b8, TB_C, 4),  # out
        cb(6, u32, 16, 2),  # done
        cb(7, u32, 16, 2),  # meta
        cb(8, u32, 2048, 1),  # misc
    ]
    idx0 = upd_rows[0]
    common_r = [k.buffer_address(), key_pool4.buffer_address(), pt.buffer_address()] + [
        upd_rows[j].buffer_address() for j in range(S)
    ]
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/ka_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core,
        compile_time_args=[0, 1, 2, 3, 4, 6, 7, 8, S, TB_IN, TB_C] + _acc(k) + _acc(key_pool4) + _acc(pt) + _acc(idx0),
        runtime_args=ttnn.RuntimeArgs(),
        common_runtime_args=common_r,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/ka_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core,
        compile_time_args=[5, 6, 7, S, TB_C] + _acc(key_pool4),
        runtime_args=ttnn.RuntimeArgs(),
        common_runtime_args=[key_pool4.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/ka_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core,
        compile_time_args=[0, 1, 2, 3, 4, 5, S],
        runtime_args=ttnn.RuntimeArgs(),
        common_runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False
        ),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = (0x3B32 << 40) | (
        hash((S, tuple(tuple(_acc(t)) for t in (k, key_pool4, pt, idx0)))) & ((1 << 40) - 1)
    )
    ttnn.generic_op([k, pt] + [upd_rows[j] for j in range(S)] + [key_pool4], prog)
