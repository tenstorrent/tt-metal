# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batch-1 decode attention epilogue as one generic_op (Laguna; kernels/ae1_*.cpp).

Replaces the gate slice, the gate reshape to [heads, 1], the softplus-gated multiply and the flatten / reshard into
WO's width-sharded input (5 small ops): one core reads the SDPA output ([heads as rows, 128]) and the gate logits
from row 0 of the fused qkv(+gate) output, computes attn * softplus(g) per head row on the tile engine and writes
each head's row into row 0 of the WO input."""

import struct
from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _cb(grid, i, pages):
    return ttnn.CBDescriptor(
        total_size=2048 * pages,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
    )


def attn_epilogue1(attn, qkv, g_col, num_heads, out_mem, beta=1.0, threshold=20.0):
    """attn: SDPA output [1, 1, H, 128] bf16 TILE; qkv: the fused qkv(+gate) output [.., 32, W] bf16 TILE whose row 0
    holds the gate logits at columns g_col .. g_col + H - 1 (g_col a multiple of 32). Returns [1, 1, 1, H * 128] bf16
    TILE in out_mem: row 0 = concat over heads of attn[h] * softplus(g[h])."""
    device = attn.device()
    H = int(num_heads)
    assert attn.dtype == ttnn.bfloat16 and qkv.dtype == ttnn.bfloat16 and H <= 32 and g_col % 32 == 0
    assert attn.padded_shape[-1] == 128 and qkv.padded_shape[-1] >= g_col + H
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 1, H * 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, out_mem
    )
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "ae1_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[H, g_col // 32]
        + list(ttnn.TensorAccessorArgs(attn).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(qkv).get_compile_time_args()),
        common_runtime_args=[attn.buffer_address(), qkv.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "ae1_writer.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[H] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
        common_runtime_args=[out.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.fp32_dest_acc_en = True
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    compute = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "ae1_compute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[_bits(beta), _bits(1.0 / beta), _bits(threshold)],
        config=cfg,
    )
    cbs = [_cb(grid, 0, 4), _cb(grid, 1, 1), _cb(grid, 2, 1), _cb(grid, 3, 1), _cb(grid, 16, 4)]
    ttnn.generic_op(
        [attn, qkv, out], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    )
    return out
