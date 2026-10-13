# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode attention epilogue (1..32 rows: batch 1, the DFlash verify, batch-32 decode and the batched verify's 32-row
chunks) as one generic_op (Laguna; kernels/ae1_*.cpp).

Replaces the gate slice, the gate reshape to [heads, 1], the softplus-gated multiply and the flatten / reshard into
WO's width-sharded input (5 small ops; at 32 rows also the head concat and its reshards): core (b, j) reads token
row b's SDPA output ([heads as rows, 128], head_dim tile group j) and gate logits (row row_in + b of the fused
qkv(+gate) output), computes attn * softplus(g) per head row on the tile engine and writes each head's row into row
row_out + b of the WO input (the batched verify: each 32-row chunk into its rows of one shared [T, H * 128])."""

import os
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


def attn_epilogue1(attn, qkv, g_col, num_heads, out_mem, beta=1.0, threshold=20.0, rows=1, row_in=0, out=None,
                   row_out=0):
    """attn: SDPA output [1, rows, H, 128] bf16 TILE; qkv: the fused qkv(+gate) output [.., R, W] bf16 TILE whose rows
    row_in .. row_in + rows - 1 hold the gate logits at columns g_col .. g_col + H - 1 (g_col a multiple of 32).
    Returns [1, 1, rows, H * 128] bf16 TILE in out_mem: row b = concat over heads of attn[b, h] * softplus(g[b, h]);
    or, given out ([1, 1, T, H * 128]), writes those rows at row_out .. of it and returns out. attn may be a list of
    one SDPA output per 32 rows (the batched verify: one epilogue over all its rows)."""
    attns = list(attn) if isinstance(attn, (list, tuple)) else [attn]
    attn = attns[0]
    device = attn.device()
    H = int(num_heads)
    assert attn.dtype == ttnn.bfloat16 and qkv.dtype == ttnn.bfloat16 and H <= 32 and g_col % 32 == 0
    assert attn.padded_shape[-1] == 128 and qkv.padded_shape[-1] >= g_col + H
    assert 1 <= rows <= 32 * len(attns) and row_in + rows <= qkv.padded_shape[-2], (rows, row_in, qkv.padded_shape)
    if out is None:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, rows, H * 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, out_mem
        )
    assert out.padded_shape[-1] == H * 128 and row_out + rows <= out.padded_shape[-2], (out.padded_shape, row_out)
    # core (b, j): token row b, head_dim tile group j of 4 // tj, cores in row-major order (TT_LAGUNA_AE1_SPLIT=0: one
    # core per row does all 4 tiles; more than 27 rows: 2 tiles per core so the cores fit the grid)
    gs = device.compute_with_storage_grid_size()
    tj = 1 if os.environ.get("TT_LAGUNA_AE1_SPLIT", "1") == "1" else 4
    while rows * (4 // tj) > gs.x * gs.y:
        tj *= 2
    grid = ttnn.num_cores_to_corerangeset(rows * (4 // tj), gs, True)
    pos = [gs.x, row_in, int(qkv.padded_shape[-1]) // 32, row_out]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "ae1_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[H, g_col // 32, tj] + pos + [len(attns)]
        + list(ttnn.TensorAccessorArgs(attn).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(qkv).get_compile_time_args()),
        common_runtime_args=[attn.buffer_address(), qkv.buffer_address()]
        + ([a.buffer_address() for a in attns] if len(attns) > 1 else []),
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "ae1_writer.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[H, tj] + pos + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
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
        compile_time_args=[_bits(beta), _bits(1.0 / beta), _bits(threshold), tj],
        config=cfg,
    )
    cbs = [_cb(grid, 0, 4), _cb(grid, 1, 1), _cb(grid, 2, 1), _cb(grid, 3, 1), _cb(grid, 16, 4)]
    ttnn.generic_op(
        attns + [qkv, out], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    )
    return out
