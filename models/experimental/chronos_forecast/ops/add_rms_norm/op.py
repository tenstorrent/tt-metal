# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm without gamma, optionally fused with the residual add: ``n = rms_norm(x)`` in a chosen dtype, or
``y = a + b`` and ``n = rms_norm(y)``.

One work unit is one tile row (32 tokens x W), streamed in DST-sized blocks; the row sum of squares is
reduced in DST and scaled by 1/W there, so the scale is exact rather than a bf16 scaler tile.

On L1-interleaved activations a row's tiles span many banks, so the fused form moves a, b, y and n across
the NoC, while a separate ``ops.add`` stays bank-local; the split form is faster there
(sweeps/bench_encoder_ops.py add_rms_norm).
"""

from __future__ import annotations

import struct

import ttnn

from models.experimental.chronos_forecast.ops import common

_OP = "add_rms_norm"
A_CB, B_CB, SCALER_CB, Y_OUT_CB, N_OUT_CB, Y_CB, SQ_CB, RSTD_CB = 0, 1, 2, 16, 17, 24, 25, 26


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def add_rms_norm(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    epsilon: float,
    memory_config=None,
    norm_dtype=None,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en: bool = True,
):
    """(a + b, rms_norm(a + b)); the sum has a's dtype, the norm ``norm_dtype`` (default a's)."""
    return _run(a, b, epsilon, memory_config, norm_dtype, math_fidelity, fp32_dest_acc_en)


def rms_norm(
    x: ttnn.Tensor,
    *,
    epsilon: float,
    memory_config=None,
    norm_dtype=None,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en: bool = True,
):
    """rms_norm(x) without gamma, in ``norm_dtype`` (default x's)."""
    return _run(x, None, epsilon, memory_config, norm_dtype, math_fidelity, fp32_dest_acc_en)[1]


def _run(a, b, epsilon, memory_config, norm_dtype, math_fidelity, fp32_dest_acc_en):
    fuse_add = b is not None
    common.require_interleaved_tile("a", a)
    if fuse_add:
        common.require_interleaved_tile("b", b)
        if tuple(a.padded_shape)[-2:] != tuple(b.padded_shape)[-2:] or common.num_tiles(a) != common.num_tiles(b):
            raise ValueError(f"add_rms_norm needs matching shapes, got {a.shape} and {b.shape}")
    width = tuple(a.padded_shape)[-1]
    if a.shape[-1] != width:
        raise ValueError(f"add_rms_norm needs a tile-aligned width, got {a.shape[-1]}")
    wt = width // common.TILE
    num_rows = common.padded_volume(a) // (common.TILE * width)

    device = a.device()
    mem = a.memory_config() if memory_config is None else memory_config
    n_dtype = a.dtype if norm_dtype is None else norm_dtype
    y = ttnn.allocate_tensor_on_device(a.shape, a.dtype, ttnn.TILE_LAYOUT, device, mem) if fuse_add else None
    n = ttnn.allocate_tensor_on_device(a.shape, n_dtype, ttnn.TILE_LAYOUT, device, mem)
    # Unused tensor slots take a's (or n's) accessor so the kernels' compile-time layout is fixed.
    b_t = b if fuse_add else a
    y_t = y if fuse_add else n

    all_cores, work = common.split_rows(device, num_rows)
    dst_tiles = 4 if fp32_dest_acc_en else 8
    blk = max(d for d in range(1, dst_tiles + 1) if wt % d == 0)
    # The row buffer the compute kernel re-reads (a, or y when fused) holds a whole row; the rest stream.
    cbs = [
        common.cb(A_CB, 2 * blk if fuse_add else 2 * wt, a.dtype, all_cores),
        common.cb(SCALER_CB, 1, ttnn.bfloat16, all_cores),
        common.cb(N_OUT_CB, 2 * blk, n_dtype, all_cores),
        common.cb(SQ_CB, wt, ttnn.bfloat16, all_cores),
        common.cb(RSTD_CB, 1, ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16, all_cores),
    ]
    if fuse_add:
        cbs += [
            common.cb(B_CB, 2 * blk, b.dtype, all_cores),
            common.cb(Y_OUT_CB, 2 * blk, a.dtype, all_cores),
            common.cb(Y_CB, wt, a.dtype, all_cores),
        ]
    a_addr, b_addr, y_addr, n_addr = (t.buffer_address() for t in (a, b_t, y_t, n))
    reader = ttnn.KernelDescriptor(
        kernel_source=common.kernel_path(_OP, "reader_add_rms_norm.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=[wt, blk, int(fuse_add), *common.accessor_args(a), *common.accessor_args(b_t)],
        runtime_args=[((c.x, c.y), [a_addr, b_addr, cnt, start]) for c, cnt, start in work],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=common.kernel_path(_OP, "writer_add_rms_norm.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=[wt, blk, int(fuse_add), *common.accessor_args(y_t), *common.accessor_args(n)],
        runtime_args=[((c.x, c.y), [y_addr, n_addr, cnt, start]) for c, cnt, start in work],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=common.kernel_path(_OP, "compute_add_rms_norm.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=[wt, blk, _f32_bits(1.0 / a.shape[-1]), _f32_bits(epsilon), int(fuse_add)],
        runtime_args=[((c.x, c.y), [cnt]) for c, cnt, _ in work],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=math_fidelity, math_approx_mode=False, fp32_dest_acc_en=fp32_dest_acc_en
        ),
    )
    io = [a, b, y, n] if fuse_add else [a, n]
    ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs))
    return y, n
