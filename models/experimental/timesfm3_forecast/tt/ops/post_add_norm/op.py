# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R2-4: ``h = residual + RMSNorm(o) * gain`` and ``xn = RMSNorm(h)`` (gain-free) in one pass, through
``ttnn.generic_op``. Replaces the post-norm, residual add and next pre-norm chain (three DRAM passes).

One work unit is one tile row (32 tokens x W). Kernels are model-local, adapted from
``chronos_forecast/ops/add_rms_norm``; their paths are absolute, from this file.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import ttnn

TILE = 32
KERNELS = Path(__file__).resolve().parent / "kernels"
O_CB, R_CB, SCALER_CB, G_CB, H_OUT_CB, N_OUT_CB, H_CB, SQ_CB, RSTD_CB, T_CB, T2_CB = 0, 1, 2, 3, 16, 17, 24, 25, 26, 27, 28
_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096}


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _cb(index, pages, dtype, cores):
    page = _TILE_BYTES[dtype]
    return ttnn.CBDescriptor(
        total_size=pages * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def _accessor(t) -> list[int]:
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _split_rows(device, num_rows: int):
    grid = device.compute_with_storage_grid_size()
    _, cores, group_1, _, per_1, per_2 = ttnn.split_work_to_cores(grid, num_rows, True)
    n1, work, start = group_1.num_cores(), [], 0
    for i, core in enumerate(ttnn.corerange_to_cores(cores, row_wise=True)):
        n = per_1 if i < n1 else per_2
        work.append((core, n, start))
        start += n
    return cores, work


def post_add_norm(o, residual, gain, *, epsilon: float, math_fidelity=ttnn.MathFidelity.HiFi4):
    """(h, xn) for bf16 interleaved tile tensors ``o`` and ``residual`` of one shape; ``gain`` is (1, W)."""
    for name, t in (("o", o), ("residual", residual), ("gain", gain)):
        if t.layout != ttnn.TILE_LAYOUT or t.is_sharded() or t.dtype != ttnn.bfloat16:
            raise ValueError(f"post_add_norm: {name} must be bf16 interleaved TILE, got {t.dtype} {t.memory_config()}")
    if tuple(o.padded_shape) != tuple(residual.padded_shape):
        raise ValueError(f"post_add_norm: shapes differ, {o.shape} vs {residual.shape}")
    width = o.padded_shape[-1]
    if o.shape[-1] != width or tuple(gain.padded_shape)[-1] != width:
        raise ValueError(f"post_add_norm: needs a tile-aligned width and a matching gain, got {o.shape}, {gain.shape}")
    wt = width // TILE
    num_rows = math.prod(o.padded_shape) // (TILE * width)
    device = o.device()
    mem = ttnn.DRAM_MEMORY_CONFIG
    h = ttnn.allocate_tensor_on_device(o.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mem)
    xn = ttnn.allocate_tensor_on_device(o.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mem)
    cores, work = _split_rows(device, num_rows)
    blk = max(d for d in range(1, 5) if wt % d == 0)  # fp32 DST: 4 tiles
    bf, f32 = ttnn.bfloat16, ttnn.float32
    cbs = [
        _cb(O_CB, wt, bf, cores),
        _cb(R_CB, 2 * blk, bf, cores),
        _cb(SCALER_CB, 1, bf, cores),
        _cb(G_CB, wt, bf, cores),
        _cb(H_OUT_CB, 2 * blk, bf, cores),
        _cb(N_OUT_CB, 2 * blk, bf, cores),
        _cb(H_CB, wt, bf, cores),
        _cb(SQ_CB, wt, bf, cores),
        _cb(RSTD_CB, 1, f32, cores),
        _cb(T_CB, wt, f32, cores),
        _cb(T2_CB, wt, f32, cores),
    ]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS / "reader_post_add_norm.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[wt, blk, *_accessor(o), *_accessor(residual), *_accessor(gain)],
        runtime_args=[
            ((c.x, c.y), [o.buffer_address(), residual.buffer_address(), gain.buffer_address(), n, s])
            for c, n, s in work
        ],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS / "writer_post_add_norm.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[wt, blk, 1, *_accessor(h), *_accessor(xn)],
        runtime_args=[((c.x, c.y), [h.buffer_address(), xn.buffer_address(), n, s]) for c, n, s in work],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS / "compute_post_add_norm.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[wt, blk, _f32_bits(1.0 / width), _f32_bits(epsilon)],
        runtime_args=[((c.x, c.y), [n]) for c, n, _ in work],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, math_approx_mode=False, fp32_dest_acc_en=True),
    )
    ttnn.generic_op([o, residual, gain, h, xn], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs))
    return h, xn
