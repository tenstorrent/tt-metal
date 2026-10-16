# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Residual add + RMSNorm in one model-local ``ttnn.generic_op``.

``sum = a + b`` (the next residual) and ``normed = rmsnorm(sum) * gamma`` are produced
by one pass: today the decoder runs ``ttnn.add`` (3 DRAM passes over [M, W]) followed by
``rms_norm`` (2 more), 72 pairs per forward. The fused op reads a and b once and writes
both outputs (4 passes), and removes half the launches. Per tile-row the compute does
    sum = a + b               -> sum output CB (in the residual dtype) + a bfp8 copy for the norm
    x2  = sum * sum           -> CB
    ms  = row-sum(x2) / W     (reduce with the 1/W scaler tile)
    inv = rsqrt(ms + eps)
    out = (sum * bcast_cols(inv)) * gamma   (gamma applied on the DST tile)
Rows are split across cores (fewest cores keeping the same per-core maximum); each core
streams whole rows, so this is the row-granular variant (M/32 >= ~cores pays best).
"""
import os

import ttnn
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads.op import (
    _TILE_BYTES,
    _core_ranges,
    _split_work_to_cores,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
READER_KERNEL = os.path.join(_HERE, "kernels", "reader_add_rmsnorm.cpp")
COMPUTE_KERNEL = os.path.join(_HERE, "kernels", "compute_add_rmsnorm.cpp")
WRITER_KERNEL = os.path.join(_HERE, "kernels", "writer_add_rmsnorm.cpp")
_BF16_TILE = _TILE_BYTES[ttnn.bfloat16]
_CHUNK = 4  # DST tiles per session with fp32 accumulation (half-sync)


def supported(a: ttnn.Tensor, b: ttnn.Tensor) -> bool:
    if a.layout != ttnn.TILE_LAYOUT or b.layout != ttnn.TILE_LAYOUT:
        return False
    if a.dtype not in _TILE_BYTES or b.dtype not in _TILE_BYTES:
        return False
    if list(a.padded_shape) != list(b.padded_shape):
        return False
    if a.is_sharded() or b.is_sharded():
        return False
    w_tiles = int(a.padded_shape[-1]) // 32
    return int(a.padded_shape[-1]) % 32 == 0 and w_tiles % _CHUNK == 0 and int(a.padded_shape[-2]) % 32 == 0


def fused_add_rmsnorm(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    gamma_tiles: ttnn.Tensor,
    scaler_tile: ttnn.Tensor,
    eps_tile: ttnn.Tensor,
    *,
    sum_dtype: ttnn.DataType | None = None,
    out_dtype: ttnn.DataType | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """``a``, ``b``: ``[..., M, W]`` TILE interleaved. Returns ``(a + b, rmsnorm(a + b) * gamma)``."""
    if memory_config is None:
        memory_config = a.memory_config()
    device = a.device()
    shape = list(a.padded_shape)
    W = shape[-1]
    Wt = W // 32
    rows = 1
    for d in shape[:-1]:
        rows *= int(d)
    rows_t = rows // 32
    sum_dtype = sum_dtype or a.dtype
    out_dtype = out_dtype or sum_dtype
    for dt in (sum_dtype, out_dtype):
        if dt not in _TILE_BYTES:
            raise ValueError(f"unsupported dtype {dt}")

    sum_tensor = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), sum_dtype, ttnn.TILE_LAYOUT, device, memory_config)
    out_tensor = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), out_dtype, ttnn.TILE_LAYOUT, device, memory_config)

    grid = device.compute_with_storage_grid_size()
    num_cores, per_core = _split_work_to_cores(rows_t, int(grid.x), int(grid.y))
    if num_cores == 0:
        raise RuntimeError("fused_add_rmsnorm: nothing to do")
    used_cores = _core_ranges(per_core)

    def cb(index, tiles, dtype, tsize):
        return ttnn.CBDescriptor(
            total_size=tiles * tsize,
            core_ranges=used_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tsize)],
        )

    b8 = ttnn.bfloat8_b
    cbs = [
        cb(0, Wt * 2, a.dtype, _TILE_BYTES[a.dtype]),  # a rows (double-buffered)
        cb(1, Wt * 2, b.dtype, _TILE_BYTES[b.dtype]),  # b rows
        cb(2, Wt, ttnn.bfloat16, _BF16_TILE),  # gamma tiles (resident)
        cb(3, 1, ttnn.bfloat16, _BF16_TILE),  # 1/W scaler
        cb(4, 1, ttnn.bfloat16, _BF16_TILE),  # eps
        cb(5, Wt, b8, _TILE_BYTES[b8]),  # sum, bfp8 copy for the norm (matches today's add->norm numerics)
        cb(6, Wt, ttnn.bfloat16, _BF16_TILE),  # sum^2
        cb(7, 1, ttnn.bfloat16, _BF16_TILE),  # mean square
        cb(8, 1, ttnn.bfloat16, _BF16_TILE),  # rsqrt
        cb(16, Wt * 2, sum_dtype, _TILE_BYTES[sum_dtype]),  # sum out
        cb(17, Wt * 2, out_dtype, _TILE_BYTES[out_dtype]),  # normed out
    ]

    reader_ct = [Wt]
    for t in (a, b, gamma_tiles, scaler_tile, eps_tile):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    compute_ct = [Wt, _CHUNK, int(sum_dtype == b8)]
    writer_ct = [Wt]
    for t in (sum_tensor, out_tensor):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    reader_rt, compute_rt, writer_rt = [], [], []
    cursor = 0
    for cx, cy, n_rows in per_core:
        core = (cx, cy)
        reader_rt.append(
            (
                core,
                [
                    a.buffer_address(),
                    b.buffer_address(),
                    gamma_tiles.buffer_address(),
                    scaler_tile.buffer_address(),
                    eps_tile.buffer_address(),
                    n_rows,
                    cursor,
                ],
            )
        )
        compute_rt.append((core, [n_rows]))
        writer_rt.append((core, [sum_tensor.buffer_address(), out_tensor.buffer_address(), n_rows, cursor]))
        cursor += n_rows

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=READER_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=used_cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=COMPUTE_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=used_cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    dst_full_sync_en=False,
                    bfp8_pack_precise=True,
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=WRITER_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=used_cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ],
        cbs=cbs,
    )
    ttnn.generic_op([a, b, gamma_tiles, scaler_tile, eps_tile, sum_tensor, out_tensor], program_descriptor)
    return sum_tensor, out_tensor
