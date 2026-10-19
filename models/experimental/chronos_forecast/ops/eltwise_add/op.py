# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Same-shape ``a + b`` on interleaved tiles with batched NoC reads and writes.

binary_ng's interleaved reader and writer wait on a barrier after every tile, and give each core a
contiguous tile range whose pages are spread over every L1 bank. Here each core moves ``batch`` tiles
per barrier, and when a, b and the output are all L1-interleaved each core takes the pages of its own
bank, so no tile crosses the NoC.
"""

from __future__ import annotations

import ttnn

from models.experimental.chronos_forecast.ops import common

_OP = "eltwise_add"
BATCH_TILES = 4
_DTYPES = (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b)
# Half-sync DST without fp32 accumulation.
_DST_TILES = 8


def add(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    memory_config=None,
    dtype=None,
    batch: int = BATCH_TILES,
    bank_local: bool = True,
) -> ttnn.Tensor:
    """``a + b`` for tile-layout interleaved tensors whose tiles line up (equal padded last two dims and
    tile count, so leading dims of 1 may differ). The output has a's shape; its dtype defaults to a's.

    ``bank_local`` applies only when all three tensors are L1-interleaved.
    """
    common.require_interleaved_tile("a", a)
    common.require_interleaved_tile("b", b)
    if tuple(a.padded_shape)[-2:] != tuple(b.padded_shape)[-2:] or common.num_tiles(a) != common.num_tiles(b):
        raise ValueError(f"add needs matching tiles, got {a.padded_shape} and {b.padded_shape}")
    out_dtype = a.dtype if dtype is None else dtype
    for name, dt in (("a", a.dtype), ("b", b.dtype), ("out", out_dtype)):
        if dt not in _DTYPES:
            raise ValueError(f"add supports {_DTYPES} for {name}, got {dt}")
    if not 1 <= batch <= _DST_TILES:
        raise ValueError(f"batch must be in [1, {_DST_TILES}], got {batch}")

    device = a.device()
    mem = a.memory_config() if memory_config is None else memory_config
    out = ttnn.allocate_tensor_on_device(a.shape, out_dtype, ttnn.TILE_LAYOUT, device, mem)
    if out.padded_shape != a.padded_shape:
        raise ValueError(f"add output padded shape {out.padded_shape} differs from input {a.padded_shape}")

    if bank_local and all(common.is_l1_interleaved(t) for t in (a, b, out)):
        all_cores, work, stride = common.split_banks(device, common.num_tiles(a))
    else:
        (all_cores, work), stride = common.split_rows(device, common.num_tiles(a)), 1
    cbs = [
        common.cb(0, 2 * batch, a.dtype, all_cores),
        common.cb(1, 2 * batch, b.dtype, all_cores),
        common.cb(2, 2 * batch, out_dtype, all_cores),
    ]

    a_addr, b_addr, out_addr = a.buffer_address(), b.buffer_address(), out.buffer_address()
    reader = ttnn.KernelDescriptor(
        kernel_source=common.kernel_path(_OP, "reader_eltwise_add.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=[batch, stride, *common.accessor_args(a), *common.accessor_args(b)],
        runtime_args=[((c.x, c.y), [a_addr, b_addr, n, start]) for c, n, start in work],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=common.kernel_path(_OP, "writer_eltwise_add.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=[batch, stride, *common.accessor_args(out)],
        runtime_args=[((c.x, c.y), [out_addr, n, start]) for c, n, start in work],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=common.kernel_path(_OP, "compute_eltwise_add.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=[batch],
        runtime_args=[((c.x, c.y), [n]) for c, n, _ in work],
        config=ttnn.ComputeConfigDescriptor(),
    )
    ttnn.generic_op([a, b, out], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs))
    return out
