# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Rotary embedding with batch-shared cos/sin on interleaved tiles (the prefill case of
``ttnn.experimental.rotary_embedding``).

Each core reads the whole cos/sin table once and keeps it in L1, reads each input row once (the
rotated half is indexed from the same tiles), batches NoC reads and writes per block of rows, and
forms every output tile from two ELWMULs accumulated in DST against a pre-negated first-half sin.
"""

from __future__ import annotations

import ttnn

from models.experimental.chronos_forecast.ops import common

_OP = "rotary_embedding"
# L1 budget per core for the resident cos/sin table plus the pre-negated first-half sin tiles.
_COS_SIN_L1_BYTES = 96 * 1024
_BLOCK_TILES = 8
_NEG_ONE_BF16 = 0xBF80

IN_CB, COS_CB, SIN_CB, SCALAR_CB, OUT_CB, NEG_SIN_CB = 0, 2, 3, 4, 16, 24


def rotary_embedding(
    x: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    *,
    memory_config=None,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en: bool = False,
) -> ttnn.Tensor:
    """``x * cos + rotate_half(x) * sin`` for x [..., S, Dh] and cos/sin [1, 1, >=S, Dh].

    The defaults match ``ttnn.experimental.rotary_embedding``'s compute config. The output keeps x's shape.
    """
    for name, t in (("x", x), ("cos", cos), ("sin", sin)):
        common.require_interleaved_tile(name, t)
    Ht = x.padded_shape[-2] // common.TILE
    Wt = x.padded_shape[-1] // common.TILE
    if Wt < 2 or Wt % 2:
        raise ValueError(f"head dim must be an even number of tiles, got {x.padded_shape[-1]}")
    for name, t in (("cos", cos), ("sin", sin)):
        if common.padded_volume(t) != t.padded_shape[-2] * t.padded_shape[-1]:
            raise ValueError(f"{name} must be batch-shared [1, 1, S, Dh], got {t.shape}")
        if t.padded_shape[-1] != x.padded_shape[-1] or t.padded_shape[-2] < x.padded_shape[-2]:
            raise ValueError(f"{name} {t.padded_shape} does not cover x {x.padded_shape}")
    HtWt, half_Wt = Ht * Wt, Wt // 2
    table_bytes = HtWt * (common.tile_bytes(cos.dtype) + common.tile_bytes(sin.dtype))
    table_bytes += Ht * half_Wt * common.tile_bytes(sin.dtype)
    if table_bytes > _COS_SIN_L1_BYTES:
        raise ValueError(f"cos/sin table needs {table_bytes} B of L1 per core, over {_COS_SIN_L1_BYTES}")

    device = x.device()
    mem = x.memory_config() if memory_config is None else memory_config
    out = ttnn.allocate_tensor_on_device(x.shape, x.dtype, ttnn.TILE_LAYOUT, device, mem)
    if out.padded_shape != x.padded_shape:
        raise ValueError(f"output padded shape {out.padded_shape} differs from input {x.padded_shape}")

    num_rows = common.num_tiles(x) // Wt
    all_cores, work = common.split_rows(device, num_rows)
    rows_per_block = max(1, _BLOCK_TILES // Wt)
    block_tiles = rows_per_block * Wt
    dst_block = min(Wt, 4 if fp32_dest_acc_en else 8)
    while Wt % dst_block:
        dst_block -= 1

    cbs = [
        common.cb(IN_CB, 2 * block_tiles, x.dtype, all_cores),
        common.cb(COS_CB, HtWt, cos.dtype, all_cores),
        common.cb(SIN_CB, HtWt, sin.dtype, all_cores),
        common.cb(SCALAR_CB, 1, ttnn.bfloat16, all_cores),
        common.cb(NEG_SIN_CB, Ht * half_Wt, sin.dtype, all_cores),
        common.cb(OUT_CB, 2 * block_tiles, x.dtype, all_cores),
    ]

    x_addr, cos_addr, sin_addr, out_addr = (t.buffer_address() for t in (x, cos, sin, out))
    reader = ttnn.KernelDescriptor(
        kernel_source=common.kernel_path(_OP, "reader_rotary_embedding.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=[
            IN_CB,
            COS_CB,
            SIN_CB,
            SCALAR_CB,
            _NEG_ONE_BF16,
            Wt,
            HtWt,
            rows_per_block,
            *common.accessor_args(x),
            *common.accessor_args(cos),
            *common.accessor_args(sin),
        ],
        runtime_args=[((c.x, c.y), [x_addr, cos_addr, sin_addr, n, start * Wt]) for c, n, start in work],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=common.kernel_path(_OP, "writer_rotary_embedding.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=[OUT_CB, block_tiles, *common.accessor_args(out)],
        runtime_args=[((c.x, c.y), [out_addr, n * Wt, start * Wt]) for c, n, start in work],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=common.kernel_path(_OP, "compute_rotary_embedding.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=[IN_CB, COS_CB, SIN_CB, SCALAR_CB, NEG_SIN_CB, OUT_CB, Wt, half_Wt, Ht, dst_block],
        runtime_args=[((c.x, c.y), [start % Ht, n]) for c, n, start in work],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=math_fidelity, math_approx_mode=True, fp32_dest_acc_en=fp32_dest_acc_en
        ),
    )
    ttnn.generic_op([x, cos, sin, out], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs))
    return out
