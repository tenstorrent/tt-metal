# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Fused QKV head split + RoPE: ``split_query_key_value_and_split_heads`` (transpose_key=False) followed by
``rotary_embedding`` on Q and K, in one program.

A work unit is one (batch, seq tile, head). Its Q and K tiles are rotated against a cos/sin table that
each core reads once, and its V tiles go straight from the reader to the writer.
"""

from __future__ import annotations

import ttnn

from models.experimental.chronos_forecast.ops import common

_OP = "qkv_heads_rope"
_NEG_ONE_BF16 = 0xBF80
_UNITS_PER_BLOCK = 2
# L1 budget per core for the resident cos/sin table plus the pre-negated first-half sin tiles.
_COS_SIN_L1_BYTES = 96 * 1024

QK_CB, V_CB, COS_CB, SIN_CB, SCALAR_CB, OUT_CB, NEG_SIN_CB = 0, 1, 2, 3, 4, 16, 24


def cos_sin_table_bytes(s_pad: int, head_dim: int, cos_dtype, sin_dtype) -> int:
    """Per-core L1 bytes for the cos/sin table plus the pre-negated first-half sin tiles."""
    seq_tiles, head_tiles = s_pad // common.TILE, head_dim // common.TILE
    table_bytes = seq_tiles * head_tiles * (common.tile_bytes(cos_dtype) + common.tile_bytes(sin_dtype))
    return table_bytes + seq_tiles * (head_tiles // 2) * common.tile_bytes(sin_dtype)


def fits_l1(s_pad: int, head_dim: int, cos_dtype, sin_dtype) -> bool:
    return cos_sin_table_bytes(s_pad, head_dim, cos_dtype, sin_dtype) <= _COS_SIN_L1_BYTES


def qkv_heads_rope(
    xqkv: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    *,
    num_heads: int,
    memory_config=None,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    units_per_block: int = _UNITS_PER_BLOCK,
):
    """[B, S, 3 * H * Dh] -> RoPE'd Q, RoPE'd K and V, each [B, H, S, Dh]. cos/sin are [1, 1, >=S, Dh]."""
    for name, t in (("xqkv", xqkv), ("cos", cos), ("sin", sin)):
        common.require_interleaved_tile(name, t)
    padded = tuple(xqkv.padded_shape)
    s_pad, width = padded[-2], padded[-1]
    if width % (3 * num_heads * common.TILE):
        raise ValueError(f"xqkv width {width} is not 3 * {num_heads} heads of whole tiles")
    head_dim = width // (3 * num_heads)
    head_tiles = head_dim // common.TILE
    if head_tiles % 2:
        raise ValueError(f"head dim must be an even number of tiles, got {head_dim}")
    seq_tiles = s_pad // common.TILE
    batch = common.padded_volume(xqkv) // (s_pad * width)
    for name, t in (("cos", cos), ("sin", sin)):
        tp = tuple(t.padded_shape)
        if common.padded_volume(t) != tp[-2] * tp[-1]:
            raise ValueError(f"{name} must be batch-shared [1, 1, S, Dh], got {t.shape}")
        if tp[-1] != head_dim or tp[-2] < s_pad:
            raise ValueError(f"{name} {tp} does not cover seq {s_pad} x head dim {head_dim}")
    cs_tiles = seq_tiles * head_tiles
    table_bytes = cos_sin_table_bytes(s_pad, head_dim, cos.dtype, sin.dtype)
    if table_bytes > _COS_SIN_L1_BYTES:
        raise ValueError(f"cos/sin table needs {table_bytes} B of L1 per core, over {_COS_SIN_L1_BYTES}")

    device = xqkv.device()
    mem = xqkv.memory_config() if memory_config is None else memory_config
    out_shape = ttnn.Shape([batch, num_heads, xqkv.shape[-2], head_dim])
    q, k, v = (ttnn.allocate_tensor_on_device(out_shape, xqkv.dtype, ttnn.TILE_LAYOUT, device, mem) for _ in range(3))

    num_units = batch * seq_tiles * num_heads
    all_cores, work = common.split_rows(device, num_units)
    upb = units_per_block
    dst_block = head_tiles
    while dst_block > 8 or head_tiles % dst_block:
        dst_block -= 1

    dt = xqkv.dtype
    cbs = [
        common.cb(QK_CB, 2 * 2 * head_tiles * upb, dt, all_cores),
        common.cb(V_CB, 2 * head_tiles * upb, dt, all_cores),
        common.cb(COS_CB, cs_tiles, cos.dtype, all_cores),
        common.cb(SIN_CB, cs_tiles, sin.dtype, all_cores),
        common.cb(SCALAR_CB, 1, ttnn.bfloat16, all_cores),
        common.cb(NEG_SIN_CB, seq_tiles * (head_tiles // 2), sin.dtype, all_cores),
        common.cb(OUT_CB, 2 * 2 * head_tiles * upb, dt, all_cores),
    ]

    src_addr, cos_addr, sin_addr = xqkv.buffer_address(), cos.buffer_address(), sin.buffer_address()
    q_addr, k_addr, v_addr = q.buffer_address(), k.buffer_address(), v.buffer_address()
    shape_args = [head_tiles, num_heads, seq_tiles, upb]
    reader = ttnn.KernelDescriptor(
        kernel_source=common.kernel_path(_OP, "reader_qkv_heads_rope.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=[
            *shape_args,
            _NEG_ONE_BF16,
            *common.accessor_args(xqkv),
            *common.accessor_args(cos),
            *common.accessor_args(sin),
        ],
        runtime_args=[((c.x, c.y), [src_addr, cos_addr, sin_addr, n, start]) for c, n, start in work],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=common.kernel_path(_OP, "writer_qkv_heads_rope.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=[*shape_args, *common.accessor_args(q), *common.accessor_args(k), *common.accessor_args(v)],
        runtime_args=[((c.x, c.y), [q_addr, k_addr, v_addr, n, start]) for c, n, start in work],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=common.kernel_path(_OP, "compute_qkv_heads_rope.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=[head_tiles, seq_tiles, num_heads, dst_block],
        runtime_args=[((c.x, c.y), [start, n]) for c, n, start in work],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, math_approx_mode=True),
    )
    ttnn.generic_op([xqkv, cos, sin, q, k, v], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs))
    return q, k, v
