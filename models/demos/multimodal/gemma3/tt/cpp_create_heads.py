# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""C++ Metalium head split for gemma3's prefill QKV (the cpp rung on NlpCreateHeadsDeviceOperation).

WHY: the stock op's interleaved program factory computes
``num_blocks = batch * 1 * seq / TILE_HEIGHT`` and hands that to ``split_work_to_cores`` -- it
parallelises over SEQ TILES ONLY. At S=128 that is 4 work units, so 4 cores of ~110 do the whole
split and the profiler tags the op ``grid=tiny``. Nothing about the work requires that: the split is
an embarrassingly parallel tile gather over (heads x seq_tiles x head_dim_tiles).

This builds the same op via ``ttnn.generic_op`` but walks the OUTPUT tile space, so the work-unit
count is 16x larger (q at S=128: 16 heads x 4 seq tiles x 8 dim tiles = 512 units) and the grid
fills. That is GUIDELINES 03 s.5(c)'s ``head_groups`` idea taken to its limit -- one work unit per
output tile.

Output tiles are consecutive per tensor, so only the READ side is a gather; the write side reuses
the stock ``writer_unary_interleaved_start_id.cpp``.
"""
from __future__ import annotations

import ttnn

TILE = 32

_READER = "models/demos/multimodal/gemma3/tt/kernels/reader_qkv_gather.cpp"
_WRITER = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"

# Bytes per 32x32 tile, by dtype. bfp formats carry a per-tile exponent section.
_TILE_BYTES = {
    ttnn.bfloat16: 32 * 32 * 2,
    ttnn.float32: 32 * 32 * 4,
    ttnn.bfloat8_b: 32 * 32 + 64,
    ttnn.bfloat4_b: 32 * 32 // 2 + 64,
}


def _gather_into(x, out, col0, s_tiles, d_tiles, in_w_tiles, all_cores):
    """One generic_op producing `out` by gathering tiles of `x` starting at tile column col0."""
    num_out_tiles = 1
    for d in out.shape:
        num_out_tiles *= int(d)
    num_out_tiles //= TILE * TILE

    (_, core_grid, group_1, group_2, work_1, work_2) = ttnn.split_work_to_cores(all_cores, num_out_tiles)

    tile_bytes = _TILE_BYTES[x.dtype]
    cb_fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=x.dtype, page_size=tile_bytes)
    cb = ttnn.CBDescriptor(total_size=2 * tile_bytes, core_ranges=core_grid, format_descriptors=[cb_fmt])

    reader_ct = [s_tiles, d_tiles, in_w_tiles, col0]
    reader_ct.extend(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    writer_ct = [0]
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    tile = 0
    for group, per_core in ((group_1, work_1), (group_2, work_2)):
        if per_core == 0:
            continue
        for core_range in group.ranges():
            for cx in range(core_range.start.x, core_range.end.x + 1):
                for cy in range(core_range.start.y, core_range.end.y + 1):
                    reader_rt[cx][cy] = [x.buffer_address(), per_core, tile]
                    writer_rt[cx][cy] = [out.buffer_address(), per_core, tile]
                    tile += per_core

    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=_READER,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_WRITER,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ],
        semaphores=[],
        cbs=[cb],
    )
    return ttnn.generic_op([x, out], program)


def can_run(x, num_heads, num_kv_heads):
    """Only the tile-aligned, non-transposed prefill shapes this gather is written for."""
    if len(x.shape) != 4 or int(x.shape[0]) != 1 or int(x.shape[1]) != 1:
        return False
    if x.dtype not in _TILE_BYTES:
        return False
    width = int(x.shape[-1])
    seq = int(x.shape[-2])
    if seq % TILE or width % TILE:
        return False
    head_dim = width // (num_heads + 2 * num_kv_heads)
    return head_dim > 0 and head_dim % TILE == 0 and head_dim * (num_heads + 2 * num_kv_heads) == width


def create_qkv_heads(x, num_heads, num_kv_heads, memory_config):
    """Drop-in for ttnn.experimental.nlp_create_qkv_heads with transpose_k_heads=False."""
    width = int(x.shape[-1])
    seq = int(x.shape[-2])
    head_dim = width // (num_heads + 2 * num_kv_heads)
    s_tiles = seq // TILE
    d_tiles = head_dim // TILE
    in_w_tiles = width // TILE

    grid = x.device().compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])

    def _alloc(heads):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, heads, seq, head_dim]), x.dtype, x.layout, x.device(), memory_config
        )

    q, k, v = _alloc(num_heads), _alloc(num_kv_heads), _alloc(num_kv_heads)
    # q occupies tile columns [0, nq*d), then k, then v -- the standard fused QKV layout.
    q_cols = num_heads * d_tiles
    kv_cols = num_kv_heads * d_tiles
    _gather_into(x, q, 0, s_tiles, d_tiles, in_w_tiles, all_cores)
    _gather_into(x, k, q_cols, s_tiles, d_tiles, in_w_tiles, all_cores)
    _gather_into(x, v, q_cols + kv_cols, s_tiles, d_tiles, in_w_tiles, all_cores)
    return q, k, v
