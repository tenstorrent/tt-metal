# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Token-dispatch weighted reduce over the top-k slots as one generic_op (Laguna; see kernels/pcr_*.cpp).

out[t, :] = sum over slots k of weights[t, k] * combined[t, k, :] (+ residual[t, :]), TILE bf16. The kernels are
ttnn.experimental.deepseek_prefill.post_combine_reduce's with three additions Laguna's token-dispatch prefill uses:

  * local slots only: the writer turns the dispatch table and the token's expert ids into a per-(token, slot) mask
    (c_4) and the reader reads only slots whose expert is on this chip. A token with no local expert gets zeros, so
    the combine buffer needs no zero fill (combine_local leaves the other slots unwritten).
  * residual: the shared-expert partial is added to the reduced tiles in-op (c_5 -> c_18) instead of by a full add.
  * column groups: when there are fewer 32-token chunks than cores, each core takes one (chunk, column group) unit.

l1_end: buffers the caller keeps resident in L1 start there; if the CBs would end past it, the residual CB holds one
chunk's tiles instead of two.
"""
from __future__ import annotations

from pathlib import Path

import ttnn

from .l1_budget import l1_allocator_base

_KDIR = Path(__file__).resolve().parent / "kernels"
_TOKENS_PER_CHUNK = 32
_TILE_ELEMS = 1024
_TILE_W = 32
_BF16 = 2
_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088, ttnn.uint32: 4096, ttnn.uint16: 2048}
_NO_SKIP_ROW = 0xFFFFFFFF


def _round_up(x, m):
    return -(-x // m) * m


def combine_reduce(combined, weights, indices, dispatch_table, *, expert_dim, residual=None, l1_end=0):
    """combined: [..., T, K, H] bf16 (expert_dim = the K axis); weights: [..., T, K, 1] (one weight per page) or
    [..., T, K] (one token's weights per page); indices: [T, K] expert ids; dispatch_table: this chip's dispatch table
    (entry >= 0: the expert is local); residual: optional [..., T, H] bf16 TILE added in-op.
    Returns [..., T, H] TILE in combined's dtype, DRAM."""
    device = combined.device()
    assert combined.layout == ttnn.ROW_MAJOR_LAYOUT and combined.dtype == ttnn.bfloat16, (
        combined.layout,
        combined.dtype,
    )
    assert weights.layout == ttnn.ROW_MAJOR_LAYOUT and weights.dtype == ttnn.bfloat16, (weights.layout, weights.dtype)
    assert indices.layout == ttnn.ROW_MAJOR_LAYOUT and indices.dtype == ttnn.uint16, (indices.layout, indices.dtype)
    assert dispatch_table.layout == ttnn.ROW_MAJOR_LAYOUT and dispatch_table.dtype == ttnn.int32
    if residual is not None:
        assert residual.layout == ttnn.TILE_LAYOUT and residual.dtype == ttnn.bfloat16, (
            residual.layout,
            residual.dtype,
        )
    shape = list(combined.shape)
    emb_dim, num_experts = shape[-1], shape[expert_dim]
    num_tokens = 1
    for d in shape[:expert_dim]:
        num_tokens *= d
    assert emb_dim % _TILE_W == 0 and -(-emb_dim // _TILE_ELEMS) <= 8, emb_dim
    assert num_tokens % _TOKENS_PER_CHUNK == 0, num_tokens
    weights_per_token = len(weights.padded_shape) + 1 == len(shape)

    gs = device.compute_with_storage_grid_size()
    cores_total = gs.x * gs.y
    total_chunks = num_tokens // _TOKENS_PER_CHUNK
    row_tiles = emb_dim // _TILE_W
    col_groups = 1
    for g in range(2, row_tiles + 1):
        if row_tiles % g == 0 and total_chunks * g <= cores_total:
            col_groups = g
    col_width = emb_dim // col_groups
    num_cores = min(total_chunks * col_groups, cores_total)
    spread = min(total_chunks, cores_total)
    base_chunks, extra_chunks = total_chunks // spread, total_chunks % spread
    cb_tiles = -(-col_width // _TILE_ELEMS)  # CB pages holding one column-group row
    out_tiles = col_width // _TILE_W  # output tiles per 32-token block of a column group
    col_bytes = col_width * _BF16

    grid = ttnn.num_cores_to_corerangeset(num_cores, gs, True)
    cores = [(i % gs.x, i // gs.x) for i in range(num_cores)]
    out_shape = [d for i, d in enumerate(shape) if i != expert_dim]
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape), combined.dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    tile = _TILE_BYTES[combined.dtype]
    w_page = weights.buffer_aligned_page_size()

    def cb(index, fmt, page, total):
        return ttnn.CBDescriptor(
            total_size=total,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
        )

    mask_page = _round_up(_TOKENS_PER_CHUNK * num_experts, 32)
    table_pages, table_page, table_apage = (
        dispatch_table.buffer_num_pages(),
        dispatch_table.buffer_page_size(),
        dispatch_table.buffer_aligned_page_size(),
    )
    idx_page, idx_apage = indices.buffer_page_size(), indices.buffer_aligned_page_size()
    cbs = [
        cb(0, combined.dtype, tile, 2 * num_experts * cb_tiles * tile),  # one token's slots per batch, x2
        cb(1, weights.dtype, tile, 2 * num_experts * tile),  # one token's weights per batch, x2
        cb(2, dispatch_table.dtype, table_apage, table_pages * table_apage),  # dispatch table
        cb(4, ttnn.uint8, mask_page, 2 * mask_page),  # per-chunk (token, slot) local mask, writer -> reader
        cb(3, indices.dtype, idx_apage, _TOKENS_PER_CHUNK * idx_apage),  # one chunk of expert ids
    ]
    if weights_per_token:
        cbs.append(cb(6, weights.dtype, _TOKENS_PER_CHUNK * w_page, _TOKENS_PER_CHUNK * w_page))
    if residual is not None:
        cbs.append(cb(5, combined.dtype, tile, 2 * out_tiles * tile))
        cbs.append(cb(18, combined.dtype, tile, out_tiles * tile))
    cbs.append(cb(16, combined.dtype, tile, _TOKENS_PER_CHUNK * cb_tiles * tile))  # output chunk
    cbs.append(cb(17, combined.dtype, tile, _TOKENS_PER_CHUNK * cb_tiles * tile))  # row-major tilize scratch
    if l1_end and residual is not None:
        if l1_allocator_base(device) + sum(_round_up(c.total_size, 64) for c in cbs) > l1_end:
            cbs = [
                (cb(5, combined.dtype, tile, out_tiles * tile) if c.format_descriptors[0].buffer_index == 5 else c)
                for c in cbs
            ]

    def acc(t):
        return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    has_res = residual is not None
    reader_ct = (
        [num_experts, cb_tiles, col_bytes, 1]
        + acc(combined)
        + [int(has_res), out_tiles, row_tiles]
        + acc(residual if has_res else combined)
        + [0, _NO_SKIP_ROW, 0, 0]  # no combine_row_index
        + acc(combined)
    )
    compute_ct = [num_experts, cb_tiles, table_pages, 1, int(has_res), out_tiles]
    writer_ct = (
        [num_experts, cb_tiles, out_tiles, table_pages, table_page, table_apage, idx_page, idx_apage]
        + acc(weights)
        + acc(out)
        + acc(dispatch_table)
        + acc(indices)
        + [1, row_tiles, int(has_res), int(weights_per_token), w_page]
    )

    rt_reader, rt_compute, rt_writer = [], [], []
    token_start = 0
    for i, core in enumerate(cores):
        chunks = 1 if col_groups > 1 else base_chunks + (1 if i < extra_chunks else 0)
        col_group = i % col_groups if col_groups > 1 else 0
        if col_groups > 1:
            token_start = (i // col_groups) * _TOKENS_PER_CHUNK
        rt_reader.append(
            (
                core,
                [
                    combined.buffer_address(),
                    token_start,
                    chunks,
                    col_group * col_bytes,
                    residual.buffer_address() if has_res else 0,
                    col_group * out_tiles,
                ],
            )
        )
        rt_compute.append((core, [token_start, chunks]))
        rt_writer.append(
            (
                core,
                [
                    weights.buffer_address(),
                    out.buffer_address(),
                    dispatch_table.buffer_address(),
                    indices.buffer_address(),
                    token_start,
                    chunks,
                    col_group * out_tiles,
                ],
            )
        )
        if col_groups == 1:
            token_start += chunks * _TOKENS_PER_CHUNK

    compute_cfg = ttnn.ComputeConfigDescriptor()
    compute_cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    compute_cfg.fp32_dest_acc_en = False
    compute_cfg.dst_full_sync_en = False

    def kernel(name, ct, rt, config):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    kernels = [
        kernel("pcr_reader.cpp", reader_ct, rt_reader, ttnn.ReaderConfigDescriptor()),
        kernel("pcr_compute.cpp", compute_ct, rt_compute, compute_cfg),
        kernel("pcr_writer.cpp", writer_ct, rt_writer, ttnn.WriterConfigDescriptor()),
    ]
    io = [combined, weights, indices, dispatch_table] + ([residual] if has_res else []) + [out]
    ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs))
    return out
