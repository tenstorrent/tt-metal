# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``GatherCombineReduce``: ``ttnn.embedding(pos, table)`` + ``post_combine_reduce`` fused as a model-local
``ttnn.generic_op`` (tt-blaze G53F-0190).

    y[t, :] = sum_j w[t, j] * table[pos[t, j], :]      (post_combine_reduce's order, zero weights skipped)

The regions MoE (tt/moe.py ``_forward_regions``) gathers every pick's output row, ``comb [T*k, H]`` (1.07 GB per
16 K-token layer call at k 8, H 4096), only for ``post_combine_reduce`` to read it back; ~3/4 of those rows are the
dummy slot of the non-local picks, which the reduce then skips (their weight is zero). This op runs
``post_combine_reduce``'s own compute and writer kernels (the ttnn files, unchanged) behind a reader that fetches
``table[pos[t, j]]`` directly, only for the rows the compute kernel uses (pick 0 of every token, and every pick not
at the dummy slot), a token's rows in flight together. Same rows, same programs: bit-identical output.

Contract: ``table`` [R, H] bf16 ROW_MAJOR DRAM interleaved (one page per row; R - 1 = the dummy slot); ``pos`` [1, T*k]
uint32 ROW_MAJOR (one page) or the [1, 1, T, k] fp32 TILE slot table (G53F-0193); ``weights`` [1, 1, T, k, 1] bf16 ROW_MAJOR with ``w == 0`` exactly where
``pos == R - 1`` (and pick 0 any value); T % 32 == 0, H % 32 == 0, H <= 8192. Returns [1, 1, T, H] bf16 TILE DRAM.
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
_PCR_DIR = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/kernels"
TOKENS_PER_CHUNK = 32
BF16_TILE_BYTES = 32 * 32 * 2


def _pcr_kernel(name: str) -> str:
    root = Path(os.environ.get("TT_METAL_HOME", Path(__file__).resolve().parents[5]))
    return str(root / _PCR_DIR / name)


class GatherCombineReduce:
    CB_IN, CB_W, CB_POS, CB_OUT, CB_RM = 0, 1, 4, 16, 17

    def __init__(self, device):
        self.device = device

    def __call__(self, table: ttnn.Tensor, pos: ttnn.Tensor, weights: ttnn.Tensor) -> ttnn.Tensor:
        R, H = int(table.shape[-2]), int(table.shape[-1])
        k = int(weights.shape[-2])
        T = int(weights.shape[-3])
        assert table.dtype == ttnn.bfloat16 and table.layout == ttnn.ROW_MAJOR_LAYOUT, (table.dtype, table.layout)
        # pos: the [1, T*k] uint32 ROW_MAJOR slot copy, or (G53F-0193) the [1,1,T,k] fp32 TILE slot table itself
        pos_tile = pos.layout == ttnn.TILE_LAYOUT
        if pos_tile:
            assert pos.dtype == ttnn.float32 and int(pos.shape[-1]) == k and int(pos.shape[-2]) == T, (
                pos.dtype,
                pos.shape,
            )
        else:
            assert pos.dtype == ttnn.uint32 and pos.layout == ttnn.ROW_MAJOR_LAYOUT, (pos.dtype, pos.layout)
            assert int(pos.shape[-1]) == T * k, (pos.shape, T, k)
        assert weights.dtype == ttnn.bfloat16 and weights.layout == ttnn.ROW_MAJOR_LAYOUT
        assert T % TOKENS_PER_CHUNK == 0 and H % 32 == 0, (T, H)
        emb_dim_cb_tiles = -(-H // 1024)
        assert emb_dim_cb_tiles <= 8, H
        emb_dim_bytes = H * 2
        assert table.buffer_aligned_page_size() == emb_dim_bytes, table.buffer_aligned_page_size()
        pos_page = pos.buffer_aligned_page_size()
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, T, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )

        grid = self.device.compute_with_storage_grid_size()
        total_chunks = T // TOKENS_PER_CHUNK
        ncores = min(total_chunks, grid.x * grid.y)
        base, extra = divmod(total_chunks, ncores)
        # row-major over the grid, as post_combine_reduce's factory (grid_to_cores(row_major=True))
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

        def cb(index, total, page, fmt):
            return ttnn.CBDescriptor(
                total_size=total,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        tile = BF16_TILE_BYTES
        pos_cb = pos_page if pos_tile else TOKENS_PER_CHUNK * k * 4 + 64
        cbs = [
            cb(self.CB_IN, 2 * k * emb_dim_cb_tiles * tile, tile, ttnn.bfloat16),  # two tokens' k rows in flight
            cb(self.CB_W, tile, tile, ttnn.bfloat16),
            cb(self.CB_POS, pos_cb, pos_cb, ttnn.uint32),
            cb(self.CB_OUT, TOKENS_PER_CHUNK * emb_dim_cb_tiles * tile, tile, ttnn.bfloat16),
            cb(self.CB_RM, TOKENS_PER_CHUNK * emb_dim_cb_tiles * tile, tile, ttnn.bfloat16),
        ]
        reader_ct = [k, emb_dim_cb_tiles, emb_dim_bytes, R - 1, self.CB_POS, pos_page, tile, int(pos_tile)]
        reader_ct += ttnn.TensorAccessorArgs(table).get_compile_time_args()
        reader_ct += ttnn.TensorAccessorArgs(pos).get_compile_time_args()
        compute_ct = [k, emb_dim_cb_tiles, 0, 0]  # no dispatch table: the weight-based skip
        writer_ct = [k, emb_dim_cb_tiles, H // 32, 0, 0, 0, 0, 0]
        w_args = ttnn.TensorAccessorArgs(weights).get_compile_time_args()
        writer_ct += w_args + ttnn.TensorAccessorArgs(out).get_compile_time_args() + w_args + w_args + [0]

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        token_start = 0
        for i, c in enumerate(cores):
            n = base + (1 if i < extra else 0)
            reader_rt[c.x][c.y] = [table.buffer_address(), pos.buffer_address(), token_start, n]
            compute_rt[c.x][c.y] = [token_start, n]
            writer_rt[c.x][c.y] = [weights.buffer_address(), out.buffer_address(), token_start, n]
            token_start += n * TOKENS_PER_CHUNK

        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_gather_combine.cpp"),
                core_ranges=crs,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_pcr_kernel("deepseek_moe_post_combine_reduce_compute.cpp"),
                core_ranges=crs,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, dst_full_sync_en=False
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_pcr_kernel("deepseek_moe_post_combine_reduce_writer.cpp"),
                core_ranges=crs,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([table, pos, weights, out], program)
