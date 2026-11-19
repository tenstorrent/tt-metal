# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""pick_combine (model-local generic op): the routed experts' pick gather + weighted combine as ONE op.

The regions prefill path (``tt/moe.py`` ``_forward_regions``) ends with ``ttnn.embedding(pos_rm, y_rm)`` (every pick's
12 KB bf16 row of the expert output table into a [T*k, H] table, 403 MB at T = 4096) and
``deepseek_prefill.post_combine_reduce`` (the weighted sum over the k picks, reading that table back): ~4-5 ms of the
25 ms sub-chunk, ~0.7 s of the 6.5 s 1 M chunk over the 42 MoE layers. This op reads each pick's row once: per job
(a block of 32 tokens x a slice of ``sw`` x 32 output columns) the reader gathers, pick by pick, the block's 32 rows
of the slice (``sw`` x 64 B contiguous per row, zero rows for zero weights: the non-local picks' table rows are
garbage) into a row-major block the compute tilizes, builds the k diagonal weight tiles diag(w[:, j]) once per
block, and the compute accumulates sum_j diag(w[:, j]) @ G_j into fp32 DST tiles, packed to the [1, 1, T, H] bf16
TILE output the writer scatters. DRAM traffic per sub-chunk at T = 4096: 403 MB read + 50 MB written against the
chain's 1.26 GB. Exact: bf16 x bf16 products in fp32, one rounding to bf16 at the pack (the chain rounds twice).

Inputs: ``y_rm`` [max_buf, H] bf16 ROW_MAJOR (the relayout'd expert output table), ``pos`` [T*k] uint32 (every pick's
table row, pick j of token t at t*k + j; the non-local picks point at the dummy row), ``w`` [T, k] float32 / bf16 (the
routing weights, 0 at the non-local picks). The op reshapes ``pos`` to [T/32, 32*k] and ``w`` to [T/32, 32*k] bf16
ROW_MAJOR (one 64-byte-aligned row per 32-token block). Output: [1, 1, T, H] bf16 TILE (what ``post_combine_reduce``
returns). T % 32 == 0, H % (32 * sw) == 0, k <= 32.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
TILE_BYTES = 2048

CB_G_RM, CB_G, CB_W, CB_POS, CB_WSLAB, CB_OUT = 0, 1, 2, 3, 4, 16


class PickCombine:
    """``sw``: output column tiles per job (8: fp32 DST holds 8 accumulating tiles in full-sync mode; the gather reads
    sw x 64 B per row)."""

    def __init__(self, device, sw: int = 8, math_fidelity=ttnn.MathFidelity.HiFi4):
        self.device = device
        self.sw = int(sw)
        self.fidelity = math_fidelity
        grid = device.compute_with_storage_grid_size()
        self.grid_x, self.grid_y = int(grid.x), int(grid.y)

    @staticmethod
    def golden(y_rm: torch.Tensor, pos: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """[T, H] float32: sum_j w[t, j] * y_rm[pos[t*k + j]] with zero-weight picks skipped (their rows may be anything)."""
        T, k = w.shape
        rows = y_rm.float()[pos.reshape(T, k).long()]  # [T, k, H]
        wz = w.float().clone()
        rows = torch.where((wz != 0).unsqueeze(-1), rows, torch.zeros_like(rows))
        return (rows * wz.unsqueeze(-1)).sum(dim=1)

    def __call__(self, y_rm: ttnn.Tensor, pos: ttnn.Tensor, w: ttnn.Tensor, *, T: int, k: int) -> ttnn.Tensor:
        assert y_rm.layout == ttnn.ROW_MAJOR_LAYOUT and y_rm.dtype == ttnn.bfloat16, (y_rm.layout, y_rm.dtype)
        max_buf, H = (int(v) for v in list(y_rm.shape)[-2:])
        S = self.sw * TILE
        assert T % TILE == 0 and H % S == 0 and 1 <= k <= TILE, (T, H, k, S)
        blocks, slices = T // TILE, H // S
        # the per-block rows: pos [blocks, 32*k] uint32 and w [blocks, 32*k] bf16, both ROW_MAJOR (>= 64-byte rows)
        pos_b = ttnn.reshape(pos, [blocks, TILE * k])
        if pos_b.layout != ttnn.ROW_MAJOR_LAYOUT or pos_b.dtype != ttnn.uint32:
            pos_b = ttnn.to_layout(ttnn.typecast(pos_b, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        w_b = w if w.dtype == ttnn.bfloat16 else ttnn.typecast(w, ttnn.bfloat16)
        if w_b.layout != ttnn.ROW_MAJOR_LAYOUT:
            w_b = ttnn.to_layout(w_b, ttnn.ROW_MAJOR_LAYOUT)
        w_b = ttnn.reshape(w_b, [blocks, TILE * k])
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, T, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )
        cores = [ttnn.CoreCoord(x, y) for y in range(self.grid_y) for x in range(self.grid_x)]
        core_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.grid_x - 1, self.grid_y - 1))]
        )
        jobs = blocks * slices  # block-major: a core's jobs walk the slices of one block before the next block
        n_cores = len(cores)
        base, extra = jobs // n_cores, jobs % n_cores

        def cb(index, page, pages, fmt=ttnn.bfloat16):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        sw = self.sw
        cbs = [
            cb(CB_G_RM, TILE_BYTES, 2 * sw),  # one pick's [32, S] row-major block (sw tile-sized pages), x2
            cb(CB_G, TILE_BYTES, 2 * k * sw),  # the k tilized pick blocks of a job, x2
            cb(CB_W, TILE_BYTES, 2 * k),  # the k diagonal weight tiles of a block, x2
            cb(CB_POS, TILE * k * 4, 1),  # a block's pick rows
            cb(CB_WSLAB, TILE * k * 2, 1),  # a block's weights
            cb(CB_OUT, TILE_BYTES, 2 * sw),
        ]
        reader_ct = [CB_G_RM, CB_W, CB_POS, CB_WSLAB, k, sw, slices, H * 2, TILE * k * 4, TILE * k * 2]
        for t in (y_rm, pos_b, w_b):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [CB_G_RM, CB_G, CB_W, CB_OUT, k, sw, slices]
        writer_ct = [CB_OUT, sw, H // TILE, slices]
        writer_ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()
        reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, core in enumerate(cores):
            j0 = i * base + min(i, extra)
            jn = base + (1 if i < extra else 0)
            reader_rt[core.x][core.y] = [y_rm.buffer_address(), pos_b.buffer_address(), w_b.buffer_address(), j0, jn]
            compute_rt[core.x][core.y] = [j0, jn]
            writer_rt[core.x][core.y] = [out.buffer_address(), j0, jn]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_pick_combine.cpp"),
                core_ranges=core_set,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_pick_combine.cpp"),
                core_ranges=core_set,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_pick_combine.cpp"),
                core_ranges=core_set,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                # fp32 DST holds 8 accumulating tiles only in full-sync mode (half-sync halves it to 4: the sw = 8 jobs
                # overran into the other half, 2026-11-02); the gather, not the math/pack overlap, bounds this op
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=self.fidelity, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=True
                ),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        ttnn.generic_op([y_rm, pos_b, w_b, out], program)
        for t in (pos_b, w_b):
            if t is not pos and t is not w:
                ttnn.deallocate(t)
        return out
