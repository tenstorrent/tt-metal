# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``TopkRows``: top-k (k = 512) indices of every row of a [B, 1, 1, W] bf16 ROW_MAJOR score tensor, each row ranked
over its OWN prefix ``valid[b]`` (a [B, 1] uint32 tensor read on device, one 4-byte page per row), as a model-local
``ttnn.generic_op``. The kernels are the shared ``topk_large_indices`` reader / compute / writer with the per-row
valid-length change (the reader derives and pushes the chunk bounds per row, compute pops them per row), so ONE
call ranks all users' score rows of the sparse DSA decode under their own causal bounds: a 262 K-wide row costs
1.14 ms alone and B rows cost the same in one call (the rows run on different cores). Contract in README.md.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn

TILE = 32
TILE_HW = 1024
FACE_WIDTH = 16
SENTINEL = 0xFFFFFFFF
KERNELS = Path(__file__).resolve().parent / "kernels"


class TopkRows:
    def __init__(self, device, *, k: int = 512):
        assert k == 512, "the K=512 LLK window is the one wired (indices CB face geometry 16 x 2)"
        self.device = device
        self.k = k
        self.llk_k = 512
        grid = device.compute_with_storage_grid_size()
        self.grid_x, self.grid_y = grid.x, grid.y
        self.cores = [ttnn.CoreCoord(x, y) for y in range(self.grid_y) for x in range(self.grid_x)]  # row-wise
        self.core_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.grid_x - 1, self.grid_y - 1))]
        )

    # ------------------------------------------------------------------ golden
    @staticmethod
    def golden(scores: torch.Tensor, valid: torch.Tensor, k: int = 512) -> torch.Tensor:
        """scores [B,1,1,W] (bf16), valid [B] int -> [B,1,1,k] int64: torch.topk of row b over its first valid[b]
        columns, descending (ties: any order; -inf survivors -> 0xFFFFFFFF). valid[b] must be >= k."""
        B, W = scores.shape[0], scores.shape[-1]
        out = torch.full((B, 1, 1, k), SENTINEL, dtype=torch.int64)
        for b in range(B):
            n = int(valid[b])
            assert k <= n <= W, (b, n, k, W)
            v, i = torch.topk(scores[b, 0, 0, :n].float(), k)
            i[~torch.isfinite(v)] = SENTINEL
            out[b, 0, 0] = i
        return out

    # ------------------------------------------------------------------ program
    def _rows_per_core(self, B: int) -> list[tuple[int, int]]:
        """(start_row, rows) per core in row-wise core order (split_work_to_cores semantics: the first B % NC
        cores take one row more when B > NC)."""
        NC = len(self.cores)
        base, extra = divmod(B, NC)
        out, start = [], 0
        for c in range(NC):
            rows = base + (1 if c < extra else 0)
            out.append((start, rows))
            start += rows
        assert start == B
        return out

    def __call__(self, scores: ttnn.Tensor, valid_rows: ttnn.Tensor, out: ttnn.Tensor) -> ttnn.Tensor:
        """scores [B,1,1,W] bf16 ROW_MAJOR DRAM (W >= k); valid_rows [B, 1] uint32 ROW_MAJOR DRAM (per-row bound in
        elements, k <= bound <= W); out [B,1,1,k] uint32 ROW_MAJOR DRAM, pre-allocated. Returns ``out``."""
        assert scores.layout == ttnn.ROW_MAJOR_LAYOUT and scores.dtype == ttnn.bfloat16, (scores.layout, scores.dtype)
        assert valid_rows.layout == ttnn.ROW_MAJOR_LAYOUT and valid_rows.dtype == ttnn.uint32
        assert out.layout == ttnn.ROW_MAJOR_LAYOUT and out.dtype == ttnn.uint32
        B, W = scores.shape[0], scores.shape[-1]
        assert list(scores.shape) == [B, 1, 1, W] and W >= self.k, scores.shape
        assert list(valid_rows.shape) == [B, 1], (valid_rows.shape, B)
        assert list(out.shape) == [B, 1, 1, self.k], out.shape
        k, llk_k = self.k, self.llk_k
        cb_in, cb_indices, cb_indices_scratch, cb_meta = 0, 1, 2, 3
        elem = 2  # bf16
        input_chunk_bytes = llk_k * elem
        input_tile_bytes = TILE_HW * elem
        tiles_per_sequence = -(-llk_k // TILE_HW)  # 1
        indices_cb_row_bytes = llk_k * 4
        indices_row_bytes = k * 4
        source_slices = llk_k // FACE_WIDTH
        output_slices = k // FACE_WIDTH
        slice_bytes = FACE_WIDTH * 4
        # compute body mode (topk_large_indices_compute_body_mode.hpp): K512 -> FusedEndToEnd up to 32 chunks, else Classic
        physical_chunks = -(-W // llk_k)
        body_mode = 1 if physical_chunks <= 32 else 0
        input_row_bytes = scores.buffer_page_size()
        assert input_row_bytes == W * elem, (input_row_bytes, W)

        fmt = ttnn.CBFormatDescriptor
        f_idx = fmt(buffer_index=cb_indices, data_format=ttnn.float32, page_size=indices_cb_row_bytes)
        f_idx.face_geometry = ttnn.FaceGeometry(FACE_WIDTH, 2)  # the K=512 LLK's 16-row, 2-face index tiles
        cbs = [
            ttnn.CBDescriptor(
                total_size=2 * tiles_per_sequence * input_tile_bytes,
                core_ranges=self.core_set,
                format_descriptors=[fmt(buffer_index=cb_in, data_format=ttnn.bfloat16, page_size=input_tile_bytes)],
            ),
            ttnn.CBDescriptor(
                total_size=2 * indices_cb_row_bytes, core_ranges=self.core_set, format_descriptors=[f_idx]
            ),
            ttnn.CBDescriptor(
                total_size=64,
                core_ranges=self.core_set,
                format_descriptors=[fmt(buffer_index=cb_meta, data_format=ttnn.uint32, page_size=64)],
            ),
        ]
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()
        reader_ct = [cb_in, input_chunk_bytes, input_tile_bytes, tiles_per_sequence] + acc(scores)
        reader_ct += [1, cb_meta, 0] + acc(valid_rows)  # metadata mode, meta CB, valid_length_offset 0
        reader_ct += acc(scores)  # valid_end accessor placeholder (unused: address 0)
        reader_ct += [1]  # per-row metadata
        compute_ct = [cb_in, cb_indices, llk_k, body_mode, 1, cb_meta, 1]
        writer_ct = [
            cb_indices,
            cb_indices_scratch,
            indices_row_bytes,
            source_slices,
            output_slices,
            slice_bytes,
        ] + acc(out)
        reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, (start, rows) in zip(self.cores, self._rows_per_core(B)):
            reader_rt[core.x][core.y] = [start, rows]
            compute_rt[core.x][core.y] = [rows]
            writer_rt[core.x][core.y] = [start, rows]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "reader_topk_rows.cpp"),
                core_ranges=self.core_set,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                common_runtime_args=[scores.buffer_address(), valid_rows.buffer_address(), W, input_row_bytes, 0],
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "compute_topk_rows.cpp"),
                core_ranges=self.core_set,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                common_runtime_args=[W],
                config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True, dst_full_sync_en=True),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "writer_topk_rows.cpp"),
                core_ranges=self.core_set,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                common_runtime_args=[out.buffer_address()],
                config=ttnn.WriterConfigDescriptor(),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([scores, valid_rows, out], program)
