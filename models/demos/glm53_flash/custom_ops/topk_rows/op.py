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

    @staticmethod
    def segments_for(W: int, max_segments: int = 8, k: int = 512, chunk: int = 256) -> int:
        """The largest S <= max_segments with S | W/chunk and W/S >= k (1 when none)."""
        best = 1
        for S in range(2, max_segments + 1):
            if (W // chunk) % S == 0 and W // S >= k and (W // S) % chunk == 0:
                best = S
        return best

    def _program(
        self, scores, valid_rows, out, *, values=None, ids1=None, seg_rows_per_user=0, translate=False, seg_len=0
    ):
        """The top-k program on ``scores`` [rows,1,1,W] bf16 RM: per-row bounds from ``valid_rows`` [rows,1] uint32 when
        given (else the full width); ``out`` the ranked ids. Segmented (two-stage) extras: ``seg_rows_per_user`` rows
        per user write user-major pages (row r -> page r // seg, column (r % seg) * K); ``values`` [B,1,1,seg*K] fp32
        receives the ranked values; ``translate`` with ``ids1`` [B,1,1,seg*K] uint32 and ``seg_len`` maps candidate
        slot v -> seg_len * (v // K) + ids1[b][v] (stage 2)."""
        B, W = scores.shape[0], scores.shape[-1]
        k, llk_k = self.k, self.llk_k
        cb_in, cb_indices, cb_indices_scratch, cb_meta, cb_values, cb_ids1 = 0, 1, 2, 3, 4, 5
        elem = 2
        input_chunk_bytes = llk_k * elem
        input_tile_bytes = TILE_HW * elem
        tiles_per_sequence = -(-llk_k // TILE_HW)
        indices_cb_row_bytes = llk_k * 4
        indices_row_bytes = k * 4
        source_slices = llk_k // FACE_WIDTH
        output_slices = k // FACE_WIDTH
        slice_bytes = FACE_WIDTH * 4
        physical_chunks = -(-W // llk_k)
        body_mode = 1 if physical_chunks <= 32 else 0
        input_row_bytes = scores.buffer_page_size()
        assert input_row_bytes == W * elem, (input_row_bytes, W)
        has_meta = valid_rows is not None
        seg = seg_rows_per_user > 0 or translate
        fmt = ttnn.CBFormatDescriptor
        f_idx = fmt(buffer_index=cb_indices, data_format=ttnn.float32, page_size=indices_cb_row_bytes)
        f_idx.face_geometry = ttnn.FaceGeometry(FACE_WIDTH, 2)
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
                total_size=indices_row_bytes,
                core_ranges=self.core_set,
                format_descriptors=[
                    fmt(buffer_index=cb_indices_scratch, data_format=ttnn.float32, page_size=indices_row_bytes)
                ],
            ),
            ttnn.CBDescriptor(
                total_size=64,
                core_ranges=self.core_set,
                format_descriptors=[fmt(buffer_index=cb_meta, data_format=ttnn.uint32, page_size=64)],
            ),
        ]
        if values is not None:
            f_val = fmt(buffer_index=cb_values, data_format=ttnn.float32, page_size=indices_cb_row_bytes)
            f_val.face_geometry = ttnn.FaceGeometry(FACE_WIDTH, 2)
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=2 * indices_cb_row_bytes, core_ranges=self.core_set, format_descriptors=[f_val]
                )
            )
        if translate:
            table_bytes = ids1.buffer_aligned_page_size()
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=table_bytes,
                    core_ranges=self.core_set,
                    format_descriptors=[fmt(buffer_index=cb_ids1, data_format=ttnn.uint32, page_size=table_bytes)],
                )
            )
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()
        reader_ct = [cb_in, input_chunk_bytes, input_tile_bytes, tiles_per_sequence] + acc(scores)
        reader_ct += [1 if has_meta else 0, cb_meta if has_meta else 0, 0] + acc(valid_rows if has_meta else scores)
        reader_ct += acc(scores)  # valid_end accessor placeholder (unused: address 0)
        reader_ct += [1 if has_meta else 0]  # per-row metadata
        compute_ct = [
            cb_in,
            cb_indices,
            llk_k,
            body_mode,
            1 if has_meta else 0,
            cb_meta if has_meta else 0,
            1 if has_meta else 0,
        ]
        if values is not None:
            compute_ct += [cb_values]
        writer_ct = [
            cb_indices,
            cb_indices_scratch,
            indices_row_bytes,
            source_slices,
            output_slices,
            slice_bytes,
        ] + acc(out)
        user_page_bytes = 0
        if seg:
            user_page_bytes = (
                (values if values is not None else ids1).buffer_aligned_page_size()
                if (values is not None or ids1 is not None)
                else 0
            )
            writer_ct += [
                seg_rows_per_user,
                cb_values if values is not None else 0,
                1 if translate else 0,
                seg_len,
                cb_ids1,
            ]
            writer_ct += acc(values if values is not None else out) + acc(ids1 if ids1 is not None else out)
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
                common_runtime_args=[
                    scores.buffer_address(),
                    valid_rows.buffer_address() if has_meta else 0,
                    W,
                    input_row_bytes,
                    0,
                ],
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(
                    KERNELS / ("compute_topk_rows_seg.cpp" if values is not None else "compute_topk_rows.cpp")
                ),
                core_ranges=self.core_set,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                common_runtime_args=[W],
                config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True, dst_full_sync_en=True),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / ("writer_topk_rows_seg.cpp" if seg else "writer_topk_rows.cpp")),
                core_ranges=self.core_set,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                common_runtime_args=[out.buffer_address()]
                + (
                    [
                        values.buffer_address() if values is not None else 0,
                        ids1.buffer_address() if ids1 is not None else 0,
                        user_page_bytes,
                    ]
                    if seg
                    else []
                ),
                config=ttnn.WriterConfigDescriptor(),
            ),
        ]
        io = (
            [scores]
            + ([valid_rows] if has_meta else [])
            + ([values] if values is not None else [])
            + ([ids1] if translate else [])
            + [out]
        )
        return ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

    def segmented(
        self, scores_seg: ttnn.Tensor, S: int, ids1: ttnn.Tensor, vals1: ttnn.Tensor, out: ttnn.Tensor
    ) -> ttnn.Tensor:
        """Two-stage top-512 of B users whose score rows are split into S segments: ``scores_seg`` [B*S,1,1,W/S] bf16 RM
        (columns beyond a user's bound already -inf: the scorer's masked writer); stage 1 ranks every segment row on its
        own core (ids1 [B,1,1,S*512] uint32 and vals1 [B,1,1,S*512] fp32, user-major, persistent); stage 2 ranks the
        S*512 candidate values of each user and translates the winning slots back to pool ids into ``out``
        [B,1,1,512]. -inf candidates rank last (sentinels only when fewer than 512 finite pools)."""
        BS, Wseg = scores_seg.shape[0], scores_seg.shape[-1]
        B = BS // S
        assert BS == B * S and Wseg >= self.k, (scores_seg.shape, S)
        assert list(ids1.shape) == [B, 1, 1, S * self.k] and list(vals1.shape) == [B, 1, 1, S * self.k], (
            ids1.shape,
            vals1.shape,
        )
        self._program(scores_seg, None, ids1, values=vals1, seg_rows_per_user=S)
        cand = ttnn.typecast(vals1, ttnn.bfloat16)  # [B,1,1,S*K] bf16 RM candidate scores
        self._program(cand, None, out, ids1=ids1, translate=True, seg_len=Wseg)
        ttnn.deallocate(cand)
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
