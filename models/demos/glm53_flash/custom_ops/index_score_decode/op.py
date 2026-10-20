# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``IndexScoreDecode``: the DSA indexer's pool scores for ONE decode query per user, batched over the users, as a
model-local ``ttnn.generic_op`` (three kernels). score[b, p] = sum_h relu(q[b, h] . k[b, p]) * w[b, h] over the
Hi = 32 index heads for every pooled key p of user b, written as one bf16 row per user. The query block of a user
is the 32 heads as the 32 ROWS of one tile row (so a user costs one Q tile row against its keys instead of a
32-row query block with one real row) and the keys stream once from DRAM; no causal mask (the top-k's per-row
bound applies it). Contract in README.md.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn

TILE = 32
KERNELS = Path(__file__).resolve().parent / "kernels"


class IndexScoreDecode:
    def __init__(self, device, *, k_chunk_tiles: int = 8):
        self.device = device
        self.KC = k_chunk_tiles  # k-tiles per chunk (<= 8: bf16 DEST half-sync holds 8 tiles)
        assert 1 <= self.KC <= 8
        grid = device.compute_with_storage_grid_size()
        self.grid_x, self.grid_y = grid.x, grid.y
        self.n_cores = grid.x * grid.y

    @staticmethod
    def golden(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """q [B, Hi, D], k [B, W, D], w [B, Hi] (bf16) -> scores [B, W] fp32 = sum_h relu(q_h . k_p) w_h."""
        s = torch.relu(torch.einsum("bhd,bpd->bhp", q.float(), k.float()))
        return torch.einsum("bhp,bh->bp", s, w.float())

    def __call__(
        self,
        q: ttnn.Tensor,
        w: ttnn.Tensor,
        pools: ttnn.Tensor,
        out: ttnn.Tensor,
        valid_rows: ttnn.Tensor | None = None,
        segments: int = 1,
    ) -> ttnn.Tensor:
        """q [1, B, 32, D] bf16 TILE (user b's 32 index heads as rows); w [1, 1, B, 32] bf16 TILE (B <= 32 per tile
        row, the head weights with the n_heads^-0.5 folded); pools [slots, 1, W, D] bf16 / bfp8_b TILE (the pooled-key
        cache view, slots >= B, user b = slot b, W % (32*KC) == 0); out bf16 ROW_MAJOR, pre-allocated: [B, 1, 1, W], or
        [B*segments, 1, 1, W/segments] with ``segments`` > 1 (the two-stage top-k's per-segment rows; W/segments %
        (32*KC) == 0). ``valid_rows`` [B, 1] uint32 (the per-user bound in pools): columns at and beyond it are written
        as -inf, so the top-k needs no per-row bound. Returns ``out``."""
        B, Hi, D = q.shape[1], q.shape[2], q.shape[3]
        W = pools.shape[2]
        assert list(q.shape) == [1, B, TILE, D] and Hi == TILE, q.shape
        assert list(w.shape) == [1, 1, B, TILE], w.shape
        assert list(pools.shape)[1:] == [1, W, D] and pools.shape[0] >= B, (pools.shape, B)
        assert segments >= 1 and W % segments == 0, (W, segments)
        seg_cols = W // segments
        assert list(out.shape) == [B * segments, 1, 1, seg_cols], (out.shape, B, segments, seg_cols)
        v2 = valid_rows is not None or segments > 1
        if valid_rows is not None:
            assert list(valid_rows.shape) == [B, 1] and valid_rows.dtype == ttnn.uint32, valid_rows.shape
        for t, lay, dt in (
            (q, ttnn.TILE_LAYOUT, ttnn.bfloat16),
            (w, ttnn.TILE_LAYOUT, ttnn.bfloat16),
            (out, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
        ):
            assert t.layout == lay and t.dtype == dt, (t.layout, t.dtype)
        assert pools.layout == ttnn.TILE_LAYOUT and pools.dtype in (ttnn.bfloat16, ttnn.bfloat8_b), (
            pools.layout,
            pools.dtype,
        )
        assert D % TILE == 0 and seg_cols % (TILE * self.KC) == 0, (D, W, segments, self.KC)
        Dt, KC = D // TILE, self.KC
        Tt = W // TILE
        total_chunks = Tt // KC
        # work items (user, k-band): about one per core, bands of whole chunks
        n_bands = max(1, min(-(-self.n_cores // B), total_chunks))
        band_chunks = -(-total_chunks // n_bands)
        n_bands = -(-total_chunks // band_chunks)
        items = B * n_bands
        n_cores = min(items, self.n_cores)
        cores = [ttnn.CoreCoord(c // self.grid_y, c % self.grid_y) for c in range(n_cores)]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        out_row_bytes = out.buffer_page_size()
        assert out_row_bytes == seg_cols * 2, (out_row_bytes, seg_cols)
        tile_bytes = 2048
        k_tile_bytes = pools.buffer_page_size()  # 2048 bf16 / 1088 bfp8_b
        assert pools.buffer_aligned_page_size() == k_tile_bytes, (k_tile_bytes, pools.buffer_aligned_page_size())

        def cb(index, page, pages, fmt=ttnn.bfloat16):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        cbs = [
            cb(0, tile_bytes, Dt),  # q block
            cb(1, k_tile_bytes, 2 * KC * Dt, pools.dtype),  # k chunks, double-buffered (the cache dtype)
            cb(2, tile_bytes, 1),  # w-row tile
            cb(3, tile_bytes, KC),  # S tiles
            cb(4, tile_bytes, 2 * KC),  # out tiles (row 0 valid), double-buffered
            cb(5, 64 * KC, 2),  # writer lines (scratch)
            cb(6, 1024, 1),  # the w row's two faces (reader scratch)
            cb(7, 64, 1),  # the user's bound page (writer v2 scratch)
        ]
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()
        reader_ct = [Dt, KC, Tt, k_tile_bytes] + acc(q) + acc(pools) + acc(w)
        compute_ct = [Dt, KC]
        writer_ct = [KC] + acc(out)
        if v2:
            vt = valid_rows if valid_rows is not None else out  # placeholder accessor when unmasked
            writer_ct += [seg_cols, segments] + acc(vt)
        reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c, core in enumerate(cores):
            n_items = (items - c + n_cores - 1) // n_cores
            reader_rt[core.x][core.y] = [
                c,
                n_items,
                n_cores,
                q.buffer_address(),
                pools.buffer_address(),
                w.buffer_address(),
                n_bands,
                band_chunks,
                total_chunks,
            ]
            compute_rt[core.x][core.y] = [c, n_items, n_cores, n_bands, band_chunks, total_chunks]
            writer_rt[core.x][core.y] = [
                c,
                n_items,
                n_cores,
                out.buffer_address(),
                out_row_bytes,
                n_bands,
                band_chunks,
                total_chunks,
            ] + (
                [
                    valid_rows.buffer_address() if valid_rows is not None else 0,
                    valid_rows.buffer_aligned_page_size() if valid_rows is not None else 64,
                ]
                if v2
                else []
            )
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "reader_index_score_decode.cpp"),
                core_ranges=core_set,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "compute_index_score_decode.cpp"),
                core_ranges=core_set,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(
                    KERNELS / ("writer_index_score_decode_v2.cpp" if v2 else "writer_index_score_decode.cpp")
                ),
                core_ranges=core_set,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([q, w, pools, out] + ([valid_rows] if valid_rows is not None else []), program)
