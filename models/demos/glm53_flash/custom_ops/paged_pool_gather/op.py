# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``PagedPoolGather``: gather pools of ``pool_size`` consecutive token rows out of a paged bf16 TILE cache into a
compact paged TILE cache (one contiguous page range per user, identity page table), as a model-local
``ttnn.generic_op`` (one data-movement kernel). The sparse DSA decode gathers the 512 top-scored 4-token pools + the
incomplete tail pool of every user (2052 rows, 1 KB of DRAM traffic per selected token on each side), which the
dense paged flash-MLA decode then attends with a per-user ``cur_pos``; the slot -> pool mapping (ranked pools, the
pools in order for a dense user, the tail pool at slot 512) is done in the kernel. Contract in README.md.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn

TILE = 32
SENTINEL = 0xFFFFFFFF
KERNEL = Path(__file__).resolve().parent / "kernels" / "reader_paged_pool_gather.cpp"


class PagedPoolGather:
    def __init__(self, device, *, pool_size: int = 4):
        assert pool_size in (1, 2, 4, 8, 16), pool_size  # a pool never crosses a 16-row tile face
        self.device = device
        self.P = pool_size
        grid = device.compute_with_storage_grid_size()
        self.grid_x, self.grid_y = grid.x, grid.y

    # ------------------------------------------------------------------ golden
    @staticmethod
    def golden(
        cache: torch.Tensor,
        page_table: torch.Tensor,
        top_ids: torch.Tensor,
        tail_pool: torch.Tensor,
        pool_size: int = 4,
    ) -> torch.Tensor:
        """cache [blocks,1,32,W], page_table [B_table,bps] int, top_ids [B,1,1,K] int64 (ranked pools), tail_pool [B]
        int (n = the user's incomplete pool) -> [B*OB,1,32,W], OB = ceil((K+1)*P/32): slot s < K = ranked pool s, or
        pool s for a dense user (n < K); slot K = pool n; output row P*s + j = cache row P*pool + j of user b
        (through its page-table row); sentinel (0xFFFFFFFF) / out-of-range pools and the padding rows are zero."""
        P = pool_size
        B, K = top_ids.shape[0], top_ids.shape[-1]
        N = K + 1
        bps = page_table.shape[1]
        OB = -(-(N * P) // TILE)
        out = torch.zeros(B * OB, 1, TILE, cache.shape[-1], dtype=cache.dtype)
        rows = out.reshape(B, OB * TILE, -1)
        ids = top_ids.reshape(B, K)
        for b in range(B):
            n = int(tail_pool[b])
            for s_ in range(N):
                p = n if s_ == K else (s_ if n < K else int(ids[b, s_]))
                if p == SENTINEL or p * P >= bps * TILE:
                    continue
                r0 = p * P
                blk = int(page_table[b, r0 // TILE])
                rows[b, s_ * P : (s_ + 1) * P] = cache[blk, 0, r0 % TILE : r0 % TILE + P]
        return out

    @staticmethod
    def output_blocks(num_ranked: int, pool_size: int = 4) -> int:
        """Compact blocks per user for ``num_ranked`` ranked pools + the tail pool."""
        return -(-((num_ranked + 1) * pool_size) // TILE)

    def output_shape(self, top_ids: ttnn.Tensor, width: int) -> list[int]:
        B, K = top_ids.shape[0], top_ids.shape[-1]
        return [B * self.output_blocks(K, self.P), 1, TILE, width]

    # ------------------------------------------------------------------ program
    def __call__(
        self,
        cache: ttnn.Tensor,
        page_table: ttnn.Tensor,
        top_ids: ttnn.Tensor,
        tail_pool: ttnn.Tensor,
        out: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """cache [max_blocks,1,32,W] bf16 TILE; page_table [B_table, bps] int32 ROW_MAJOR; top_ids [B,1,1,K] uint32
        ROW_MAJOR (B <= B_table; the ranked pools, 0xFFFFFFFF = none); tail_pool [B,1,1,1] uint32 ROW_MAJOR (n = the
        user's incomplete pool; n < K marks a dense user: pools 0..K-1 in order); out [B*OB,1,32,W] bf16 TILE,
        pre-allocated (OB = ceil((K+1)*P/32)). Returns ``out``."""
        P = self.P
        assert cache.layout == ttnn.TILE_LAYOUT and cache.dtype == ttnn.bfloat16, (cache.layout, cache.dtype)
        assert page_table.layout == ttnn.ROW_MAJOR_LAYOUT and page_table.dtype == ttnn.int32
        assert top_ids.layout == ttnn.ROW_MAJOR_LAYOUT and top_ids.dtype == ttnn.uint32
        assert tail_pool.layout == ttnn.ROW_MAJOR_LAYOUT and tail_pool.dtype == ttnn.uint32
        cs = list(cache.shape)
        B, K = top_ids.shape[0], top_ids.shape[-1]
        assert len(cs) == 4 and cs[1] == 1 and cs[2] == TILE and cs[3] % TILE == 0, cs
        assert list(top_ids.shape) == [B, 1, 1, K] and list(tail_pool.shape) == [B, 1, 1, 1], (
            top_ids.shape,
            tail_pool.shape,
        )
        assert 1 <= B <= page_table.shape[0], (B, page_table.shape)
        OB = self.output_blocks(K, P)
        assert list(out.shape) == [B * OB, 1, TILE, cs[3]], (list(out.shape), [B * OB, 1, TILE, cs[3]])
        assert out.layout == ttnn.TILE_LAYOUT and out.dtype == ttnn.bfloat16
        Wt = cs[3] // TILE
        bps = page_table.shape[1]
        tile_bytes = 2048
        pt_row_bytes = page_table.buffer_aligned_page_size()
        ids_row_bytes = top_ids.buffer_aligned_page_size()
        tail_row_bytes = tail_pool.buffer_aligned_page_size()
        assert cache.buffer_aligned_page_size() == tile_bytes and out.buffer_aligned_page_size() == tile_bytes
        spb = TILE // P  # pool slots per output block
        items = B * OB
        n_cores = min(items, self.grid_x * self.grid_y)
        cores = [ttnn.CoreCoord(c // self.grid_y, c % self.grid_y) for c in range(n_cores)]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

        def cb(index, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=page)
                ],
            )

        # cb 0: two tile blocks + one spare tile (64 B alignment slack); cb 1: the user's ranked-id row + its tail
        # page; cb 2: one 64 B page-table fragment per pool slot (x2 for alignment slack)
        cbs = [
            cb(0, tile_bytes, 2 * Wt + 1),
            cb(1, 64, (-(-ids_row_bytes // 64)) + (-(-tail_row_bytes // 64)) + 2),
            cb(2, 64, 2 * spb + 1),
        ]
        ct = [Wt, P, K, OB, bps, pt_row_bytes, ids_row_bytes, spb, tail_row_bytes]
        for t in (cache, page_table, top_ids, tail_pool, out):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        for c, core in enumerate(cores):
            n_items = (items - c + n_cores - 1) // n_cores
            rt[core.x][core.y] = [
                c,
                n_items,
                n_cores,
                cache.buffer_address(),
                page_table.buffer_address(),
                top_ids.buffer_address(),
                out.buffer_address(),
                tail_pool.buffer_address(),
            ]
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL),
            core_ranges=core_set,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
        return ttnn.generic_op([cache, page_table, top_ids, tail_pool, out], program)
