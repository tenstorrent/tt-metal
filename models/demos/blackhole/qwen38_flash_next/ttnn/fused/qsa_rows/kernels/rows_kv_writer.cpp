// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail_rows, the KV core's writer (one core): the current block's 32 row-major rows (CB_BLOCK, the read-back block
// with this pass's rows placed) to the cache at P & ~31 (page = row), then, unless single_row, the next block's 32 rows
// (CB_NEXT: zeros with the crossing rows) at (P & ~31) + 32 -- the chain's update_padded_kv_cache of `staged` and of
// `placed_next`.  Every row of both blocks is rewritten, as the chain rewrites them.
// Compile-time args: TensorAccessorArgs cache.  Runtime args: 0 cache address, 1 single_row.
// PAGED (compile-time define; the unpaged program is unchanged): + TensorAccessorArgs page table, + runtime args
// 2 page table address, 3 shift, 4 bs.  The cache is the page pool; the current and the next block each start at the
// pool row pt[b >> shift] * bs + (b & (bs - 1)) (b = P & ~31 and P & ~31 + 32; the next block may sit in the next
// page or in a padding entry) -- two window reads, one barrier (rows_kv_cbs.h).
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../qsa_block/kernels/main_tail_cbs.h"
#include "../../kernels/zones.h"
#include "rows_kv_cbs.h"
using namespace main_tail;
using namespace rows_kv;

void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t single_row = get_arg_val<uint32_t>(1);
    constexpr auto cache_args = TensorAccessorArgs<0>();
    const auto cache = TensorAccessor(cache_args, cache_addr);
#ifdef PAGED
    constexpr auto pt_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    const auto pt = TensorAccessor(pt_args, get_arg_val<uint32_t>(2));
    const uint32_t shift = get_arg_val<uint32_t>(3);
    const uint32_t bs = get_arg_val<uint32_t>(4);
#endif
    cb_wait_front(CB_KV_POS, 1);
    const uint32_t block = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_KV_POS) + 4);
#ifdef PAGED
    const uint32_t pt_cur_l1 = pt_window_l1(get_write_ptr(CB_KV_PT) + PT_WRITER_OFFSET, 0);
    const uint32_t pt_next_l1 = pt_window_l1(get_write_ptr(CB_KV_PT) + PT_WRITER_OFFSET, 1);
    const uint32_t next_block = block + tile_rows::TILE_ROWS;
    pt_window_read(pt, block, shift, pt_cur_l1);
    pt_window_read(pt, next_block, shift, pt_next_l1);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t cur_base = pt_base_row(pt_cur_l1, block, shift, bs);
    const uint32_t next_base = pt_base_row(pt_next_l1, next_block, shift, bs);
#else
    const uint32_t cur_base = block, next_base = block + tile_rows::TILE_ROWS;
#endif
#ifdef QWEN38_ROWS_KV_FAST
    // QWEN38_QSA_ROWS_KV_FAST: only the rows the reader placed (see rows_kv_reader.cpp): n0 rows from slot s0 of the
    // current block, n1 rows from slot 0 of the next block, one barrier.
    const volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_KV_POS));
    const uint32_t s0 = meta[2], n0 = meta[3], n1 = meta[4];
    cb_wait_front(CB_BLOCK, BLOCK_PAGES);
    cb_wait_front(CB_NEXT, BLOCK_PAGES);
    {
        FUSED_ZONE("fz_qr_kv_write_block");
        const uint32_t l1 = get_read_ptr(CB_BLOCK);
        for (uint32_t k = s0; k < s0 + n0; ++k) {
            noc_async_write(l1 + k * QUERY_ROW_BYTES, cache.get_noc_addr(cur_base + k, 0), QUERY_ROW_BYTES);
        }
        const uint32_t next_l1 = get_read_ptr(CB_NEXT), next = next_base;
        for (uint32_t k = 0; k < n1; ++k) {
            noc_async_write(next_l1 + k * QUERY_ROW_BYTES, cache.get_noc_addr(next + k, 0), QUERY_ROW_BYTES);
        }
        noc_async_write_barrier();
    }
    cb_pop_front(CB_BLOCK, BLOCK_PAGES);
    cb_pop_front(CB_NEXT, BLOCK_PAGES);
    cb_pop_front(CB_KV_POS, 1);
}
#else
    cb_wait_front(CB_BLOCK, BLOCK_PAGES);
    {
        FUSED_ZONE("fz_qr_kv_write_block");
        const uint32_t l1 = get_read_ptr(CB_BLOCK);
        for (uint32_t k = 0; k < tile_rows::TILE_ROWS; ++k) {
            noc_async_write(l1 + k * QUERY_ROW_BYTES, cache.get_noc_addr(cur_base + k, 0), QUERY_ROW_BYTES);
        }
        noc_async_write_barrier();
    }
    cb_pop_front(CB_BLOCK, BLOCK_PAGES);
    cb_wait_front(CB_NEXT, BLOCK_PAGES);
    if (!single_row) {
        FUSED_ZONE("fz_qr_kv_write_next");
        const uint32_t l1 = get_read_ptr(CB_NEXT);
        const uint32_t next = next_base;
        for (uint32_t k = 0; k < tile_rows::TILE_ROWS; ++k) {
            noc_async_write(l1 + k * QUERY_ROW_BYTES, cache.get_noc_addr(next + k, 0), QUERY_ROW_BYTES);
        }
        noc_async_write_barrier();
    }
    cb_pop_front(CB_NEXT, BLOCK_PAGES);
    cb_pop_front(CB_KV_POS, 1);
}
#endif
