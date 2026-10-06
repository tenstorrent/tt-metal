// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail_rows, the KV core's writer (one core): the current block's 32 row-major rows (CB_BLOCK, the read-back block
// with this pass's rows placed) to the cache at P & ~31 (page = row), then, unless single_row, the next block's 32 rows
// (CB_NEXT: zeros with the crossing rows) at (P & ~31) + 32 -- the chain's update_padded_kv_cache of `staged` and of
// `placed_next`.  Every row of both blocks is rewritten, as the chain rewrites them.
// Compile-time args: TensorAccessorArgs cache.  Runtime args: 0 cache address, 1 single_row.
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
    cb_wait_front(CB_KV_POS, 1);
    const uint32_t block = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_KV_POS) + 4);
    cb_wait_front(CB_BLOCK, BLOCK_PAGES);
    {
        FUSED_ZONE("fz_qr_kv_write_block");
        const uint32_t l1 = get_read_ptr(CB_BLOCK);
        for (uint32_t k = 0; k < tile_rows::TILE_ROWS; ++k) {
            noc_async_write(l1 + k * QUERY_ROW_BYTES, cache.get_noc_addr(block + k, 0), QUERY_ROW_BYTES);
        }
        noc_async_write_barrier();
    }
    cb_pop_front(CB_BLOCK, BLOCK_PAGES);
    cb_wait_front(CB_NEXT, BLOCK_PAGES);
    if (!single_row) {
        FUSED_ZONE("fz_qr_kv_write_next");
        const uint32_t l1 = get_read_ptr(CB_NEXT);
        const uint32_t next = block + tile_rows::TILE_ROWS;
        for (uint32_t k = 0; k < tile_rows::TILE_ROWS; ++k) {
            noc_async_write(l1 + k * QUERY_ROW_BYTES, cache.get_noc_addr(next + k, 0), QUERY_ROW_BYTES);
        }
        noc_async_write_barrier();
    }
    cb_pop_front(CB_NEXT, BLOCK_PAGES);
    cb_pop_front(CB_KV_POS, 1);
}
