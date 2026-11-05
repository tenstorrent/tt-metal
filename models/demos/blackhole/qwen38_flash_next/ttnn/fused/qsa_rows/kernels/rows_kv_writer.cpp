// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail_rows, the KV core's writer (one core): the current block's 32 row-major rows (cb_block) to the cache at
// P & ~31 (page = row), then, unless single_row, the next block's 32 rows (cb_next) at (P & ~31) + 32.  Every row of
// both blocks is rewritten, as the op chain rewrites them.
// Compile-time args: TensorAccessorArgs cache.  Runtime args: 0 cache address, 1 single_row.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "../../qsa_block/kernels/main_tail_cbs.h"
#include "rows_kv_cbs.h"
using namespace main_tail;
using namespace rows_kv;

void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t single_row = get_arg_val<uint32_t>(1);
    constexpr auto cache_args = TensorAccessorArgs<0>();
    const auto cache = TensorAccessor(cache_args, cache_addr);
    const Noc noc;
    CircularBuffer(cb_kv_pos).wait_front(1);
    const uint32_t block = CoreLocalMem<volatile uint32_t>(CircularBuffer(cb_kv_pos).get_read_ptr())[1];  // word 1
    CircularBuffer(cb_block).wait_front(block_pages);
    {
        for (uint32_t k = 0; k < tile_rows::rows_per_tile; ++k) {
            noc.async_write(
                CircularBuffer(cb_block),
                cache,
                query_row_bytes,
                {.offset_bytes = k * query_row_bytes},
                {.page_id = block + k, .offset_bytes = 0});
        }
        noc.async_write_barrier();
    }
    CircularBuffer(cb_block).pop_front(block_pages);
    CircularBuffer(cb_next).wait_front(block_pages);
    if (!single_row) {
        const uint32_t next = block + tile_rows::rows_per_tile;
        for (uint32_t k = 0; k < tile_rows::rows_per_tile; ++k) {
            noc.async_write(
                CircularBuffer(cb_next),
                cache,
                query_row_bytes,
                {.offset_bytes = k * query_row_bytes},
                {.page_id = next + k, .offset_bytes = 0});
        }
        noc.async_write_barrier();
    }
    CircularBuffer(cb_next).pop_front(block_pages);
    CircularBuffer(cb_kv_pos).pop_front(1);
}
