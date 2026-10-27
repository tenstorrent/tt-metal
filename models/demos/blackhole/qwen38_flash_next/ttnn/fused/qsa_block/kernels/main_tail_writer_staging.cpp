// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, staging core writer: per lane the staging tiles back to DRAM in place and the 32 untilized rows into the
// lane's KV cache block at lane * lane_rows + (P & ~31) (page = row).
// Compile-time args: TensorAccessorArgs staging, cache, kv_block_start, kv_row_hit.
// Runtime args: 0 staging, 1 cache, 2 kv_block_start, 3 kv_row_hit addresses, 4 rows, 5 lane rows (cache rows per
// lane), 6 lane_first, 7 lane_count.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "main_tail_cbs.h"
#include "positions.h"

using namespace main_tail;

void kernel_main() {
    const uint32_t staging_addr = get_arg_val<uint32_t>(0);
    const uint32_t cache_addr = get_arg_val<uint32_t>(1);
    const uint32_t block_start_addr = get_arg_val<uint32_t>(2);
    const uint32_t row_hit_addr = get_arg_val<uint32_t>(3);
    const uint32_t rows = get_arg_val<uint32_t>(4);
    const uint32_t lane_rows = get_arg_val<uint32_t>(5);
    const uint32_t lane_first = get_arg_val<uint32_t>(6);
    const uint32_t lane_count = get_arg_val<uint32_t>(7);
    constexpr auto staging_args = TensorAccessorArgs<0>();
    constexpr auto cache_args = TensorAccessorArgs<staging_args.next_compile_time_args_offset()>();
    constexpr auto block_start_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto row_hit_args = TensorAccessorArgs<block_start_args.next_compile_time_args_offset()>();
    const auto staging = TensorAccessor(staging_args, staging_addr);
    const auto cache = TensorAccessor(cache_args, cache_addr);
    const auto block_start = TensorAccessor(block_start_args, block_start_addr);
    const auto row_hit = TensorAccessor(row_hit_args, row_hit_addr);

    CircularBuffer(cb_pos_w).reserve_back(1);
    CircularBuffer(cb_poscw).reserve_back(1);
    const uint32_t positions_l1 = get_write_ptr(cb_pos_w);
    tile_rows::read_positions(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(positions_l1),
        block_start,
        row_hit,
        rows,
        get_write_ptr(cb_poscw));
    volatile tt_l1_ptr uint32_t* positions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(positions_l1);

    for (uint32_t i = 0; i < lane_count; ++i) {
        const uint32_t lane = lane_first + i;
        CircularBuffer(cb_stgw).wait_front(pack_tiles);
        const uint32_t l1 = get_read_ptr(cb_stgw);
        for (uint32_t c = 0; c < pack_tiles; ++c) {
            noc_async_write_page(lane * pack_tiles + c, staging, l1 + c * tile_bytes);
        }
        noc_async_write_barrier();
        CircularBuffer(cb_stgw).pop_front(pack_tiles);

        CircularBuffer(cb_rm).wait_front(pack_tiles);
        const uint32_t cache_rows_l1 = get_read_ptr(cb_rm);
        const uint32_t block_first_row = lane * lane_rows + (positions[lane] & ~31u);
        for (uint32_t k = 0; k < tile_rows::rows_per_tile; ++k) {
            noc_async_write(
                cache_rows_l1 + k * query_row_bytes, cache.get_noc_addr(block_first_row + k, 0), query_row_bytes);
        }
        noc_async_write_barrier();
        CircularBuffer(cb_rm).pop_front(pack_tiles);
    }
}
