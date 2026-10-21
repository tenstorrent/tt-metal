// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, staging core writer (one staging core per lane): for this core's lanes the canonical staging tiles back
// to DRAM (in place) and the 32 untilized rows into the lane's KV cache block at lane * lane_rows + (P & ~31) (page =
// row).
// Compile-time args: TensorAccessorArgs staging, cache, kv_block_start, kv_row_hit.
// Runtime args: 0 staging, 1 cache, 2 kv_block_start, 3 kv_row_hit addresses, 4 rows, 5 lane rows (cache rows per
// lane), 6 lane_first, 7 lane_count.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "main_tail_cbs.h"
#include "positions.h"

using namespace main_tail;

void kernel_main() {
    const uint32_t stg_addr = get_arg_val<uint32_t>(0);
    const uint32_t cache_addr = get_arg_val<uint32_t>(1);
    const uint32_t pos_addr = get_arg_val<uint32_t>(2);
    const uint32_t hit_addr = get_arg_val<uint32_t>(3);
    const uint32_t rows = get_arg_val<uint32_t>(4);
    const uint32_t lane_rows = get_arg_val<uint32_t>(5);
    const uint32_t lane_first = get_arg_val<uint32_t>(6);
    const uint32_t lane_count = get_arg_val<uint32_t>(7);
    constexpr auto stg_args = TensorAccessorArgs<0>();
    constexpr auto cache_args = TensorAccessorArgs<stg_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto hit_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    const auto stg = TensorAccessor(stg_args, stg_addr);
    const auto cache = TensorAccessor(cache_args, cache_addr);
    const auto pos = TensorAccessor(pos_args, pos_addr);
    const auto hit = TensorAccessor(hit_args, hit_addr);

    cb_reserve_back(CB_POS_W, 1);
    cb_reserve_back(CB_POSCW, 1);
    const uint32_t pos_l1 = get_write_ptr(CB_POS_W);
    tile_rows::read_positions(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1), pos, hit, rows, get_write_ptr(CB_POSCW));
    volatile tt_l1_ptr uint32_t* positions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1);

    for (uint32_t i = 0; i < lane_count; ++i) {
        const uint32_t lane = lane_first + i;
        cb_wait_front(CB_STGW, PACK_TILES);
        const uint32_t l1 = get_read_ptr(CB_STGW);
        for (uint32_t c = 0; c < PACK_TILES; ++c) {
            noc_async_write_page(lane * PACK_TILES + c, stg, l1 + c * TILE_BYTES);
        }
        noc_async_write_barrier();
        cb_pop_front(CB_STGW, PACK_TILES);

        cb_wait_front(CB_RM, PACK_TILES);
        const uint32_t rm = get_read_ptr(CB_RM);
        const uint32_t base = lane * lane_rows + (positions[lane] & ~31u);
        for (uint32_t k = 0; k < tile_rows::TILE_ROWS; ++k) {
            noc_async_write(rm + k * QUERY_ROW_BYTES, cache.get_noc_addr(base + k, 0), QUERY_ROW_BYTES);
        }
        noc_async_write_barrier();
        cb_pop_front(CB_RM, PACK_TILES);
    }
}
