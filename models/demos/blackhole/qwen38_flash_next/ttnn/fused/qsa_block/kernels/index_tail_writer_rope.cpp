// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// index_tail, rope core writer (one core pair per lane): the rotated query tiles 0, 1 into the index query (the first
// pair); per lane of this pair row 0 of the rotated key tiles into the lane's compressed cache row (P // 4), columns
// 0..63.
// Compile-time args: TensorAccessorArgs index_query, cache, kv_block_start, kv_row_hit.
// Runtime args: 0 index_query, 1 cache, 2 kv_block_start, 3 kv_row_hit addresses, 4 rows, 5 lane tile rows,
// 6 lane_first, 7 lane_count, 8 do_query.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "index_tail_cbs.h"
#include "positions.h"
#include "../../kernels/zones.h"

using namespace index_tail;

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t cache_addr = get_arg_val<uint32_t>(1);
    const uint32_t pos_addr = get_arg_val<uint32_t>(2);
    const uint32_t hit_addr = get_arg_val<uint32_t>(3);
    const uint32_t rows = get_arg_val<uint32_t>(4);
    const uint32_t lane_tile_rows = get_arg_val<uint32_t>(5);
    const uint32_t lane_first = get_arg_val<uint32_t>(6);
    const uint32_t lane_count = get_arg_val<uint32_t>(7);
    const uint32_t do_query = get_arg_val<uint32_t>(8);
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto cache_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto hit_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);
    const auto cache = TensorAccessor(cache_args, cache_addr);
    const auto pos = TensorAccessor(pos_args, pos_addr);
    const auto hit = TensorAccessor(hit_args, hit_addr);

    cb_reserve_back(CB_POS_W, 1);
    cb_reserve_back(CB_POSCW, 1);
    const uint32_t pos_l1 = get_write_ptr(CB_POS_W);
    tile_rows::read_positions(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1), pos, hit, rows, get_write_ptr(CB_POSCW));
    volatile tt_l1_ptr uint32_t* positions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1);

    if (do_query) {
        FUSED_ZONE("fz_qs_it_wr_query");
        cb_wait_front(CB_OUT_Q, ROPE_TILES);
        for (uint32_t t = 0; t < ROPE_TILES; ++t) {
            noc_async_write_page(t, out, get_read_ptr(CB_OUT_Q) + t * TILE_BYTES);
        }
        noc_async_write_barrier();
        cb_pop_front(CB_OUT_Q, ROPE_TILES);
    }

    for (uint32_t i = 0; i < lane_count; ++i) {
        FUSED_ZONE("fz_qs_it_wr_lane");
        const uint32_t lane = lane_first + i;
        cb_wait_front(CB_OUT_K, ROPE_TILES);
        const uint32_t l1 = get_read_ptr(CB_OUT_K);
        const uint32_t row = positions[lane] >> 2;
        const uint32_t tile_row = lane * lane_tile_rows + (row >> 5);
        for (uint32_t t = 0; t < ROPE_TILES; ++t) {
            for (uint32_t half = 0; half < 2; ++half) {
                noc_async_write(
                    l1 + t * TILE_BYTES + chunk_offset(0, half),
                    cache.get_noc_addr(tile_row * HEAD_TILES + t, chunk_offset(row & 31, half)),
                    ROW_BYTES);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(CB_OUT_K, ROPE_TILES);
    }
}
