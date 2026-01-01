// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// index_tail, rope core writer: the rotated query tiles 0, 1 into the index query (first pair only); per lane row 0 of
// the rotated key tiles into the lane's compressed cache row (P // 4), columns 0..63.
// Compile-time args: TensorAccessorArgs index_query, cache, kv_block_start, kv_row_hit.
// Runtime args: 0 index_query, 1 cache, 2 kv_block_start, 3 kv_row_hit addresses, 4 rows, 5 lane tile rows,
// 6 lane_first, 7 lane_count, 8 do_query.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "index_tail_cbs.h"
#include "positions.h"

using namespace index_tail;

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t cache_addr = get_arg_val<uint32_t>(1);
    const uint32_t block_start_addr = get_arg_val<uint32_t>(2);
    const uint32_t row_hit_addr = get_arg_val<uint32_t>(3);
    const uint32_t rows = get_arg_val<uint32_t>(4);
    const uint32_t lane_tile_rows = get_arg_val<uint32_t>(5);
    const uint32_t lane_first = get_arg_val<uint32_t>(6);
    const uint32_t lane_count = get_arg_val<uint32_t>(7);
    const uint32_t do_query = get_arg_val<uint32_t>(8);
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto cache_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto block_start_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto row_hit_args = TensorAccessorArgs<block_start_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);
    const auto cache = TensorAccessor(cache_args, cache_addr);
    const auto block_start = TensorAccessor(block_start_args, block_start_addr);
    const auto row_hit = TensorAccessor(row_hit_args, row_hit_addr);
    const Noc noc;

    CircularBuffer(cb_pos_w).reserve_back(1);
    CircularBuffer(cb_poscw).reserve_back(1);
    // cb_pos_w is scratch: reserved once and never pushed
    const CoreLocalMem<volatile uint32_t> positions(CircularBuffer(cb_pos_w).get_write_ptr());
    tile_rows::read_positions(positions, block_start, row_hit, rows, CircularBuffer(cb_poscw).get_write_ptr());

    if (do_query) {
        CircularBuffer(cb_out_q).wait_front(rope_tiles);
        for (uint32_t t = 0; t < rope_tiles; ++t) {
            noc.async_write(
                CircularBuffer(cb_out_q),
                out,
                out.get_aligned_page_size(),
                {.offset_bytes = t * tile_bytes},
                {.page_id = t});
        }
        noc.async_write_barrier();
        CircularBuffer(cb_out_q).pop_front(rope_tiles);
    }

    for (uint32_t i = 0; i < lane_count; ++i) {
        const uint32_t lane = lane_first + i;
        CircularBuffer(cb_out_k).wait_front(rope_tiles);
        const uint32_t row = positions[lane] >> 2;
        const uint32_t tile_row = lane * lane_tile_rows + (row >> 5);
        for (uint32_t t = 0; t < rope_tiles; ++t) {
            for (uint32_t half = 0; half < 2; ++half) {
                noc.async_write(
                    CircularBuffer(cb_out_k),
                    cache,
                    half_row_bytes,
                    {.offset_bytes = t * tile_bytes + chunk_offset(0, half)},
                    {.page_id = tile_row * head_tiles + t, .offset_bytes = chunk_offset(row & 31, half)});
            }
        }
        noc.async_write_barrier();
        CircularBuffer(cb_out_k).pop_front(rope_tiles);
    }
}
