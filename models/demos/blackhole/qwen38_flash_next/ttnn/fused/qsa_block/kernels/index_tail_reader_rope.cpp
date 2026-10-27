// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// index_tail, rope core reader: the -1 scalar tile, the query cos/sin tiles and the query's RoPE tiles (first pair
// only), then per lane the pooled key's RoPE tiles from the norm core (after signalling room) with block-start cos/sin
// tiles whose row 0 is the lane's row.
// Compile-time args: TensorAccessorArgs cos, sin, block_cos, block_sin.
// Runtime args: 0 cos, 1 sin, 2 block_cos, 3 block_sin addresses, 4 rows, 5 peer x, 6 peer y, 7 lane_first,
// 8 lane_count, 9 do_query.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "index_tail_cbs.h"

using namespace index_tail;

void kernel_main() {
    const uint32_t cos_addr = get_arg_val<uint32_t>(0);
    const uint32_t sin_addr = get_arg_val<uint32_t>(1);
    const uint32_t block_cos_addr = get_arg_val<uint32_t>(2);
    const uint32_t block_sin_addr = get_arg_val<uint32_t>(3);
    const uint32_t rows = get_arg_val<uint32_t>(4);
    const uint32_t peer_x = get_arg_val<uint32_t>(5);
    const uint32_t peer_y = get_arg_val<uint32_t>(6);
    const uint32_t lane_first = get_arg_val<uint32_t>(7);
    const uint32_t lane_count = get_arg_val<uint32_t>(8);
    const uint32_t do_query = get_arg_val<uint32_t>(9);
    constexpr auto cos_args = TensorAccessorArgs<0>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto block_cos_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    constexpr auto block_sin_args = TensorAccessorArgs<block_cos_args.next_compile_time_args_offset()>();
    const auto cos = TensorAccessor(cos_args, cos_addr);
    const auto sin = TensorAccessor(sin_args, sin_addr);
    const auto block_cos = TensorAccessor(block_cos_args, block_cos_addr);
    const auto block_sin = TensorAccessor(block_sin_args, block_sin_addr);
    volatile tt_l1_ptr uint32_t* sem_q =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(query_semaphore_id));
    volatile tt_l1_ptr uint32_t* sem_k =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(key_semaphore_id));
    const uint64_t peer_ready = get_noc_addr(peer_x, peer_y, get_semaphore(ready_semaphore_id));

    generate_bcast_col_scalar(CircularBuffer(cb_scalar), 0xBF800000u);
    if (do_query) {
        CircularBuffer(cb_cos).reserve_back(rope_tiles);
        CircularBuffer(cb_sin).reserve_back(rope_tiles);
        for (uint32_t t = 0; t < rope_tiles; ++t) {
            noc_async_read_page(t, cos, get_write_ptr(cb_cos) + t * tile_bytes);
            noc_async_read_page(t, sin, get_write_ptr(cb_sin) + t * tile_bytes);
        }
        noc_async_read_barrier();
        CircularBuffer(cb_cos).push_back(rope_tiles);
        CircularBuffer(cb_sin).push_back(rope_tiles);

        CircularBuffer(cb_in_q).reserve_back(rope_tiles);
        CircularBuffer(cb_rot_q).reserve_back(rope_tiles);
        noc_semaphore_wait_min(sem_q, 1);
        CircularBuffer(cb_in_q).push_back(rope_tiles);
        CircularBuffer(cb_rot_q).push_back(rope_tiles);
    }

    for (uint32_t i = 0; i < lane_count; ++i) {
        const uint32_t lane = lane_first + i;
        CircularBuffer(cb_in_k).reserve_back(rope_tiles);
        CircularBuffer(cb_rot_k).reserve_back(rope_tiles);
        noc_semaphore_inc(peer_ready, 1);
        CircularBuffer(cb_bcos).reserve_back(rope_tiles);
        CircularBuffer(cb_bsin).reserve_back(rope_tiles);
        const uint32_t block_cos_l1 = get_write_ptr(cb_bcos), block_sin_l1 = get_write_ptr(cb_bsin);
        for (uint32_t t = 0; t < rope_tiles; ++t) {
            noc_async_read_page(t, block_cos, block_cos_l1 + t * tile_bytes);
            noc_async_read_page(t, block_sin, block_sin_l1 + t * tile_bytes);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        if (lane != 0) {  // the pooled row of the lane is row 0 of its tiles: give it the lane's block-start row
            for (uint32_t t = 0; t < rope_tiles; ++t) {
                copy_tile_row(block_cos_l1 + t * tile_bytes, lane, block_cos_l1 + t * tile_bytes, 0, false);
                copy_tile_row(block_sin_l1 + t * tile_bytes, lane, block_sin_l1 + t * tile_bytes, 0, false);
            }
        }
        CircularBuffer(cb_bcos).push_back(rope_tiles);
        CircularBuffer(cb_bsin).push_back(rope_tiles);
        noc_semaphore_wait_min(sem_k, i + 1);
        CircularBuffer(cb_in_k).push_back(rope_tiles);
        CircularBuffer(cb_rot_k).push_back(rope_tiles);
    }
    noc_semaphore_set(sem_q, 0);
    noc_semaphore_set(sem_k, 0);
    noc_async_atomic_barrier();
}
