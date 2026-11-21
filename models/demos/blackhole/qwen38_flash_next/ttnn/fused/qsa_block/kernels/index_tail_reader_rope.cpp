// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// index_tail, rope core reader: the -1 scalar tile, the query cos/sin tiles and the query's RoPE tiles (first pair
// only), then per lane the pooled key's RoPE tiles from the norm core (after signalling room) with block-start cos/sin
// tiles whose row 0 is the lane's row.
// Compile-time args: TensorAccessorArgs cos, sin, block_cos, block_sin.
// Runtime args: 0 cos, 1 sin, 2 block_cos, 3 block_sin addresses, 4 rows (not read here), 5 peer x, 6 peer y,
// 7 lane_first, 8 lane_count, 9 do_query.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "index_tail_cbs.h"

using namespace index_tail;

void kernel_main() {
    const uint32_t cos_addr = get_arg_val<uint32_t>(0);
    const uint32_t sin_addr = get_arg_val<uint32_t>(1);
    const uint32_t block_cos_addr = get_arg_val<uint32_t>(2);
    const uint32_t block_sin_addr = get_arg_val<uint32_t>(3);
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
    const Noc noc;
    // this core's query / key hand-off counters, and the norm core's (the peer's) room counter
    Semaphore<> query_handoff(query_semaphore_id), key_handoff(key_semaphore_id), rope_ready(ready_semaphore_id);

    generate_bcast_col_scalar(CircularBuffer(cb_scalar), 0xBF800000u);
    if (do_query) {
        CircularBuffer(cb_cos).reserve_back(rope_tiles);
        CircularBuffer(cb_sin).reserve_back(rope_tiles);
        for (uint32_t t = 0; t < rope_tiles; ++t) {
            noc.async_read(
                cos,
                CircularBuffer(cb_cos),
                cos.get_aligned_page_size(),
                {.page_id = t},
                {.offset_bytes = t * tile_bytes});
            noc.async_read(
                sin,
                CircularBuffer(cb_sin),
                sin.get_aligned_page_size(),
                {.page_id = t},
                {.offset_bytes = t * tile_bytes});
        }
        noc.async_read_barrier();
        CircularBuffer(cb_cos).push_back(rope_tiles);
        CircularBuffer(cb_sin).push_back(rope_tiles);

        CircularBuffer(cb_in_q).reserve_back(rope_tiles);
        CircularBuffer(cb_rot_q).reserve_back(rope_tiles);
        query_handoff.wait_min(1);
        CircularBuffer(cb_in_q).push_back(rope_tiles);
        CircularBuffer(cb_rot_q).push_back(rope_tiles);
    }

    for (uint32_t i = 0; i < lane_count; ++i) {
        const uint32_t lane = lane_first + i;
        CircularBuffer(cb_in_k).reserve_back(rope_tiles);
        CircularBuffer(cb_rot_k).reserve_back(rope_tiles);
        rope_ready.up(noc, peer_x, peer_y, 1);
        CircularBuffer(cb_bcos).reserve_back(rope_tiles);
        CircularBuffer(cb_bsin).reserve_back(rope_tiles);
        const uint32_t block_cos_l1 = CircularBuffer(cb_bcos).get_write_ptr();
        const uint32_t block_sin_l1 = CircularBuffer(cb_bsin).get_write_ptr();
        for (uint32_t t = 0; t < rope_tiles; ++t) {
            noc.async_read(
                block_cos,
                CircularBuffer(cb_bcos),
                block_cos.get_aligned_page_size(),
                {.page_id = t},
                {.offset_bytes = t * tile_bytes});
            noc.async_read(
                block_sin,
                CircularBuffer(cb_bsin),
                block_sin.get_aligned_page_size(),
                {.page_id = t},
                {.offset_bytes = t * tile_bytes});
        }
        noc.async_read_barrier();
        invalidate_l1_cache();
        if (lane != 0) {  // the pooled row of the lane is row 0 of its tiles: give it the lane's block-start row
            for (uint32_t t = 0; t < rope_tiles; ++t) {
                copy_tile_row(block_cos_l1 + t * tile_bytes, lane, block_cos_l1 + t * tile_bytes, 0, false);
                copy_tile_row(block_sin_l1 + t * tile_bytes, lane, block_sin_l1 + t * tile_bytes, 0, false);
            }
        }
        CircularBuffer(cb_bcos).push_back(rope_tiles);
        CircularBuffer(cb_bsin).push_back(rope_tiles);
        key_handoff.wait_min(i + 1);
        CircularBuffer(cb_in_k).push_back(rope_tiles);
        CircularBuffer(cb_rot_k).push_back(rope_tiles);
    }
    query_handoff.set(0);
    key_handoff.set(0);
    noc.async_atomic_barrier();
}
