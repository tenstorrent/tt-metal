// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, staging core reader: the ones tile, the positions, the value tiles into pack tiles 0..7 (the key's
// rotated tiles 8, 9 and normalized tiles 10..15 arrive from the key cores); per lane the KV staging tiles with row
// (P % 32) replaced by the lane's packed row [v | k] (-0 -> +0, as the SFPU one-hot select does).
// Compile-time args: TensorAccessorArgs staging, v, kv_block_start, kv_row_hit.  Runtime args: 0 staging, 1 v, 2
// kv_block_start, 3 kv_row_hit, 4 rows, 5 first tile of the value in v, 6 lane_first, 7 lane_count.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "main_tail_cbs.h"
#include "positions.h"

using namespace main_tail;

void kernel_main() {
    const uint32_t staging_addr = get_arg_val<uint32_t>(0);
    const uint32_t value_addr = get_arg_val<uint32_t>(1);
    const uint32_t block_start_addr = get_arg_val<uint32_t>(2);
    const uint32_t row_hit_addr = get_arg_val<uint32_t>(3);
    const uint32_t rows = get_arg_val<uint32_t>(4);
    const uint32_t value_first_tile = get_arg_val<uint32_t>(5);
    const uint32_t lane_first = get_arg_val<uint32_t>(6);
    const uint32_t lane_count = get_arg_val<uint32_t>(7);
    constexpr auto staging_args = TensorAccessorArgs<0>();
    constexpr auto value_args = TensorAccessorArgs<staging_args.next_compile_time_args_offset()>();
    constexpr auto block_start_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto row_hit_args = TensorAccessorArgs<block_start_args.next_compile_time_args_offset()>();
    const auto staging = TensorAccessor(staging_args, staging_addr);
    const auto value = TensorAccessor(value_args, value_addr);
    const auto block_start = TensorAccessor(block_start_args, block_start_addr);
    const auto row_hit = TensorAccessor(row_hit_args, row_hit_addr);
    const Noc noc;
    Semaphore<> key_rotated_handoff(key_rotated_semaphore_id), key_normalized_handoff(key_normalized_semaphore_id);

    CircularBuffer(cb_ones).reserve_back(1);
    tile_rows::fill_words(CircularBuffer(cb_ones).get_write_ptr(), tile_bytes / 4, 0x3F803F80u);
    CircularBuffer(cb_ones).push_back(1);
    CircularBuffer(cb_pos_r).reserve_back(1);
    CircularBuffer(cb_poscr).reserve_back(1);
    CircularBuffer(cb_pack).reserve_back(pack_tiles);
    // cb_pos_r and cb_pack are scratch: reserved once and never pushed (the key cores write pack tiles 8..15 into
    // cb_pack at the same address on this core)
    const CoreLocalMem<volatile uint32_t> positions(CircularBuffer(cb_pos_r).get_write_ptr());
    const uint32_t pack_l1 = CircularBuffer(cb_pack).get_write_ptr();
    tile_rows::read_positions(positions, block_start, row_hit, rows, CircularBuffer(cb_poscr).get_write_ptr());
    {
        for (uint32_t c = 0; c < head_tiles; ++c) {
            noc.async_read(
                value,
                CoreLocalMem<uint32_t>(pack_l1),
                value.get_aligned_page_size(),
                {.page_id = value_first_tile + c},
                {.offset_bytes = c * tile_bytes});
        }
        noc.async_read_barrier();
        invalidate_l1_cache();
    }
    {
        key_rotated_handoff.wait_min(1);
        key_normalized_handoff.wait_min(1);
        invalidate_l1_cache();
    }

    for (uint32_t i = 0; i < lane_count; ++i) {
        const uint32_t lane = lane_first + i;
        const uint32_t slot = positions[lane] & 31;
        CircularBuffer(cb_stg).reserve_back(pack_tiles);
        const uint32_t staging_l1 = CircularBuffer(cb_stg).get_write_ptr();
        for (uint32_t c = 0; c < pack_tiles; ++c) {
            noc.async_read(
                staging,
                CircularBuffer(cb_stg),
                staging.get_aligned_page_size(),
                {.page_id = lane * pack_tiles + c},
                {.offset_bytes = c * tile_bytes});
        }
        noc.async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t c = 0; c < pack_tiles; ++c) {
            copy_tile_row(pack_l1 + c * tile_bytes, lane, staging_l1 + c * tile_bytes, slot, true);
        }
        CircularBuffer(cb_stg).push_back(pack_tiles);
    }
    key_rotated_handoff.set(0);
    key_normalized_handoff.set(0);
}
