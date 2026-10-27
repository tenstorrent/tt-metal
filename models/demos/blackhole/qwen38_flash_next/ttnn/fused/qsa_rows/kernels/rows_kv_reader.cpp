// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail_rows, the KV core's reader (one core): the packed KV write on the 32-row tile at R real rows. The value
// tiles (8) come from the v projection window, the key's rotated (8, 9) and normalized tiles (10..15) from the key
// cores (into cb_pack, each hand-off counted on its semaphore). The current cache block (32 row-major rows at P & ~31,
// P = first_position) is read back, row j < R in that block replaces row (P + j) & 31 with the packed row [v_j | k_j]
// (-0 -> +0), its rows past P % 32 + R - 1 are zeroed, and the 32 rows go to the writer (cb_block); then, unless
// single_row, the next block: 32 zero rows with the rows crossing the block edge at their slots (cb_next). Rows past R
// are never read; R is read from a uint32 [1, 1, 1, 1] page and clamped to 32.
// Compile-time args: TensorAccessorArgs cache, v, position, rows.  Runtime args: 0 cache, 1 v, 2 position, 3 rows
// addresses, 4 first tile of the value in v, 5 single_row.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "../../qsa_block/kernels/main_tail_cbs.h"
#include "rows_kv_cbs.h"
using namespace main_tail;
using namespace rows_kv;

void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t value_addr = get_arg_val<uint32_t>(1);
    const uint32_t position_addr = get_arg_val<uint32_t>(2);
    const uint32_t rows_addr = get_arg_val<uint32_t>(3);
    const uint32_t value_first_tile = get_arg_val<uint32_t>(4);
    const uint32_t single_row = get_arg_val<uint32_t>(5);
    constexpr auto cache_args = TensorAccessorArgs<0>();
    constexpr auto value_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto position_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto rows_args = TensorAccessorArgs<position_args.next_compile_time_args_offset()>();
    const auto cache = TensorAccessor(cache_args, cache_addr);
    const auto value = TensorAccessor(value_args, value_addr);
    const auto position_page = TensorAccessor(position_args, position_addr);
    const auto rows_page = TensorAccessor(rows_args, rows_addr);
    Semaphore<> key_rotated_handoff(key_rotated_semaphore_id), key_normalized_handoff(key_normalized_semaphore_id);

    // the tile's first position P and the row count R (uint32 [1, 1, 1, 1] each) and the value tiles into pack slots
    // 0..7
    CircularBuffer(cb_kv_pos).reserve_back(1);
    CircularBuffer(cb_pack).reserve_back(pack_tiles);
    const uint32_t kv_pos_l1 = get_write_ptr(cb_kv_pos), pack_l1 = get_write_ptr(cb_pack);
    noc_async_read_page(0, position_page, kv_pos_l1);
    noc_async_read_page(0, rows_page, kv_pos_l1 + row_count_offset_bytes);
    {
        for (uint32_t c = 0; c < head_tiles; ++c) {
            noc_async_read_page(value_first_tile + c, value, pack_l1 + c * tile_bytes);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
    }
    const uint32_t first_position = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kv_pos_l1);
    const uint32_t rows_read = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kv_pos_l1 + row_count_offset_bytes);
    const uint32_t rows = rows_read < tile_rows::rows_per_tile ? rows_read : tile_rows::rows_per_tile;
    const uint32_t block = first_position & ~31u;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kv_pos_l1 + 4) = block;  // the writer reads the block start here
    CircularBuffer(cb_kv_pos).push_back(1);

    // the current block, read back
    CircularBuffer(cb_block).reserve_back(block_pages);
    const uint32_t block_l1 = get_write_ptr(cb_block);
    {
        for (uint32_t k = 0; k < tile_rows::rows_per_tile; ++k) {
            noc_async_read(cache.get_noc_addr(block + k, 0), block_l1 + k * query_row_bytes, query_row_bytes);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
    }
    {
        key_rotated_handoff.wait_min(1);
        key_normalized_handoff.wait_min(1);
        invalidate_l1_cache();
    }
    // this pass's rows into the current block; the crossing rows into the next block's zero rows
    CircularBuffer(cb_next).reserve_back(block_pages);
    const uint32_t next_l1 = get_write_ptr(cb_next);
    if (!single_row) {
        tile_rows::fill_words(next_l1, block_pages * query_row_bytes / 4, 0u);
    }
    {
        for (uint32_t j = 0; j < rows; ++j) {
            const uint32_t position = first_position + j;
            const uint32_t slot = position & 31u;
            if ((position & ~31u) == block) {
                pack_row_to_rm(pack_l1, j, block_l1 + slot * query_row_bytes);
            } else if (!single_row) {
                pack_row_to_rm(pack_l1, j, next_l1 + slot * query_row_bytes);
            }
        }
        // keep the block's rows below P % 32, place rows P % 32 .. P % 32 + R - 1, and zero the rows past them (+0, as
        // 0 * x + 0 in the op chain; no sparse row names them before a later pass rewrites them)
        for (uint32_t slot = (first_position & 31u) + rows; slot < tile_rows::rows_per_tile; ++slot) {
            tile_rows::fill_words(block_l1 + slot * query_row_bytes, query_row_bytes / 4, 0u);
        }
    }
    CircularBuffer(cb_block).push_back(block_pages);
    CircularBuffer(cb_next).push_back(block_pages);
    key_rotated_handoff.set(0);
    key_normalized_handoff.set(0);
}
