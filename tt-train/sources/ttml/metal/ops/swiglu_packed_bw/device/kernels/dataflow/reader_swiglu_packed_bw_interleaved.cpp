// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/common/swiglu_packed_layout.hpp"

constexpr uint32_t cb_gate_idx = tt::CBIndex::c_0;  // gate branch (silu'd), from packed[:, :I]
constexpr uint32_t cb_up_idx = tt::CBIndex::c_1;    // up branch, from packed[:, I:]
constexpr uint32_t cb_dh_idx = tt::CBIndex::c_2;    // upstream grad dL/dh

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);  // half width in tiles (one branch)
constexpr uint32_t blocks_per_row = get_compile_time_arg_val(2);
constexpr uint32_t Wt_dh = get_compile_time_arg_val(3);  // dL/dh row stride in tiles (>= Wt); Ht when transposed
#ifdef TRANSPOSED_GRAD
constexpr uint32_t cb_dh_t_idx = tt::CBIndex::c_8;   // dh^T tiles as read
constexpr uint32_t cb_a_idx = tt::CBIndex::c_9;      // lora_a tiles of the block's columns
constexpr uint32_t cb_tail_idx = tt::CBIndex::c_10;  // the row's s_dh^T tile
#endif

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    const uint32_t packed_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t dh_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    [[maybe_unused]] const uint32_t a_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_blocks_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_block = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_gate_idx);
    constexpr auto packed_args = TensorAccessorArgs<4>();
    constexpr auto dh_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
    const auto packed_gen = TensorAccessor(packed_args, packed_addr);
    const auto dh_gen = TensorAccessor(dh_args, dh_addr);
#ifdef TRANSPOSED_GRAD
    constexpr auto a_args = TensorAccessorArgs<dh_args.next_compile_time_args_offset()>();
    const auto a_gen = TensorAccessor(a_args, a_addr);
#endif

    const uint32_t end_block = start_block + num_blocks_to_process;
    for (uint32_t b = start_block; b < end_block; ++b) {
#ifdef TRANSPOSED_GRAD
        // Column-major blocks (iteration 38): block b is column block b / Ht of token row b % Ht, so a core's
        // consecutive blocks share their lora_a tiles (read once per column block) and read dh^T's column tiles
        // (pages (c + j) * Ht + r) consecutively; the row's s_dh^T tile (tile row Wt of dh^T) comes every block.
        const uint32_t dh_row = b % Wt_dh;
        const uint32_t dh_col = (b / Wt_dh) * block_size;
        const uint32_t gate_start = dh_row * (2U * Wt) + dh_col;
#else
        const uint32_t gate_start = swiglu_packed_gate_tile(b, Wt, block_size);
        const uint32_t dh_row = b / blocks_per_row;
        const uint32_t dh_col = (b - dh_row * blocks_per_row) * block_size;
#endif

        read_tiles_by_row<false>(cb_gate_idx, packed_gen, gate_start, block_size, tile_bytes, block_size);
        read_tiles_by_row<false>(cb_up_idx, packed_gen, gate_start + Wt, block_size, tile_bytes, block_size);
#ifdef TRANSPOSED_GRAD
        const bool new_col = (b == start_block) || (dh_row == 0);
        cb_reserve_back(cb_dh_t_idx, block_size);
        cb_reserve_back(cb_tail_idx, 1);
        if (new_col) {
            cb_reserve_back(cb_a_idx, block_size);
        }
        {
            uint32_t l1_dh = get_write_ptr(cb_dh_t_idx);
            uint32_t l1_a = get_write_ptr(cb_a_idx);
            for (uint32_t j = 0; j < block_size; ++j) {
                noc_async_read_page((dh_col + j) * Wt_dh + dh_row, dh_gen, l1_dh);
                l1_dh += tile_bytes;
                if (new_col) {
                    noc_async_read_page(dh_col + j, a_gen, l1_a);
                    l1_a += tile_bytes;
                }
            }
            noc_async_read_page(Wt * Wt_dh + dh_row, dh_gen, get_write_ptr(cb_tail_idx));
        }
#else
        read_tiles_by_row<false>(cb_dh_idx, dh_gen, dh_row * Wt_dh + dh_col, block_size, tile_bytes, block_size);
#endif
        noc_async_read_barrier();
        cb_push_back(cb_gate_idx, block_size);
        cb_push_back(cb_up_idx, block_size);
#ifdef TRANSPOSED_GRAD
        cb_push_back(cb_dh_t_idx, block_size);
        cb_push_back(cb_tail_idx, 1);
        if (new_col) {
            cb_push_back(cb_a_idx, block_size);
        }
#else
        cb_push_back(cb_dh_idx, block_size);
#endif
    }
}
