// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr uint32_t cb_out_idx = tt::CBIndex::c_2;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t blocks_per_row = get_compile_time_arg_val(1);
constexpr uint32_t Wt_out = get_compile_time_arg_val(2);  // output row stride in tiles (>= the half width); the
                                                          // token count Ht under TRANSPOSED_OUTPUT
#ifdef TRANSPOSED_OUTPUT
constexpr uint32_t cb_zero_idx = tt::CBIndex::c_5;
#endif

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_blocks_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_block = get_arg_val<uint32_t>(runtime_args_counter++);
    [[maybe_unused]] const uint32_t zero_start = get_arg_val<uint32_t>(runtime_args_counter++);
    [[maybe_unused]] const uint32_t zero_tiles = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_out_idx);
    constexpr auto out_args = TensorAccessorArgs<3>();
    const auto out_gen = TensorAccessor(out_args, out_addr);

    // Block b is row b / blocks_per_row, tiles [(b % blocks_per_row) * block_size, +block_size) of it; the output
    // row may be wider than the half width (Wt_out).
    const uint32_t end_block = start_block + num_blocks_to_process;
    for (uint32_t b = start_block; b < end_block; ++b) {
        const uint32_t row = b / blocks_per_row;
        const uint32_t col = (b - row * blocks_per_row) * block_size;
#ifdef TRANSPOSED_OUTPUT
        // h^T: the block's tiles (row, col + j) go to pages (col + j) * Ht + row (iteration 38)
        cb_wait_front(cb_out_idx, block_size);
        uint32_t l1 = get_read_ptr(cb_out_idx);
        for (uint32_t j = 0; j < block_size; ++j) {
            noc_async_write_page((col + j) * Wt_out + row, out_gen, l1);
            l1 += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out_idx, block_size);
#else
        write_tiles_by_row(cb_out_idx, out_gen, row * Wt_out + col, block_size, tile_bytes, block_size);
#endif
    }
#ifdef TRANSPOSED_OUTPUT
    if (zero_tiles > 0) {  // the tail rows of the transposed wide buffer (the LoRA h^T rows are written later)
        generate_tile_with_bfloat16_value(cb_zero_idx, 0);
        const uint32_t zero_l1 = get_read_ptr(cb_zero_idx);
        for (uint32_t t = 0; t < zero_tiles; ++t) {
            noc_async_write_page(zero_start + t, out_gen, zero_l1);
        }
        noc_async_write_barrier();
    }
#endif
}
