// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr uint32_t cb_out_idx = tt::CBIndex::c_2;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t blocks_per_row = get_compile_time_arg_val(1);
constexpr uint32_t Wt_out = get_compile_time_arg_val(2);  // output row stride in tiles (>= the half width)

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_blocks_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_block = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_out_idx);
    constexpr auto out_args = TensorAccessorArgs<3>();
    const auto out_gen = TensorAccessor(out_args, out_addr);

    // Block b is row b / blocks_per_row, tiles [(b % blocks_per_row) * block_size, +block_size) of it; the output
    // row may be wider than the half width (Wt_out).
    const uint32_t end_block = start_block + num_blocks_to_process;
    for (uint32_t b = start_block; b < end_block; ++b) {
        const uint32_t row = b / blocks_per_row;
        const uint32_t col = (b - row * blocks_per_row) * block_size;
        write_tiles_by_row(cb_out_idx, out_gen, row * Wt_out + col, block_size, tile_bytes, block_size);
    }
}
