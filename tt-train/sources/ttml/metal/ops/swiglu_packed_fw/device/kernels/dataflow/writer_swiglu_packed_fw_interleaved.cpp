// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr uint32_t cb_out_idx = tt::CBIndex::c_2;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);  // output width in tiles (one branch)

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_out_idx);
    constexpr auto out_args = TensorAccessorArgs<2>();
    const auto out_gen = TensorAccessor(out_args, out_addr);

    const uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        const uint32_t out_row_start = r * Wt;
        for (uint32_t c = 0; c < Wt; c += block_size) {
            const uint32_t current_block_size = std::min(block_size, Wt - c);
            write_tiles_by_row(cb_out_idx, out_gen, out_row_start + c, current_block_size, tile_bytes, block_size);
        }
    }
}
