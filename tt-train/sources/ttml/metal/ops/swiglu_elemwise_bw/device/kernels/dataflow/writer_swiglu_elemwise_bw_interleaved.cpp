// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr uint32_t cb_dL_dlinear1_idx = tt::CBIndex::c_3;
constexpr uint32_t cb_dL_dgate_idx = tt::CBIndex::c_4;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t dL_dlinear1_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t dL_dgate_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_dL_dlinear1_idx);
    constexpr auto dL_dlinear1_args = TensorAccessorArgs<2>();
    constexpr auto dL_dgate_args = TensorAccessorArgs<dL_dlinear1_args.next_compile_time_args_offset()>();
    const auto dL_dlinear1_gen = TensorAccessor(dL_dlinear1_args, dL_dlinear1_addr, tile_bytes);
    const auto dL_dgate_gen = TensorAccessor(dL_dgate_args, dL_dgate_addr, tile_bytes);

    const uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        for (uint32_t c = 0; c < Wt; c += block_size) {
            const uint32_t row_tile_idx = r * Wt + c;
            const uint32_t current_block_size = std::min(block_size, Wt - c);

            write_tiles_by_row(cb_dL_dgate_idx, dL_dgate_gen, row_tile_idx, current_block_size, tile_bytes, block_size);
            write_tiles_by_row(
                cb_dL_dlinear1_idx, dL_dlinear1_gen, row_tile_idx, current_block_size, tile_bytes, block_size);
        }
    }
}
