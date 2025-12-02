// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// CBs with input data
constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_1;
// CBs with output data
constexpr uint32_t cb_dL_da_idx = tt::CBIndex::c_2;
// CBs with intermediate computations
constexpr uint32_t cb_sigmoid_idx = tt::CBIndex::c_3;
constexpr uint32_t cb_one_minus_sigmoid_idx = tt::CBIndex::c_4;
constexpr uint32_t cb_times_input_plus_one_idx = tt::CBIndex::c_5;
constexpr uint32_t cb_times_sigmoid_idx = tt::CBIndex::c_6;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

#ifdef DO_MASK_W
constexpr bool do_mask_w = true;
#else
constexpr bool do_mask_w = false;
#endif

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dL_out_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);
    constexpr auto input_args = TensorAccessorArgs<2>();
    constexpr auto dL_out_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto input_address_generator = TensorAccessor(input_args, input_address, tile_bytes);
    const auto dL_out_address_generator = TensorAccessor(dL_out_args, dL_out_address, tile_bytes);

    // Read input tensors row by row, reading each row's data twice due to compute requirements
    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t row_tile_idx = (r * Wt) + c;
            uint32_t current_block_size = (c + block_size <= Wt) ? block_size : (Wt - c);

            read_tiles_by_row</* UseBarrier = */ false>(
                cb_input_idx, input_address_generator, row_tile_idx, current_block_size, tile_bytes, block_size);
            read_tiles_by_row(
                cb_dL_out_idx, dL_out_address_generator, row_tile_idx, current_block_size, tile_bytes, block_size);
            // Barrier called by read_tiles_by_row with UseBarrier=true above
            cb_push_back(cb_input_idx, block_size);
        }
    }
}
