// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t arg_idx = 0U;
    const uint32_t input_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dL_dout_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t scaler_fp32_bits = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eps_fp32_bits = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t packed_w0 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t packed_w1 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t packed_w2 = get_arg_val<uint32_t>(arg_idx++);

    // CBs with input data / scalar parameters
    constexpr auto cb_input_pass_1 = tt::CBIndex::c_0;
    constexpr auto cb_input_pass_2 = tt::CBIndex::c_1;
    constexpr auto cb_input_pass_3 = tt::CBIndex::c_2;
    constexpr auto cb_input_pass_4 = tt::CBIndex::c_3;
    constexpr auto cb_input_pass_5 = tt::CBIndex::c_4;
    constexpr auto cb_input_pass_6 = tt::CBIndex::c_5;
    constexpr auto cb_dL_dout_pass_1 = tt::CBIndex::c_6;
    constexpr auto cb_dL_dout_pass_2 = tt::CBIndex::c_7;
    constexpr auto cb_dL_dout_pass_3 = tt::CBIndex::c_8;
    constexpr auto cb_dL_dout_pass_4 = tt::CBIndex::c_9;
    constexpr auto cb_scaler = tt::CBIndex::c_10;
    constexpr auto cb_eps = tt::CBIndex::c_11;
    constexpr auto cb_one = tt::CBIndex::c_12;
    constexpr auto cb_w0 = tt::CBIndex::c_13;
    constexpr auto cb_w1 = tt::CBIndex::c_14;
    constexpr auto cb_w2 = tt::CBIndex::c_15;

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    generate_tile_with_uint32_value(cb_scaler, scaler_fp32_bits);
    generate_tile_with_uint32_value(cb_eps, eps_fp32_bits);
    generate_tile_with_uint32_value(cb_one, 0x3F800000U);
    generate_tile_with_packed_bfloat16_value(cb_w0, packed_w0);
    generate_tile_with_packed_bfloat16_value(cb_w1, packed_w1);
    generate_tile_with_packed_bfloat16_value(cb_w2, packed_w2);

    const uint32_t tile_bytes = get_tile_size(cb_input_pass_1);
    constexpr auto input_args = TensorAccessorArgs<2>();
    constexpr auto dL_dout_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto input_address_generator = TensorAccessor(input_args, input_address, tile_bytes);
    const auto dL_dout_address_generator = TensorAccessor(dL_dout_args, dL_dout_address, tile_bytes);

    for (uint32_t row = 0; row < num_rows_to_process; ++row) {
        const uint32_t row_start_idx = (start_row + row) * Wt;

        for (uint32_t col = 0; col < Wt; col += block_size) {
            const uint32_t current_block_size = std::min(block_size, Wt - col);
            const uint32_t block_start_idx = row_start_idx + col;
            read_tiles_by_row(
                cb_input_pass_1, input_address_generator, block_start_idx, current_block_size, tile_bytes, block_size);
        }

        for (uint32_t col = 0; col < Wt; col += block_size) {
            const uint32_t current_block_size = std::min(block_size, Wt - col);
            const uint32_t block_start_idx = row_start_idx + col;
            read_tiles_by_row(
                cb_input_pass_2, input_address_generator, block_start_idx, current_block_size, tile_bytes, block_size);
        }

        for (uint32_t col = 0; col < Wt; col += block_size) {
            const uint32_t current_block_size = std::min(block_size, Wt - col);
            const uint32_t block_start_idx = row_start_idx + col;
            read_tiles_by_row(
                cb_input_pass_3, input_address_generator, block_start_idx, current_block_size, tile_bytes, block_size);
            read_tiles_by_row(
                cb_dL_dout_pass_1,
                dL_dout_address_generator,
                block_start_idx,
                current_block_size,
                tile_bytes,
                block_size);
        }

        for (uint32_t col = 0; col < Wt; col += block_size) {
            const uint32_t current_block_size = std::min(block_size, Wt - col);
            const uint32_t block_start_idx = row_start_idx + col;
            read_tiles_by_row(
                cb_input_pass_4, input_address_generator, block_start_idx, current_block_size, tile_bytes, block_size);
            read_tiles_by_row(
                cb_dL_dout_pass_2,
                dL_dout_address_generator,
                block_start_idx,
                current_block_size,
                tile_bytes,
                block_size);
        }

        for (uint32_t col = 0; col < Wt; col += block_size) {
            const uint32_t current_block_size = std::min(block_size, Wt - col);
            const uint32_t block_start_idx = row_start_idx + col;
            read_tiles_by_row(
                cb_input_pass_5, input_address_generator, block_start_idx, current_block_size, tile_bytes, block_size);
            read_tiles_by_row(
                cb_dL_dout_pass_3,
                dL_dout_address_generator,
                block_start_idx,
                current_block_size,
                tile_bytes,
                block_size);
        }

        for (uint32_t col = 0; col < Wt; col += block_size) {
            const uint32_t current_block_size = std::min(block_size, Wt - col);
            const uint32_t block_start_idx = row_start_idx + col;
            read_tiles_by_row(
                cb_input_pass_6, input_address_generator, block_start_idx, current_block_size, tile_bytes, block_size);
            read_tiles_by_row(
                cb_dL_dout_pass_4,
                dL_dout_address_generator,
                block_start_idx,
                current_block_size,
                tile_bytes,
                block_size);
        }
    }
}
