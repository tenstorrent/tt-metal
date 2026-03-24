// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// Reader kernel: emits constants and three input streams in compute-consumption order.
void kernel_main() {
    uint32_t arg_idx = 0U;
    const uint32_t input_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t weight_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t bias_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t scaler_fp32_bits = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eps_fp32_bits = get_arg_val<uint32_t>(arg_idx++);

    // CBs with input data / scalar parameters
    constexpr auto cb_input_pass_1 = tt::CBIndex::c_0;
    constexpr auto cb_input_pass_2 = tt::CBIndex::c_1;
    constexpr auto cb_input_pass_3 = tt::CBIndex::c_2;
    constexpr auto cb_scaler = tt::CBIndex::c_3;
    constexpr auto cb_eps = tt::CBIndex::c_4;
    constexpr auto cb_w0 = tt::CBIndex::c_5;
    constexpr auto cb_w1 = tt::CBIndex::c_6;
    constexpr auto cb_w2 = tt::CBIndex::c_7;
    constexpr auto cb_bias = tt::CBIndex::c_8;

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    generate_tile_with_uint32_value(cb_scaler, scaler_fp32_bits);
    generate_tile_with_uint32_value(cb_eps, eps_fp32_bits);

    const uint32_t tile_bytes = get_tile_size(cb_input_pass_1);
    constexpr auto input_args = TensorAccessorArgs<2>();
    constexpr auto weight_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    const auto input_address_generator = TensorAccessor(input_args, input_address, tile_bytes);
    const auto weight_address_generator = TensorAccessor(weight_args, weight_address, tile_bytes);
    const auto bias_address_generator = TensorAccessor(bias_args, bias_address, tile_bytes);

    constexpr uint32_t cb_scratch = tt::CBIndex::c_5;
    const uint16_t w0_bf16 =
        read_bfloat16_scalar_from_tile(weight_address_generator, /*row=*/0U, /*col=*/0U, cb_scratch);
    const uint16_t w1_bf16 =
        read_bfloat16_scalar_from_tile(weight_address_generator, /*row=*/0U, /*col=*/1U, cb_scratch);
    const uint16_t w2_bf16 =
        read_bfloat16_scalar_from_tile(weight_address_generator, /*row=*/0U, /*col=*/2U, cb_scratch);
    const uint16_t bias_bf16 =
        read_bfloat16_scalar_from_tile(bias_address_generator, /*row=*/0U, /*col=*/0U, cb_scratch);
    generate_tile_with_bfloat16_value(cb_w0, w0_bf16);
    generate_tile_with_bfloat16_value(cb_w1, w1_bf16);
    generate_tile_with_bfloat16_value(cb_w2, w2_bf16);
    generate_tile_with_bfloat16_value(cb_bias, bias_bf16);

    for (uint32_t row = 0; row < num_rows_to_process; ++row) {
        const uint32_t row_start_idx = (start_row + row) * Wt;

        // Pass A: feed stream consumed first by compute (pass_1).
        for (uint32_t col = 0; col < Wt; col += block_size) {
            const uint32_t current_block_size = std::min(block_size, Wt - col);
            const uint32_t block_start_idx = row_start_idx + col;

            read_tiles_by_row(
                cb_input_pass_1, input_address_generator, block_start_idx, current_block_size, tile_bytes, block_size);
        }

        // Pass B: feed pass_3 after pass_1, matching compute's stage order.
        for (uint32_t col = 0; col < Wt; col += block_size) {
            const uint32_t current_block_size = std::min(block_size, Wt - col);
            const uint32_t block_start_idx = row_start_idx + col;
            read_tiles_by_row(
                cb_input_pass_3, input_address_generator, block_start_idx, current_block_size, tile_bytes, block_size);
        }

        // Pass C: feed pass_2 last. This stream is only consumed in emit_output(),
        // so producing it earlier can deadlock for Wt > block_size.
        for (uint32_t col = 0; col < Wt; col += block_size) {
            const uint32_t current_block_size = std::min(block_size, Wt - col);
            const uint32_t block_start_idx = row_start_idx + col;
            read_tiles_by_row(
                cb_input_pass_2, input_address_generator, block_start_idx, current_block_size, tile_bytes, block_size);
        }
    }
}
