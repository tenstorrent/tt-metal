// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// PolyNorm3 backward — reader kernel
//
// Reads weight scalars (w0, w1, w2) directly from the weight tensor on DRAM and
// generates constant tiles (scaler, epsilon, one, w0, w1, w2) once at startup,
// then streams input (x) and upstream gradient (dout) tiles to the compute
// kernel in two sequential passes per row:
//
//   Pass 1 — feeds accumulate_all_sums_for_row() in compute:
//     For each tile block: read x block, read dout block.
//
//   Pass 2 — feeds emit_output_for_row() in compute:
//     Same data re-read in the same order (x block, dout block).
//
// The two passes read the same DRAM data but push to the same CBs (cb_x, cb_dout)
// which are consumed and popped by the compute kernel between passes.
// ============================================================================

#include <algorithm>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t arg_idx = 0U;
    const uint32_t input_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dL_dout_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t weight_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t scaler_fp32_bits = get_arg_val<uint32_t>(arg_idx++);
    (void)get_arg_val<uint32_t>(arg_idx++);  // eps_fp32_bits — now consumed by compute kernel directly.

    constexpr auto cb_x = tt::CBIndex::c_0;
    constexpr auto cb_dout = tt::CBIndex::c_1;
    constexpr auto cb_scaler = tt::CBIndex::c_3;
    constexpr auto cb_one = tt::CBIndex::c_5;
    constexpr auto cb_w0 = tt::CBIndex::c_6;
    constexpr auto cb_w1 = tt::CBIndex::c_7;
    constexpr auto cb_w2 = tt::CBIndex::c_8;

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    // Set up DRAM address generators
    const uint32_t tile_bytes = get_tile_size(cb_x);
    constexpr auto input_args = TensorAccessorArgs<2>();
    constexpr auto dL_dout_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weight_args = TensorAccessorArgs<dL_dout_args.next_compile_time_args_offset()>();
    const auto input_address_generator = TensorAccessor(input_args, input_address, tile_bytes);
    const auto dL_dout_address_generator = TensorAccessor(dL_dout_args, dL_dout_address, tile_bytes);
    const auto weight_address_generator = TensorAccessor(weight_args, weight_address, tile_bytes);

    // Generate constant tiles (consumed once by compute kernel at startup)
    generate_tile_with_uint32_value(cb_scaler, scaler_fp32_bits);
    generate_tile_with_uint32_value(cb_one, 0x3F800000U);

    // Read weight scalars directly from the weight tensor on DRAM (no host roundtrip)
    constexpr uint32_t cb_scratch = tt::CBIndex::c_23;
    const uint16_t w0_bf16 =
        read_bfloat16_scalar_from_tile(weight_address_generator, /*row=*/0U, /*col=*/0U, cb_scratch);
    const uint16_t w1_bf16 =
        read_bfloat16_scalar_from_tile(weight_address_generator, /*row=*/0U, /*col=*/1U, cb_scratch);
    const uint16_t w2_bf16 =
        read_bfloat16_scalar_from_tile(weight_address_generator, /*row=*/0U, /*col=*/2U, cb_scratch);
    generate_tile_with_bfloat16_value(cb_w0, w0_bf16);
    generate_tile_with_bfloat16_value(cb_w1, w1_bf16);
    generate_tile_with_bfloat16_value(cb_w2, w2_bf16);

    for (uint32_t row = 0; row < num_rows_to_process; ++row) {
        const uint32_t row_start_idx = (start_row + row) * Wt;

        // Pass 1: x + dout for single-pass accumulation of all sums
        for (uint32_t col = 0; col < Wt; col += block_size) {
            const uint32_t current_block_size = std::min(block_size, Wt - col);
            const uint32_t block_start_idx = row_start_idx + col;
            read_tiles_by_row(
                cb_x, input_address_generator, block_start_idx, current_block_size, tile_bytes, block_size);
            read_tiles_by_row(
                cb_dout, dL_dout_address_generator, block_start_idx, current_block_size, tile_bytes, block_size);
        }

        // Pass 2: x + dout for emit_output (grad_x)
        for (uint32_t col = 0; col < Wt; col += block_size) {
            const uint32_t current_block_size = std::min(block_size, Wt - col);
            const uint32_t block_start_idx = row_start_idx + col;
            read_tiles_by_row(
                cb_x, input_address_generator, block_start_idx, current_block_size, tile_bytes, block_size);
            read_tiles_by_row(
                cb_dout, dL_dout_address_generator, block_start_idx, current_block_size, tile_bytes, block_size);
        }
    }
}
