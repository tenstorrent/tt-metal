// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// CBs with input data
constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_0;  // 1/N scaler
constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;  // mask for width dimension
constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_2;   // gamma (scale parameter)
constexpr uint32_t cb_rstd_idx = tt::CBIndex::c_4;    // rstd from forward pass
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;  // upstream gradient
constexpr uint32_t cb_input_idx = tt::CBIndex::c_6;   // input tensor
constexpr uint32_t cb_mean_idx = tt::CBIndex::c_7;    // mean from forward pass

constexpr uint32_t packed_scaler = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t mask_w = get_compile_time_arg_val(2);
constexpr uint32_t Wt = get_compile_time_arg_val(3);

#ifdef DO_MASK_W
constexpr bool do_mask_w = true;
#else
constexpr bool do_mask_w = false;
#endif

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t gamma_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t input_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t mean_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t rstd_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dL_out_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
    constexpr uint16_t zero = 0x0;
    // Generate mask tile.
    if constexpr (do_mask_w) {
        generate_mask_tile(cb_mask_w_idx, one, zero, mask_w);
    }
    // Generate tile with scalar (1/N).
    generate_tile_with_packed_bfloat16_value(cb_scaler_idx, packed_scaler);

    const uint32_t tile_bytes = get_tile_size(cb_scaler_idx);
    constexpr auto gamma_args = TensorAccessorArgs<4>();
    constexpr auto input_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    constexpr auto mean_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto rstd_args = TensorAccessorArgs<mean_args.next_compile_time_args_offset()>();
    constexpr auto dL_out_args = TensorAccessorArgs<rstd_args.next_compile_time_args_offset()>();

    const auto gamma_address_generator = TensorAccessor(gamma_args, gamma_address, tile_bytes);
    const auto input_address_generator = TensorAccessor(input_args, input_address, tile_bytes);
    const auto mean_address_generator = TensorAccessor(mean_args, mean_address, tile_bytes);
    const auto rstd_address_generator = TensorAccessor(rstd_args, rstd_address, tile_bytes);
    const auto dL_out_address_generator = TensorAccessor(dL_out_args, dL_out_address, tile_bytes);

    // Read input tensors row by row
    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        // Read rstd and mean once per row - both have shape [B,1,S,1]
        read_tiles_by_row</* UseBarrier = */ false>(
            cb_rstd_idx, rstd_address_generator, r, onetile, tile_bytes, onetile);
        read_tiles_by_row(cb_mean_idx, mean_address_generator, r, onetile, tile_bytes, onetile);
        // Barrier called by read_tiles_by_row with UseBarrier=true above
        cb_push_back(cb_rstd_idx, onetile);

#ifdef EVERYTHING_FITS_IN_L1
        // If everything fits in L1, read all data for the row at once
        read_tiles_by_row</* UseBarrier = */ false>(cb_input_idx, input_address_generator, r * Wt, Wt, tile_bytes, Wt);
        read_tiles_by_row(cb_dL_out_idx, dL_out_address_generator, r * Wt, Wt, tile_bytes, Wt);
        // Barrier called by read_tiles_by_row with UseBarrier=true above
        cb_push_back(cb_input_idx, Wt);
        if (r == start_row) {
            // Read gamma only once for all rows when everything fits in L1
            read_tiles_by_row(cb_gamma_idx, gamma_address_generator, 0, Wt, tile_bytes, Wt);
        }

#else
        // If not everything fits in L1, we need to read data multiple times per row

        // First pass: for computing sum(dy * gamma)
        for (uint32_t c = 0; c < Wt; c += block_size) {
            const uint32_t current_block_size = std::min(block_size, Wt - c);
            uint32_t row_tile_idx = (r * Wt) + c;

            read_tiles_by_row</* UseBarrier = */ false>(
                cb_dL_out_idx, dL_out_address_generator, row_tile_idx, current_block_size, tile_bytes, block_size);
            read_tiles_by_row(cb_gamma_idx, gamma_address_generator, c, current_block_size, tile_bytes, block_size);
            // Barrier called by read_tiles_by_row with UseBarrier=true above
            cb_push_back(cb_dL_out_idx, block_size);
        }

        // Second pass: for computing sum(dy * gamma * x_normalized)
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t row_tile_idx = (r * Wt) + c;
            const uint32_t current_block_size = std::min(block_size, Wt - c);

            read_tiles_by_row</* UseBarrier = */ false>(
                cb_input_idx, input_address_generator, row_tile_idx, current_block_size, tile_bytes, block_size);
            read_tiles_by_row</* UseBarrier = */ false>(
                cb_dL_out_idx, dL_out_address_generator, row_tile_idx, current_block_size, tile_bytes, block_size);
            read_tiles_by_row(cb_gamma_idx, gamma_address_generator, c, current_block_size, tile_bytes, block_size);
            // Barrier called by read_tiles_by_row with UseBarrier=true above
            cb_push_back(cb_input_idx, block_size);
            cb_push_back(cb_dL_out_idx, block_size);
        }

        // Three passes: for computing dx, dgamma_components, and dbeta_components
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t row_tile_idx = (r * Wt) + c;
            const uint32_t current_block_size = std::min(block_size, Wt - c);

            read_tiles_by_row</* UseBarrier = */ false>(
                cb_input_idx, input_address_generator, row_tile_idx, current_block_size, tile_bytes, block_size);
            read_tiles_by_row</* UseBarrier = */ false>(
                cb_gamma_idx, gamma_address_generator, c, current_block_size, tile_bytes, block_size);
            read_tiles_by_row(
                cb_dL_out_idx, dL_out_address_generator, row_tile_idx, current_block_size, tile_bytes, block_size);
            // Barrier called by read_tiles_by_row with UseBarrier=true above
            cb_push_back(cb_input_idx, block_size);
            cb_push_back(cb_gamma_idx, block_size);
        }
#endif
    }
}
