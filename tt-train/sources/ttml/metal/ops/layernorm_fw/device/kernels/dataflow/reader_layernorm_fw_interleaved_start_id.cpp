// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// CBs with input data
constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_0;  // 1/N scaler
constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;  // mask for width dimension
constexpr uint32_t cb_eps_idx = tt::CBIndex::c_2;     // epsilon
constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_3;   // gamma (scale parameter)
constexpr uint32_t cb_beta_idx = tt::CBIndex::c_4;    // beta (shift parameter)
constexpr uint32_t cb_input_idx = tt::CBIndex::c_5;   // input tensor

constexpr uint32_t packed_scaler = get_compile_time_arg_val(0);
constexpr uint32_t packed_eps = get_compile_time_arg_val(1);
constexpr uint32_t block_size = get_compile_time_arg_val(2);
constexpr uint32_t mask_w = get_compile_time_arg_val(3);
constexpr uint32_t Wt = get_compile_time_arg_val(4);
constexpr uint32_t closest_to_Wt_multiple_of_block_size = ((Wt + block_size - 1) / block_size) * block_size;

#ifdef DO_MASK_W
constexpr bool do_mask_w = true;
#else
constexpr bool do_mask_w = false;
#endif

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t gamma_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t beta_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
    constexpr uint16_t zero = 0x0;

    // Generate mask tile if needed
    if constexpr (do_mask_w) {
        generate_mask_tile(cb_mask_w_idx, one, zero, mask_w);
    }

    // Generate tile with scalar (1/N)
    generate_tile_with_packed_bfloat16_value(cb_scaler_idx, packed_scaler);

    // Generate tile with epsilon
    generate_tile_with_packed_bfloat16_value(cb_eps_idx, packed_eps);

    const uint32_t tile_bytes = get_tile_size(cb_scaler_idx);
    constexpr auto input_args = TensorAccessorArgs<5>();
    constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const auto input_address_generator = TensorAccessor(input_args, input_address, tile_bytes);
    const auto gamma_address_generator = TensorAccessor(gamma_args, gamma_address, tile_bytes);
    const auto beta_address_generator = TensorAccessor(beta_args, beta_address, tile_bytes);

#ifdef EVERYTHING_FITS_IN_L1
    // Read gamma and beta once for all rows when everything fits in L1
    read_tiles_by_row</* UseBarrier = */ false>(cb_gamma_idx, gamma_address_generator, 0, Wt, tile_bytes, Wt);
    read_tiles_by_row(cb_beta_idx, beta_address_generator, 0, Wt, tile_bytes, Wt);
    cb_push_back(cb_gamma_idx, Wt);
#endif

    // Read input tensors row by row
    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
#ifdef EVERYTHING_FITS_IN_L1
        // If everything fits in L1, read all input for the row at once
        read_tiles_by_row(cb_input_idx, input_address_generator, r * Wt, Wt, tile_bytes, Wt);
#else
        // If not everything fits in L1, read data in blocks
        // Note: For forward pass, we need to read the input multiple times:
        // 1. For computing sum (mean)
        // 2. For computing variance
        // 3. For computing x_hat and output

        // First pass: for computing sum (mean)
        read_full_row_tiles(cb_input_idx, input_address_generator, Wt, block_size, tile_bytes, r * Wt);

        // Second pass: for computing rstd
        read_full_row_tiles(cb_input_idx, input_address_generator, Wt, block_size, tile_bytes, r * Wt);

        // Third pass: for computing x_hat and output
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t row_tile_idx = (r * Wt) + c;
            const uint32_t current_block_size = std::min(block_size, Wt - c);

            read_tiles_by_row</* UseBarrier = */ false>(
                cb_input_idx, input_address_generator, row_tile_idx, current_block_size, tile_bytes, block_size);
            read_tiles_by_row</* UseBarrier = */ false>(
                cb_gamma_idx, gamma_address_generator, c, current_block_size, tile_bytes, block_size);
            read_tiles_by_row(cb_beta_idx, beta_address_generator, c, current_block_size, tile_bytes, block_size);
            // Barrier called by read_tiles_by_row with UseBarrier=true above
            cb_push_back(cb_input_idx, block_size);
            cb_push_back(cb_gamma_idx, block_size);
        }
#endif
    }
}
