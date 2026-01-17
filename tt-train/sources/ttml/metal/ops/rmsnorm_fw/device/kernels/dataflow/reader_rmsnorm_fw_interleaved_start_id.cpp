// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t gamma_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_eps_idx = tt::CBIndex::c_3;
    constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_4;

    constexpr uint32_t packed_scaler = get_compile_time_arg_val(0);
    constexpr uint32_t packed_eps = get_compile_time_arg_val(1);
    constexpr uint32_t mask_w = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t block_size = get_compile_time_arg_val(4);

    constexpr uint32_t onetile = 1U;

#ifdef DO_MASK_W
    constexpr bool do_mask_w = true;
#else
    constexpr bool do_mask_w = false;
#endif

    // generate mask tile
    if constexpr (do_mask_w) {
        constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
        constexpr uint16_t zero = 0x0;
        generate_mask_tile(cb_mask_w_idx, one, zero, mask_w);
    }

    // generate tiles to include scalar and epsilon
    generate_tile_with_packed_bfloat16_value(cb_scaler_idx, packed_scaler);
    generate_tile_with_packed_bfloat16_value(cb_eps_idx, packed_eps);

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);
    constexpr auto input_args = TensorAccessorArgs<5>();
    constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto input_address_generator = TensorAccessor(input_args, input_address, tile_bytes);
    const auto gamma_address_generator = TensorAccessor(gamma_args, gamma_address, tile_bytes);

    const uint32_t max_block_size = 4;

    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        uint32_t idx = (start_row + i) * Wt;

#ifdef EVERYTHING_FITS_IN_L1
        read_tiles_by_row(cb_input_idx, input_address_generator, idx, Wt, tile_bytes, Wt);

        if (i == 0) {
            read_tiles_by_row(cb_gamma_idx, gamma_address_generator, 0, Wt, tile_bytes, Wt);
        }

#elif defined(EVERYTHING_EXCEPT_GAMMA_FITS_IN_L1)
        read_tiles_by_row(cb_input_idx, input_address_generator, idx, Wt, tile_bytes, Wt);

        read_full_row_tiles(cb_gamma_idx, gamma_address_generator, Wt, block_size, tile_bytes, 0);
#else
        read_full_row_tiles(cb_input_idx, input_address_generator, Wt, block_size, tile_bytes, idx);

        for (uint32_t j = 0; j < Wt; j += block_size) {
            read_tiles_by_row(cb_input_idx, input_address_generator, idx + j, block_size, tile_bytes, block_size);

            // reading gamma to L1
            read_tiles_by_row(cb_gamma_idx, gamma_address_generator, j, block_size, tile_bytes, block_size);
        }
#endif
    }
}
