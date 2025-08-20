// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

// CBs with input data
constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;
constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_3;
constexpr uint32_t cb_rms_a_idx = tt::CBIndex::c_4;
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;
constexpr uint32_t cb_mat_mul_reduce = tt::CBIndex::c_6;
// CBs with output data
constexpr uint32_t cb_dL_da_idx = tt::CBIndex::c_7;
constexpr uint32_t cb_dL_dgamma_components = tt::CBIndex::c_8;
// CBs with intermediate computations
constexpr uint32_t cb_recip_rms_a_bcasted_idx = tt::CBIndex::c_9;
constexpr uint32_t cb_scale_idx = tt::CBIndex::c_10;
constexpr uint32_t cb_scale_bcasted_idx = tt::CBIndex::c_11;

constexpr uint32_t packed_scaler = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t mask_w = get_compile_time_arg_val(2);
constexpr uint32_t Wt = get_compile_time_arg_val(3);

#ifdef DO_MASK_W
constexpr bool do_mask_w = true;
#else
constexpr bool do_mask_w = false;
#endif

inline void read_tiles(
    const uint32_t cb_idx,
    const InterleavedAddrGenFast<true>& addr_gen,
    const uint32_t start_tile,
    const uint32_t num_tiles,
    const uint32_t tile_bytes) {
    // Reads `num_tiles` tiles from DRAM starting at logical tile index `start_tile` into circular buffer `cb_idx`.
    cb_reserve_back(cb_idx, num_tiles);
    uint32_t l1_write_addr = get_write_ptr(cb_idx);
    for (uint32_t k = 0; k < num_tiles; ++k) {
        noc_async_read_tile(start_tile + k, addr_gen, l1_write_addr);
        l1_write_addr += tile_bytes;
    }
}

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t gamma_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t rms_a_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dL_out_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
    constexpr uint16_t zero = 0x0;
    constexpr uint16_t minus_inf = 0xFF80;  // (bfloat16)-inf -> uint16_t
    // Generate mask tile.
    if constexpr (do_mask_w) {
        generate_mask_tile(cb_mask_w_idx, one, zero, mask_w);
    }
    // Generate tile with scalar.
    generate_tile_with_packed_bfloat16_value(cb_scaler_idx, packed_scaler);
    // Generate tile for matmul row reduce.
    generate_matmul_row_reduce_tile(cb_mat_mul_reduce);

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);
    const DataFormat data_format = get_dataformat(cb_input_idx);

    const InterleavedAddrGenFast</* is_dram */ true> input_address_generator = {
        .bank_base_address = input_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> gamma_address_generator = {
        .bank_base_address = gamma_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> rms_a_address_generator = {
        .bank_base_address = rms_a_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> dL_out_address_generator = {
        .bank_base_address = dL_out_address, .page_size = tile_bytes, .data_format = data_format};

    // Read input tensors row by row, reading each row's data twice due to compute requirements
    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        // Read RMS(a) once per row - shape [B,1,S,1], one scalar per row
        read_tiles(cb_rms_a_idx, rms_a_address_generator, r, 1, tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_rms_a_idx, 1);

        // First pass: read row data for first compute phase
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t row_tile_idx = (r * Wt) + c;

            read_tiles(cb_input_idx, input_address_generator, row_tile_idx, block_size, tile_bytes);
            read_tiles(cb_dL_out_idx, dL_out_address_generator, row_tile_idx, block_size, tile_bytes);
#ifndef EVERYTHING_FITS_IN_L1
            // If not everything fits in L1, we need to reread gamma for each row.
            read_tiles(cb_gamma_idx, gamma_address_generator, c, block_size, tile_bytes);
#else
            // If everything fits in L1, we can read gamma only once.
            if (r == start_row) {
                read_tiles(cb_gamma_idx, gamma_address_generator, c, block_size, tile_bytes);
            }
#endif

            noc_async_read_barrier();
            cb_push_back(cb_input_idx, block_size);
            cb_push_back(cb_dL_out_idx, block_size);
#ifndef EVERYTHING_FITS_IN_L1
            cb_push_back(cb_gamma_idx, block_size);
#else
            if (r == start_row) {
                cb_push_back(cb_gamma_idx, block_size);
            }
#endif
        }

#ifndef EVERYTHING_FITS_IN_L1
        // Second pass: read row data again for second compute phase
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t row_tile_idx = (r * Wt) + c;

            read_tiles(cb_input_idx, input_address_generator, row_tile_idx, block_size, tile_bytes);
            read_tiles(cb_dL_out_idx, dL_out_address_generator, row_tile_idx, block_size, tile_bytes);
            read_tiles(cb_gamma_idx, gamma_address_generator, c, block_size, tile_bytes);

            noc_async_read_barrier();
            cb_push_back(cb_input_idx, block_size);
            cb_push_back(cb_dL_out_idx, block_size);
            cb_push_back(cb_gamma_idx, block_size);
        }
#endif
    }
}
