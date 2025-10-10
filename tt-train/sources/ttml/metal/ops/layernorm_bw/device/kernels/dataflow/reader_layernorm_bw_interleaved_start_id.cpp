// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

// CBs with input data
constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_0;      // 1/N scaler
constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_1;       // gamma (scale parameter)
constexpr uint32_t cb_x_hat_idx = tt::CBIndex::c_2;       // x_hat (normalized input) from forward pass
constexpr uint32_t cb_rstd_idx = tt::CBIndex::c_3;        // rstd from forward pass
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_4;      // upstream gradient
constexpr uint32_t cb_mat_mul_reduce = tt::CBIndex::c_5;  // reduction vector

constexpr uint32_t packed_scaler = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);

template <typename AddrGen>
inline void read_tiles(
    const uint32_t cb_idx,
    const AddrGen& addr_gen,
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
    uint32_t gamma_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t x_hat_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t rstd_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dL_out_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);

    // Generate tile with scalar (1/N).
    generate_tile_with_packed_bfloat16_value(cb_scaler_idx, packed_scaler);
    // Generate tile for matmul row reduce.
    generate_matmul_row_reduce_tile(cb_mat_mul_reduce);

    const uint32_t tile_bytes = get_tile_size(cb_scaler_idx);
    constexpr auto gamma_args = TensorAccessorArgs<3>();
    constexpr auto x_hat_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    constexpr auto rstd_args = TensorAccessorArgs<x_hat_args.next_compile_time_args_offset()>();
    constexpr auto dL_out_args = TensorAccessorArgs<rstd_args.next_compile_time_args_offset()>();

    const auto gamma_address_generator = TensorAccessor(gamma_args, gamma_address, tile_bytes);
    const auto x_hat_address_generator = TensorAccessor(x_hat_args, x_hat_address, tile_bytes);
    const auto rstd_address_generator = TensorAccessor(rstd_args, rstd_address, tile_bytes);
    const auto dL_out_address_generator = TensorAccessor(dL_out_args, dL_out_address, tile_bytes);

    // Read input tensors row by row
    // LayerNorm backward needs multiple passes over the data for sum computations
    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        // Read rstd once per row - rstd has shape [B,1,S,1]
        read_tiles(cb_rstd_idx, rstd_address_generator, r, 1, tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_rstd_idx, 1);

        // DEBUG: Print row number
        if (r == start_row) {
            DPRINT << "READER: Processing row " << r << ENDL();
        }

#ifdef EVERYTHING_FITS_IN_L1
        // Read x_hat for the entire row when everything fits in L1
        read_tiles(cb_x_hat_idx, x_hat_address_generator, r * Wt, Wt, tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_x_hat_idx, Wt);
        // If everything fits in L1, read all data for the row at once
        if (r == start_row) {
            // Read gamma only once for all rows when everything fits in L1
            for (uint32_t c = 0; c < Wt; c += block_size) {
                read_tiles(cb_gamma_idx, gamma_address_generator, c, block_size, tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma_idx, Wt);

            // DEBUG: Print first gamma tile (inputs to dy*gamma computation)
            DPRINT << "READER: gamma tile 0:" << ENDL();
            print_tile(cb_gamma_idx, 0, false);
        }

        // Read all row data at once
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t row_tile_idx = (r * Wt) + c;

            read_tiles(cb_dL_out_idx, dL_out_address_generator, row_tile_idx, block_size, tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_dL_out_idx, Wt);

        // DEBUG: Print first dL_out (dy) tile for this row
        if (r == start_row) {
            DPRINT << "READER: dy tile 0:" << ENDL();
            print_tile(cb_dL_out_idx, 0, false);
        }
#else
        // If not everything fits in L1, we need to read data multiple times per row

        // First pass: for computing sum(dy * gamma)
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t row_tile_idx = (r * Wt) + c;

            read_tiles(cb_dL_out_idx, dL_out_address_generator, row_tile_idx, block_size, tile_bytes);
            read_tiles(cb_gamma_idx, gamma_address_generator, c, block_size, tile_bytes);

            noc_async_read_barrier();
            cb_push_back(cb_dL_out_idx, block_size);
            cb_push_back(cb_gamma_idx, block_size);
        }

        // Second pass: for computing sum(dy * gamma * x_normalized)
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t row_tile_idx = (r * Wt) + c;

            read_tiles(cb_x_hat_idx, x_hat_address_generator, row_tile_idx, block_size, tile_bytes);
            read_tiles(cb_dL_out_idx, dL_out_address_generator, row_tile_idx, block_size, tile_bytes);
            read_tiles(cb_gamma_idx, gamma_address_generator, c, block_size, tile_bytes);

            noc_async_read_barrier();
            cb_push_back(cb_x_hat_idx, block_size);
            cb_push_back(cb_dL_out_idx, block_size);
            cb_push_back(cb_gamma_idx, block_size);
        }

        // Third pass: for computing dx
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t row_tile_idx = (r * Wt) + c;

            read_tiles(cb_x_hat_idx, x_hat_address_generator, row_tile_idx, block_size, tile_bytes);
            read_tiles(cb_gamma_idx, gamma_address_generator, c, block_size, tile_bytes);
            read_tiles(cb_dL_out_idx, dL_out_address_generator, row_tile_idx, block_size, tile_bytes);

            noc_async_read_barrier();
            cb_push_back(cb_x_hat_idx, block_size);
            cb_push_back(cb_gamma_idx, block_size);
            cb_push_back(cb_dL_out_idx, block_size);
        }

        // Fourth pass: for computing dgamma_components
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t row_tile_idx = (r * Wt) + c;

            read_tiles(cb_x_hat_idx, x_hat_address_generator, row_tile_idx, block_size, tile_bytes);
            read_tiles(cb_dL_out_idx, dL_out_address_generator, row_tile_idx, block_size, tile_bytes);

            noc_async_read_barrier();
            cb_push_back(cb_x_hat_idx, block_size);
            cb_push_back(cb_dL_out_idx, block_size);
        }

        // Fifth pass: for computing dbeta_components
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t row_tile_idx = (r * Wt) + c;

            read_tiles(cb_dL_out_idx, dL_out_address_generator, row_tile_idx, block_size, tile_bytes);

            noc_async_read_barrier();
            cb_push_back(cb_dL_out_idx, block_size);
        }
#endif
    }
}
