// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <debug/dprint.h>

// constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
// constexpr uint32_t block_size = get_compile_time_arg_val(1);     // Number of tiles in the inner dimention of the
// input tensor.constexpr uint32_t mask_w = get_compile_time_arg_val(2);  // Unused atm. constexpr uint32_t Wt =
// get_compile_time_arg_val(3);

// Think about move this to compile args to ainline void mess while adjusting indicies
//  CBs with input data
constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;  // Unused atm
constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_3;  // Number of activations, i.e. c in the paper
constexpr uint32_t cb_rms_a_idx = tt::CBIndex::c_4;
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;
// CBs with output data
// Create more intermedaite-output CBs that will be used exclusively by the writer. Do not compute anything on them
constexpr uint32_t cb_dL_da_idx = tt::CBIndex::c_6;
constexpr uint32_t cb_dL_dgamma_idx = tt::CBIndex::c_7;
// CBs with intermediate computations
constexpr uint32_t cb_scaled_gain = tt::CBIndex::c_8;
constexpr uint32_t cb_gained_dL_dout = tt::CBIndex::c_9;
constexpr uint32_t cb_scale = tt::CBIndex::c_10;
constexpr uint32_t cb_ms_a = tt::CBIndex::c_11;
constexpr uint32_t cb_c_by_ms_a = tt::CBIndex::c_12;
constexpr uint32_t cb_rhs = tt::CBIndex::c_13;
constexpr uint32_t cb_a_over_rms_a = tt::CBIndex::c_14;
constexpr uint32_t cb_dL_dgamma_components = tt::CBIndex::c_15;

constexpr uint32_t onetile = 1;

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr_left = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 16, .ws = 1};
        SliceRange sr_right = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 16, .w1 = 32, .ws = 1};
        DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
               << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL();
        break;
    }
}

void kernel_main() {
    // DPRINT << "Starting writer" << ENDL();
    uint32_t runtime_args_counter = 0;
    uint32_t dx_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dgamma_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    DPRINT << "Block size: " << block_size << ENDL();

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_dL_da_idx);
    const DataFormat data_format = get_dataformat(cb_dL_da_idx);

    const InterleavedAddrGenFast</* is dram */ true> dx_output_addr_generator = {
        .bank_base_address = dx_output_addr, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is dram */ true> dgamma_output_addr_generator = {
        .bank_base_address = dgamma_output_addr, .page_size = tile_bytes, .data_format = data_format};

    // NOTE: it is better to copy data to another CB and do not reuse CB between writer and kernel

    // cb_wait_front(cb_gained_dL_dout, 1);
    // DPRINT << "cb_gained_dL_dout" << ENDL();
    // print_full_tile(cb_gained_dL_dout, 0);

    // cb_wait_front(cb_scale, 1);
    // DPRINT << "cb_scale" << ENDL();
    // print_full_tile(cb_scale, 0);

    // cb_wait_front(cb_c_by_ms_a, 1);
    // DPRINT << "cb_c_by_ms_a" << ENDL();
    // print_full_tile(cb_c_by_ms_a, 0);

    // cb_wait_front(cb_rhs, 1);
    // DPRINT << "cb_rhs" << ENDL();
    // print_full_tile(cb_rhs, 0);

    /// SECTION: dL_da

    cb_wait_front(cb_dL_da_idx, onetile);
    DPRINT << "cb_dL_da_idx" << ENDL();
    print_full_tile(cb_dL_da_idx, 0);

    DPRINT << "cb_gained_dL_dout" << ENDL();
    print_full_tile(cb_gained_dL_dout, 0);

    DPRINT << "cb_scale" << ENDL();
    print_full_tile(cb_scale, 0);

    DPRINT << "cb_c_by_ms_a" << ENDL();
    print_full_tile(cb_c_by_ms_a, 0);

    DPRINT << "cb_rhs" << ENDL();
    print_full_tile(cb_rhs, 0);

    DPRINT << "cb_input_idx" << ENDL();
    print_full_tile(cb_input_idx, 0);

    DPRINT << "cb_rms_a_idx" << ENDL();
    print_full_tile(cb_rms_a_idx, 0);

    DPRINT << "cb_dL_out_idx" << ENDL();
    print_full_tile(cb_dL_out_idx, 0);

    /// SECTION: dL_dgamma_components

    // cb_wait_front(cb_dL_dgamma_components, onetile);
    // DPRINT << "cb_dL_dgamma_components" << ENDL();
    // print_full_tile(cb_dL_dgamma_components, 0);

    // DPRINT << "cb_input_idx" << ENDL();
    // print_full_tile(cb_input_idx, 0);
    // DPRINT << "cb_rms_a_idx" << ENDL();
    // print_full_tile(cb_rms_a_idx, 0);
    // DPRINT << "cb_a_over_rms_a" << ENDL();
    // print_full_tile(cb_a_over_rms_a, 0);

    // DPRINT << "cb_dL_out_idx" << ENDL();
    // print_full_tile(cb_dL_out_idx, 0);

    uint32_t end_row = start_row + num_rows_to_process;
    // DPRINT << "Writing dx and dgamma for rows from " << start_row << " to " << end_row << ENDL();
    for (uint32_t r = start_row; r < end_row; ++r) {
        // Write dx (grad w.r.t. input)
        for (uint32_t c = 0, idx = r * Wt; c < Wt; c += block_size) {
            // cb_wait_front(cb_dL_da_idx, block_size);
            uint32_t l1_read_addr = get_read_ptr(cb_dL_da_idx);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx, ++idx) {
                noc_async_write_tile(idx, dx_output_addr_generator, l1_read_addr);
                l1_read_addr += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_dL_da_idx, block_size);
        }

        // DPRINT << "Wrote dx for row " << r << ENDL();

        // Write dgamma (grad w.r.t. gamma)
        for (uint32_t c = 0, idx = r * Wt; c < Wt; c += block_size) {
            cb_wait_front(cb_dL_dgamma_components, block_size);
            uint32_t l1_read_addr = get_read_ptr(cb_dL_dgamma_components);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx, ++idx) {
                noc_async_write_tile(idx, dgamma_output_addr_generator, l1_read_addr);
                l1_read_addr += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_dL_dgamma_components, block_size);
        }
    }
}
