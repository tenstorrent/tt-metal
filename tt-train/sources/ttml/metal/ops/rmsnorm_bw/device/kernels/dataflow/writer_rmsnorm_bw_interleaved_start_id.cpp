// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

// TODO: Consider moving these constants to compile-time arguments to avoid index-mismatch issues while developing.
//  CBs with input data
constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;
constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;  // 1/c - used for scaling
constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_3;
constexpr uint32_t cb_rms_a_idx = tt::CBIndex::c_4;
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;
constexpr uint32_t cb_one_idx = tt::CBIndex::c_6;  // Used to reduce scale to a single value
// CBs with output data
constexpr uint32_t cb_dL_da_idx = tt::CBIndex::c_7;
constexpr uint32_t cb_dL_dgamma_components = tt::CBIndex::c_8;
// CBs with intermediate computations
constexpr uint32_t cb_recip_rms_a_bcasted_idx = tt::CBIndex::c_9;
constexpr uint32_t cb_scale_idx = tt::CBIndex::c_10;
constexpr uint32_t cb_scale_bcasted = tt::CBIndex::c_11;

constexpr uint32_t onetile = 1;

// DEBUG: TO be removed just before merge
inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr_left = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 16, .ws = 1};
        SliceRange sr_right = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 16, .w1 = 32, .ws = 1};
        DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
               << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL();
        // break;
    }
}

inline void write_cb_to_dram(
    uint32_t cb_idx,
    const InterleavedAddrGenFast</* is dram */ true>& addr_gen,
    uint32_t start_idx,
    uint32_t num_tiles,
    uint32_t block_size,
    uint32_t tile_bytes) {
    for (uint32_t c = 0, idx = start_idx; c < num_tiles; c += block_size) {
        cb_wait_front(cb_idx, block_size);
        uint32_t l1_read_addr = get_read_ptr(cb_idx);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx, ++idx) {
            noc_async_write_tile(idx, addr_gen, l1_read_addr);
            l1_read_addr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_idx, block_size);
    }
}

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t dx_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dgamma_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_dL_da_idx);
    const DataFormat data_format = get_dataformat(cb_dL_da_idx);

    const InterleavedAddrGenFast</* is dram */ true> dx_output_addr_generator = {
        .bank_base_address = dx_output_addr, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is dram */ true> dgamma_output_addr_generator = {
        .bank_base_address = dgamma_output_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        // Write da (gradient w.r.t. input) and dgamma_components (partial gradients w.r.t. gamma).
        // Note: The final dL_dgamma (gradient w.r.t. gamma) requires reduction across batches, which is performed on
        // the host. Here, we output the per-tile components for host-side reduction.
        write_cb_to_dram(cb_dL_da_idx, dx_output_addr_generator, r * Wt, Wt, block_size, tile_bytes);
        write_cb_to_dram(cb_dL_dgamma_components, dgamma_output_addr_generator, r * Wt, Wt, block_size, tile_bytes);
    }
}
