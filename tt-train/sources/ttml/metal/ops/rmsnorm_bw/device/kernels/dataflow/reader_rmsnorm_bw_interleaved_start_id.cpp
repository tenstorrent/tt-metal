// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compile_time_args.h>

#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"

// inline void print_loop(uint32_t count) {
//     // UNPACK(DPRINT << "U-LOOP:" << (uint32_t)count << ENDL());
//     // MATH(DPRINT << "M-LOOP:" << (uint32_t)count << ENDL());
//     // PACK(DPRINT << "P-LOOP:" << (uint32_t)count << ENDL());
// }

// inline void print_full_tile_column0(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
//     for (uint8_t r = 0; r < 32; ++r) {
//         SliceRange sr_left = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 1, .ws = 1};
//         DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " ";
//     }
// }

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr_left = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 16, .ws = 1};
        SliceRange sr_right = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 17, .w1 = 32, .ws = 1};
        DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
               << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL();
        break;
    }
}

// TODO: improve with a more efficient implementation
// using noc_async_writes
void generate_tile_with_value(uint32_t cb, uint32_t packed_value) {
    constexpr uint32_t onetile = 1U;
    cb_reserve_back(cb, onetile);
    uint32_t* ptr = reinterpret_cast<uint32_t*>(get_write_ptr(cb));
    // 512 = 32x16
    for (uint32_t i = 0; i < 512U; ++i, ++ptr) {
        *ptr = packed_value;
    }
    cb_push_back(cb, onetile);
}

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t gamma_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t rms_a_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dL_out_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;  // Unused atm
    constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_3;
    constexpr uint32_t cb_rms_a_idx = tt::CBIndex::c_4;
    constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;

    constexpr uint32_t packed_scaler = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);
    constexpr uint32_t mask_w = get_compile_time_arg_val(2);  // Unused atm
    constexpr uint32_t Wt = get_compile_time_arg_val(3);

    constexpr uint32_t onetile = 1U;  // Unused atm

#ifdef DO_MASK_W
    constexpt bool do_mask_w = true;
#else
    constexpr bool do_mask_w = false;
#endif

    // generate mask tile
    if constexpr (do_mask_w) {
        // TODO
    }

    // generate tiles to include scalar and epsilon
    generate_tile_with_value(cb_scaler_idx, packed_scaler);

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);
    const DataFormat data_format = get_dataformat(cb_input_idx);

    // Is it okay to assume that the tile bytes and data format will be the same?
    const InterleavedAddrGenFast</* is_dram */ true> input_address_generator = {
        .bank_base_address = input_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> gamma_address_generator = {
        .bank_base_address = gamma_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> rms_a_address_generator = {
        .bank_base_address = rms_a_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> dL_out_address_generator = {
        .bank_base_address = dL_out_address, .page_size = tile_bytes, .data_format = data_format};

    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        uint32_t idx = (start_row + i) * Wt;

#define EVERYTHING_FITS_IN_L1 1
#ifdef EVERYTHING_FITS_IN_L1
        // Input tensor a. Input tensor is [B,1,S,C], so we read it in Wt tiles. Hence id is idx + j. Its original shape
        // is [B,1,S,C].
        cb_reserve_back(cb_input_idx, Wt);
        uint32_t l1_input_write_addr = get_write_ptr(cb_input_idx);
        for (uint32_t j = 0; j < Wt; j++) {
            noc_async_read_tile(idx + j, input_address_generator, l1_input_write_addr);
            l1_input_write_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_idx, Wt);

        // Gamma. Gamma is constant for all rows, so we read it only once. Hence id is j, not idx + j. Its original
        // shape is [1,1,1,C].
        if (i == 0) {
            cb_reserve_back(cb_gamma_idx, Wt);
            uint32_t l1_gamma_write_addr = get_write_ptr(cb_gamma_idx);
            for (uint32_t j = 0; j < Wt; j++) {
                noc_async_read_tile(j, gamma_address_generator, l1_gamma_write_addr);
                l1_gamma_write_addr += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma_idx, Wt);
        }

        // RMS(a). RMS(a) is not constant for all rows, but for each row it is a single, scalar value. It is alread
        // tiled with one value at [0, 0], so we read it only once. Hence id is start_row + i. Its original shape is
        // [B,1,S,1].
        cb_reserve_back(cb_rms_a_idx, onetile);
        uint32_t l1_rms_a_write_addr = get_write_ptr(cb_rms_a_idx);
        noc_async_read_tile(start_row + i, rms_a_address_generator, l1_rms_a_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_rms_a_idx, onetile);

        // dL_out. dL_out is the gradient w.r.t. output, so it is the same shape as input, i.e. [B,1,S,C]. We read it in
        // Wt tiles. Hence id is idx + j.
        cb_reserve_back(cb_dL_out_idx, Wt);
        uint32_t l1_dL_out_write_addr = get_write_ptr(cb_dL_out_idx);
        for (uint32_t j = 0; j < Wt; j++) {
            noc_async_read_tile(idx + j, dL_out_address_generator, l1_dL_out_write_addr);
            l1_dL_out_write_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_dL_out_idx, Wt);
        if (i == 0) {
            DPRINT << "Print from reader" << ENDL();
            print_full_tile(cb_dL_out_idx, 0, true);
        }
    }
#elif defined(SOME_OTHER_OPTIONS_TBD)
        // TODO
#else
        // TODO
#endif
}
