// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compile_time_args.h>

#include <cstdint>

#include "debug/dprint.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

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
// THIS SHOULD BE REMOVED AND CALLED FROM COMMONS
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

// const char* data_format_to_string(DataFormat fmt) {
//     switch (fmt) {
//         case DataFormat::Float32: return "Float32";
//         case DataFormat::Float16: return "Float16";
//         case DataFormat::Bfp8: return "Bfp8";
//         case DataFormat::Bfp4: return "Bfp4";
//         case DataFormat::Bfp2: return "Bfp2";
//         case DataFormat::Float16_b: return "Float16_b";
//         case DataFormat::Bfp8_b: return "Bfp8_b";
//         case DataFormat::Bfp4_b: return "Bfp4_b";
//         case DataFormat::Bfp2_b: return "Bfp2_b";
//         case DataFormat::Lf8: return "Lf8";
//         case DataFormat::Int8: return "Int8";
//         case DataFormat::UInt8: return "UInt8";
//         case DataFormat::UInt16: return "UInt16";
//         case DataFormat::Int32: return "Int32";
//         case DataFormat::UInt32: return "UInt32";
//         case DataFormat::Tf32: return "Tf32";
//         case DataFormat::testMan7: return "testMan7";
//         case DataFormat::testMan2: return "testMan2";
//         case DataFormat::Invalid: return "Invalid";
//         default: return "Unknown";
//     }
// }

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t gamma_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t rms_a_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dL_out_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_3;
    constexpr uint32_t cb_rms_a_idx = tt::CBIndex::c_4;
    constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;
    // constexpr uint32_t cb_mask_garbage_idx = tt::CBIndex::c_25;

    constexpr uint32_t packed_scaler = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);
    constexpr uint32_t mask_w = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);

    constexpr uint32_t onetile = 1U;

#ifdef DO_MASK_W
    constexpr bool do_mask_w = true;
#else
    constexpr bool do_mask_w = false;
#endif
    // generate scaler and mask tile
    constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
    constexpr uint16_t zero = 0x0;
    constexpr uint16_t minus_inf = 0xFF80;  // (bfloat16)-inf -> uint16_t

    // generate mask tile
    if constexpr (do_mask_w) {
        DPRINT << "Generating mask tile with value: " << mask_w << ENDL();
        generate_mask_tile(cb_mask_w_idx, one, zero, mask_w);
        // generate_mask_tile(cb_mask_garbage_idx, zero, minus_inf, mask_w);
        DPRINT << "Generated mask tile done" << ENDL();
    }

    // generate tiles to include scalar and epsilon
    generate_tile_with_value(cb_scaler_idx, packed_scaler);

    // constexpr uint32_t val = 6789U;
    // generate_tile_with_value(cb_rms_a_idx, val);
    // DPRINT << "Generated dL_out tile with value: " << val << ENDL();
    // print_full_tile(cb_rms_a_idx, 0, true);
    // DPRINT << "END OF dL_out tile generation" << ENDL();

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);
    const DataFormat data_format = get_dataformat(cb_input_idx);

    // DPRINT << "DataFormat: " << data_format_to_string(data_format) << ENDL();
    // DPRINT << "rms_a_address: " << rms_a_address << ENDL();
    // Is it okay to assume that the tile bytes and data format will be the same?
    const InterleavedAddrGenFast</* is_dram */ true> input_address_generator = {
        .bank_base_address = input_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> gamma_address_generator = {
        .bank_base_address = gamma_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> rms_a_address_generator = {
        .bank_base_address = rms_a_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> dL_out_address_generator = {
        .bank_base_address = dL_out_address, .page_size = tile_bytes, .data_format = data_format};

    // DPRINT << dL_out_address << " dL_out_address" << ENDL();

    // DPRINT << "Print from reader" << ENDL();

    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        // DPRINT << "i: " << i << ENDL();
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
        // if (i == 0) {
        //     DPRINT << "Printing input from reader" << ENDL();
        //     print_full_tile(cb_input_idx, 0, true);
        // }

        // // Gamma. Gamma is constant for all rows, so we read it only once. Hence id is j, not idx + j. Its original
        // // shape is [1,1,1,C].
        if (i == 0) {
            cb_reserve_back(cb_gamma_idx, Wt);
            uint32_t l1_gamma_write_addr = get_write_ptr(cb_gamma_idx);
            for (uint32_t j = 0; j < Wt; j++) {
                noc_async_read_tile(j, gamma_address_generator, l1_gamma_write_addr);
                l1_gamma_write_addr += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma_idx, Wt);

            // DPRINT << "Printing gamma from reader" << ENDL();
            // print_full_tile(cb_gamma_idx, 0, true);
        }

        // // RMS(a). RMS(a) is not constant for all rows, but for each row it is a single, scalar value. It is alread
        // // tiled with one value at [0, 0], so we read it only once. Hence id is start_row + i. Its original shape is
        // // [B,1,S,1].
        cb_reserve_back(cb_rms_a_idx, onetile);
        uint32_t l1_rms_a_write_addr = get_write_ptr(cb_rms_a_idx);
        noc_async_read_tile(start_row + i, rms_a_address_generator, l1_rms_a_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_rms_a_idx, onetile);
        // if (i == 0) {
        //     DPRINT << "Printing RMS(a) from reader" << ENDL();
        //     print_full_tile(cb_rms_a_idx, 0, true);
        // }

        // dL_out. dL_out is the gradient w.r.t. output, so it is the same shape as input, i.e. [B,1,S,C]. We read it in
        // Wt tiles. Hence id is idx + j.
        cb_reserve_back(cb_dL_out_idx, Wt);
        uint32_t l1_dL_out_write_addr = get_write_ptr(cb_dL_out_idx);
        // DPRINT << "Wt: " << Wt << ENDL();
        // DPRINT << idx << " idx, l1_dL_out_write_addr: " << l1_dL_out_write_addr << ENDL();
        // DPRINT << "start_row: " << start_row << ENDL();
        // DPRINT << "num_rows_to_process: " << num_rows_to_process << ENDL();
        for (uint32_t j = 0; j < Wt; j++) {
            // DPRINT << "j: " << j << ENDL();
            noc_async_read_tile(idx + j, dL_out_address_generator, l1_dL_out_write_addr);
            l1_dL_out_write_addr += tile_bytes;
        }
        noc_async_read_barrier();
        // uint16_t* l1_ptr = (uint16_t*)l1_dL_out_write_addr;
        // for (uint32_t k = 0; k < 5; k++) {
        //     DPRINT << "l1_ptr[" << k << "]: " << l1_ptr[k] << ENDL();
        // }

        cb_push_back(cb_dL_out_idx, Wt);
        // cb_wait_front(cb_dL_out_idx, Wt);  // Wait for the dL_out tile to be ready before proceeding?
        // if (i == 0) {
        // DPRINT << "Printing dL_out from reader" << ENDL();
        // print_full_tile(cb_dL_out_idx, 0, true);
        // }
    }
#elif defined(SOME_OTHER_OPTIONS_TBD)
        // TODO
#else
        // TODO
#endif
}
