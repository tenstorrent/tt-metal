// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/dprint_pages.h"

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(runtime_args_counter++);        // input buffer address
    uint32_t target_address = get_arg_val<uint32_t>(runtime_args_counter++);       // target buffer address
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);  // rows to process in this kernel
    uint32_t start_row =
        get_arg_val<uint32_t>(runtime_args_counter++);  // pre calculated num_rows_written in program factory

    constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_target_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_mask_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_max_mask_idx = tt::CBIndex::c_3;
    constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_4;

    constexpr uint32_t num_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t mask_w = get_compile_time_arg_val(3);

    constexpr uint32_t onetile = 1U;
#ifdef DO_MASK_W
    constexpr bool do_mask_w = true;
#else
    constexpr bool do_mask_w = false;
#endif

    // generate mask tile
    if constexpr (do_mask_w) {
        cb_reserve_back(cb_mask_idx, onetile);
        cb_reserve_back(cb_max_mask_idx, onetile);
        cb_reserve_back(cb_scaler_idx, onetile);
        uint16_t* mask_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_mask_idx));          // write mask tile
        uint16_t* max_mask_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_max_mask_idx));  // write max mask tile
        uint16_t* scaler_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_scaler_idx));      // write scalar tile
        constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
        constexpr uint16_t zero = 0x0;
        constexpr uint16_t minus_inf = 0xFF80;  // (bfloat16)-inf -> uint16_t
        for (uint32_t face = 0; face < 4; ++face) {
            uint32_t offset = (face & 1U) << 4U;
            for (uint32_t h = 0; h < 16; ++h) {
                for (uint32_t w = 0; w < 16; ++w, ++mask_ptr, ++max_mask_ptr, ++scaler_ptr) {
                    *mask_ptr = (offset + w < mask_w) ? one : zero;  // how to create the proper mask?
                    *max_mask_ptr = (offset + w < mask_w) ? zero : minus_inf;

                    *scaler_ptr = one;
                }
            }
        }
        cb_push_back(cb_mask_idx, onetile);
        cb_push_back(cb_max_mask_idx, onetile);
        cb_push_back(cb_scaler_idx, onetile);
    }

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);
    const DataFormat data_format = get_dataformat(cb_input_idx);

    const InterleavedAddrGenFast</* is_dram */ true> input_address_generator = {
        .bank_base_address = input_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> target_address_generator = {
        .bank_base_address = target_address, .page_size = tile_bytes, .data_format = data_format};

    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        // calculate the address of the first tile in the row
        // start_row is the number of rows already processed in other cores
        uint32_t idx = (start_row + i) * Wt;  // (take already processed rows + current row)*Wt(number of tiles in row)

        // read input buffer
        cb_reserve_back(cb_input_idx, Wt);  // reserve Wt tiles in input buffer ==  wait until cb will has Wt tiles
        uint32_t l1_write_addr = get_write_ptr(cb_input_idx);  // get the address of the first tile in the input buffer

        for (uint32_t j = 0; j < Wt; ++j) {
            noc_async_read_tile(
                idx + j, input_address_generator, l1_write_addr);  // read the tile from the input buffer
            l1_write_addr += tile_bytes;                           // move to the next tile
        }
        noc_async_read_barrier();        // wait until all tiles are read
        cb_push_back(cb_input_idx, Wt);  // push the tile to the back of the input buffer

        // read target buffer
        cb_reserve_back(cb_target_idx, Wt);  // reserve Wt tiles in target buffer == wait until cb will has Wt tiles
        uint32_t l1_target_write_addr =
            get_write_ptr(cb_target_idx);  // get the address of the first tile in the target buffer
        for (uint32_t j = 0; j < Wt; ++j) {
            noc_async_read_tile(
                idx + j, target_address_generator, l1_target_write_addr);  // read the tile from the target buffer
            l1_target_write_addr += tile_bytes;                            // move to the next tile
        }
        noc_async_read_barrier();         // wait until all tiles are read
        cb_push_back(cb_target_idx, Wt);  // push the tile to the back of the target buffer
    }
}
