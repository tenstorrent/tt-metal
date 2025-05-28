// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dataflow_api_addrgen.h>
#include <hostdevcommon/kernel_structs.h>

#include <cstdint>
#include <cstring>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

void read_block_tiles(
    const uint32_t cb_input_idx,
    const InterleavedAddrGenFast<true>& input_address_generator,
    const uint32_t Wt,
    const uint32_t block_size,
    const uint32_t tile_bytes,
    const uint32_t idx) {
    for (uint32_t j = 0; j < Wt; j += block_size) {
        cb_reserve_back(cb_input_idx, block_size);
        uint32_t l1_write_addr = get_write_ptr(cb_input_idx);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            noc_async_read_tile(idx + j + block_idx, input_address_generator, l1_write_addr);
            l1_write_addr += tile_bytes;
        }

        noc_async_read_barrier();
        cb_push_back(cb_input_idx, block_size);
    }
}

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
    constexpr uint32_t cb_reduction_scaler_idx = tt::CBIndex::c_4;  // used for reduction
    constexpr uint32_t cb_matmul_reduce = tt::CBIndex::c_9;

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t mask_w = get_compile_time_arg_val(2);
    constexpr uint32_t target_indexes_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t tiled_H = get_compile_time_arg_val(4);
    constexpr uint32_t target_indexes_read_page_size = get_compile_time_arg_val(5);

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

    if constexpr (do_mask_w) {
        generate_mask_tile(cb_mask_idx, one, zero, mask_w);
        generate_mask_tile(cb_max_mask_idx, zero, minus_inf, mask_w);
    }

    generate_tile_with_bfloat16_value(
        cb_reduction_scaler_idx, one);                  // generate tile with bfloat16 value 1.0 for reduction scaler
    generate_matmul_row_reduce_tile(cb_matmul_reduce);  // generate tile for matmul row reduce

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);
    const DataFormat data_format = get_dataformat(cb_input_idx);

    const InterleavedAddrGenFast</* is_dram */ true> input_address_generator = {
        .bank_base_address = input_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGen</* is_dram */ true> target_indexes_address_generator = {
        .bank_base_address = target_address, .page_size = target_indexes_page_size};

    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        // calculate the address of the first tile in the row
        // start_row is the number of rows already processed in other cores
        uint32_t idx = (start_row + i) * Wt;  // (take already processed rows + current row)*Wt(number of tiles in row)

        // read target indexes
        cb_reserve_back(cb_target_idx, onetile);
        uint32_t l1_target_indexes_write_addr =
            get_write_ptr(cb_target_idx);  // get the address of the first tile in the target buffer

        auto [page, offset] = get_page_and_offset(start_row + i, tiled_H);

        auto noc_async_target_indexes_page_addr = get_noc_addr(page, target_indexes_address_generator, offset);
        noc_async_read(
            noc_async_target_indexes_page_addr,
            l1_target_indexes_write_addr,
            target_indexes_read_page_size);    // read the page from the target buffer
        noc_async_read_barrier();              // wait until all tiles are read
        cb_push_back(cb_target_idx, onetile);  // push the tile to the back of the target buffer

        // read input buffer by blocks
        read_block_tiles(cb_input_idx, input_address_generator, Wt, block_size, tile_bytes, idx);

#ifndef EVERYTHING_FITS_IN_L1
        // read input buffer by blocks to calculate sum(exp(x - max(x))) in row
        read_block_tiles(cb_input_idx, input_address_generator, Wt, block_size, tile_bytes, idx);

        // read input buffer by blocks to calculate softmax in row
        read_block_tiles(cb_input_idx, input_address_generator, Wt, block_size, tile_bytes, idx);
#endif
    }
}
