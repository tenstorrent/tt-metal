// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

// #define THREE_PACKS true

// CBs with input data
constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_1;
// CBs with output data
constexpr uint32_t cb_dL_da_idx = tt::CBIndex::c_2;
// CBs with intermediate computations
#ifdef THREE_PACKS
constexpr uint32_t cb_neg_sigmoid_idx = tt::CBIndex::c_3;
constexpr uint32_t cb_x_plus_x_times_neg_sigmoid_plus_one_idx = tt::CBIndex::c_4;
constexpr uint32_t cb_times_neg_sigmoid_and_neg_idx = tt::CBIndex::c_5;
#else
constexpr uint32_t cb_sigmoid_idx = tt::CBIndex::c_3;
constexpr uint32_t cb_one_minus_sigmoid_idx = tt::CBIndex::c_4;
constexpr uint32_t cb_times_input_plus_one_idx = tt::CBIndex::c_5;
constexpr uint32_t cb_times_sigmoid_idx = tt::CBIndex::c_6;
#endif

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

#ifdef DO_MASK_W
constexpr bool do_mask_w = true;
#else
constexpr bool do_mask_w = false;
#endif

inline void read_tiles(
    uint32_t cb_idx,
    const InterleavedAddrGenFast</* is dram */ true>& addr_gen,
    uint32_t start_idx,
    uint32_t block_size,
    uint32_t current_block_size,
    const uint32_t tile_bytes) {
    // Reads `num_tiles` tiles from DRAM starting at logical tile index `start_tile` into circular buffer `cb_idx`.
    cb_reserve_back(cb_idx, block_size);
    uint32_t l1_write_addr = get_write_ptr(cb_idx);
    for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
        noc_async_read_tile(start_idx + block_idx, addr_gen, l1_write_addr);
        l1_write_addr += tile_bytes;
    }
}

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dL_out_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
    constexpr uint16_t five = 0x40A0;     // (bfloat16)5.0 -> uint16_t
    // Here might be called generate tile for one, but hope it's not needed because we fiugure this out in a different
    // way

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);
    const DataFormat data_format = get_dataformat(cb_input_idx);

    const InterleavedAddrGenFast</* is_dram */ true> input_address_generator = {
        .bank_base_address = input_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> dL_out_address_generator = {
        .bank_base_address = dL_out_address, .page_size = tile_bytes, .data_format = data_format};

    // Read input tensors row by row, reading each row's data twice due to compute requirements
    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t row_tile_idx = (r * Wt) + c;
            uint32_t current_block_size = (c + block_size <= Wt) ? block_size : (Wt - c);

            read_tiles(cb_input_idx, input_address_generator, row_tile_idx, block_size, current_block_size, tile_bytes);
            read_tiles(
                cb_dL_out_idx, dL_out_address_generator, row_tile_idx, block_size, current_block_size, tile_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_input_idx, block_size);
            cb_push_back(cb_dL_out_idx, block_size);
        }
    }
}
