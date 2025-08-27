
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <debug/dprint.h>

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"  // this is prob obsolete once debugged

// CBs with input data
constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_1;
// CBs with output data
constexpr uint32_t cb_dL_da_idx = tt::CBIndex::c_2;
// CBs with intermediate computations
constexpr uint32_t cb_sigmoid_idx = tt::CBIndex::c_3;
constexpr uint32_t cb_one_minus_sigmoid_idx = tt::CBIndex::c_4;
constexpr uint32_t cb_times_input_plus_one_idx = tt::CBIndex::c_5;
constexpr uint32_t cb_times_sigmoid_idx = tt::CBIndex::c_6;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

template <typename AddrGen>
inline void write_cb_block_to_dram(
    uint32_t cb_idx,
    const AddrGen& addr_gen,
    uint32_t start_idx,
    uint32_t block_size,
    uint32_t current_block_size,
    uint32_t tile_bytes) {
    cb_wait_front(cb_idx, block_size);
    uint32_t l1_read_addr = get_read_ptr(cb_idx);
    // We wait for the block_size tiles to be available in the CB, but we only write the current_block_size tiles
    // because C % block_size may not be zero.
    for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
        noc_async_write_tile(start_idx + block_idx, addr_gen, l1_read_addr);
        l1_read_addr += tile_bytes;
    }
}

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t da_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_dL_da_idx);
    constexpr auto da_args = TensorAccessorArgs<2>();
    const auto da_output_addr_generator = TensorAccessor(da_args, da_output_addr, tile_bytes);

    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        // Write da (gradient w.r.t. input) in blocks.
        // We interleave the writes to avoid waiting for the entire Wt tiles at once.
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t current_block_size = (c + block_size <= Wt) ? block_size : (Wt - c);
            uint32_t start_idx = (r * Wt) + c;

            // Write dL_da block
            write_cb_block_to_dram(
                cb_dL_da_idx, da_output_addr_generator, start_idx, block_size, current_block_size, tile_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_dL_da_idx, block_size);
        }
    }
}
