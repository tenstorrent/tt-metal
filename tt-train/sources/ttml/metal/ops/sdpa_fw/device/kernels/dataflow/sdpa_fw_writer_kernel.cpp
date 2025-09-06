// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compile_time_args.h>
#include <debug/dprint.h>

#include <cstdint>

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t intermediates_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_intermediates = tt::CBIndex::c_4;
    constexpr uint32_t cb_output = tt::CBIndex::c_15;

    constexpr uint32_t qWt = get_compile_time_arg_val(0);  // number of tiles in inner dimension
    constexpr uint32_t Ht = get_compile_time_arg_val(1);   // number of tiles in sequence dimension
    constexpr uint32_t block_size = get_compile_time_arg_val(2);
    constexpr uint32_t q_tiles_per_head = get_compile_time_arg_val(3);  // num of tiles per head in query
    constexpr uint32_t q_heads = get_compile_time_arg_val(4);           // num of heads in query
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(5);   // num of heads per group

    constexpr uint32_t onetile = 1U;

    const uint32_t tile_bytes = get_tile_size(cb_output);
    const DataFormat data_format = get_dataformat(cb_output);

    const InterleavedAddrGenFast</* is dram */ true> output_addr_generator = {
        .bank_base_address = output_addr, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is dram */ true> intermediates_addr_generator = {
        .bank_base_address = intermediates_addr, .page_size = tile_bytes, .data_format = data_format};

    const uint32_t tiles_per_head = q_tiles_per_head;

    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; r++) {
        uint32_t idx = r * qWt;
        uint32_t intermediate_idx = r;

        for (uint32_t q_head_idx = 0; q_head_idx < q_heads; ++q_head_idx) {
            cb_wait_front(cb_output, tiles_per_head);
            uint32_t l1_read_addr = get_read_ptr(cb_output);
            for (uint32_t col = 0; col < tiles_per_head; ++col) {
                noc_async_write_tile(idx + col, output_addr_generator, l1_read_addr);
                l1_read_addr += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_output, tiles_per_head);
            idx += tiles_per_head;

            if (q_head_idx == 0) {
                cb_wait_front(cb_intermediates, onetile);
                uint32_t l1_intermediates_read_addr = get_read_ptr(cb_intermediates);
                noc_async_write_tile(intermediate_idx, intermediates_addr_generator, l1_intermediates_read_addr);
                noc_async_write_barrier();
                cb_pop_front(cb_intermediates, onetile);
            }
        }
    }
}
