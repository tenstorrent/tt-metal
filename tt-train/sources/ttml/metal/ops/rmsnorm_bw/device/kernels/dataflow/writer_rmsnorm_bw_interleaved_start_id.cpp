// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t dx_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dgamma_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_dx_idx =
        tt::CBIndex::c_4;  // NOTE: those numbers may change once compute kernel will be implemented
    constexpr uint32_t cb_dgamma_idx = tt::CBIndex::c_5;

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_dx_idx);
    const DataFormat data_format = get_dataformat(cb_dx_idx);

    const InterleavedAddrGenFast</* is dram */ true> dx_output_addr_generator = {
        .bank_base_address = dx_output_addr, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is dram */ true> dgamma_output_addr_generator = {
        .bank_base_address = dgamma_output_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t end_row = start_row + num_rows_to_process;

    for (uint32_t r = start_row; r < end_row; ++r) {
        // Write dx (grad w.r.t. input)
        for (uint32_t c = 0, idx = r * Wt; c < Wt; c += block_size) {
            cb_wait_front(cb_dx_idx, block_size);
            uint32_t l1_read_addr = get_read_ptr(cb_dx_idx);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx, ++idx) {
                noc_async_write_tile(idx, dx_output_addr_generator, l1_read_addr);
                l1_read_addr += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_dx_idx, block_size);
        }

        // Write dgamma (grad w.r.t. gamma)
        for (uint32_t c = 0, idx = r * Wt; c < Wt; c += block_size) {
            cb_wait_front(cb_dgamma_idx, block_size);
            uint32_t l1_read_addr = get_read_ptr(cb_dgamma_idx);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx, ++idx) {
                noc_async_write_tile(idx, dgamma_output_addr_generator, l1_read_addr);
                l1_read_addr += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_dgamma_idx, block_size);
        }
    }
}
