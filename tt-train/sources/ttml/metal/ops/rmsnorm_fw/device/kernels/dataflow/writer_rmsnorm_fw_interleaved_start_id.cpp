// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t rms_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_output_idx = tt::CBIndex::c_8;
    constexpr uint32_t cb_rms_output_idx = tt::CBIndex::c_9;

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_output_idx);
    const DataFormat data_format = get_dataformat(cb_output_idx);

    const InterleavedAddrGenFast</* is dram */ true> output_addr_generator = {
        .bank_base_address = output_addr, .page_size = tile_bytes, .data_format = data_format};

#ifdef RETURN_RMS
    const InterleavedAddrGenFast</* is dram */ true> rms_output_addr_generator = {
        .bank_base_address = rms_output_addr, .page_size = tile_bytes, .data_format = data_format};
#endif

    uint32_t end_row = start_row + num_rows_to_process;

    for (uint32_t r = start_row; r < end_row; ++r) {
#ifdef RETURN_RMS
        cb_wait_front(cb_rms_output_idx, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_rms_output_idx);
        noc_async_write_tile(r, rms_output_addr_generator, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_rms_output_idx, onetile);
#endif

        for (uint32_t c = 0, idx = r * Wt; c < Wt; c += block_size) {
            cb_wait_front(cb_output_idx, block_size);
            uint32_t l1_read_addr = get_read_ptr(cb_output_idx);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx, ++idx) {
                noc_async_write_tile(idx, output_addr_generator, l1_read_addr);
                l1_read_addr += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_output_idx, block_size);
        }
    }
}
