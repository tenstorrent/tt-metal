// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_output_idx = tt::CBIndex::c_10;

    constexpr uint32_t block_size = get_compile_time_arg_val(0);  // size of block == 1 tile
    constexpr uint32_t Wt = get_compile_time_arg_val(1);          // number of tiles in inner dimension

    constexpr uint32_t onetile = 1U;

    const uint32_t tile_bytes = get_tile_size(cb_output_idx);
    const DataFormat data_format = get_dataformat(cb_output_idx);

    const InterleavedAddrGenFast</* is dram */ true> output_addr_generator = {
        .bank_base_address = output_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t end_row = start_row + num_rows_to_process;

    for (uint32_t r = start_row; r < end_row; r++) {
        // uint32_t idx = r * Wt;
        cb_wait_front(cb_output_idx, onetile);                // wait until cb has block_size tiles(1 tile)
        uint32_t l1_read_addr = get_read_ptr(cb_output_idx);  // get the address output buffer

        // write block_size tiles to output buffer
        noc_async_write_tile(r, output_addr_generator, l1_read_addr);  // write the tile to the output buffer
        noc_async_write_barrier();                                     // wait until all tiles are written
        cb_pop_front(cb_output_idx, onetile);                          // pop the tiles from the cb
    }
}
