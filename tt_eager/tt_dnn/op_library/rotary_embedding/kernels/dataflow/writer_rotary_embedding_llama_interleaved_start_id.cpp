// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {

    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles_written = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t num_tiles =  get_compile_time_arg_val(2); // The number of tiles outputted by each core

    // single-tile ublocks
    constexpr uint32_t onetile = 1;

    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    uint32_t end_id = num_tiles_written + num_tiles; // num_tiles = (num_rows / total_cores) * Wt

    const uint32_t granularity = 4;
    constexpr uint32_t loop_count = num_tiles/granularity;
    uint32_t write_id = num_tiles_written;

    for (uint32_t i = 0; i < loop_count; ++i) {
        cb_wait_front(cb_id_out, granularity);

        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        for (uint32_t j = 0; j < granularity; ++j) {
            noc_async_write_tile(write_id++, s, l1_read_addr);
            l1_read_addr += tile_bytes;
        }

        noc_async_write_barrier();
        cb_pop_front(cb_id_out, granularity);
    }
}
