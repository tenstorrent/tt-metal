// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(1);
    const uint32_t block_width_tiles = get_arg_val<uint32_t>(2);
    const uint32_t output_width_offset_tiles = get_arg_val<uint32_t>(3); // input width in tiles - block width in tiles
    const uint32_t block_num_tiles = get_arg_val<uint32_t>(4); // block_height_tiles * block_width_tiles
    uint32_t start_id = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;

    // single-tile ublocks
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    uint32_t tile_id = start_id;
    cb_wait_front(cb_id_out, block_num_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out);
    for (uint32_t h = 0; h < block_height_tiles; h++) {
        for (uint32_t w = 0; w < block_width_tiles; w++) {
            noc_async_write_tile(tile_id, s, l1_read_addr);
            tile_id++;
            l1_read_addr += tile_bytes;
            noc_async_write_barrier();
        }
        tile_id += output_width_offset_tiles;
    }
    cb_pop_front(cb_id_out, block_num_tiles);
}
