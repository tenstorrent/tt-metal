// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

//#include "debug/dprint.h"

void kernel_main() {
    const uint32_t src_addr  = get_arg_val<uint32_t>(0);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(1);
    const uint32_t block_width_tiles = get_arg_val<uint32_t>(2);
    const uint32_t input_width_offset_tiles = get_arg_val<uint32_t>(3); // input width in tiles - block width in tiles
    const uint32_t block_num_tiles = get_arg_val<uint32_t>(4); // block_height_tiles * block_width_tiles
    const uint32_t start_id = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr bool src_is_dram = get_compile_time_arg_val(1) == 1;

    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    uint32_t curr_tile_id = start_id;
    cb_reserve_back(cb_id_in0, block_num_tiles);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
    for (uint32_t h = 0; h < block_height_tiles; h++) {
        uint32_t tile_id = curr_tile_id;
        for (uint32_t w = 0; w < block_width_tiles; w++) {
            noc_async_read_tile(tile_id, s, l1_write_addr);
            tile_id++;
            l1_write_addr += tile_bytes;
            noc_async_read_barrier();
        }
        curr_tile_id += input_width_offset_tiles;
    } 
    cb_push_back(cb_id_in0, block_num_tiles);
}
