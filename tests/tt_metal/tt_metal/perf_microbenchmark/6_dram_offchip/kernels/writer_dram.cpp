// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t input_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t num_blocks = get_arg_val<uint32_t>(2);
    uint32_t num_cb_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id = 0;
    constexpr bool is_dram = true;
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id);
    const DataFormat data_format = get_dataformat(cb_id);

    const InterleavedAddrGenFast<is_dram> s = {
        .bank_base_address = input_addr,
        .page_size = single_tile_size_bytes,
        .data_format = data_format,
    };

    uint32_t block_size = num_cb_tiles;
    cb_reserve_back(cb_id, num_cb_tiles);
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        uint32_t cb_addr = get_write_ptr(cb_id);
        for (uint32_t i = 0; i < block_size; ++i) {
            noc_async_write_tile(input_start_tile_id, s, cb_addr);
            cb_addr += single_tile_size_bytes;
            input_start_tile_id++;
        }
        noc_async_read_barrier();
    }
}
