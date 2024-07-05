// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t out_num_blocks_w_per_core = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t out_num_blocks_h = get_arg_val<uint32_t>(3);
    uint32_t out_total_blocks_w = get_arg_val<uint32_t>(4);


    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;


    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    for(uint32_t block_h_id = 0; block_h_id < out_num_blocks_h; block_h_id++){
        uint32_t end_id = start_id + out_num_blocks_w_per_core;
        for (uint32_t i = start_id; i < end_id; ++ i) {
            cb_wait_front(cb_id_out, onetile);
            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
            noc_async_write_tile((block_h_id * out_total_blocks_w) + i, s, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_out, onetile);
        }
    }
}
