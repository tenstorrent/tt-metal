// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {

    const uint32_t src_addr                             = get_arg_val<uint32_t>(0);
    const uint32_t start_id                             = get_arg_val<uint32_t>(1);

    constexpr bool src0_is_dram                         = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t num_tiles                        = get_compile_time_arg_val(1);
    constexpr uint32_t num_unpadded_tiles_head_dim      = get_compile_time_arg_val(2);
    constexpr uint32_t num_unpadded_tiles_seqlen_dim    = get_compile_time_arg_val(3);
    constexpr uint32_t num_padded_tiles_seqlen_dim      = get_compile_time_arg_val(4);

    constexpr uint32_t cb_id_in0 = 0;

    const uint32_t tile_size = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);


    // In and out are assumed to be same dataformat
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src_addr,
        .page_size = tile_size,
        .data_format = data_format
    };

    uint32_t src_tile_id = start_id;
    cb_reserve_back(cb_id_in0, num_tiles);
    uint32_t src_buffer_l1_addr = get_write_ptr(cb_id_in0);
    uint32_t seqlen_dim_id = 0;

    // method 1
    // uint32_t num_iterations = num_tiles / num_unpadded_tiles_head_dim;
    // for(uint32_t i = 0; i < num_iterations; i++) {
    //     // Copy Input
    //     for (uint32_t j = 0; j < num_unpadded_tiles_head_dim; j++) {
    //         noc_async_read_tile(src_tile_id, s0, src_buffer_l1_addr);
    //         src_buffer_l1_addr += tile_size;
    //         src_tile_id++;
    //     }
    //     seqlen_dim_id++;
    //     if (seqlen_dim_id == num_unpadded_tiles_seqlen_dim) {
    //         seqlen_dim_id = 0;
    //         src_tile_id += num_padded_tiles_seqlen_dim;
    //     }
    // }

    // method 2
    uint32_t internal_tile_id = 0;
    for(uint32_t i = 0; i < num_tiles; i++) {
        // Copy Input
        noc_async_read_tile(src_tile_id, s0, src_buffer_l1_addr);
        src_buffer_l1_addr += tile_size;
        src_tile_id++;
        internal_tile_id++;
        if (internal_tile_id == num_unpadded_tiles_head_dim) {
            internal_tile_id = 0;
            seqlen_dim_id++;
            if (seqlen_dim_id == num_unpadded_tiles_seqlen_dim) {
                seqlen_dim_id = 0;
                src_tile_id += num_padded_tiles_seqlen_dim;
            }
        }
    }

    noc_async_read_barrier();
    cb_push_back(cb_id_in0, num_tiles);
}
