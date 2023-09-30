// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    // in0 tensor args
    uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t in0_tensor_stride_w = get_arg_val<uint32_t>(2);
    uint32_t in0_tensor_stride_h = get_arg_val<uint32_t>(3);
    uint32_t in0_tensor_next_block_stride = get_arg_val<uint32_t>(4);

    // in0 block args
    uint32_t in0_block_w = get_arg_val<uint32_t>(5);
    uint32_t in0_block_h = get_arg_val<uint32_t>(6);
    uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(7);

    // in0/in1 common args
    uint32_t num_blocks = get_arg_val<uint32_t>(8);

    uint32_t noc_x = get_arg_val<uint32_t>(9);
    uint32_t noc_y = get_arg_val<uint32_t>(10);
    uint32_t last_block_h = get_arg_val<uint32_t>(11);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in2 = 2;

    // Fill tile with zeros
    cb_reserve_back(cb_id_in2, 1);
    uint32_t l1_zeros_addr_in2_noc = get_noc_addr(get_write_ptr(cb_id_in2));

    // in0 reader
    const uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    uint32_t l1_write_addr_in0;

    uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
    for (uint32_t block = 0; block < num_blocks; block++) {
        cb_reserve_back(cb_id_in0, in0_block_num_tiles);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);

        uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
        for (uint32_t h = 0; h < in0_block_h; h++) {
            uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;

            for (uint32_t w = 0; w < in0_block_w; w++) {
                uint32_t l1_buffer_addr = in0_tensor_addr + (in0_tensor_tile_id * in0_single_tile_size_bytes);
                uint64_t l1_buffer_noc_addr = get_noc_addr(noc_x, noc_y, l1_buffer_addr);
                if (h < last_block_h) {
                    noc_async_read(l1_buffer_noc_addr, l1_write_addr_in0, in0_single_tile_size_bytes);
                } else {
                    noc_async_read(l1_zeros_addr_in2_noc, l1_write_addr_in0, in0_single_tile_size_bytes);
                }
                l1_write_addr_in0 += in0_single_tile_size_bytes;
                in0_tensor_tile_id += in0_tensor_stride_w;
            }
            in0_tensor_row_start_tile_id += in0_tensor_stride_h;
        }

        // We commented this line to reuse the first block of in0
        // in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride;
        noc_async_read_barrier();

        cb_push_back(cb_id_in0, in0_block_num_tiles);
    }
}
