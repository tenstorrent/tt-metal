// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t src_addr           = get_arg_val<uint32_t>(0);
    uint32_t src_noc_x          = get_arg_val<uint32_t>(1);
    uint32_t src_noc_y          = get_arg_val<uint32_t>(2);
    uint32_t src_buffer_size    = get_arg_val<uint32_t>(3);

    uint32_t local_addr         = get_arg_val<uint32_t>(4);

    uint32_t dst_addr           = get_arg_val<uint32_t>(5);
    uint32_t dst_noc_x_start    = get_arg_val<uint32_t>(6);
    uint32_t dst_noc_y_start    = get_arg_val<uint32_t>(7);
    uint32_t dst_noc_x_end      = get_arg_val<uint32_t>(8);
    uint32_t dst_noc_y_end      = get_arg_val<uint32_t>(9);
    uint32_t num_dests          = get_arg_val<uint32_t>(10);
    uint32_t exclude_start_x    = get_arg_val<uint32_t>(11);
    uint32_t exclude_start_y    = get_arg_val<uint32_t>(12);
    uint32_t exclude_dir_x      = get_arg_val<uint32_t>(13);
    uint32_t exclude_dir_y      = get_arg_val<uint32_t>(14);


    // Read src buffer into local L1 buffer
    uint64_t src_buffer_noc_addr = get_noc_addr(src_noc_x, src_noc_y, src_addr);
    noc_async_read(src_buffer_noc_addr, local_addr, src_buffer_size);
    noc_async_read_barrier();

    // multicast local L1 buffer to all destination cores
    uint64_t dst_noc_multicast_addr = get_noc_multicast_addr(
        dst_noc_x_start,
        dst_noc_y_start,
        dst_noc_x_end,
        dst_noc_y_end,
        dst_addr);
    uint32_t noc_exclude_addr = get_noc_exclude_addr(
        exclude_start_x, 
        exclude_start_y,
        exclude_dir_x,
        exclude_dir_y);
    noc_async_write_multicast_exclude_region(local_addr, dst_noc_multicast_addr, src_buffer_size, num_dests, noc_exclude_addr);
    noc_async_write_barrier();
}
