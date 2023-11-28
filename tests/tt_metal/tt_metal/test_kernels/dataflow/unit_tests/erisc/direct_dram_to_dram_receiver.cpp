// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    std::uint32_t dram_buffer_dst_addr = get_arg_val<uint32_t>(0);
    std::uint32_t dram_dst_noc_x = get_arg_val<uint32_t>(1);
    std::uint32_t dram_dst_noc_y = get_arg_val<uint32_t>(2);
    std::uint32_t remaining_bytes = get_arg_val<uint32_t>(3);
    std::uint32_t num_loops = get_arg_val<uint32_t>(4);
    std::uint32_t num_bytes = get_arg_val<uint32_t>(5);

    // DRAM NOC dst address
    for (uint32_t i = 0; i < num_loops; i++) {
        eth_wait_for_bytes(num_bytes);

        std::uint64_t dram_buffer_dst_noc_addr = get_noc_addr(dram_dst_noc_x, dram_dst_noc_y, dram_buffer_dst_addr);
        noc_async_write(eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE, dram_buffer_dst_noc_addr, num_bytes);
        noc_async_write_barrier();

        eth_receiver_done();
        dram_buffer_dst_addr += num_bytes;
    }

    eth_wait_for_bytes(remaining_bytes);
    std::uint64_t dram_buffer_dst_noc_addr = get_noc_addr(dram_dst_noc_x, dram_dst_noc_y, dram_buffer_dst_addr);
    noc_async_write(eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE, dram_buffer_dst_noc_addr, remaining_bytes);
    noc_async_write_barrier();

    eth_receiver_done();
}
