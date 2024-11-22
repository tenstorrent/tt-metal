// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    std::uint32_t dram_buffer_src_addr    = get_arg_val<uint32_t>(0);
    std::uint32_t src_bank_id             = get_arg_val<uint32_t>(1);
    std::uint32_t l1_buffer_src_addr_base    = get_arg_val<uint32_t>(2);
    std::uint32_t l1_buffer_dst_addr_base    = get_arg_val<uint32_t>(3);
    std::uint32_t l1_dst_noc_x        = get_arg_val<uint32_t>(4);
    std::uint32_t l1_dst_noc_y        = get_arg_val<uint32_t>(5);
    std::uint32_t num_tiles                  = get_arg_val<uint32_t>(6);
    std::uint32_t single_tile_size_bytes     = get_arg_val<uint32_t>(7);
    std::uint32_t total_tiles_size_bytes     = get_arg_val<uint32_t>(8);

     // DRAM NOC src address
    std::uint64_t dram_buffer_src_noc_addr = get_noc_addr_from_bank_id<true>(src_bank_id, dram_buffer_src_addr);
    noc_async_read(dram_buffer_src_noc_addr, l1_buffer_src_addr_base, total_tiles_size_bytes);
    noc_async_read_barrier();

    for (uint32_t i = 0; i < 1000; i++) {
        // L1 NOC dst address
        std::uint64_t l1_buffer_dst_noc_addr = get_noc_addr(l1_dst_noc_x, l1_dst_noc_y, l1_buffer_dst_addr_base);
        noc_async_write(l1_buffer_src_addr_base, l1_buffer_dst_noc_addr, total_tiles_size_bytes);
        noc_async_write_barrier();
    }
}
