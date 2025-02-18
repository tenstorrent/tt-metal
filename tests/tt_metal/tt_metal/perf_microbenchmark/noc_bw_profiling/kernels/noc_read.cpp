// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    std::uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);
    std::uint32_t dram_buffer_addr = get_arg_val<uint32_t>(1);
    std::uint32_t dst_x = get_arg_val<uint32_t>(2);
    std::uint32_t dst_y = get_arg_val<uint32_t>(3);
    std::uint32_t page_size = get_arg_val<uint32_t>(4);

    uint32_t read_ptr = l1_buffer_addr;
    uint32_t write_ptr = l1_buffer_addr;

    uint64_t noc_addr_bank0 = NOC_XY_ADDR(NOC_X(dst_x), NOC_Y(dst_y), dram_buffer_addr);
    int32_t page_count = 65536 / page_size;

    for (int j = 0; j < page_count; j++) {
        noc_async_write(read_ptr, noc_addr_bank0, page_size);
    }
    noc_async_write_barrier();
}
