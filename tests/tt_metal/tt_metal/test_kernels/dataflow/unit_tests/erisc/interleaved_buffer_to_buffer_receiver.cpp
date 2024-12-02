// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t page_size = get_arg_val<uint32_t>(1);
    uint32_t max_buffer_size = get_arg_val<uint32_t>(2);
    uint32_t num_loops = get_arg_val<uint32_t>(3);
    uint32_t pages_per_loop = get_arg_val<uint32_t>(4);
    uint32_t remaining_bytes = get_arg_val<uint32_t>(5);
    uint32_t remaining_pages = get_arg_val<uint32_t>(6);

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;

    const InterleavedAddrGen<dst_is_dram> s = {.bank_base_address = dst_addr, .page_size = page_size};

    uint32_t page_idx = 0;
    for (uint32_t i = 0; i < num_loops; ++i) {
        uint32_t l1_read_addr = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
        eth_wait_for_bytes(max_buffer_size);

        for (uint32_t j = 0; j < pages_per_loop; ++j) {
            uint64_t dst_noc_addr = get_noc_addr(page_idx, s);
            noc_async_write(l1_read_addr, dst_noc_addr, page_size);
            page_idx++;
            l1_read_addr += page_size;
            noc_async_write_barrier();
        }
        eth_receiver_done();
    }
    if (remaining_bytes > 0) {
        uint32_t l1_read_addr = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
        eth_wait_for_bytes(remaining_bytes);
        for (uint32_t j = 0; j < remaining_pages; ++j) {
            uint64_t dst_noc_addr = get_noc_addr(page_idx, s);
            noc_async_write(l1_read_addr, dst_noc_addr, page_size);
            page_idx++;
            l1_read_addr += page_size;
            noc_async_write_barrier();
        }
        eth_receiver_done();
    }
}
