// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

void kernel_main() {
#ifdef PAGE_SIZE
    uint32_t page_size = PAGE_SIZE;
#else
    uint32_t page_size = get_arg_val<uint32_t>(0);
#endif

    cb_reserve_back(0, PAGE_COUNT);
    uint32_t cb_addr = get_write_ptr(0);
    for (int i = 0; i < ITERATIONS; i++) {
        uint32_t read_ptr = cb_addr;
        uint32_t write_ptr = cb_addr;
        for (int j = 0; j < PAGE_COUNT; j++) {

#if DRAM_BANKED
            uint64_t noc_addr = get_dram_noc_addr(j, page_size, 0);
#else
            uint64_t noc_addr = NOC_XY_ADDR(NOC_X(NOC_ADDR_X), NOC_Y(NOC_ADDR_Y), NOC_MEM_ADDR);
#endif

#if ISSUE_MCAST
            const uint32_t num_dests = (MCAST_NOC_END_ADDR_Y - NOC_ADDR_Y + 1) * (MCAST_NOC_END_ADDR_X - NOC_ADDR_X + 1);
            uint64_t dst_noc_multicast_addr =
                get_noc_multicast_addr(NOC_ADDR_X, NOC_ADDR_Y, MCAST_NOC_END_ADDR_X, MCAST_NOC_END_ADDR_Y, NOC_MEM_ADDR);
            noc_async_write_multicast(write_ptr, dst_noc_multicast_addr, page_size, num_dests);
#elif READ_ONE_PACKET
            noc_async_read_one_packet(noc_addr, read_ptr, page_size);
#else
            noc_async_read(noc_addr, read_ptr, page_size);
#endif

#if LATENCY
            noc_async_read_barrier();
            noc_async_write_barrier();
#endif

            read_ptr += page_size;
            write_ptr += page_size;
        }
    }
#if !LATENCY
    noc_async_read_barrier();
    noc_async_write_barrier();
#endif
}
