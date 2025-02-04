// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

void kernel_main() {
#if NOP_COUNT
    for (int i = 0; i < ITERATIONS; i++) {
#pragma GCC unroll 4096
        for (int j = 0; j < NOP_COUNT; j++) {
            asm("nop");
        }
    }
#else
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
            uint64_t dst_noc_multicast_addr =
                get_noc_multicast_addr(NOC_ADDR_X, NOC_ADDR_Y, MCAST_NOC_END_ADDR_X, MCAST_NOC_END_ADDR_Y, write_ptr);
            noc_async_write_multicast(read_ptr, dst_noc_multicast_addr, page_size, NUM_MCAST_DESTS, LINKED);
#elif WRITE
            uint64_t noc_write_addr = NOC_XY_ADDR(NOC_X(NOC_ADDR_X), NOC_Y(NOC_ADDR_Y), write_ptr);
            noc_async_write(NOC_MEM_ADDR, noc_write_addr, page_size);
#elif READ_ONE_PACKET
            noc_async_read_one_packet(noc_addr, read_ptr, page_size);
#else
            noc_async_read(noc_addr, read_ptr, page_size);
#endif

#if LATENCY
#if WRITE
#if LINKED
            noc_async_write_multicast(cb_addr, dst_noc_multicast_addr, page_size, NUM_MCAST_DESTS, false);
#endif
            noc_async_write_barrier();
#else
            noc_async_read_barrier();
#endif
#endif
            read_ptr += page_size;
            write_ptr += page_size;
        }
    }
#if !LATENCY
#if WRITE
#if LINKED
    uint64_t dst_noc_multicast_addr =
        get_noc_multicast_addr(NOC_ADDR_X, NOC_ADDR_Y, MCAST_NOC_END_ADDR_X, MCAST_NOC_END_ADDR_Y, cb_addr);
    noc_async_write_multicast(cb_addr, dst_noc_multicast_addr, page_size, NUM_MCAST_DESTS, false);
#endif
    noc_async_write_barrier();
#else
    noc_async_read_barrier();
#endif
#endif
#endif
}
