// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

void kernel_main() {
    cb_reserve_back(0, PAGE_COUNT);
    uint32_t cb_addr = get_write_ptr(0);
    uint64_t noc_addr = NOC_XY_ADDR(NOC_X(NOC_ADDR_X), NOC_Y(NOC_ADDR_Y), NOC_MEM_ADDR);
    for (int i = 0; i < ITERATIONS; i++) {
        uint32_t read_ptr = cb_addr;
        for (int j = 0; j < PAGE_COUNT; j++) {
#if READ_ONE_PACKET
            noc_async_read_one_packet(noc_addr, read_ptr, PAGE_SIZE);
#else
            noc_async_read(noc_addr, read_ptr, PAGE_SIZE);
#endif
#if LATENCY
            noc_async_read_barrier();
#endif
            read_ptr += PAGE_SIZE;
        }
    }
#if LATENCY
    noc_async_read_barrier();
#endif
}
