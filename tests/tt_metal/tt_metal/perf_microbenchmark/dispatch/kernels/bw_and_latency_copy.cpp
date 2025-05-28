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
    uint32_t read_ptr = cb_addr;
    uint32_t write_ptr = cb_addr;
    uint64_t noc_addr = NOC_XY_ADDR(NOC_X(NOC_ADDR_X), NOC_Y(NOC_ADDR_Y), NOC_MEM_ADDR);

    for (int i = 0; i < ITERATIONS; i++) {
        for (int j = 0; j < PAGE_COUNT; j++) {
            noc_async_read(noc_addr, read_ptr, page_size);
            noc_async_read_barrier();
            read_ptr += page_size;
        }
    }
}
