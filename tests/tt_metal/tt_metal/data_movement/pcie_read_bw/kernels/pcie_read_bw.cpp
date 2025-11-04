// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t test_id = get_arg_val<uint32_t>(0);
    uint32_t page_size = get_arg_val<uint32_t>(1);

    cb_reserve_back(0, PAGE_COUNT);
    uint32_t cb_addr = get_write_ptr(0);
    {
        DeviceZoneScopedN("RISCV0");
        for (int i = 0; i < ITERATIONS; i++) {
            uint32_t read_ptr = cb_addr;

            for (int j = 0; j < PAGE_COUNT; j++) {
                // Read from PCIe memory
                uint64_t noc_addr = NOC_XY_ADDR(NOC_X(NOC_ADDR_X), NOC_Y(NOC_ADDR_Y), NOC_MEM_ADDR);
                noc_async_read(noc_addr, read_ptr, page_size);

                read_ptr += page_size;
            }
        }

        // Wait for all reads to complete
        noc_async_read_barrier();
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of transactions", PAGE_COUNT * ITERATIONS);
    DeviceTimestampedData("Transaction size in bytes", page_size);
}
