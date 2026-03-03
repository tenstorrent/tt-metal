// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Kernel for testing L2 cache flush operations on Quasar DM cores.
// Writes values to cacheable memory and flushes using various L2 flush functions.

#include "api/dataflow/dataflow_api.h"
#include "risc_common.h"

void kernel_main() {
    // Runtime args
    uint32_t base_addr = get_arg_val<uint32_t>(0);
    uint32_t test_mode = get_arg_val<uint32_t>(1);  // 0=line, 1=range, 2=full

    // Common args
    uint32_t value = get_common_arg_val<uint32_t>(0);
    uint32_t num_words = get_common_arg_val<uint32_t>(1);

    // Write values to cacheable addresses
    volatile uint32_t* ptr = (volatile uint32_t*)base_addr;
    for (uint32_t i = 0; i < num_words; i++) {
        ptr[i] = value + i;
    }

    // Flush based on test mode
    switch (test_mode) {
        case 0:
            // Single line flush - flush each cache line individually
            for (uint32_t i = 0; i < num_words; i++) {
                flush_l2_cache_line(base_addr + i * sizeof(uint32_t));
            }
            break;
        case 1:
            // Range flush
            flush_l2_cache_range(base_addr, num_words * sizeof(uint32_t));
            break;
        case 2:
            // Full cache flush
            flush_l2_cache_full();
            break;
    }
}
