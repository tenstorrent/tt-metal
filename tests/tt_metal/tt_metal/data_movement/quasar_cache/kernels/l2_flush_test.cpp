// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Kernel for testing L2 cache operations on Quasar DM cores.
// Writes values to cacheable memory and flushes/invalidates using various L2 functions.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "dev_mem_map.h"
#include "risc_common.h"

void kernel_main() {
    // Runtime args
    uint32_t base_addr = get_arg_val<uint32_t>(0);
    // 0=flush_line, 1=flush_range, 2=flush_full, 3=invalidate_line, 4=invalidate_fresh_read
    uint32_t test_mode = get_arg_val<uint32_t>(1);

    // Common args
    uint32_t value = get_common_arg_val<uint32_t>(0);
    uint32_t num_words = get_common_arg_val<uint32_t>(1);

    DPRINT << "START mode=" << test_mode << " words=" << num_words << ENDL();
    DEVICE_PRINT("START mode={} words={}\n", test_mode, num_words);

    // Write values to cacheable addresses
    volatile uint32_t* ptr = (volatile uint32_t*)(uintptr_t)base_addr;
    for (uint32_t i = 0; i < num_words; i++) {
        ptr[i] = value + i;
    }

    DPRINT << "WRITES DONE" << ENDL();
    DEVICE_PRINT("WRITES DONE\n");

    // Flush/invalidate based on test mode
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
        case 3:
            // Invalidate line - flush L1 to L2, then invalidate L2 (no writeback to TL1)
            flush_l1_dcache(0);  // Flush L1 D$ to L2
            for (uint32_t i = 0; i < num_words; i++) {
                invalidate_l2_cache_line(base_addr + i * sizeof(uint32_t));
            }
            break;

        case 4: {
            // Test that invalidation causes fresh read from TL1:
            // 1. Read from cacheable address (caches in L2)
            // 2. Write new value via uncached address (directly to TL1)
            // 3. Invalidate L2 line
            // 4. Read again - should get new value
            uint32_t uncached_addr = base_addr + MEM_L1_UNCACHED_BASE;
            volatile uint32_t* uncached_ptr = (volatile uint32_t*)(uintptr_t)uncached_addr;

            for (uint32_t i = 0; i < num_words; i++) {
                // Read to cache the old value (host pre-populated)
                volatile uint32_t cached_val = ptr[i];
                (void)cached_val;

                // Write new value directly to TL1 via uncached alias
                uncached_ptr[i] = value + i;

                // Invalidate the L2 cache line
                invalidate_l2_cache_line(base_addr + i * sizeof(uint32_t));
            }
            // Now when host reads, it should see the new values
            // (they were written via uncached path, L2 was invalidated so no stale data)
            break;
        }

        default:
            ASSERT(false);
            while (1);
    }

    DPRINT << "DONE" << ENDL();
    DEVICE_PRINT("DONE\n");
}
