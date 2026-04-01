// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Kernel for testing L1 data cache operations on Quasar DM cores.
// Tests flush and invalidate operations with two-stage verification.

#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"
#include "risc_common.h"

void kernel_main() {
    // Runtime args
    uint32_t base_addr = get_arg_val<uint32_t>(0);
    // 0=flush_line, 1=flush_full, 2=invalidate_line, 3=invalidate_full, 4=invalidate_fresh_read
    uint32_t test_mode = get_arg_val<uint32_t>(1);

    // Common args
    uint32_t value = get_common_arg_val<uint32_t>(0);
    uint32_t num_words = get_common_arg_val<uint32_t>(1);

    // Write values to cacheable addresses (goes through L1 D$ -> L2)
    volatile uint32_t* ptr = (volatile uint32_t*)(uintptr_t)base_addr;
    for (uint32_t i = 0; i < num_words; i++) {
        ptr[i] = value + i;
    }

    switch (test_mode) {
        case 0:
            // Flush L1 D$ line(s) to L2, then flush L2 to TL1
            for (uint32_t i = 0; i < num_words; i++) {
                flush_l1_dcache(base_addr + i * sizeof(uint32_t));
            }
            // Now flush L2 to make data visible to host
            flush_l2_cache_range(base_addr, num_words * sizeof(uint32_t));
            break;

        case 1:
            // Flush entire L1 D$ to L2, then flush L2 to TL1
            flush_l1_dcache(0);  // 0 = entire cache
            flush_l2_cache_range(base_addr, num_words * sizeof(uint32_t));
            break;

        case 2:
            // Invalidate L1 D$ line(s) WITHOUT writeback, then flush L2
            // Since L1 didn't write back, L2 should have stale/no data
            // This tests that invalidate does NOT write back
            for (uint32_t i = 0; i < num_words; i++) {
                invalidate_l1_dcache(base_addr + i * sizeof(uint32_t));
            }
            // Flush L2 - should NOT contain the new values since L1 didn't write back
            flush_l2_cache_range(base_addr, num_words * sizeof(uint32_t));
            break;

        case 3:
            // Invalidate entire L1 D$ WITHOUT writeback, then flush L2
            invalidate_l1_dcache(0);  // 0 = entire cache
            flush_l2_cache_range(base_addr, num_words * sizeof(uint32_t));
            break;

        case 4: {
            // Test that invalidation causes fresh read from memory:
            // 1. Read from cacheable address (caches in L1 D$)
            // 2. Write new value via uncached address (directly to TL1)
            // 3. Invalidate L1 D$ and L2 lines
            // 4. Read again - should get new value from TL1
            uint32_t uncached_addr = base_addr + MEM_L1_UNCACHED_BASE;
            volatile uint32_t* uncached_ptr = (volatile uint32_t*)(uintptr_t)uncached_addr;

            for (uint32_t i = 0; i < num_words; i++) {
                // Read to cache the old value (host pre-populated)
                volatile uint32_t cached_val = ptr[i];
                (void)cached_val;

                // Write new value directly to TL1 via uncached alias
                uncached_ptr[i] = value + i;

                // Invalidate both L1 D$ and L2 to ensure fresh read from TL1
                invalidate_l1_dcache(base_addr + i * sizeof(uint32_t));
                invalidate_l2_cache_line(base_addr + i * sizeof(uint32_t));
            }
            // Now when host reads, it should see the new values
            break;
        }

        default:
            ASSERT(false);
            while (1);
    }
}
