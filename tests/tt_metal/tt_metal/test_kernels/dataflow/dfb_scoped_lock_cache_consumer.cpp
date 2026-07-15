// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "dev_mem_map.h"
#include "risc_common.h"
#include "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_scoped_lock_cache_common.h"

void kernel_main() {
    DataflowBuffer dfb(dfb::in);
    const uint32_t is_active = get_arg(args::is_active);
    const auto mode =
        static_cast<DfbCacheTestMode>(get_arg(args::test_mode));  // flush-on-release vs invalidate-on-acquire
    const uint32_t lock_n = get_arg(args::lock_n);            // entries to lock (held window)
    const uint32_t num_entries = get_arg(args::num_entries);  // total ring entries to drive
    const uint32_t ring_base = get_arg(args::ring_base);      // cacheable L1 byte addr of the DFB ring
    const uint32_t result_addr = get_arg(args::result_addr);  // cacheable L1 byte addr for per-slot results
    const uint32_t new_val = get_arg(args::new_val);

    const uint32_t entry_size = dfb.get_entry_size();
    const uint32_t wpe = entry_size / sizeof(uint32_t);  // words per entry
    volatile uint32_t* cached = (volatile uint32_t*)(uintptr_t)ring_base;
    volatile uint32_t* uncached = (volatile uint32_t*)(uintptr_t)(ring_base + MEM_L1_UNCACHED_BASE);
    volatile uint32_t* result_uncached = (volatile uint32_t*)(uintptr_t)(result_addr + MEM_L1_UNCACHED_BASE);

#if defined(DFB_CACHE_MULTI_ALL)
    // WAITER (1-producer / multi-consumer ALL invalidate test). ALL N consumer threads wait on the
    // publisher (producer) to seed the shared stale state (shared L2 = OLD, TL1 = NEW), then concurrently
    // invalidate the SHARED held entries via scoped_lock and re-read into their own result block.
    (void)is_active;
    (void)mode;
    (void)new_val;
    (void)uncached;
    (void)result_uncached;
    const uint32_t cidx = get_my_thread_id();  // 0 .. num_consumers-1
    Semaphore release(sem::release);
    release.wait(1);  // wait until the publisher has seeded shared-L2=OLD / TL1=NEW
    {
        auto lk = dfb.scoped_lock(lock_n);
    }  // acquire invalidates the held (shared) entries
    volatile uint32_t* my_result =
        (volatile uint32_t*)(uintptr_t)(result_addr + cidx * num_entries * sizeof(uint32_t) + MEM_L1_UNCACHED_BASE);
    for (uint32_t s = 0; s < num_entries; ++s) {
        my_result[s] = cached[s * wpe];  // held -> NEW (invalidated->refetch), non-held -> OLD (shared-L2 hit)
    }
#else
    if (!is_active || get_my_thread_id() != 0) {
        (void)dfb;
        return;
    }

    if (mode == DfbCacheTestMode::FlushOnRelease) {
        // FLUSH-on-release through a CONSUMER lock. Write inside the lock (after acquire-invalidate); a
        // CONSUMER release does NOT flush, so every slot stays cache-resident and TL1 stays OLD for all.
        {
            auto lk = dfb.scoped_lock(lock_n);
            for (uint32_t s = 0; s < num_entries; ++s) {
                cached[s * wpe] = new_val + s;
            }
        }
        // Verify in-kernel via the non-cacheable alias (reads TL1): all slots OLD (consumer never flushes).
        for (uint32_t s = 0; s < num_entries; ++s) {
            result_uncached[s] = uncached[s * wpe];
        }
    } else {
        // INVALIDATE-on-acquire: both roles invalidate, so the held entries' cache lines are discarded.
        invalidate_l2_cache_range(ring_base, num_entries * entry_size);
        for (uint32_t s = 0; s < num_entries; ++s) {
            volatile uint32_t v = cached[s * wpe];  // fetch OLD (host pre-filled TL1) into the coherent cache
            (void)v;
        }
        for (uint32_t s = 0; s < num_entries; ++s) {
            uncached[s * wpe] = new_val + s;  // TL1 = NEW; cache still OLD (uncached write not snooped)
        }
        {
            auto lk = dfb.scoped_lock(lock_n);
        }  // acquire invalidates held entries' cache lines
        for (uint32_t s = 0; s < num_entries; ++s) {
            result_uncached[s] = cached[s * wpe];  // held -> NEW (invalidated->refetch), others -> OLD (hit)
        }
    }
#endif  // DFB_CACHE_MULTI_ALL
}
