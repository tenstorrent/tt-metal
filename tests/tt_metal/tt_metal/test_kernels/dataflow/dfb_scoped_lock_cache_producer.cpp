// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Validates that DataflowBuffer::scoped_lock() applies the Quasar L2 cache ops to *exactly the held
// entries it walks*: invalidate on acquire (both roles) and flush on release (producer only). The
// held entries are wr_ptr + k*stride_size for k in [0, lock_n) — i.e. STRIDED neighbours must NOT be
// touched.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "dev_mem_map.h"
#include "risc_common.h"

void kernel_main() {
    DataflowBuffer dfb(dfb::out);
    const uint32_t is_active = get_arg(args::is_active);
    const uint32_t mode = get_arg(args::test_mode);           // 0 = flush-on-release, 1 = invalidate-on-acquire
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
    // PUBLISHER (1-producer / multi-consumer ALL invalidate test). The single producer thread seeds the
    // shared stale state for every ring slot — OLD in the shared L2 (read it in) and NEW in TL1 (uncached
    // write) — then releases the consumers via a semaphore. The consumers then concurrently invalidate
    // the SHARED held entries (the ALL redundant-invalidate path) and each verifies it reads fresh.
    (void)is_active;
    (void)mode;
    (void)lock_n;
    (void)result_uncached;
    Semaphore release(sem::release);
    invalidate_l2_cache_range(ring_base, num_entries * entry_size);
    for (uint32_t s = 0; s < num_entries; ++s) {
        volatile uint32_t v = cached[s * wpe];  // OLD (host pre-filled TL1) -> shared L2
        (void)v;
    }
    for (uint32_t s = 0; s < num_entries; ++s) {
        uncached[s * wpe] = new_val + s;  // TL1 = NEW; shared L2 still OLD
    }
    release.up(1);  // release the waiting consumers
#else
    // Only the chosen role's thread 0 drives the ring + locks; every other endpoint/thread is a no-op
    // that merely satisfies the DFB's producer/consumer binding.
    if (!is_active || get_my_thread_id() != 0) {
        (void)dfb;
        return;
    }

    if (mode == 0) {
        // FLUSH-on-release. scoped_lock invalidates the held entries on construction and flushes on
        // destruction, so the producer must write the data INSIDE the lock. Release then flushes the
        // HELD entries (producer) L2->TL1; non-held stores stay cache-resident -> TL1 stays OLD.
        {
            auto lk = dfb.scoped_lock(lock_n);
            for (uint32_t s = 0; s < num_entries; ++s) {
                cached[s * wpe] = new_val + s;
            }
        }
        // Verify in-kernel via the non-cacheable alias (reads TL1): held -> NEW (flushed), non-held -> OLD.
        for (uint32_t s = 0; s < num_entries; ++s) {
            result_uncached[s] = uncached[s * wpe];
        }
    } else {
        // INVALIDATE-on-acquire. Drop any prior cached lines, fetch the host's OLD sentinels into the
        // cache, then overwrite TL1 with NEW via the non-cacheable alias (cache keeps the stale OLD; an uncached
        // write is not snooped). scoped_lock acquire invalidates the HELD entries' lines; a cacheable re-read
        // then misses for held slots (refetching NEW from TL1) and hits stale OLD for the rest. Results go to
        // the scratch region via the uncached alias.
        invalidate_l2_cache_range(ring_base, num_entries * entry_size);
        for (uint32_t s = 0; s < num_entries; ++s) {
            volatile uint32_t v = cached[s * wpe];  // fetch OLD (host pre-filled TL1) into the cache
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
