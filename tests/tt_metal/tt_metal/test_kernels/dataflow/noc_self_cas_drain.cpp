// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A NoC CAS spin-lock (0 free, 1 held) around a check-then-decrement.
// Mode 0 is a pure drain: every user hart decrements under the lock. Mode 1 pairs the
// harts: evens produce with unlocked increments, odds consume under the lock.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

#if defined(ARCH_QUASAR)

inline volatile tt_l1_ptr uint32_t* uncached(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(addr) + MEM_L1_UNCACHED_BASE);
}

void kernel_main() {
    const uint32_t sem_addr = get_arg(args::sem_addr);
    const uint32_t lock_addr = get_arg(args::lock_addr);
    const uint32_t ret_base = get_arg(args::ret_base);
    const uint32_t increment_times = get_arg(args::increment_times);
    const uint32_t mode = get_arg(args::mode);
    const uint32_t pairs = get_arg(args::pairs);

    uint64_t hart;
    asm volatile("csrr %0, mhartid" : "=r"(hart));
    const uint32_t user_idx = static_cast<uint32_t>(hart) - 2;

    constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
    const uint32_t ret_slot = ret_base + static_cast<uint32_t>(hart) * 4;
    const uint64_t lock_noc_addr = get_noc_addr(lock_addr);
    const uint64_t sem_noc_addr = get_noc_addr(sem_addr);

    // One CAS on the lock word; returns its pre-op value (0 or 1).
    auto lock_cas = [&](uint32_t cmp, uint32_t swap) -> uint32_t {
        *uncached(ret_slot) = SENTINEL;
        noc_fast_atomic_cas4<DM_DEDICATED_NOC>(noc_index, lock_noc_addr, NOC_UNICAST_WRITE_VC, cmp, swap, ret_slot);
        noc_async_atomic_barrier();
        while (*uncached(ret_slot) == SENTINEL) {
        }
        return *uncached(ret_slot);
    };

    // An increment also returns its pre-op word to ret_slot, consume it.
    auto consumed_inc = [&](uint64_t noc_addr, uint32_t incr) {
        *uncached(ret_slot) = SENTINEL;
        noc_semaphore_inc(noc_addr, incr);
        noc_async_atomic_barrier();
        while (*uncached(ret_slot) == SENTINEL) {
        }
    };

    // Release: pre-op must be 1 (we held the lock); anything else means the lock broke,
    // so poison the count to fail the host's exact-0 check.
    auto release_lock = [&]() {
        if (lock_cas(1 /*cmp*/, 0 /*swap*/) != 1) {
            consumed_inc(sem_noc_addr, 0x10000001u);
        }
    };

    // One lock-protected conditional decrement
    auto locked_decrement = [&]() {
        while (true) {
            while (*uncached(sem_addr) < 1) {
            }
            if (lock_cas(0 /*cmp*/, 1 /*swap*/) != 0) {
                continue;
            }
            if (*uncached(sem_addr) < 1) {
                release_lock();
                continue;
            }
            consumed_inc(sem_noc_addr, (uint32_t)(-1));
            release_lock();
            return;
        }
    };

    if (mode == 0) {
        // Only drains
        for (uint32_t i = 0; i < increment_times; i++) {
            locked_decrement();
        }
    } else {
        // Mixed
        if (user_idx / 2 >= pairs) {
            return;
        }
        if ((user_idx & 1) == 0) {
            // Producer: unlocked increments
            for (uint32_t i = 0; i < increment_times; i++) {
                noc_semaphore_inc(sem_noc_addr, 1);
                noc_async_atomic_barrier();
            }
        } else {
            // Consumer: lock-protected decrements
            for (uint32_t i = 0; i < increment_times; i++) {
                locked_decrement();
            }
        }
    }
}

#else

void kernel_main() {}

#endif
