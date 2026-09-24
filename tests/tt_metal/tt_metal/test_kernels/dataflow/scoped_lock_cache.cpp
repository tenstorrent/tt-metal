// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/kernel_args.h"
#include "dev_mem_map.h"  // MEM_L1_UNCACHED_BASE
#include "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_cache_common.h"

#if defined(USE_CORE_LOCAL_MEM)
#include "api/core_local_mem.h"
#elif defined(USE_LOCAL_TENSOR_ACCESSOR)
#include "api/tensor/local_tensor_accessor.h"
#elif !defined(USE_SCRATCHPAD)
#error "scoped_lock_cache.cpp requires one of -DUSE_CORE_LOCAL_MEM / -DUSE_SCRATCHPAD / -DUSE_LOCAL_TENSOR_ACCESSOR"
#endif

void kernel_main() {
    const auto mode = static_cast<ScopedLockCacheMode>(get_arg(args::mode));
    const uint32_t result_addr = get_arg(args::result_addr);
    const uint32_t lock_off_lines = get_arg(args::lock_off_lines);
    const uint32_t lock_n_lines = get_arg(args::lock_n_lines);

    constexpr uint32_t WPL = SCOPED_LOCK_CACHE_WORDS_PER_LINE;
    constexpr uint32_t N = SCOPED_LOCK_CACHE_NUM_LINES;
    const uint32_t lock_off_elems = lock_off_lines * WPL;
    const uint32_t lock_n_elems = lock_n_lines * WPL;

#if defined(USE_CORE_LOCAL_MEM)
    // The lock starts where `mem` points, so a middle chunk means offsetting it. That makes every
    // off>0 case a temporary-pointer test too: operator+ yields a prvalue that dies at the end of the
    // full-expression, so a guard capturing `this` or a reference would dangle.
    const uint32_t region_base = get_arg(args::region_base);
    CoreLocalMem<volatile uint32_t> mem(static_cast<uintptr_t>(region_base));
    auto make_lock = [&](uint32_t off, uint32_t n) { return (mem + off).scoped_lock(n); };
#elif defined(USE_LOCAL_TENSOR_ACCESSOR)
    const uint32_t region_base = get_arg(args::region_base);
    LocalTensorAccessor<volatile uint32_t> acc(region_base);
    auto make_lock = [&](uint32_t off, uint32_t n) { return acc.scoped_lock(off, n); };
#else  // USE_SCRATCHPAD
    // No region_base runtime arg here: the host binds the scratchpad, so its address is only known
    // in-kernel and the probe region is derived from the binding instead.
    Scratchpad<volatile uint32_t> pad(scratch::pad);
    const uint32_t region_base = pad.get_base_address();
    auto make_lock = [&](uint32_t off, uint32_t n) { return pad.scoped_lock(off, n); };
#endif

    volatile uint32_t* cached = reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(region_base));
    volatile uint32_t* uncached =
        reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(region_base) + MEM_L1_UNCACHED_BASE);
    volatile uint32_t* result =
        reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(result_addr) + MEM_L1_UNCACHED_BASE);
    // Establish the OLD baseline in TL1. Written through the uncached alias so it lands in TL1 directly
    // rather than sitting dirty in L2, where only a flush would publish it.
    for (uint32_t l = 0; l < N; ++l) {
        uncached[l * WPL] = SCOPED_LOCK_CACHE_OLD_BASE + l;
    }
    switch (mode) {
        case ScopedLockCacheMode::InvalidateOnAcquire: {
            // An uncached write is not snooped, so TL1 can be moved to NEW while the cache still holds
            // OLD. The acquire invalidates only the LOCKED lines, so the re-read below distinguishes
            // them: locked lines miss and refetch NEW, the rest hit stale OLD.
            for (uint32_t l = 0; l < N; ++l) {
                volatile uint32_t v = cached[l * WPL];
                (void)v;
            }
            for (uint32_t l = 0; l < N; ++l) {
                uncached[l * WPL] = SCOPED_LOCK_CACHE_NEW_BASE + l;
            }
            {
                auto lk = make_lock(lock_off_elems, lock_n_elems);
                (void)lk;
            }
            for (uint32_t l = 0; l < N; ++l) {
                result[l] = cached[l * WPL];
            }
            break;
        }
        case ScopedLockCacheMode::FlushOnRelease: {
            // Every line is stored THROUGH THE CACHE inside the lock, but release writes back only the
            // locked ones -- so TL1 shows NEW exactly where the flush reached, and OLD elsewhere.
            for (uint32_t l = 0; l < N; ++l) {
                volatile uint32_t v = cached[l * WPL];
                (void)v;
            }
            {
                auto lk = make_lock(lock_off_elems, lock_n_elems);
                (void)lk;
                for (uint32_t l = 0; l < N; ++l) {
                    cached[l * WPL] = SCOPED_LOCK_CACHE_NEW_BASE + l;
                }
            }
            for (uint32_t l = 0; l < N; ++l) {
                result[l] = uncached[l * WPL];
            }
            break;
        }
    }
}
