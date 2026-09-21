// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)

#include <stddef.h>
#include <stdint.h>

#include "internal/tt-2xx/quasar/overlay/overlay_addresses.h"

// Keep these primitives independent of assertion and runtime headers so they can
// also flush Watcher diagnostics from the assertion handler.

// Flush a single 64B cache line from L2 to TL1 (node memory).
// Probes L1 D$ for dirty data before flushing - no need to flush L1 first.
inline __attribute__((always_inline)) void flush_l2_cache_line(uintptr_t addr) {
    __asm__ __volatile__("fence" ::: "memory");
    volatile uint64_t* flush_reg = (volatile uint64_t*)L2_FLUSH_ADDR;
    *flush_reg = (uint64_t)addr;
    __asm__ __volatile__("fence" ::: "memory");
}

// Flush a range of addresses from L2 to TL1.
// Flushes all cache lines covering [start_addr, start_addr + size).
inline __attribute__((always_inline)) void flush_l2_cache_range(uintptr_t start_addr, size_t size) {
    if (size == 0) {
        return;
    }
    uintptr_t aligned_start = start_addr & ~(uintptr_t)63;  // align to 64B
    uintptr_t end_addr = start_addr + size;

    __asm__ __volatile__("fence" ::: "memory");
    volatile uint64_t* flush_reg = (volatile uint64_t*)L2_FLUSH_ADDR;
    for (uintptr_t addr = aligned_start; addr < end_addr; addr += 64) {
        *flush_reg = (uint64_t)addr;
    }
    __asm__ __volatile__("fence" ::: "memory");
}

#endif  // ARCH_QUASAR && COMPILE_FOR_DM
