// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * kernels/shared/tensix_live_timer.h
 *
 * Concurrent-safe live-high/low/live-high 64-bit hardware timestamper reader.
 * Specialized at compile time for RV32 local MMIO or RV64 remote window MMIO.
 * No virtual dispatch or function pointers in device code.
 */
#ifndef X280_KERNELS_SHARED_TENSIX_LIVE_TIMER_H
#define X280_KERNELS_SHARED_TENSIX_LIVE_TIMER_H

#include <stdint.h>
#include <stddef.h>

/* Debug timestamper live words and the 2 MiB NOC page containing them. */
#define TENSIX_TIMER_LOW_OFFSET 0x001121F0u
#define TENSIX_TIMER_HIGH_OFFSET 0x001121F4u
#define TENSIX_TIMER_LOCAL_LOW_ADDR 0xFFB121F0u
#define TENSIX_TIMER_LOCAL_HIGH_ADDR 0xFFB121F4u
#define TENSIX_TIMER_NOC_PAGE_ADDR UINT64_C(0xFFA00000)

#if !defined(__riscv)
#ifdef __cplusplus
extern "C" {
#endif
/* Hook for host-only unit testing */
extern uint32_t (*g_tensix_live_timer_mock_read)(uintptr_t addr);
#ifdef __cplusplus
}
#endif
#endif

/**
 * Read the 64-bit live timestamper using high1/low/high2 consistency check.
 *
 * @param low_addr      Address of the 32-bit low word (high word must be at low_addr + 4).
 * @param max_attempts  Maximum retry iterations if high word rolls over during reading.
 * @param ticks         Receives the 64-bit timestamp on success. Unmodified on failure.
 * @return 1 on success, 0 on retry exhaustion.
 */
static inline int tensix_live_timer_read(uintptr_t low_addr, uint32_t max_attempts, uint64_t* ticks) {
#if defined(__riscv) && __riscv_xlen == 64
    // RV64 X280 remote window MMIO access
    volatile void* p_low = (volatile void*)low_addr;
    volatile void* p_high = (volatile void*)(low_addr + 4u);

    __asm__ volatile("fence iorw, iorw" ::: "memory");
    for (uint32_t attempt = 0; attempt < max_attempts; ++attempt) {
        uint32_t high1 = *(volatile const uint32_t*)p_high;
        __asm__ volatile("fence ir, ir" ::: "memory");
        uint32_t low = *(volatile const uint32_t*)p_low;
        __asm__ volatile("fence ir, ir" ::: "memory");
        uint32_t high2 = *(volatile const uint32_t*)p_high;

        if (high1 == high2) {
            __asm__ volatile("fence iorw, iorw" ::: "memory");
            *ticks = (((uint64_t)high1) << 32) | (uint64_t)low;
            return 1;
        }
    }
    __asm__ volatile("fence iorw, iorw" ::: "memory");
    return 0;

#elif defined(__riscv) && __riscv_xlen == 32
    // RV32 Tensix BabyRISCV local MMIO access
    const volatile uint32_t* p_low = (volatile const uint32_t*)low_addr;
    const volatile uint32_t* p_high = (volatile const uint32_t*)(low_addr + 4u);

    __asm__ volatile("fence iorw, iorw" ::: "memory");
    for (uint32_t attempt = 0; attempt < max_attempts; ++attempt) {
        uint32_t high1 = *p_high;
        __asm__ volatile("" ::: "memory");
        uint32_t low = *p_low;
        __asm__ volatile("" ::: "memory");
        uint32_t high2 = *p_high;

        if (high1 == high2) {
            __asm__ volatile("fence iorw, iorw" ::: "memory");
            *ticks = (((uint64_t)high1) << 32) | (uint64_t)low;
            return 1;
        }
    }
    __asm__ volatile("fence iorw, iorw" ::: "memory");
    return 0;

#else
    // Host / simulation fallback with optional mock
    if (g_tensix_live_timer_mock_read != NULL) {
        for (uint32_t attempt = 0; attempt < max_attempts; ++attempt) {
            uint32_t high1 = g_tensix_live_timer_mock_read(low_addr + 4u);
            uint32_t low = g_tensix_live_timer_mock_read(low_addr);
            uint32_t high2 = g_tensix_live_timer_mock_read(low_addr + 4u);
            if (high1 == high2) {
                *ticks = (((uint64_t)high1) << 32) | (uint64_t)low;
                return 1;
            }
        }
        return 0;
    }

    const volatile uint32_t* p_low = (volatile const uint32_t*)low_addr;
    const volatile uint32_t* p_high = (volatile const uint32_t*)(low_addr + 4u);

    for (uint32_t attempt = 0; attempt < max_attempts; ++attempt) {
        uint32_t high1 = *p_high;
        uint32_t low = *p_low;
        uint32_t high2 = *p_high;
        if (high1 == high2) {
            *ticks = (((uint64_t)high1) << 32) | (uint64_t)low;
            return 1;
        }
    }
    return 0;
#endif
}

#endif /* X280_KERNELS_SHARED_TENSIX_LIVE_TIMER_H */
