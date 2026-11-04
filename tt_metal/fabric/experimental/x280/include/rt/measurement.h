// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_RT_COUNTER_MEASUREMENT_H
#define X280_RT_COUNTER_MEASUREMENT_H

#include <stdint.h>

/* Same-instance, same-generation modulo-64 interval. The caller must bound
 * duration below half-range independently. Zero is below resolution, not a
 * rate denominator. Returns zero on invalid delta, leaving output unchanged. */
static inline int measurement_delta(uint64_t start, uint64_t end, uint64_t* delta) {
    const uint64_t value = end - start;
    if (value >= (UINT64_C(1) << 63)) {
        return 0;
    }
    *delta = value;
    return 1;
}

#if defined(__riscv) && __riscv_xlen == 32
/* Per-RISC native cycle CSR snapshot, with compiler ordering. Bounded retries
 * return zero on failure without changing output. No frequency assumption. */
static inline __attribute__((always_inline)) int measurement_cpu_cycles(uint64_t* ticks) {
    for (unsigned attempt = 0; attempt < 8; ++attempt) {
        uint32_t high, low, check;
        __asm__ volatile("rdcycleh %0\n\trdcycle %1\n\trdcycleh %2" : "=r"(high), "=r"(low), "=r"(check)::"memory");
        if (high == check) {
            *ticks = ((uint64_t)high << 32) | low;
            return 1;
        }
    }
    return 0;
}
#endif

#endif
