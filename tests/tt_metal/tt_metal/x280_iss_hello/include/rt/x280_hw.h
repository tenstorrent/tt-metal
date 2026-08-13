// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ISS stand-in for rt/x280_hw.h. Spike is not given a Blackhole CLINT, so
// mtime is rdcycle (ordering only — not a 50 MHz wall clock).

#ifndef X280_RT_HW_H
#define X280_RT_HW_H

#include <stdint.h>

static inline uint64_t x280_rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

#define X280_MTIME_HZ 50000000ULL

static inline uint64_t x280_mtime(void) { return x280_rdcycle(); }

static inline void x280_pause(void) { __asm__ volatile("nop" ::: "memory"); }

static inline __attribute__((noreturn)) void x280_cease(void) {
    for (;;) {
        __asm__ volatile("wfi");
    }
}

#endif
