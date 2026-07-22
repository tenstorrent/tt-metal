// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// perf-debug profiler workload: data-movement RISCs (BRISC = RISCV_0, NCRISC = RISCV_1). Each iteration
// enters 10 DIFFERENTLY-NAMED DeviceZoneScopedN scopes with INCREASING durations, so the perf-debug (X280)
// profiler captures a variety of named zones across all RISCs. The name carries a per-RISC tag (BR_/NC_)
// so each RISC's 10 zones are distinct. N_ITERS controls how many times the 10-zone sweep repeats.
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"

#ifndef N_ITERS
#define N_ITERS 50u
#endif

#if defined(COMPILE_FOR_BRISC)
#define ZTAG "BR"
#else
#define ZTAG "NC"
#endif

// One named zone whose body busy-waits DUR nop iterations (-> a distinct on-device duration per zone).
#define ZONE(NAME, DUR)                                           \
    {                                                             \
        DeviceZoneScopedN(NAME);                                  \
        for (volatile uint32_t j = 0; j < (uint32_t)(DUR); j++) { \
            asm volatile("nop");                                  \
        }                                                         \
    }

void kernel_main() {
    for (uint32_t it = 0; it < (uint32_t)N_ITERS; it++) {
        ZONE(ZTAG "_Zone0", 100u);
        ZONE(ZTAG "_Zone1", 200u);
        ZONE(ZTAG "_Zone2", 300u);
        ZONE(ZTAG "_Zone3", 400u);
        ZONE(ZTAG "_Zone4", 500u);
        ZONE(ZTAG "_Zone5", 600u);
        ZONE(ZTAG "_Zone6", 700u);
        ZONE(ZTAG "_Zone7", 800u);
        ZONE(ZTAG "_Zone8", 900u);
        ZONE(ZTAG "_Zone9", 1000u);
    }
}
