// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// perf-debug profiler workload: compute RISCs (TRISC0/1/2). Same 10-zone sweep as zones_dm.cpp, tagged per
// TRISC (T0_/T1_/T2_) so each of the 3 compute RISCs emits its own 10 distinctly-named zones.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifndef N_ITERS
#define N_ITERS 50u
#endif

#if COMPILE_FOR_TRISC == 0
#define ZTAG "T0"
#elif COMPILE_FOR_TRISC == 1
#define ZTAG "T1"
#else
#define ZTAG "T2"
#endif

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
