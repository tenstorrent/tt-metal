// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Demo kernel for the continuous (SPSC-streaming) profiler. Runs N_ITERS
// iterations of nested zones; each ContinuousZoneScopedN ctor/dtor streams a
// START/END marker into the L1 SPSC flit-ring for the X280 (or host) to drain.
//
// Per outer iteration = 4 markers (outer-START, inner-START, inner-END,
// outer-END) -> two iterations fill one 64B flit (8 markers). Keep N_ITERS even
// so production ends on a flit boundary (N_ITERS=1000 -> 4000 markers = 500
// flits). WORK_CYCLES of busy-wait between markers simulates real kernel work
// and sets the marker rate. After N_ITERS the kernel returns; the ring contents
// persist in L1 for the consumer to finish draining.

#include <cstdint>
#include "tools/profiler/continous_profiler.hpp"

#ifndef N_ITERS
#define N_ITERS 1000  // number of outer-loop iterations (keep even)
#endif
#ifndef WORK_CYCLES
#define WORK_CYCLES 100  // wall-clock ticks of "work" between markers (~74 ns @ 1.35 GHz)
#endif

static inline uint32_t cp_wall_lo() { return *reinterpret_cast<volatile uint32_t*>(0xFFB121F0u); }

static inline void busy(uint32_t cycles) {
    const uint32_t deadline = cp_wall_lo() + cycles;
    while (static_cast<int32_t>(cp_wall_lo() - deadline) < 0) {
    }
}

void kernel_main() {
    continuous_profiler::init();
    for (uint32_t i = 0; i < N_ITERS; i++) {
        ContinuousZoneScopedN("outer");
        busy(WORK_CYCLES);
        {
            ContinuousZoneScopedN("inner");
            busy(WORK_CYCLES);
        }
        busy(WORK_CYCLES);
    }
}
