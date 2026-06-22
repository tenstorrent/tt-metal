// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Demo kernel for the continuous (SPSC-streaming) profiler. Runs a forever loop
// of nested zones; each ContinuousZoneScopedN ctor/dtor streams a START/END
// marker into the L1 SPSC flit-ring for the X280 to drain live.
//
// Per outer iteration = 4 markers (outer-START, inner-START, inner-END,
// outer-END) -> two iterations fill one 64B flit (8 markers). WORK_CYCLES of
// busy-wait between markers simulates real kernel work and sets the marker rate.

#include <cstdint>
#include "tools/profiler/continous_profiler.hpp"

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
    while (true) {
        ContinuousZoneScopedN("outer");
        busy(WORK_CYCLES);
        {
            ContinuousZoneScopedN("inner");
            busy(WORK_CYCLES);
        }
        busy(WORK_CYCLES);
    }
}
