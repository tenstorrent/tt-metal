// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// REAL kernel_profiler marker generator for the compute RISCs (TRISC0-2). Emits genuine DeviceZoneScopedN
// zones so the X280 drain + host decode are exercised against the production 2-word marker format. The
// runtime host-id (STICKY_PROG) is pushed by BRISC (see realprof_dm.cpp); compute RISCs only emit zones.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"  // get_arg_val for compute kernels
#include "tools/profiler/kernel_profiler.hpp"

#ifndef MARKER_COUNT
#define MARKER_COUNT 4096u
#endif
#ifndef WORK_SIZE
#define WORK_SIZE 64u
#endif

void kernel_main() {
    uint32_t stagger = get_arg_val<uint32_t>(0);  // per-core start-stagger cycles (0 = none); see realprof_dm.cpp
    for (volatile uint32_t s = 0; s < stagger; s++) {
        asm volatile("nop");
    }
    for (uint32_t i = 0; i < (uint32_t)MARKER_COUNT; i++) {
        DeviceZoneScopedN("REALPROF-COMPUTE");
        for (volatile uint32_t j = 0; j < (uint32_t)WORK_SIZE; j++) {
            asm volatile("nop");
        }
    }
}
