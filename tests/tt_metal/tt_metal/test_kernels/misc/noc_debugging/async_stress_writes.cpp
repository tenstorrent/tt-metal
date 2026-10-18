// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Optional on-device idle: model a long-running kernel that spends time NOT emitting events (compute/idle) between
// bursts, so the host's periodic debug-dump poll runs while the kernel is still executing. Defaults keep the original
// behavior (no idle, a single burst).
#ifndef WAIT_ITERS
#define WAIT_ITERS 0
#endif
#ifndef BURST_SIZE
#define BURST_SIZE NUM_ITERATIONS
#endif

// Diagnostic stress kernel: issue NUM_ITERATIONS non-posted writes with NO barrier. Sources cycle through SRC_SLOTS
// distinct 32B slots so a large NUM_ITERATIONS stays inside the reserved L1 buffer, while each write still gets a
// distinct device timestamp -> a distinct profiler event/marker. On the host this exercises how large the
// NOCDebugState maps and the profiler marker set grow, and whether they are reclaimed.
void kernel_main() {
    const uint64_t dst_noc_addr = get_noc_addr(OTHER_CORE_X, OTHER_CORE_Y, DST_ADDR);
    constexpr uint32_t num_bytes = 32;

    for (uint32_t i = 0; i < NUM_ITERATIONS; ++i) {
        // Wrap the source within SRC_SLOTS 32B slots so a large NUM_ITERATIONS stays inside the reserved L1 buffer.
        // Each write still gets a distinct device timestamp, so it is still a distinct profiler marker/event.
        uint32_t src = SRC_BASE_ADDR + (i % SRC_SLOTS) * num_bytes;
        noc_async_write_one_packet(src, dst_noc_addr, num_bytes);

#if WAIT_ITERS > 0
        // After each burst, idle on-device (no NOC events) so the host background poll runs mid-kernel. The volatile
        // counter keeps the compiler from eliding the spin.
        if ((i % BURST_SIZE) == (BURST_SIZE - 1)) {
            for (volatile uint32_t w = 0; w < WAIT_ITERS; ++w) {
            }
        }
#endif
    }

    // Intentionally NO write barrier / flush: the writes are left outstanding so the host-side
    // nonposted_writes_pending map keeps all NUM_ITERATIONS entries when the test inspects it.
}
