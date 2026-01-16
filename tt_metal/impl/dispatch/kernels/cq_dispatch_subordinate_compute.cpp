// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#define DISPATCH_KERNEL 1
#include "tools/profiler/kernel_profiler.hpp"
#include "api/debug/dprint.h"

// Stream register definitions
#define NOC_OVERLAY_START_ADDR 0xFFB40000
#define NOC_STREAM_REG_SPACE_SIZE 0x1000
#define STREAM_REG_ADDR(stream_id, reg_id) \
    ((NOC_OVERLAY_START_ADDR) + (((uint32_t)(stream_id)) * (NOC_STREAM_REG_SPACE_SIZE)) + (((uint32_t)(reg_id)) << 2))

// For Blackhole: STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX = 297
// For Wormhole: STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX = 64
#if defined(ARCH_BLACKHOLE)
#define STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX 297
#else
#define STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX 64
#endif

constexpr uint32_t WAIT_STREAM = 48;
constexpr uint32_t WAIT_COUNT = 550;

namespace NAMESPACE {
void MAIN {
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0

    volatile uint32_t* worker_sem =
        (volatile uint32_t*)STREAM_REG_ADDR(WAIT_STREAM, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);

    // First wait for the semaphore to be zero
    DPRINT << "Waiting for semaphore to be zero..." << ENDL();
    while (*worker_sem != 0) {
    }
    DPRINT << "Semaphore is zero, now waiting for worker done count" << ENDL();

    // Now wait for the count to reach target
    uint32_t last_count = 0;
    while (true) {
        uint32_t current_count = *worker_sem;
        if (current_count != last_count) {
            DeviceZoneScopedN("TT");
            DPRINT << "Worker done count: " << current_count << ENDL();
            last_count = current_count;
        }
        if (current_count >= WAIT_COUNT) {
            DPRINT << "Reached target count " << WAIT_COUNT << ENDL();
            break;
        }
    }

#endif
}
}  // namespace NAMESPACE
