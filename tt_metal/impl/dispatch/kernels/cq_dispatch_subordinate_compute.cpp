// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#define DISPATCH_KERNEL 1
#include "tools/profiler/kernel_profiler.hpp"
#include "hostdevcommon/profiler_common.h"
#include "hostdev/dev_msgs.h"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler.hpp"

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

// Compile-time args from host (set in dispatch_s.cpp)
constexpr uint32_t first_stream_index = FIRST_STREAM_INDEX;
constexpr uint32_t num_streams_to_monitor = NUM_STREAMS_TO_MONITOR;

namespace NAMESPACE {
void MAIN {
// Real-time profiler stream monitoring is only functional on Blackhole due to HW constraints.
// On other architectures, TRISC0 exits immediately.
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0 && defined(ARCH_BLACKHOLE)

    // Array to track last seen count for each stream
    uint32_t last_counts[num_streams_to_monitor] = {0};

    // Pointer to real-time profiler config for reading terminate flag
    volatile tt_l1_ptr realtime_profiler_msg_t* realtime_profiler_mailbox =
        reinterpret_cast<volatile tt_l1_ptr realtime_profiler_msg_t*>(GET_MAILBOX_ADDRESS_DEV(realtime_profiler));

    // Main loop: runs until dispatch_s signals terminate
    while (realtime_profiler_mailbox->realtime_profiler_state != REALTIME_PROFILER_STATE_TERMINATE) {
        // Loop over all streams we're monitoring
        for (uint32_t i = 0; i < num_streams_to_monitor; i++) {
            uint32_t stream_id = first_stream_index + i;
            volatile uint32_t* stream_reg =
                (volatile uint32_t*)STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);

            uint32_t current_count = *stream_reg;
            if (current_count != last_counts[i]) {
                DeviceZoneScopedN("Count_changed");
                // Stream count changed - record the event
                last_counts[i] = current_count;
                record_realtime_timestamp(realtime_profiler_mailbox, false);
            }
        }
    }

#endif
}
}  // namespace NAMESPACE
