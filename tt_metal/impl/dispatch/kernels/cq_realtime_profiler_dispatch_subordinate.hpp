// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "hostdev/realtime_profiler_msgs.h"
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

FORCE_INLINE void dispatch_subordinate_realtime_profiler() {
    // Dispatch-core-local L1 region carved by DispatchMemMap (CommandQueueDeviceAddrType::
    // REALTIME_PROFILER_MSG). Address is supplied by host via the REALTIME_PROFILER_MSG_ADDR
    // compile-time define; mirrors cq_dispatch.cpp / cq_dispatch_subordinate.cpp on the same core.
    volatile tt_l1_ptr realtime_profiler_msg_t* rt_profiler_msg =
        reinterpret_cast<volatile tt_l1_ptr realtime_profiler_msg_t*>(REALTIME_PROFILER_MSG_ADDR);

    // Clear stale RT-profiler carve-out state left in L1 from prior runs.
    rt_profiler_msg->realtime_profiler_core_noc_xy = 0;
    rt_profiler_msg->realtime_profiler_remote_state_addr = 0;
    rt_profiler_msg->realtime_profiler_state = REALTIME_PROFILER_STATE_IDLE;

    // Wait until host explicitly enables RT profiler, or terminate if RT is not used.
    while (rt_profiler_msg->realtime_profiler_core_noc_xy == 0) {
        if (rt_profiler_msg->realtime_profiler_state == REALTIME_PROFILER_STATE_TERMINATE) {
            return;
        }
    }

    if (rt_profiler_msg->realtime_profiler_state == REALTIME_PROFILER_STATE_TERMINATE) {
        rt_profiler_msg->realtime_profiler_state = REALTIME_PROFILER_STATE_IDLE;
    }

    uint32_t last_counts[num_streams_to_monitor];
    for (uint32_t i = 0; i < num_streams_to_monitor; i++) {
        uint32_t stream_id = first_stream_index + i;
        volatile uint32_t* stream_reg =
            (volatile uint32_t*)STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
        last_counts[i] = *stream_reg;
    }

    while (rt_profiler_msg->realtime_profiler_state != REALTIME_PROFILER_STATE_TERMINATE) {
        for (uint32_t i = 0; i < num_streams_to_monitor; i++) {
            uint32_t stream_id = first_stream_index + i;
            volatile uint32_t* stream_reg =
                (volatile uint32_t*)STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);

            uint32_t current_count = *stream_reg;
            if (current_count != last_counts[i]) {
                DeviceZoneScopedN("TRISC0-record-end-ts");
                last_counts[i] = current_count;
                record_realtime_timestamp(rt_profiler_msg, false);
            }
        }
    }
}
