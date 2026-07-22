// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Real-time profiler BRISC kernel (fast path)
// Reads timestamp data from dispatch_s A/B buffers and writes it into an L1
// ring buffer. The companion NCRISC kernel drains the ring buffer to the host
// via PCIe. This split decouples the NOC read from the PCIe push, allowing
// dispatch_s to proceed without waiting.

#include <cstdint>
#include "risc_common.h"
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "hostdev/realtime_profiler_msgs.h"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler.hpp"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "api/debug/dprint.h"

// Size of timestamp data to read from dispatch core (kernel_start + kernel_end)
constexpr uint32_t realtime_profiler_timestamp_size = 2 * sizeof(realtime_profiler_timestamp_t);  // 32 bytes

// Compile-time defines set by host:
// DISPATCH_CORE_NOC_X  - NOC X coordinate of dispatch_s core
// DISPATCH_CORE_NOC_Y  - NOC Y coordinate of dispatch_s core
// DISPATCH_DATA_ADDR_A - Address of kernel_start_a in dispatch_s's L1 mailbox
// DISPATCH_DATA_ADDR_B - Address of kernel_start_b in dispatch_s's L1 mailbox
// RING_BUFFER_ADDR     - L1 address of the shared ring buffer

// L1 region carved by DispatchMemMap (CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG) on this
// reserved RT-profiler tensix core. The matching dispatch cores use the same define to address
// this structure; host propagates the value via the REALTIME_PROFILER_MSG_ADDR compile-time define.
volatile tt_l1_ptr realtime_profiler_msg_t* rt_profiler_msg =
    reinterpret_cast<volatile tt_l1_ptr realtime_profiler_msg_t*>(REALTIME_PROFILER_MSG_ADDR);

volatile RtProfilerRingBuffer* ring_buffer = reinterpret_cast<volatile RtProfilerRingBuffer*>(RING_BUFFER_ADDR);

// Read timestamps from dispatch_s into the next ring buffer slot
__attribute__((noinline)) void realtime_profiler_read_and_enqueue(bool buffer_a) {
    // Heartbeat: ring_full_wait_count increments once per enqueue blocked on a full ring.
    // Host post-mortems can pair it with ncrisc_debug.socket_reserve_pages_{enter,exit}_count.
    if (rt_ring_full(ring_buffer)) {
        ring_buffer->ring_full_wait_count++;
        while (rt_ring_full(ring_buffer)) {
            invalidate_l1_cache();
        }
    }

    uint32_t slot_addr = rt_ring_data_addr(ring_buffer, ring_buffer->write_index);

    uint32_t dispatch_data_addr = buffer_a ? DISPATCH_DATA_ADDR_A : DISPATCH_DATA_ADDR_B;
    uint64_t dispatch_noc_addr = get_noc_addr(DISPATCH_CORE_NOC_X, DISPATCH_CORE_NOC_Y, dispatch_data_addr);

    noc_async_read(dispatch_noc_addr, slot_addr, realtime_profiler_timestamp_size);
    noc_async_read_barrier();

    const uint32_t id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr)[2];
    if (id != REALTIME_PROFILER_UNPROFILED_PROGRAM_HOST_ID) {
        ring_buffer->write_index++;
    }
}

void kernel_main() {
    DPRINT("REALTIME BRISC: kernel started\n");

    // Initialize ring buffer
    ring_buffer->write_index = 0;
    ring_buffer->read_index = 0;
    ring_buffer->terminate = 0;

    rt_profiler_msg->realtime_profiler_state = REALTIME_PROFILER_STATE_IDLE;

    while (true) {
        invalidate_l1_cache();

        RealtimeProfilerState state = static_cast<RealtimeProfilerState>(rt_profiler_msg->realtime_profiler_state);

        switch (state) {
            case REALTIME_PROFILER_STATE_IDLE: continue;

            case REALTIME_PROFILER_STATE_PUSH_A:
                realtime_profiler_read_and_enqueue(true);
                rt_profiler_msg->realtime_profiler_state = REALTIME_PROFILER_STATE_IDLE;
                break;

            case REALTIME_PROFILER_STATE_PUSH_B:
                realtime_profiler_read_and_enqueue(false);
                rt_profiler_msg->realtime_profiler_state = REALTIME_PROFILER_STATE_IDLE;
                break;

            case REALTIME_PROFILER_STATE_TERMINATE: ring_buffer->terminate = 1; return;
        }
    }
}
