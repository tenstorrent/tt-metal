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

// Handle a host sync request: capture one device wall-clock timestamp and enqueue a sync marker for the NCRISC
// pusher. One-shot (no spin loop): dispatch_s hands off program timestamps through the double-buffered
// PUSH_A/PUSH_B state and never waits on this core, so any stall here silently drops those records. Blocking until
// the host clears the request would drop a burst of records on every periodic sync; instead we capture at most one
// timestamp per host write and return immediately, and the main loop re-checks on its next pass.
__attribute__((noinline)) void realtime_profiler_sync() {
    uint32_t host_time = rt_profiler_msg->sync_host_timestamp;
    if (host_time == 0) {
        return;
    }

    while (rt_ring_full(ring_buffer)) {
        invalidate_l1_cache();
    }

    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t slot_addr = rt_ring_data_addr(ring_buffer, ring_buffer->write_index);
    tt_l1_ptr uint32_t* l1_data = reinterpret_cast<tt_l1_ptr uint32_t*>(slot_addr);

    uint32_t time_lo = p_reg[WALL_CLOCK_LOW_INDEX];
    uint32_t time_hi = p_reg[WALL_CLOCK_HIGH_INDEX];

    // Publish device_time to L1 before the token ACK, so once the host observes the token it can read a stable
    // device_time directly (no dependence on the slower FIFO sync marker for re-anchoring).
    rt_profiler_msg->sync_ack_device_time[0] = time_lo;
    rt_profiler_msg->sync_ack_device_time[1] = time_hi;

    // NOC-write the token straight to the host's pinned ACK word (device->host, bypassing the record FIFO) so the host
    // times the round trip by polling its own memory. Stage it in sync_request as the L1 source; Blackhole PCIe uses
    // the 64-bit noc_wwrite_with_state form (mirrors socket_notify_receiver).
    rt_profiler_msg->sync_request = host_time;
    if (rt_profiler_msg->sync_ack_pcie_xy_enc != 0) {
        const uint64_t ack_pcie_addr = (static_cast<uint64_t>(rt_profiler_msg->sync_ack_host_addr_hi) << 32) |
                                       static_cast<uint64_t>(rt_profiler_msg->sync_ack_host_addr_lo);
        noc_write_init_state<write_cmd_buf>(noc_index, NOC_UNICAST_WRITE_VC);
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            noc_index,
            reinterpret_cast<uint32_t>(&rt_profiler_msg->sync_request),
            rt_profiler_msg->sync_ack_pcie_xy_enc,
            ack_pcie_addr,
            sizeof(uint32_t),
            1);
        noc_async_write_barrier();
    }

    l1_data[0] = time_hi;
    l1_data[1] = time_lo;
    l1_data[2] = host_time;
    l1_data[3] = REALTIME_PROFILER_SYNC_MARKER_ID;
    l1_data[4] = 0;
    l1_data[5] = 0;
    l1_data[6] = 0;
    l1_data[7] = 0;

    ring_buffer->write_index++;
    rt_profiler_msg->sync_host_timestamp = 0;
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
            case REALTIME_PROFILER_STATE_IDLE: realtime_profiler_sync(); continue;

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
