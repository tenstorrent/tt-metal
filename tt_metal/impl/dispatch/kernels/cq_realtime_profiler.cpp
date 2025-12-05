// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Real-time profiler BRISC kernel (fast path)
// Reads timestamp data from dispatch_s A/B buffers and writes it into an L1
// ring buffer. The companion NCRISC kernel drains the ring buffer to the host
// via PCIe. This split decouples the fast NOC read (~0.3 µs) from the slow
// PCIe push (~50-80 µs), allowing dispatch_s to proceed without waiting.

#include <cstdint>
#include "risc_common.h"
#include "api/dataflow/dataflow_api.h"
#include "hostdev/dev_msgs.h"
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

volatile tt_l1_ptr realtime_profiler_msg_t* realtime_profiler_mailbox =
    reinterpret_cast<volatile tt_l1_ptr realtime_profiler_msg_t*>(GET_MAILBOX_ADDRESS_DEV(realtime_profiler));

volatile RtProfilerRingBuffer* ring_buffer = reinterpret_cast<volatile RtProfilerRingBuffer*>(RING_BUFFER_ADDR);

// Read timestamps from dispatch_s into the next ring buffer slot
__attribute__((noinline)) void realtime_profiler_read_and_enqueue(bool buffer_a) {
    if (buffer_a) {
        DeviceZoneScopedN("A");
    } else {
        DeviceZoneScopedN("B");
    }

    // Spin until ring buffer has space
    while (rt_ring_full(ring_buffer)) {
        invalidate_l1_cache();
    }

    uint32_t slot_addr = rt_ring_data_addr(ring_buffer, ring_buffer->write_index);

    uint32_t dispatch_data_addr = buffer_a ? DISPATCH_DATA_ADDR_A : DISPATCH_DATA_ADDR_B;
    uint64_t dispatch_noc_addr = get_noc_addr(DISPATCH_CORE_NOC_X, DISPATCH_CORE_NOC_Y, dispatch_data_addr);

    noc_async_read(dispatch_noc_addr, slot_addr, realtime_profiler_timestamp_size);
    noc_async_read_barrier();

    ring_buffer->write_index++;
}

// Handle sync requests from host: capture device timestamp and enqueue
// a sync marker record into the ring buffer for the NCRISC pusher.
__attribute__((noinline)) void realtime_profiler_sync() {
    DPRINT << "REALTIME: entering sync" << ENDL();

    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);

    uint32_t sync_count = 0;
    while (realtime_profiler_mailbox->sync_request) {
        invalidate_l1_cache();

        uint32_t host_time = realtime_profiler_mailbox->sync_host_timestamp;
        if (host_time > 0) {
            DPRINT << "REALTIME: sync got host_time=" << host_time << ENDL();

            // Spin until ring buffer has space
            while (rt_ring_full(ring_buffer)) {
                invalidate_l1_cache();
            }

            uint32_t slot_addr = rt_ring_data_addr(ring_buffer, ring_buffer->write_index);
            tt_l1_ptr uint32_t* l1_data = reinterpret_cast<tt_l1_ptr uint32_t*>(slot_addr);

            uint32_t time_lo = p_reg[WALL_CLOCK_LOW_INDEX];
            uint32_t time_hi = p_reg[WALL_CLOCK_HIGH_INDEX];

            l1_data[0] = time_hi;
            l1_data[1] = time_lo;
            l1_data[2] = host_time;
            l1_data[3] = REALTIME_PROFILER_SYNC_MARKER_ID;
            l1_data[4] = 0;
            l1_data[5] = 0;
            l1_data[6] = 0;
            l1_data[7] = 0;

            ring_buffer->write_index++;

            realtime_profiler_mailbox->sync_host_timestamp = 0;
            sync_count++;
            DPRINT << "REALTIME: sync pushed count=" << sync_count << ENDL();
        }
    }
    DPRINT << "REALTIME: exiting sync, total=" << sync_count << ENDL();
}

void kernel_main() {
    DPRINT << "REALTIME BRISC: kernel started" << ENDL();

    // Initialize ring buffer
    ring_buffer->write_index = 0;
    ring_buffer->read_index = 0;
    ring_buffer->terminate = 0;

    realtime_profiler_mailbox->realtime_profiler_state = REALTIME_PROFILER_STATE_IDLE;

    while (true) {
        invalidate_l1_cache();

        RealtimeProfilerState state =
            static_cast<RealtimeProfilerState>(realtime_profiler_mailbox->realtime_profiler_state);

        switch (state) {
            case REALTIME_PROFILER_STATE_IDLE:
                if (realtime_profiler_mailbox->sync_request) {
                    DPRINT << "REALTIME: sync_request detected!" << ENDL();
                    realtime_profiler_sync();
                }
                continue;

            case REALTIME_PROFILER_STATE_PUSH_A:
                realtime_profiler_read_and_enqueue(true);
                realtime_profiler_mailbox->realtime_profiler_state = REALTIME_PROFILER_STATE_IDLE;
                break;

            case REALTIME_PROFILER_STATE_PUSH_B:
                realtime_profiler_read_and_enqueue(false);
                realtime_profiler_mailbox->realtime_profiler_state = REALTIME_PROFILER_STATE_IDLE;
                break;

            case REALTIME_PROFILER_STATE_TERMINATE: ring_buffer->terminate = 1; return;
        }
    }
}
