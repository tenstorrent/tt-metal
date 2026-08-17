// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Real-time profiler BRISC kernel (fast path)
// Reads timestamp data from dispatch_s mailbox B and writes it into an L1
// ring buffer. The companion NCRISC kernel drains the ring buffer to the host
// via PCIe. This split decouples the NOC read from the PCIe push, allowing
// dispatch_s to proceed without waiting.

#include <cstdint>
#include "risc_common.h"
#include "api/dataflow/dataflow_api.h"
#include "hostdev/realtime_profiler_msgs.h"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler.hpp"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "api/debug/dprint.h"

// Size of timestamp data to read from dispatch core (kernel_start + kernel_end)
constexpr uint32_t realtime_profiler_timestamp_size = 2 * sizeof(realtime_profiler_timestamp_t);  // 32 bytes
static_assert(
    DISPATCH_DATA_ADDR_B == REALTIME_PROFILER_MSG_ADDR + __builtin_offsetof(realtime_profiler_msg_t, kernel_start_b));
static_assert((DISPATCH_DATA_ADDR_B % L1_ALIGNMENT) == 0);

// Compile-time defines set by host:
// DISPATCH_CORE_NOC_X  - NOC X coordinate of dispatch_s core
// DISPATCH_CORE_NOC_Y  - NOC Y coordinate of dispatch_s core
// DISPATCH_DATA_ADDR_B - Address of kernel_start_b in dispatch_s's L1 mailbox
// DISPATCH_PROFILER_STATE_ADDR - dispatch_s state word acknowledged after each read
// RING_BUFFER_ADDR     - L1 address of the shared ring buffer

// L1 region carved by DispatchMemMap (CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG) on this
// reserved RT-profiler tensix core. The matching dispatch cores use the same define to address
// this structure; host propagates the value via the REALTIME_PROFILER_MSG_ADDR compile-time define.
volatile tt_l1_ptr realtime_profiler_msg_t* rt_profiler_msg =
    reinterpret_cast<volatile tt_l1_ptr realtime_profiler_msg_t*>(REALTIME_PROFILER_MSG_ADDR);

volatile RtProfilerRingBuffer* ring_buffer = reinterpret_cast<volatile RtProfilerRingBuffer*>(RING_BUFFER_ADDR);

// Read timestamps from dispatch_s into the next ring buffer slot
__attribute__((noinline)) void realtime_profiler_read_and_enqueue() {
    // A completely full ring is transient because interval publication leaves
    // one control slot reserved. Do not acknowledge the mailbox until NCRISC
    // frees that slot; dispatch_s never waits for this acknowledgement.
    if (rt_ring_full(ring_buffer) && rt_profiler_msg->terminate_requested == 0) {
        return;
    }

    if (rt_ring_full(ring_buffer)) {
        // Termination cannot retain an occupied mailbox after the profiler
        // core exits. Attribute the terminal control/interval loss to the same
        // transport stage and acknowledge dispatch_s explicitly.
        ring_buffer->transport_drop_count++;
    } else {
        const uint32_t scratch_addr = reinterpret_cast<uint32_t>(&rt_profiler_msg->kernel_start_b);
        const uint64_t dispatch_noc_addr = get_noc_addr(DISPATCH_CORE_NOC_X, DISPATCH_CORE_NOC_Y, DISPATCH_DATA_ADDR_B);
        noc_async_read(dispatch_noc_addr, scratch_addr, realtime_profiler_timestamp_size);
        noc_async_read_barrier();
        // Blackhole has no uncached L1 alias. The NOC read updated this fixed
        // scratch region, so invalidate before BRISC reads the payload back.
        invalidate_l1_cache();

        volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr);
        const uint32_t record_type = (scratch[3] >> 16) & 0xff;
        const bool is_watermark = record_type == REALTIME_PROFILER_RECORD_TYPE_WATERMARK;
        if (!is_watermark && rt_ring_interval_full(ring_buffer)) {
            ring_buffer->transport_drop_count++;
        } else if (scratch[2] != REALTIME_PROFILER_UNPROFILED_PROGRAM_HOST_ID) {
            const uint32_t slot_addr = rt_ring_data_addr(ring_buffer, ring_buffer->write_index);
            volatile tt_l1_ptr uint32_t* slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr);
            for (uint32_t i = 0; i < REALTIME_PROFILER_RECORD_WORDS; ++i) {
                slot[i] = scratch[i];
            }
            if (is_watermark) {
                slot[7] = ring_buffer->transport_drop_count;
            }
            asm volatile("fence w, w" ::: "memory");
            ring_buffer->write_index++;
        }
    }

    rt_profiler_msg->realtime_profiler_state = REALTIME_PROFILER_STATE_IDLE;
    const uint64_t dispatch_state_noc_addr =
        get_noc_addr(DISPATCH_CORE_NOC_X, DISPATCH_CORE_NOC_Y, DISPATCH_PROFILER_STATE_ADDR);
    noc_async_write_one_packet(
        reinterpret_cast<uint32_t>(&rt_profiler_msg->realtime_profiler_state),
        dispatch_state_noc_addr,
        sizeof(uint32_t));
    noc_async_write_barrier();
}

// Handle sync requests from host: capture device timestamp and enqueue
// a sync marker record into the ring buffer for the NCRISC pusher.
__attribute__((noinline)) void realtime_profiler_sync() {
    DPRINT("REALTIME: entering sync\n");

    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);

    uint32_t sync_count = 0;
    while (rt_profiler_msg->sync_request && rt_profiler_msg->terminate_requested == 0) {
        invalidate_l1_cache();

        uint32_t host_time = rt_profiler_msg->sync_host_timestamp;
        if (host_time > 0) {
            DPRINT("REALTIME: sync got host_time={}\n", host_time);

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

            rt_profiler_msg->sync_host_timestamp = 0;
            sync_count++;
            DPRINT("REALTIME: sync pushed count={}\n", sync_count);
        }
    }
    DPRINT("REALTIME: exiting sync, total={}\n", sync_count);
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

        if (rt_profiler_msg->terminate_requested != 0) {
            // A notification already accepted by dispatch_s must be consumed
            // (or counted as a transport drop) before the reserved core exits.
            if (rt_profiler_msg->realtime_profiler_state == REALTIME_PROFILER_STATE_PUSH_B) {
                realtime_profiler_read_and_enqueue();
            }
            ring_buffer->terminate = 1;
            return;
        }

        RealtimeProfilerState state = static_cast<RealtimeProfilerState>(rt_profiler_msg->realtime_profiler_state);

        switch (state) {
            case REALTIME_PROFILER_STATE_IDLE:
                if (rt_profiler_msg->sync_request) {
                    DPRINT("REALTIME: sync_request detected!\n");
                    realtime_profiler_sync();
                }
                continue;

            case REALTIME_PROFILER_STATE_PUSH_B: realtime_profiler_read_and_enqueue(); break;
        }
    }
}
