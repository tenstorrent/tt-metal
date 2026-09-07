// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Real-time profiler BRISC kernel (fast path)
// Copies completed dispatch records into an L1
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

// Compile-time defines set by host:
// DISPATCH_CORE_NOC_X  - NOC X coordinate of dispatch_s core
// DISPATCH_CORE_NOC_Y  - NOC Y coordinate of dispatch_s core
// RING_BUFFER_ADDR     - L1 address of the shared ring buffer

// L1 region carved by DispatchMemMap (CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG) on this
// reserved RT-profiler tensix core. The matching dispatch cores use the same define to address
// this structure; host propagates the value via the REALTIME_PROFILER_MSG_ADDR compile-time define.
volatile tt_l1_ptr realtime_profiler_msg_t* rt_profiler_msg =
    reinterpret_cast<volatile tt_l1_ptr realtime_profiler_msg_t*>(REALTIME_PROFILER_MSG_ADDR);

volatile RtProfilerRingBuffer* ring_buffer = reinterpret_cast<volatile RtProfilerRingBuffer*>(RING_BUFFER_ADDR);

static_assert(sizeof(realtime_profiler_record_t) == RT_PROFILER_ENTRY_SIZE);
static_assert(offsetof(realtime_profiler_msg_t, records) % 16 == 0);

// Read one slot. READY records are immutable until this reader acknowledges
// them. Scan all slots so a long program cannot hold up another subdevice's
// completed records. No dispatch or worker is blocked by reader backpressure.
bool poll_record(uint32_t index) {
    if (rt_ring_full(ring_buffer)) {
        return true;
    }
    const uint32_t slot_addr = rt_ring_data_addr(ring_buffer, ring_buffer->write_index);
    const uint32_t remote_addr = REALTIME_PROFILER_MSG_ADDR + offsetof(realtime_profiler_msg_t, records) +
                                 index * sizeof(realtime_profiler_record_t);
    const uint64_t remote = get_noc_addr(DISPATCH_CORE_NOC_X, DISPATCH_CORE_NOC_Y, remote_addr);
    // Acquire ownership before copying payload. A single read spanning payload
    // and state could sample old timestamps followed by a newly published READY.
    constexpr uint32_t state_offset = offsetof(realtime_profiler_record_t, state);
    noc_async_read(remote + state_offset, slot_addr + state_offset, sizeof(uint32_t));
    noc_async_read_barrier();
    const auto* record = reinterpret_cast<volatile tt_l1_ptr realtime_profiler_record_t*>(slot_addr);
    const uint32_t state = record->state;
    if (state == REALTIME_PROFILER_RECORD_READY) {
        // Only this reader can free a READY slot, so its payload stays immutable
        // throughout this second read and until the acknowledgement below.
        noc_async_read(remote, slot_addr, sizeof(realtime_profiler_record_t));
        noc_async_read_barrier();
        asm volatile("fence rw, rw" ::: "memory");
        ring_buffer->write_index++;
        noc_inline_dw_write(remote + offsetof(realtime_profiler_record_t, state), REALTIME_PROFILER_RECORD_FREE);
    }
    return state != REALTIME_PROFILER_RECORD_FREE;
}

void report_dropped_records() {
    if (rt_ring_full(ring_buffer)) {
        return;
    }
    static uint32_t reported = 0;
    const uint32_t offset = offsetof(realtime_profiler_msg_t, dropped_records);
    noc_async_read(
        get_noc_addr(DISPATCH_CORE_NOC_X, DISPATCH_CORE_NOC_Y, REALTIME_PROFILER_MSG_ADDR + offset),
        REALTIME_PROFILER_MSG_ADDR + offset,
        sizeof(uint32_t));
    noc_async_read_barrier();
    const uint32_t dropped = rt_profiler_msg->dropped_records;
    if (dropped == reported) {
        return;
    }
    auto* page =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rt_ring_data_addr(ring_buffer, ring_buffer->write_index));
    for (uint32_t i = 0; i < RT_PROFILER_ENTRY_SIZE / sizeof(uint32_t); ++i) {
        page[i] = 0;
    }
    page[2] = dropped - reported;
    page[3] = REALTIME_PROFILER_DROPPED_MARKER_ID;
    asm volatile("fence rw, rw" ::: "memory");
    ring_buffer->write_index++;
    reported = dropped;
}

// Service one handshake step and return to record draining. A host sync request
// must not stop the reader while dispatch continues to fill the record queue.
void realtime_profiler_sync() {
    const uint32_t host_time = rt_profiler_msg->sync_host_timestamp;
    if (host_time == 0 || rt_ring_full(ring_buffer)) {
        return;
    }
    auto* page =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rt_ring_data_addr(ring_buffer, ring_buffer->write_index));
    const uint64_t timestamp = realtime_profiler_read_timestamp();
    page[0] = timestamp >> 32;
    page[1] = static_cast<uint32_t>(timestamp);
    page[2] = host_time;
    page[3] = REALTIME_PROFILER_SYNC_MARKER_ID;
    for (uint32_t i = 4; i < RT_PROFILER_ENTRY_SIZE / sizeof(uint32_t); ++i) {
        page[i] = 0;
    }
    asm volatile("fence rw, rw" ::: "memory");
    ring_buffer->write_index++;
    rt_profiler_msg->sync_host_timestamp = 0;
}

void kernel_main() {
    ring_buffer->write_index = 0;
    ring_buffer->read_index = 0;
    ring_buffer->terminate = 0;
    rt_profiler_msg->realtime_profiler_state = REALTIME_PROFILER_STATE_IDLE;
    uint32_t index = 0;
    bool pass_nonempty = false;
    bool terminating = false;
    while (true) {
        invalidate_l1_cache();
        if (rt_profiler_msg->sync_request) {
            realtime_profiler_sync();
        }
        if (rt_profiler_msg->realtime_profiler_state == REALTIME_PROFILER_STATE_TERMINATE && !terminating) {
            terminating = true;
            index = 0;
            pass_nonempty = false;
        }
        if (index == 0) {
            // Complete FREE acknowledgements before revisiting these slots.
            // Reads and writes use separate NOC channels; an outstanding FREE
            // must not let the next scan deliver the same READY record again.
            // This also covers a partial scan restarted during termination.
            noc_async_write_barrier();
        }
        pass_nonempty |= poll_record(index);
        index++;
        if (index == REALTIME_PROFILER_RECORD_CAPACITY) {
            report_dropped_records();
            if (terminating && !pass_nonempty) {
                ring_buffer->terminate = 1;
                return;
            }
            index = 0;
            pass_nonempty = false;
        }
    }
}
