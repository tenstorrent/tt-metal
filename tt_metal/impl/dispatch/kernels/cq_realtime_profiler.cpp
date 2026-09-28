// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Real-time profiler BRISC kernel (fast path)
// Reads timestamp data from dispatch_s A/B buffers and writes it into an L1
// ring buffer. The companion NCRISC kernel drains the ring buffer to the host
// via PCIe. This split decouples the NOC read from the PCIe push, allowing
// dispatch_s to proceed without waiting.

#include <cstddef>
#include <cstdint>
#include "risc_common.h"
#include "api/dataflow/dataflow_api.h"
#include "hostdev/realtime_profiler_msgs.h"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler.hpp"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "api/debug/dprint.h"

// Size of timestamp data to read from dispatch core (kernel_start + kernel_end)
constexpr uint32_t realtime_profiler_timestamp_size = 2 * sizeof(realtime_profiler_timestamp_t);  // 32 bytes

static_assert(realtime_profiler_timestamp_size == sizeof(realtime_profiler_record_t));
static_assert((REALTIME_PROFILER_RECORD_SLOTS & (REALTIME_PROFILER_RECORD_SLOTS - 1)) == 0);
static_assert(offsetof(realtime_profiler_msg_t, records) % 16 == 0, "record slots are NOC-read into 16 B-aligned L1");

// Compile-time defines set by host:
// DISPATCH_CORE_NOC_X        - NOC X coordinate of dispatch_s core
// DISPATCH_CORE_NOC_Y        - NOC Y coordinate of dispatch_s core
// DISPATCH_RECORDS_ADDR      - Address of records[0] in dispatch_s's L1 mailbox
// DISPATCH_RECORD_RD_IDX_ADDR - Address of record_rd_idx in dispatch_s's L1 mailbox
// RING_BUFFER_ADDR           - L1 address of the shared ring buffer

// L1 region carved by DispatchMemMap (CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG) on this
// reserved RT-profiler tensix core. The matching dispatch cores use the same define to address
// this structure; host propagates the value via the REALTIME_PROFILER_MSG_ADDR compile-time define.
volatile tt_l1_ptr realtime_profiler_msg_t* rt_profiler_msg =
    reinterpret_cast<volatile tt_l1_ptr realtime_profiler_msg_t*>(REALTIME_PROFILER_MSG_ADDR);

volatile RtProfilerRingBuffer* ring_buffer = reinterpret_cast<volatile RtProfilerRingBuffer*>(RING_BUFFER_ADDR);

// Latest end time among the dispatch_s records read so far.
uint64_t last_record_end = 0;

// Read one record slot from dispatch_s into the next ring buffer slot
__attribute__((noinline)) void realtime_profiler_read_and_enqueue(uint32_t record_idx) {
    // Heartbeat: ring_full_wait_count increments once per enqueue blocked on a full ring.
    // Host post-mortems can pair it with ncrisc_debug.socket_reserve_pages_{enter,exit}_count.
    if (rt_ring_full(ring_buffer)) {
        ring_buffer->ring_full_wait_count++;
        while (rt_ring_full(ring_buffer)) {
            invalidate_l1_cache();
        }
    }

    uint32_t slot_addr = rt_ring_data_addr(ring_buffer, ring_buffer->write_index);

    uint32_t dispatch_data_addr = DISPATCH_RECORDS_ADDR + (record_idx & (REALTIME_PROFILER_RECORD_SLOTS - 1)) *
                                                              sizeof(realtime_profiler_record_t);
    uint64_t dispatch_noc_addr = get_noc_addr(DISPATCH_CORE_NOC_X, DISPATCH_CORE_NOC_Y, dispatch_data_addr);

    noc_async_read(dispatch_noc_addr, slot_addr, realtime_profiler_timestamp_size);
    noc_async_read_barrier();

    // A record ends at the latest worker completion seen, so end times never go backwards. A slot that saw no
    // completion while open (dispatch_s was held up and its wait for workers returned at once) still holds its
    // end from a full ring ago; replace it with the previous record's end. Done here rather than in dispatch_s
    // to keep it off the dispatch path. Covers unprofiled records too, since they carry the chain forward.
    volatile tt_l1_ptr realtime_profiler_record_t* record =
        reinterpret_cast<volatile tt_l1_ptr realtime_profiler_record_t*>(slot_addr);
    const uint64_t end = (static_cast<uint64_t>(record->kernel_end.time_hi) << 32) | record->kernel_end.time_lo;
    if (end < last_record_end) {
        record->kernel_end.time_hi = static_cast<uint32_t>(last_record_end >> 32);
        record->kernel_end.time_lo = static_cast<uint32_t>(last_record_end);
    } else {
        last_record_end = end;
    }

    const uint32_t id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr)[2];
    if (id != REALTIME_PROFILER_UNPROFILED_PROGRAM_HOST_ID) {
        ring_buffer->write_index++;
    }
}

// Consumer index into the dispatch_s record ring; this kernel is its only writer.
uint32_t record_rd_idx = 0;

// Drain every record dispatch_s has published, then hand the slots back.
// Returns true once dispatch_s has terminated and every record up to its final count has been read.
__attribute__((noinline)) bool realtime_profiler_drain_records() {
    const uint32_t published = rt_profiler_msg->record_wr_idx;
    const uint32_t wr_idx = published & REALTIME_PROFILER_RECORD_WR_IDX_MASK;
    const uint32_t pending = (wr_idx - record_rd_idx) & REALTIME_PROFILER_RECORD_WR_IDX_MASK;
    // dispatch_s publishes at most SLOTS - 1 records past our last ack, plus its final record at terminate,
    // so a larger gap can only be a stale index; reading it would replay old slots.
    if (pending != 0 && pending <= REALTIME_PROFILER_RECORD_SLOTS) {
        while (record_rd_idx != wr_idx) {
            realtime_profiler_read_and_enqueue(record_rd_idx);
            record_rd_idx = (record_rd_idx + 1) & REALTIME_PROFILER_RECORD_WR_IDX_MASK;
        }
        // Every read above has landed (noc_async_read_barrier), so dispatch_s may now reuse these slots.
        noc_inline_dw_write(
            get_noc_addr(DISPATCH_CORE_NOC_X, DISPATCH_CORE_NOC_Y, DISPATCH_RECORD_RD_IDX_ADDR), record_rd_idx);
    }
    return (published & REALTIME_PROFILER_RECORD_WR_IDX_TERMINATE) && record_rd_idx == wr_idx;
}

// Handle sync requests from host: capture device timestamp and enqueue
// a sync marker record into the ring buffer for the NCRISC pusher.
__attribute__((noinline)) void realtime_profiler_sync() {
    DPRINT("REALTIME: entering sync\n");

    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);

    uint32_t sync_count = 0;
    while (rt_profiler_msg->sync_request) {
        invalidate_l1_cache();

        // Keep serving dispatch_s during a sync, or it would stall once the record ring fills.
        realtime_profiler_drain_records();

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

    // record_wr_idx on this core is written only by dispatch_s (host zeroes it before launch).
    while (true) {
        invalidate_l1_cache();

        if (realtime_profiler_drain_records()) {
            noc_async_write_barrier();  // last record_rd_idx ack
            ring_buffer->terminate = 1;
            return;
        }

        if (rt_profiler_msg->sync_request) {
            DPRINT("REALTIME: sync_request detected!\n");
            realtime_profiler_sync();
        }
    }
}
