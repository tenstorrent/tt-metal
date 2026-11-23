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

// Enqueue a dispatch-stall marker: the host shows it as a stall ending at stall_end, stall_cycles long.
__attribute__((noinline)) void realtime_profiler_enqueue_stall_marker(uint64_t stall_end, uint32_t stall_cycles) {
    while (rt_ring_full(ring_buffer)) {
        invalidate_l1_cache();
    }
    tt_l1_ptr uint32_t* l1_data =
        reinterpret_cast<tt_l1_ptr uint32_t*>(rt_ring_data_addr(ring_buffer, ring_buffer->write_index));
    l1_data[0] = static_cast<uint32_t>(stall_end >> 32);
    l1_data[1] = static_cast<uint32_t>(stall_end);
    l1_data[2] = stall_cycles;
    l1_data[3] = REALTIME_PROFILER_DISPATCH_STALL_MARKER_ID;
    l1_data[4] = 0;
    l1_data[5] = 0;
    l1_data[6] = 0;
    l1_data[7] = 0;
    ring_buffer->write_index++;
}

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

    // A stamp's low word is always right, but around a wrap of the clock's low word its high word can be off by one.
    // One too new: another core on the dispatch tile read the clock between the stamp's two reads (the high word
    // comes from a latch that any low-word read updates). One too old: two cores' writes of an end interleaved.
    // A record needing no repair has equal high words and does not end before it starts, unless it spans a wrap, so
    // the clock is rarely read.
    volatile tt_l1_ptr realtime_profiler_record_t* record =
        reinterpret_cast<volatile tt_l1_ptr realtime_profiler_record_t*>(slot_addr);
    const uint32_t start_hi = record->kernel_start.time_hi;
    const uint32_t start_lo = record->kernel_start.time_lo;
    const uint32_t end_hi = record->kernel_end.time_hi;
    const uint32_t end_lo = record->kernel_end.time_lo;
    if (end_hi != start_hi || end_lo < start_lo) [[unlikely]] {
        volatile tt_reg_ptr uint32_t* p_reg =
            reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
        const uint32_t now_lo = p_reg[WALL_CLOCK_LOW_INDEX];
        const uint32_t now_hi = p_reg[WALL_CLOCK_HIGH_INDEX];
        const uint64_t now = (static_cast<uint64_t>(now_hi) << 32) | now_lo;
        // A real stamp lies between its earliest possible time and the clock; one outside is moved by 2^32 when that
        // puts it inside.
        auto repair = [now](volatile realtime_profiler_timestamp_t* stamp, uint64_t earliest) {
            uint64_t time = (static_cast<uint64_t>(stamp->time_hi) << 32) | stamp->time_lo;
            if (time > now + (1ull << 31)) {
                time -= 1ull << 32;
            } else if (time < earliest && time + (1ull << 32) <= now) {
                time += 1ull << 32;
            }
            stamp->time_hi = static_cast<uint32_t>(time >> 32);
            return time;
        };
        const uint64_t repaired_start = repair(&record->kernel_start, 0);
        repair(&record->kernel_end, repaired_start);
    }

    // A nonzero kernel_end.header is how long dispatch_s waited for a free slot, starting at this record's start.
    // Zero it in dispatch_s's slot before the slot is handed back (the ack below follows on the same NOC path),
    // so a reused slot never reports the same wait twice.
    const uint32_t stall_cycles = record->kernel_end.header;
    if (stall_cycles != 0) {
        record->kernel_end.header = 0;
        noc_inline_dw_write(
            get_noc_addr(
                DISPATCH_CORE_NOC_X,
                DISPATCH_CORE_NOC_Y,
                dispatch_data_addr + offsetof(realtime_profiler_record_t, kernel_end) +
                    offsetof(realtime_profiler_timestamp_t, header)),
            0);
    }
    const uint64_t start = (static_cast<uint64_t>(record->kernel_start.time_hi) << 32) | record->kernel_start.time_lo;

    const uint32_t id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr)[2];
    if (id != REALTIME_PROFILER_UNPROFILED_PROGRAM_HOST_ID) {
        ring_buffer->write_index++;
    }

    if (stall_cycles != 0) {
        realtime_profiler_enqueue_stall_marker(start + stall_cycles, stall_cycles);
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
    }
}
