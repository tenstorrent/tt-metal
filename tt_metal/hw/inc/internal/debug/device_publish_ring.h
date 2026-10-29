// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DevicePublish's ring (internal/debug/device_publish_encoding.h writes records on it): each RISC streams its packets
// into its own single-producer/single-consumer ring in L1 and a resident relay empties it; a full ring blocks the
// producer, so the stream is lossless. Per RISC r: storage profiler_data_buffer[r].data[0..PROFILER_L1_VECTOR_SIZE),
// tail = control[SPSC_RING_TAIL_0 + r] (producer), head = control[SPSC_RING_HEAD_0 + r] (relay); both are monotonic
// word counts, index = count % capacity. The wire types (ppfmt) and SpscControlBuffer slots are its own, in
// hostdev/streaming_profiler_common.h; ControlBuffer/PacketTypes belong to the DRAM backend, and sharing a control word
// breaks both. Firmware wraps every kernel in its launch lifecycle (LaunchScope), which publishes the
// launch's packets.
//
// Quasar has no DRISC drainer, so a producer there would block forever on the first full ring. There is deliberately NO
// fallback to the DRAM producer: the host refuses TT_METAL_DEVICE_OBSERVABILITY on Quasar (jit_build/build.cpp,
// distributed/mesh_device.cpp), and this guard is the device-side backstop.
#pragma once

#if defined(ARCH_QUASAR)
#error "Device observability has no Quasar producer; TT_METAL_DEVICE_OBSERVABILITY is not supported on Quasar."
#endif

#include <algorithm>

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_ERISC) || \
    defined(COMPILE_FOR_IDLE_ERISC) || defined(COMPILE_FOR_AERISC) || defined(COMPILE_FOR_DM)
#include "risc_common.h"
#else
#include "ckernel.h"
#endif

#include "hostdev/streaming_profiler_common.h"
#include "internal/risc_attribs.h"

#include "hostdev/dev_msgs.h"

#include "internal/ethernet/erisc.h"

namespace device_observability {

using namespace ::kernel_profiler;

// The RISC's producer state is defined here for firmware builds rather than in each hw/firmware/src/*.cc, which only
// reach this producer through the shared DeviceZone* and DeviceValidateProfiler macros. A kernel build takes it from
// its firmware ELF (--just-symbols).
//
// wIndex is the producer tail: a monotonic word count, kept across launches.
//
// publish_tail() advances the tail consumers see only while launchValid is true. The RISCs that validate launches clear
// it in begin_launch(), and DeviceValidateProfiler sets it once the launch is known to be real, so an idle launch's
// records are rewound instead of published.
#if !defined(KERNEL_BUILD)
[[gnu::used]] uint32_t wIndex;
[[gnu::used]] bool launchValid = true;
#else
extern uint32_t wIndex;
extern bool launchValid;
#endif

// The RISCs whose firmware loop decides per-launch validity. Only these defer the first publish. On an active eth core
// only ERISC0's firmware runs the launch loop.
#if defined(COMPILE_FOR_BRISC) || (defined(COMPILE_FOR_AERISC) && (COMPILE_FOR_AERISC == 0)) || defined(COMPILE_FOR_DM)
inline constexpr bool VALIDATES_LAUNCH = true;
#else
inline constexpr bool VALIDATES_LAUNCH = false;
#endif

constexpr int WALL_CLOCK_HIGH_INDEX = 2;  // WALL_CLOCK_H, the snapshot; 1 is the live half
constexpr int WALL_CLOCK_LOW_INDEX = 0;

volatile tt_l1_ptr uint32_t* const profiler_control_buffer =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.control_vector));

volatile tt_l1_ptr profiler_msg_buffer_t* const profiler_data_buffer =
    reinterpret_cast<volatile tt_l1_ptr profiler_msg_buffer_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.buffer));

constexpr uint32_t myRiscID = PROCESSOR_INDEX;

// This RISC's ring geometry (SpscControlBuffer slots, not the DRAM profiler's).
constexpr uint32_t RING_CAPACITY = PROFILER_L1_VECTOR_SIZE;  // words
constexpr uint32_t TAIL_INDEX = SPSC_RING_TAIL_0 + myRiscID;
constexpr uint32_t HEAD_INDEX = SPSC_RING_HEAD_0 + myRiscID;
constexpr uint32_t STATE_TIMER_INDEX = SPSC_STATE_TIMER_0 + myRiscID;
constexpr uint32_t STATE_PROG_INDEX = spsc_state_prog_word(myRiscID);
static_assert(myRiscID < PROFILER_SPSC_MAX_RISC, "this processor has no slot in the SPSC control layout");

// A value no clock high half or runtime id takes, which these words hold when they have none: no timer packet yet, no
// runtime id awaiting publish, no cursor.
constexpr uint32_t kUnset = 0xFFFFFFFFu;

// The clock's high half in this RISC's last timer packet, kUnset before its first, and that packet's write index.
// Firmware owns them, as it does wIndex, so they carry over launches: a timer packet goes only where the high half
// changes, about every 3.2 s, or after an idle launch rewinds the last one away (set_launch_valid).
#if !defined(KERNEL_BUILD)
[[gnu::used]] uint32_t g_prev_timer_hi = kUnset;
[[gnu::used]] uint32_t g_timer_written_at = 0;
#else
extern uint32_t g_prev_timer_hi;
extern uint32_t g_timer_written_at;
#endif
[[maybe_unused]] static uint32_t g_prog_pending = kUnset;  // validators: the runtime id awaiting the launch's publish

// A stall waited out but not yet written: its end, and its duration saturated to 32 bits. A duration of 0 is none.
struct PendingStall {
    uint32_t end_hi, end_lo, duration;
};

// The write index past which the smallest reservation no longer fits, from a cached copy of the relay's head. The head
// only advances, so a stale copy is conservative; the fast path compares against this word and reaches L1 once per
// drained batch. 0 is the safe floor.
//
// It and the pending stall are the RISC's, shared by its firmware and kernels, so a stall one of them waited out is
// written by whichever records next. A kernel build takes both from its firmware ELF, as it does wIndex.
#if !defined(KERNEL_BUILD)
[[gnu::used]] uint32_t g_room_end = 0;
[[gnu::used]] PendingStall g_stall;
#else
extern uint32_t g_room_end;
extern PendingStall g_stall;
#endif

// Reading L snapshots the high half into WALL_CLOCK_H and H returns it, so the order is the protocol. Measured on
// Blackhole (Tensix and DRISC): the snapshot holds only while the H load follows the L load immediately, so these two
// loads must stay adjacent. Another RISC's L read landing in that gap across a 2^32 boundary puts a marker ~3.2 s in
// the future, at ~1e-9 per read, which is cheaper than a retry branch here. One asm statement holds them together:
// as two volatile loads, a spill could still be scheduled between them.
FORCE_INLINE void read_wall_clock(uint32_t& hi, uint32_t& lo) {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    asm volatile("lw %0, %2\n\tlw %1, %3"
                 : "=&r"(lo), "=r"(hi)
                 : "m"(p_reg[WALL_CLOCK_LOW_INDEX]), "m"(p_reg[WALL_CLOCK_HIGH_INDEX]));
}

// Publishes `tail`, the write index past the records written so far: a writer passes the one it holds, so the store
// after the fence doesn't wait on a load of wIndex.
FORCE_INLINE void publish_tail(uint32_t tail) {
    if constexpr (VALIDATES_LAUNCH) {
        if (!launchValid) {
            return;
        }
    }
    // The relay reads TAIL then the slots over the NoC, and stores can reach L1 SRAM out of order.
    asm volatile("fence" ::: "memory");
    profiler_control_buffer[TAIL_INDEX] = tail;
}
FORCE_INLINE void publish_tail() { publish_tail(wIndex); }

// A lane-state slot must never show a value the published tail does not cover, so the tail goes first and a fence
// separates the two stores (see SPSC_STATE_TIMER_0).
FORCE_INLINE void publish_state(uint32_t index, uint32_t value) {
    if constexpr (VALIDATES_LAUNCH) {
        if (!launchValid) {
            return;
        }
    }
    publish_tail();
    asm volatile("fence" ::: "memory");
    profiler_control_buffer[index] = value;
}

// The fence pays the posted ring stores' latency, so paying it once per SPSC_PUBLISH_BATCH_WORDS is most of
// the close-side saving; the trigger is wIndex crossing a batch boundary. Visibility lags by at most one
// batch within a launch; launch boundaries and the stall path publish unconditionally, and blocking is
// head-vs-wIndex, so losslessness does not depend on the published tail.
FORCE_INLINE void publish_tail_batched(uint32_t start, uint32_t end) {
    constexpr uint32_t kBatchShift = __builtin_ctz(SPSC_PUBLISH_BATCH_WORDS);
    static_assert((1u << kBatchShift) == SPSC_PUBLISH_BATCH_WORDS, "batch must be a power of two");
    if (__builtin_expect((end >> kBatchShift) != (start >> kBatchShift), 0)) {
        publish_tail(end);
    }
}

// Whether `words` words from write index `index` lie before the ring's end, so they store as one block from
// ring_slots(index); otherwise each goes to ring_slots of its own index, which wraps.
FORCE_INLINE bool ring_fits_before_end(uint32_t index, uint32_t words) {
    return index % RING_CAPACITY <= RING_CAPACITY - words;
}
FORCE_INLINE volatile tt_l1_ptr uint32_t* ring_slots(uint32_t index) {
    return &profiler_data_buffer[myRiscID].data[index % RING_CAPACITY];
}

using AliasingWord [[gnu::may_alias]] = uint32_t;

// Copies `words` words to `to` in batches of 4, each batch's loads ahead of its stores so they overlap.
FORCE_INLINE void ring_copy_block(volatile tt_l1_ptr uint32_t* to, const AliasingWord* from, uint32_t words) {
    uint32_t k = 0;
    for (; k + 4 <= words; k += 4) {
        const uint32_t a = from[k], b = from[k + 1], c = from[k + 2], d = from[k + 3];
        to[k] = a;
        to[k + 1] = b;
        to[k + 2] = c;
        to[k + 3] = d;
    }
    for (; k < words; ++k) {
        to[k] = from[k];
    }
}

// Copies words into the ring from write index `index` on, which may wrap: as at most two blocks, up to the ring's end
// and then from its start.
[[gnu::noinline, gnu::cold]] void ring_copy_wrapping(uint32_t index, const AliasingWord* from, uint32_t words) {
    volatile tt_l1_ptr uint32_t* const ring = profiler_data_buffer[myRiscID].data;
    for (uint32_t at = index % RING_CAPACITY; words != 0; at = 0) {
        const uint32_t block = std::min(words, RING_CAPACITY - at);
        ring_copy_block(ring + at, from, block);
        from += block;
        words -= block;
    }
}

// Words written from a write index held in a register, each at its own ring index, which wraps, so a ring store doesn't
// make the compiler reload wIndex for the next word. Its writer moves wIndex past them once.
struct WrapStore {
    uint32_t index;

    FORCE_INLINE void write_word(uint32_t word, uint32_t value) const { *ring_slots(index + word) = value; }
    template <uint32_t kWords>
    FORCE_INLINE void copy_run(uint32_t word, const AliasingWord* from) const {
        ring_copy_wrapping(index + word, from, kWords);
    }
};

// Writes `words` at the write index and moves wIndex past them.
template <typename... Words>
FORCE_INLINE void ring_write_words(Words... words) {
    const uint32_t index = wIndex;
    uint32_t word = 0;
    (WrapStore{index}.write_word(word++, words), ...);
    wIndex = index + sizeof...(Words);
}

// A timer packet for the clock's high half, and the lane-state slot the host restarts a lane's decoding from. The high
// half moves about once per 3.2 s, so test before writing; every caller's room reservation already covers the timer
// packet. Inline, so a call doesn't make the caller reload wIndex and launchValid after the test.
FORCE_INLINE void ring_write_sticky_timer(uint32_t time_hi) {
    if (__builtin_expect(time_hi != g_prev_timer_hi, 0)) {
        g_timer_written_at = wIndex;
        ring_write_words(ppfmt::w0(PacketType::TimerHigh, time_hi));
        g_prev_timer_hi = time_hi;
        publish_state(STATE_TIMER_INDEX, time_hi);
    }
}

// This ELF's copy of the lane's cursor, the time of its last short or medium record of a call site with a short form
// (internal/debug/device_publish_encoding.h), which the decoder keeps too. A high word of kUnset is no cursor, which no
// short record matches.
[[maybe_unused]] static uint32_t g_cursor_lo = 0;
[[maybe_unused]] static uint32_t g_cursor_hi = kUnset;

// A stall writes into a ring that is by definition full, so its words come from a reserve ordinary records may not fill
// into: the timer's high word, word 0 and the stall's body.
constexpr uint32_t STALL_RESERVE_WORDS = 1 + ppfmt::STALL_PACKET_WORDS;
constexpr uint32_t RING_USABLE = RING_CAPACITY - STALL_RESERVE_WORDS;
static_assert(
    ppfmt::MAX_RECORD_WORDS == RING_USABLE - STALL_RESERVE_WORDS, "a record ppfmt allows fits the ring's room");
static_assert(RING_USABLE > STALL_RESERVE_WORDS, "the ring is too small to carry a stall reserve");

// Every check reserves at least a zone record's longest packet (word 0, the time's two words and the start's 64-bit
// distance), so the commonest check compares wIndex itself.
constexpr uint32_t MIN_RESERVE_WORDS = 5;

#if defined(DEVICE_OBSERVABILITY_INLINE_ENABLED)
#define WRITER_INLINE_ATTR [[gnu::always_inline]] inline
constexpr bool kWriterInlined = true;
#else
#define WRITER_INLINE_ATTR [[gnu::noinline]]
constexpr bool kWriterInlined = false;
#endif

// The pending stall, straight into the reserve its wait left room for, as ppfmt's stall record of the stall's call
// site, which no ELF describes (llrt/call_sites.cpp describes it from ppfmt): the end is the record's time and the
// start a 32-bit distance back. A stall of 2^32 cycles or more saturates it, because the reserve has no room for a full
// record. A 3.2 s wait is a wedged relay, not a measurement.
FORCE_INLINE void write_pending_stall(const PendingStall stall) {
    static_assert(
        ppfmt::STALL_PACKET_WORDS == 3 && ppfmt::STALL_START_DELTA_BITS == 32,
        "a stall's body is its end's low word, then its start's 32-bit distance back");
    ring_write_sticky_timer(stall.end_hi);
    ring_write_words(ppfmt::record_w0(ppfmt::STALL_PACKET_WORDS) | TT_STALL_CALL_SITE_ID, stall.end_lo, stall.duration);
    g_stall.duration = 0;
    // The stall is the back-pressure signal and its wait already cost a full stall, so it publishes at once.
    publish_tail();
}

// The caller read its time first, so its record goes into the stream before the stall, and a stall never sits inside
// the zone it interrupted. After a wait the cached head looks full, so the next record's room check comes back here and
// writes the stall first. A record's check (kForRecord) brings the record's time, which waits in the pending stall's
// end, meaningless while no stall is pending, rather than in a register either side would save, and starts the stall.
template <bool kForRecord>
FORCE_INLINE uint64_t ring_ensure_room_slow_body(uint32_t reserve, uint64_t time) {
    const PendingStall pending = g_stall;
    if constexpr (kForRecord) {
        g_stall.end_hi = static_cast<uint32_t>(time >> 32);
        g_stall.end_lo = static_cast<uint32_t>(time);
    }
    // Invalidate before the refresh: the relay's head write-back arrives over the NoC, which the core's L1 read cache
    // does not observe.
    invalidate_l1_cache();
    const uint32_t head = profiler_control_buffer[HEAD_INDEX];
    g_room_end = head + (RING_USABLE - MIN_RESERVE_WORDS);
    if (pending.duration != 0) {
        write_pending_stall(pending);
    }
    if ((wIndex - head) <= (RING_USABLE - reserve)) {
        return kForRecord ? uint64_t{g_stall.end_hi} << 32 | g_stall.end_lo : 0;
    }
    if constexpr (myRiscID < SPSC_STALL_COUNT_MAX) {
        profiler_control_buffer[SPSC_STALL_COUNT_0 + myRiscID]++;
    }
    uint32_t start_hi, start_lo;
    if constexpr (!kForRecord) {
        read_wall_clock(start_hi, start_lo);
    }
    // The relay can only free words up to the published tail, so waiting on unpublished words deadlocks.
    publish_tail();
    while ((wIndex - profiler_control_buffer[HEAD_INDEX]) > (RING_USABLE - reserve - STALL_RESERVE_WORDS)) {
        invalidate_l1_cache();  // re-read the relay-updated head (and the arm flag)
        if (!profiler_control_buffer[PROFILER_ARMED]) {
            break;  // nobody drains this ring (or teardown disarmed it): overwrite rather than wait
        }
    }
    uint32_t end_hi, end_lo;
    read_wall_clock(end_hi, end_lo);
    if constexpr (kForRecord) {
        start_hi = g_stall.end_hi;
        start_lo = g_stall.end_lo;
    }
    const uint32_t duration_hi = end_hi - start_hi - (end_lo < start_lo);
    const uint32_t duration = (end_lo - start_lo) | (0u - static_cast<uint32_t>(duration_hi != 0));
    g_stall = {end_hi, end_lo, duration | (duration == 0)};  // never 0, which is no stall
    g_room_end = wIndex - RING_CAPACITY;
    return uint64_t{start_hi} << 32 | start_lo;
}
[[gnu::noinline, gnu::cold]] void ring_ensure_room_slow(uint32_t reserve) {
    ring_ensure_room_slow_body<false>(reserve, 0);
}
[[gnu::noinline, gnu::cold]] uint64_t ring_ensure_room_slow_for_record(uint64_t time, uint32_t reserve) {
    return ring_ensure_room_slow_body<true>(reserve, time);
}

// A record's room check: its time, read first, goes through the slow check and back.
FORCE_INLINE void ring_ensure_room(uint32_t words, uint32_t& time_hi, uint32_t& time_lo) {
    const uint32_t reserve = std::max(words, MIN_RESERVE_WORDS);
    if (__builtin_expect(static_cast<int32_t>(wIndex + (reserve - MIN_RESERVE_WORDS) - g_room_end) > 0, 0)) {
        const uint64_t time = ring_ensure_room_slow_for_record(uint64_t{time_hi} << 32 | time_lo, reserve);
        time_hi = static_cast<uint32_t>(time >> 32);
        time_lo = static_cast<uint32_t>(time);
    }
}

// One local compare against the cached room end, signed because wIndex can pass it by a record.
FORCE_INLINE void ring_ensure_room(uint32_t words) {
    const uint32_t reserve = std::max(words, MIN_RESERVE_WORDS);
    if (__builtin_expect(static_cast<int32_t>(wIndex + (reserve - MIN_RESERVE_WORDS) - g_room_end) > 0, 0)) {
        ring_ensure_room_slow(reserve);
    }
}

// Writes one record timed now, which Writer builds from `header`, its call site's word 0, and `inputs`. The clock is
// read before the room check, so a stall the check waits out follows the record rather than lengthening it. Inlined at
// its call site (kAtCallSite), the record's time goes through the slow check and back, so the call site keeps no
// register across that call; an out-of-line writer's frame lasts only its call, so it keeps the time itself.
template <typename Writer, bool kImmediately, bool kAtCallSite, typename... Inputs>
FORCE_INLINE void write_record(uint32_t header, Inputs... inputs) {
    uint32_t time_hi, time_lo;
    read_wall_clock(time_hi, time_lo);
    if constexpr (kAtCallSite) {
        ring_ensure_room(Writer::kMaxWords, time_hi, time_lo);
    } else {
        ring_ensure_room(Writer::kMaxWords);
    }
    const uint32_t start = wIndex;
    const uint32_t end = Writer::write(header, time_hi, time_lo, inputs...);
    if constexpr (kImmediately) {
        publish_tail(end);
    } else {
        publish_tail_batched(start, end);
    }
}
template <typename Writer, bool kImmediately, typename... Inputs>
[[gnu::noinline]] void write_record_out_of_line(uint32_t header, Inputs... inputs) {
    write_record<Writer, kImmediately, false>(header, inputs...);
}

// write_record, inlined into the call site when the writer is inlined, unless the call site asked for kNoInline, which
// only an inlined writer passes.
template <typename Writer, bool kImmediately = false, bool kNoInline = false, typename... Inputs>
WRITER_INLINE_ATTR void publish_with(uint32_t header, Inputs... inputs) {
    if constexpr (kNoInline) {
        write_record_out_of_line<Writer, kImmediately>(header, inputs...);
    } else {
        write_record<Writer, kImmediately, kWriterInlined>(header, inputs...);
    }
}

// DeviceZoneSetCounter hook: the runtime host-id goes in band as a STICKY_PROG the host forward-fills onto
// this lane's following markers. Every RISC emits one at its own launch point; a sweep-granular id
// misassigns about twice as many zones on back-to-back launches.
FORCE_INLINE void set_host_counter(uint32_t counter_value) {
    // An id that reaches the type field takes the two-word sticky.
    if (counter_value > ppfmt::VALUE_MASK) {
        ring_ensure_room(2);
        ring_write_words(ppfmt::w0(PacketType::RuntimeIdWide, 0), counter_value);
    } else {
        ring_ensure_room(1);
        ring_write_words(ppfmt::w0(PacketType::RuntimeId, counter_value));
    }
    if constexpr (VALIDATES_LAUNCH) {
        g_prog_pending = counter_value;
    }
    publish_state(STATE_PROG_INDEX, counter_value);
}

FORCE_INLINE void set_launch_valid(bool condition) {
    launchValid = condition;
    if (condition) {
        publish_tail();
        asm volatile("fence" ::: "memory");
        if (g_prev_timer_hi != kUnset) {
            profiler_control_buffer[STATE_TIMER_INDEX] = g_prev_timer_hi;
        }
        if (g_prog_pending != kUnset) {
            profiler_control_buffer[STATE_PROG_INDEX] = g_prog_pending;
            g_prog_pending = kUnset;
        }
    } else {
        // Idle launch: rewind to the last committed tail; the rewound words may have moved our cursor past anything
        // the decoder sees, so invalidate it, and a timer packet among them never reaches the decoder, so the next
        // record writes another.
        wIndex = profiler_control_buffer[TAIL_INDEX];
        g_cursor_hi = kUnset;
        if (static_cast<int32_t>(g_timer_written_at - wIndex) >= 0) {
            g_prev_timer_hi = kUnset;
        }
        g_stall.duration = 0;
    }
}

[[gnu::noinline]] void begin_launch() {
    // Seeded from TAIL_INDEX once per FW session, then monotonic across launches: the relay tracks its own head,
    // so re-reading the per-program-reset TAIL would rewind below it and duplicate zones.
    static bool s_windex_seeded = false;
    if (!s_windex_seeded) {
        wIndex = profiler_control_buffer[TAIL_INDEX];
        s_windex_seeded = true;
    }

    // After a rewind the decoder's cursor is not where ours is; drop ours so the first record isn't short.
    g_cursor_hi = kUnset;

    // Validators defer publishing until DeviceValidateProfiler resolves the launch.
    if constexpr (VALIDATES_LAUNCH) {
        launchValid = false;
    }

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_TRISC)
    // Each lane records its own runtime id. The host applies a lane's last STICKY_PROG to the markers that follow it on
    // that lane, so every producing RISC must emit one per launch. The firmware only calls DeviceZoneSetCounter on
    // BRISC and the ERISCs, so NCRISC and TRISC read the launch message themselves, where DeviceZoneScopedMainN opens.
    // The firmware reads launch_msg_rd_ptr at that same point, right after the go signal that made the message valid.
    {
        volatile tt_l1_ptr uint32_t* rd_ptr = GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
        volatile tt_l1_ptr launch_msg_t* launch = GET_MAILBOX_ADDRESS_DEV(launch[0]);
        set_host_counter(launch[*rd_ptr].kernel_config.host_assigned_id);
    }
#endif
}

// A stall still pending at the launch's end has no later room check to write it, so the slow one runs, which writes it
// and publishes the tail.
[[gnu::noinline]] void end_launch() {
    if (__builtin_expect(g_stall.duration != 0, 0)) {
        ring_ensure_room_slow(0);
        return;
    }
    publish_tail();
}

// Lifecycle only, no markers. Every kernel must be wrapped or nothing it records is published.
struct LaunchScope {
    FORCE_INLINE LaunchScope() { begin_launch(); }
    FORCE_INLINE ~LaunchScope() { end_launch(); }
};

}  // namespace device_observability
