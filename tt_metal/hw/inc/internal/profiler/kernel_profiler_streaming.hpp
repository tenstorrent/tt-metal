// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SPSC device kernel profiler: each RISC streams markers into its own single-producer/single-consumer ring in
// L1 and a resident relay empties it; a full ring blocks the producer, so the stream is lossless. Per
// RISC r: storage profiler_data_buffer[r].data[0..PROFILER_L1_VECTOR_SIZE), tail = control[SPSC_RING_TAIL_0 + r]
// (producer), head = control[SPSC_RING_HEAD_0 + r] (relay); both are monotonic word counts, index = count %
// capacity. The wire types (ppfmt) and SpscControlBuffer slots are its own, in hostdev/streaming_profiler_common.h;
// ControlBuffer/PacketTypes belong to the DRAM backend, and sharing a control word breaks both.
//
// SELECTION: this header is reached ONLY through api/debug/kernel_profiler.hpp, which includes it when
// the JIT build defines PROFILE_STREAMING (TT_METAL_STREAMING_PROFILER=1, jit_build/build.cpp). Without that
// define kernel_profiler.hpp is the DRAM profiler's own text, untouched. The two are never compiled into the
// same process: llrt/rtoptions.cpp TT_FATALs when both env vars are set. Never include this file directly.
//
// Quasar has no DRISC drainer, so an SPSC producer there would block forever on the first full ring. There
// is deliberately NO fallback to the DRAM producer: the host refuses TT_METAL_STREAMING_PROFILER on Quasar
// (jit_build/build.cpp, distributed/mesh_device.cpp), and this guard is the device-side backstop.

#pragma once

#if defined(ARCH_QUASAR)
#error "The streaming profiler has no Quasar producer; TT_METAL_STREAMING_PROFILER is not supported on Quasar."
#endif
#if !defined(PROFILE_STREAMING)
#error \
    "kernel_profiler_streaming.hpp is selected by kernel_profiler.hpp under PROFILE_STREAMING; do not include it directly."
#endif

#include <algorithm>
#include <climits>

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_ERISC) || \
    defined(COMPILE_FOR_IDLE_ERISC) || defined(COMPILE_FOR_AERISC) || defined(COMPILE_FOR_DM)
#include "risc_common.h"
#include "internal/dataflow/dataflow_api_addrgen.h"
#include "api/tensor/tensor_accessor.h"
#else
#include "ckernel.h"
#endif

#include "hostdev/streaming_profiler_common.h"
#include "internal/risc_attribs.h"

#include "hostdev/dev_msgs.h"

#include "internal/ethernet/erisc.h"

// PROFILE_KERNEL is a global JIT define, so dispatch kernels get it too; no relay serves a dispatch core, so a
// producer there would fill its ring and wedge the next relay bring-up. The relay kernel is excluded as well:
// this producer is ~1 KB it has no code room for (its self-profiling is its own staging-slot markers, see
// impl/streaming_profiler/kernels/drisc_relay.cpp).
#if defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL) && !defined(STREAMING_PROFILER_RELAY_KERNEL)

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_ERISC)
// Global scope: a block-scope extern inside the namespace would look for kernel_profiler::__stack_base and
// fail to link.
extern uint32_t __stack_base[];
#endif

namespace kernel_profiler {

extern uint32_t wIndex;  // producer tail: monotonic word count, lives in FW .bss across launches

// publish_tail() advances the tail consumers see only while this is true. The RISCs that validate launches clear it in
// init_profiler(), and DeviceValidateProfiler sets it once the launch is known to be real, so an idle launch's markers
// are rewound instead of published.
//
// It is defined here for firmware builds rather than in each hw/firmware/src/*.cc, which only reach this producer
// through the shared DeviceZone* and DeviceValidateProfiler macros. A kernel build takes the symbol from its firmware
// ELF (--just-symbols), as it does wIndex.
#if !defined(KERNEL_BUILD)
bool zoneValid __attribute__((used)) = true;
#else
extern bool zoneValid;
#endif

// The RISCs whose firmware loop decides per-launch validity. Only these defer the first publish. On an active eth core
// only ERISC0's firmware runs the launch loop.
#if defined(COMPILE_FOR_BRISC) || (defined(COMPILE_FOR_AERISC) && (COMPILE_FOR_AERISC == 0)) || defined(COMPILE_FOR_DM)
inline constexpr bool PROFILER_VALIDATES_ZONE = true;
#else
inline constexpr bool PROFILER_VALIDATES_ZONE = false;
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

constexpr uint32_t PROFILER_STALL_ZONE_ID = TT_ZONE_STALL_ID;

// A value no clock high half or runtime id takes, which these words hold when they have none: no timer word yet this
// launch, no runtime id awaiting publish, no cursor, no pending stall.
constexpr uint32_t kUnset = 0xFFFFFFFFu;

// Last high half emitted in a STICKY_TIMER; kUnset forces a fresh sticky on a launch's first marker.
[[maybe_unused]] static uint32_t g_prev_timer_hi = kUnset;
[[maybe_unused]] static uint32_t g_prog_pending = kUnset;  // validators: the runtime id awaiting the launch's publish

struct PendingStall {
    uint32_t start_hi = kUnset, start_lo = 0, end_hi = 0, end_lo = 0;
};

// The write index past which the smallest reservation no longer fits, from a cached copy of the relay's head. The head
// only advances, so a stale copy is conservative; the fast path compares against this word and reaches L1 once per
// drained batch. 0 is the safe floor.
//
// It and the pending stall are the RISC's, shared by its firmware and kernels, so a stall one of them waited out is
// written by whichever records next. A kernel build takes both from its firmware ELF, as it does wIndex.
#if !defined(KERNEL_BUILD)
uint32_t g_room_end __attribute__((used)) = 0;
PendingStall g_stall __attribute__((used));
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

FORCE_INLINE void publish_tail() {
    if constexpr (PROFILER_VALIDATES_ZONE) {
        if (!zoneValid) {
            return;
        }
    }
    // The relay reads TAIL then the slots over the NoC, and stores can reach L1 SRAM out of order.
    asm volatile("fence" ::: "memory");
    profiler_control_buffer[TAIL_INDEX] = wIndex;
}

// A lane-state slot must never show a value the published tail does not cover, so the tail goes first and a fence
// separates the two stores (see SPSC_STATE_TIMER_0).
FORCE_INLINE void publish_state(uint32_t index, uint32_t value) {
    if constexpr (PROFILER_VALIDATES_ZONE) {
        if (!zoneValid) {
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
FORCE_INLINE void publish_tail_batched(uint32_t words_written) {
    constexpr uint32_t kBatchShift = __builtin_ctz(SPSC_PUBLISH_BATCH_WORDS);
    static_assert((1u << kBatchShift) == SPSC_PUBLISH_BATCH_WORDS, "batch must be a power of two");
    if (__builtin_expect((wIndex >> kBatchShift) != ((wIndex - words_written) >> kBatchShift), 0)) {
        publish_tail();
    }
}

FORCE_INLINE void ring_write_word(uint32_t word) {
    profiler_data_buffer[myRiscID].data[wIndex % RING_CAPACITY] = word;
    wIndex++;
}

// Whether `words` words from write index `index` lie before the ring's end, so they store as one block from
// ring_slots(index); otherwise they go through ring_write_word, which wraps.
FORCE_INLINE bool ring_fits_before_end(uint32_t index, uint32_t words) {
    return index % RING_CAPACITY <= RING_CAPACITY - words;
}
FORCE_INLINE volatile tt_l1_ptr uint32_t* ring_slots(uint32_t index) {
    return &profiler_data_buffer[myRiscID].data[index % RING_CAPACITY];
}

using AliasingWord __attribute__((may_alias)) = uint32_t;

// Copies words into the ring one at a time, so they may wrap.
__attribute__((noinline, cold)) void ring_copy_words(const AliasingWord* from, uint32_t words) {
    for (uint32_t k = 0; k < words; ++k) {
        ring_write_word(from[k]);
    }
}

// The high half moves about once per 3.2 s, so test before storing; every caller's room reservation already
// covers the sticky word.
FORCE_INLINE void ring_write_sticky_timer(uint32_t time_hi) {
    if (__builtin_expect(time_hi != g_prev_timer_hi, 0)) {
        ring_write_word(ppfmt::w0(ppfmt::T_STICKY_TIMER, time_hi));
        g_prev_timer_hi = time_hi;
        publish_state(STATE_TIMER_INDEX, time_hi);
    }
}

// This ELF's copy of the lane's cursor, the time of its last short or medium record of a call site with a short form
// (internal/debug/device_publish.h), which the decoder keeps too. A high word of kUnset is no cursor, which no short
// record matches.
[[maybe_unused]] static uint32_t g_cursor_lo = 0;
[[maybe_unused]] static uint32_t g_cursor_hi = kUnset;

// A stall writes into a ring that is by definition full, so its words come from a reserve ordinary records may not fill
// into: the timer's high word, word 0 and the stall's body.
constexpr uint32_t STALL_RESERVE_WORDS = 2 + ppfmt::STALL_BODY_WORDS;
constexpr uint32_t RING_USABLE = RING_CAPACITY - STALL_RESERVE_WORDS;
static_assert(RING_USABLE > STALL_RESERVE_WORDS, "the ring is too small to carry a stall reserve");

// Every check reserves at least a zone record's longest packet (word 0, the time's two words and the start's 64-bit
// distance), so the commonest check compares wIndex itself.
constexpr uint32_t MIN_RESERVE_WORDS = 5;

#if defined(PROFILE_INLINE_ENABLED)
#define PROFILER_INLINE_ATTR inline __attribute__((always_inline))
#else
#define PROFILER_INLINE_ATTR __attribute__((noinline))
#endif

// The pending stall, straight into the reserve its wait left room for, as a zone's medium record
// (internal/debug/device_publish.h) at the stall's call site, which has no ELF entry: the end is the record's time and
// the start a 32-bit distance back. A stall of 2^32 cycles or more saturates it, because the reserve has no room for a
// full record. A 3.2 s wait is a wedged relay, not a measurement.
FORCE_INLINE void write_pending_stall() {
    const uint32_t end_hi = g_stall.end_hi, end_lo = g_stall.end_lo;
    const uint32_t duration_lo = end_lo - g_stall.start_lo;
    const uint32_t duration_hi = end_hi - g_stall.start_hi - (end_lo < g_stall.start_lo);
    const uint32_t duration = duration_lo | (0u - static_cast<uint32_t>(duration_hi != 0));
    ring_write_sticky_timer(end_hi);
    ring_write_word(ppfmt::record_w0(ppfmt::STALL_BODY_WORDS) | PROFILER_STALL_ZONE_ID);
    ring_write_word(end_lo);
    ring_write_word(duration);
    g_stall.start_hi = kUnset;
    // The stall is the back-pressure signal and its wait already cost a full stall, so it publishes at once.
    publish_tail();
}

// The caller read its time first, so its record goes into the stream before the stall, and a stall never sits inside
// the zone it interrupted. After a wait the cached head looks full, so the next record's room check comes back here and
// writes the stall first.
__attribute__((noinline, cold)) void ring_ensure_room_slow(uint32_t reserve) {
    // Invalidate before the refresh: the relay's head write-back arrives over the NoC, which the core's L1 read cache
    // does not observe.
    invalidate_l1_cache();
    const uint32_t head = profiler_control_buffer[HEAD_INDEX];
    g_room_end = head + (RING_USABLE - MIN_RESERVE_WORDS);
    if (g_stall.start_hi != kUnset) {
        write_pending_stall();
    }
    if ((wIndex - head) <= (RING_USABLE - reserve)) {
        return;
    }
    if constexpr (myRiscID < SPSC_STALL_COUNT_MAX) {
        profiler_control_buffer[SPSC_STALL_COUNT_0 + myRiscID]++;
    }
    read_wall_clock(g_stall.start_hi, g_stall.start_lo);
    // The relay can only free words up to the published tail, so waiting on unpublished words deadlocks.
    publish_tail();
    while ((wIndex - profiler_control_buffer[HEAD_INDEX]) > (RING_USABLE - reserve - STALL_RESERVE_WORDS)) {
        invalidate_l1_cache();  // re-read the relay-updated head (and the arm flag)
        if (!profiler_control_buffer[PROFILER_ARMED]) {
            break;  // nobody drains this ring (or teardown disarmed it): overwrite rather than wait
        }
    }
    read_wall_clock(g_stall.end_hi, g_stall.end_lo);
    g_room_end = wIndex - RING_CAPACITY;
}

// One local compare against the cached room end, signed because wIndex can pass it by a record.
FORCE_INLINE void ring_ensure_room(uint32_t words) {
    const uint32_t reserve = std::max(words, MIN_RESERVE_WORDS);
    if (__builtin_expect(static_cast<int32_t>(wIndex + (reserve - MIN_RESERVE_WORDS) - g_room_end) > 0, 0)) {
        ring_ensure_room_slow(reserve);
    }
}

// Writes one record timed now, which Writer builds from `header`, its call site's word 0, and `inputs`. The clock is
// read before the room check, so a stall the check waits out follows the record rather than lengthening it.
template <typename Writer, bool kImmediately = false, typename... Inputs>
PROFILER_INLINE_ATTR void publish_with(uint32_t header, Inputs... inputs) {
    uint32_t time_hi, time_lo;
    read_wall_clock(time_hi, time_lo);
    ring_ensure_room(Writer::kMaxWords);
    const uint32_t start = wIndex;
    Writer::write(header, time_hi, time_lo, inputs...);
    if constexpr (kImmediately) {
        publish_tail();
    } else {
        publish_tail_batched(wIndex - start);
    }
}

// DeviceZoneSetCounter hook: the runtime host-id goes in band as a STICKY_PROG the host forward-fills onto
// this lane's following markers. Every RISC emits one at its own launch point; a sweep-granular id
// misassigns about twice as many zones on back-to-back launches.
FORCE_INLINE void set_host_counter(uint32_t counter_value) {
    // An id that reaches the type field takes the two-word sticky.
    if (counter_value >> ppfmt::TYPE_SHIFT) {
        ring_ensure_room(2);
        ring_write_word(ppfmt::w0(ppfmt::T_STICKY_PROG_EXT, 0));
        ring_write_word(counter_value);
    } else {
        ring_ensure_room(1);
        ring_write_word(ppfmt::w0(ppfmt::T_STICKY_PROG, counter_value));
    }
    if constexpr (PROFILER_VALIDATES_ZONE) {
        g_prog_pending = counter_value;
    }
    publish_state(STATE_PROG_INDEX, counter_value);
}

FORCE_INLINE void set_profiler_zone_valid(bool condition) {
    zoneValid = condition;
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
        // the decoder sees, so invalidate it.
        wIndex = profiler_control_buffer[TAIL_INDEX];
        g_cursor_hi = kUnset;
        g_stall.start_hi = kUnset;
    }
}

__attribute__((noinline)) void init_profiler(
    uint16_t briscKernelID = 0, uint16_t ncriscKernelID = 0, uint16_t triscsKernelID = 0) {
    // Seeded from TAIL_INDEX once per FW session, then monotonic across launches: the relay tracks its own head,
    // so re-reading the per-program-reset TAIL would rewind below it and duplicate zones.
    static bool s_windex_seeded = false;
    if (!s_windex_seeded) {
        wIndex = profiler_control_buffer[TAIL_INDEX];
        s_windex_seeded = true;
    }

    // Fresh STICKY_TIMER on this launch's first marker (guards the idle-launch rewind).
    g_prev_timer_hi = kUnset;
    // After a rewind the decoder's cursor is not where ours is; drop ours so the first record isn't short.
    g_cursor_hi = kUnset;

    // Validators defer publishing until DeviceValidateProfiler resolves the launch.
    if constexpr (PROFILER_VALIDATES_ZONE) {
        zoneValid = false;
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

// The slow room check, asked for no room, only writes a stall still pending.
__attribute__((noinline)) void finish_profiler() {
    ring_ensure_room_slow(0);
    publish_tail();
}

// Lifecycle only, no markers. Every kernel must be wrapped or nothing it records is published.
struct profileScopeLifecycle {
    FORCE_INLINE profileScopeLifecycle() { init_profiler(); }
    FORCE_INLINE ~profileScopeLifecycle() { finish_profiler(); }
};

}  // namespace kernel_profiler

#include "noc_event_profiler.hpp"
#include "perf_counters.hpp"
#include "internal/profiler/synchronization_event_profiler.hpp"
#include "internal/debug/device_publish.h"

namespace kernel_profiler {

// Watcher-off net for a globals-heavy kernel whose loader-guaranteed stack floor is only MEM_*_STACK_MIN_SIZE
// (192-256 B): plant one word at __stack_base when the kernel zone opens, check it at close, publish a
// STACK-OVERFLOW event if frame data overwrote it. The pattern is the watcher's stack_usage_pattern on purpose, so an
// intact canary reads as painted-and-unused to measure_stack_usage(). Compiled out for firmware
// (__stack_base is a kernel-link symbol) and active ERISC (stack guarded by -Werror=stack-usage).
#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_ERISC)
constexpr uint32_t STACK_CANARY_PATTERN = 0xBABABABA;  // == watcher stack_usage_pattern
struct stackCanaryScope {
    FORCE_INLINE stackCanaryScope() { ::__stack_base[0] = STACK_CANARY_PATTERN; }
    FORCE_INLINE ~stackCanaryScope() {
        if (__builtin_expect(::__stack_base[0] != STACK_CANARY_PATTERN, 0)) {
            DevicePublish<sp::EventMetadata{"STACK-OVERFLOW"_s}>();
        }
    }
};
#else
struct stackCanaryScope {};  // FW builds and active ERISC: no kernel stack floor to watch
#endif

// A zone gated on a bool evaluated once at entry: false reads no clock and writes nothing, and a constant argument
// folds to the plain zone or to no code. Its open state is only its 8-byte start, since more would be register pressure
// across the user code inside the zone, and a globals-maxed kernel on the 192-256 B loader stack floor already gets
// tight around 10-20 open zones.
template <typename Site, auto kMetadata>
struct profileScope {
    bool on;
    sp::Timestamp start;
    FORCE_INLINE profileScope(bool active) : on(active) {
        if (on) {
            start = sp::Timestamp::now();
        }
    }
    FORCE_INLINE ~profileScope() {
        if (on) {
            Site::template publish<kMetadata>(records::ZoneStart{start});
        }
    }
};

}  // namespace kernel_profiler

#define DeviceZoneScopedNIf(name, active)                                                                         \
    kernel_profiler::profileScope<DevicePublishSite, ::kernel_profiler::sp::ZoneMetadata{TT_RECORD_STRING(name)}> \
        zone{active};

#define DeviceZoneScopedN(name) DeviceZoneScopedNIf(name, true)

#define DeviceTimestampedData(name, data) \
    DevicePublish<::kernel_profiler::sp::TimestampedDataMetadata{TT_RECORD_STRING(name)}>(static_cast<uint64_t>(data))

#define DeviceRecordEvent(name) DevicePublish<::kernel_profiler::sp::EventMetadata{TT_RECORD_STRING(name)}>()

#define DeviceValidateProfiler(condition) kernel_profiler::set_profiler_zone_valid(condition);

#define DeviceZoneScopedMainN(name) \
    kernel_profiler::profileScopeLifecycle zone_fw_lifecycle = kernel_profiler::profileScopeLifecycle();

// The canary is declared second so its check runs first at scope exit, inside the still-open kernel zone.
#define DeviceZoneScopedMainChildN(name) \
    DeviceZoneScopedN(name);             \
    kernel_profiler::stackCanaryScope zone_stack_canary = kernel_profiler::stackCanaryScope();

#define DeviceZoneSetCounter(counter) kernel_profiler::set_host_counter(counter);

// Shared firmware calls these unconditionally; they are real on the DRAM backend (trace replay) and empty
// here.
#define DeviceProfilerInit()
#define DeviceTraceOnlyProfilerInit()
#define DeviceIncrementTraceCount()

// NOT SUPPORTED on this backend, kept as null macros so the shared call sites compile: sum zones
// (llk_io_{pack,unpack}.h "CB-COMPUTE-*" accumulators) are a DRAM-profiler feature; the streaming wire ships
// every zone whole instead of summing on device.
#define DeviceZoneScopedSumN1(name) (void(name))
#define DeviceZoneScopedSumN2(name) (void(name))

#else

// Zero codegen when profiling is off.
#define DeviceValidateProfiler(condition) (void(sizeof(condition)))

#define DeviceZoneScopedMainN(name) (void(name))

#define DeviceZoneScopedMainChildN(name) (void(name))

#define DeviceZoneScopedN(name) (void(name))

#define DeviceZoneScopedNIf(name, active) (void(name), void(sizeof(active)))

#define DeviceZoneScopedSumN1(name) (void(name))

#define DeviceZoneScopedSumN2(name) (void(name))

#define DeviceRecordEvent(event_id) (void(sizeof(event_id)))

#define DeviceTraceOnlyProfilerInit()

#define DeviceZoneSetCounter(counter) (void(sizeof(counter)))

#define DeviceTimestampedData(data_id, data) (void(sizeof(data_id) + sizeof(data)))

#define DeviceProfilerInit()

#define DeviceIncrementTraceCount()

// null macros when noc tracing is disabled
#define RECORD_NOC_EVENT_WITH_ADDR(type, local_addr, noc_addr, num_bytes, vc, posted, noc)
#define RECORD_NOC_EVENT_WITH_ID(type, local_addr, noc_id, addrgen, offset, num_bytes, vc, posted, noc)
#define RECORD_NOC_EVENT(type, posted, noc)
#define NOC_TRACE_QUICK_PUSH_IF_LINKED(cmd_buf, linked)

// null macros when noc debugging is disabled
#define RECORD_SCOPED_LOCK_EVENT(event_type, locked_address_base, num_bytes)

// null macros when perf counters are disabled
#define StartPerfCounters()
#define StopPerfCounters()
#define RecordPerfCounters()

#define SYNC_WAIT(name, key) (void(sizeof(key)))
#define SYNC_WAIT_ZONE(name)
#define SYNC_WAIT_KEY(name, key) (void(sizeof(key)))
#define SYNC_SIGNAL(name, key) (void(sizeof(key)))
#define SYNC_SIGNAL_NOC_ADDR(name, addr, noc) (void(sizeof(addr) + sizeof(noc)))

#endif
