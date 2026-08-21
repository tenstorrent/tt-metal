// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SPSC device kernel profiler: each RISC streams markers into its own single-producer/single-consumer
// ring in L1, and a resident DRISC drainer continuously empties the rings. A full ring BLOCKS the
// producer (spin on the consumer head), so the stream is lossless and flow-controlled; no DRAM traffic.
//
//   Per RISC r: storage = profiler_data_buffer[r].data[0..PROFILER_L1_VECTOR_SIZE-1],
//   tail = profiler_control_buffer[SPSC_RING_TAIL_0 + r] (producer),
//   head = profiler_control_buffer[SPSC_RING_HEAD_0 + r] (drainer).
//   tail/head are MONOTONIC word counts; storage index = count % capacity.
//
// This file owns the PP_* wire types and the SpscControlBuffer slots; ControlBuffer/PacketTypes belong
// to the DRAM backend and must not appear here (sharing a control word has silently broken both).

#pragma once

// Quasar has no DRISC drainer, so an SPSC producer there would block forever on the first full ring;
// it keeps the DRAM producer. Both headers define the same public macro API.
#if defined(ARCH_QUASAR)
#include "tools/profiler/kernel_profiler_push.hpp"
#else

#include <climits>

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_ERISC) || \
    defined(COMPILE_FOR_IDLE_ERISC) || defined(COMPILE_FOR_AERISC) || defined(COMPILE_FOR_DM)
#include "risc_common.h"
#include "internal/dataflow/dataflow_api_addrgen.h"
#include "api/tensor/tensor_accessor.h"
#else
#include "ckernel.h"
#endif

#include "hostdevcommon/profiler_common.h"
#include "hostdevcommon/profiler_zone_id.h"
#include "internal/risc_attribs.h"

#include "hostdev/dev_msgs.h"

#include "internal/ethernet/erisc.h"

// LOAD-BEARING gate: PROFILE_KERNEL is a GLOBAL jit define (dispatch kernels get it too), dispatch
// kernels contain DeviceZoneScoped* sites, and no drainer serves dispatch cores. Without the
// !DISPATCH_KERNEL clause a dispatch core fills its blocking ring and the NEXT device open's drainer
// bring-up wedges at its write barrier (heartbeat stuck, phase=11).
#if defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL)
namespace kernel_profiler {

extern uint32_t wIndex;  // producer tail: monotonic word count, lives in FW .bss across launches

// Publish gate: publish_tail() only advances the consumer-visible tail while true. Validator RISCs
// clear it in init_profiler() and resolve it via DeviceValidateProfiler, so an idle launch's markers
// are rewound instead of published.
extern bool zoneValid;

// The RISCs whose FW loop resolves per-launch validity; only these defer the first publish.
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC) || \
    defined(COMPILE_FOR_AERISC) || defined(COMPILE_FOR_DM)
inline constexpr bool PROFILER_VALIDATES_ZONE = true;
#else
inline constexpr bool PROFILER_VALIDATES_ZONE = false;
#endif

constexpr int WALL_CLOCK_HIGH_INDEX = 1;
constexpr int WALL_CLOCK_LOW_INDEX = 0;

volatile tt_l1_ptr uint32_t* profiler_control_buffer =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.control_vector));

volatile tt_l1_ptr profiler_msg_buffer_t* profiler_data_buffer =
    reinterpret_cast<volatile tt_l1_ptr profiler_msg_buffer_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.buffer));

constexpr uint32_t myRiscID = PROCESSOR_INDEX;

// This RISC's ring geometry (SpscControlBuffer slots, not the DRAM profiler's).
constexpr uint32_t RING_CAPACITY = PROFILER_L1_VECTOR_SIZE;  // words
constexpr uint32_t TAIL_INDEX = SPSC_RING_TAIL_0 + myRiscID;
constexpr uint32_t HEAD_INDEX = SPSC_RING_HEAD_0 + myRiscID;
static_assert(myRiscID < PROFILER_SPSC_MAX_RISC, "this processor has no slot in the SPSC control layout");

enum class ZoneKind : uint32_t { Start = 0, End = 1 };

// The producer's back-pressure zone: an ordinary structural id, ELF-named like any kernel zone.
// At namespace scope only because profileScopeStall needs it.
TT_ZONE_DEFINE_ID(PROFILER_STALL_ZONE_ID, "PRODUCER-STALL");

// Wire encode; MUST stay in sync with tt_metal/tools/profiler/spsc_packet.h (inlined because the JIT
// build lacks that include path). word0 = type(5) | low27. A zone marker is 2 words: type|id27 +
// timer_low. Lane identity and time's high half are host-reconstructed from stickies: STICKY_PROG
// (runtime host-id, 1 word; 2-word PROG_EXT past 2^27), STICKY_TIMER (timer_hi, on high-half tick),
// STICKY_SRC (lane, injected by the drainer reader, never by the producer).
struct ppfmt {
    static constexpr uint32_t TYPE_SHIFT = 27;
    static constexpr uint32_t TYPE_MASK = 0x1Fu;
    static constexpr uint32_t LOW27_MASK = 0x7FFFFFFu;
    // This wire's OWN type space -- never pass a hostdevcommon PacketTypes value through (that aliased
    // unrelated types on this wire before). Retired values (11 = ZONE_TOTAL) are never reused.
    static constexpr uint32_t T_ZONE_START = 0u;        // PP_ZONE_START
    static constexpr uint32_t T_ZONE_END = 1u;          // PP_ZONE_END
    static constexpr uint32_t T_STICKY_PROG = 8u;       // PP_STICKY_PROG
    static constexpr uint32_t T_STICKY_PROG_EXT = 14u;  // PP_STICKY_PROG_EXT
    static constexpr uint32_t T_STICKY_TIMER = 9u;      // PP_STICKY_TIMER
    static constexpr uint32_t T_DATA = 10u;             // PP_DATA
    static constexpr uint32_t T_EVENT = 12u;            // PP_EVENT
    static constexpr uint32_t DATA_SIZE_SHIFT = 25u;    // PP_DATA_SIZE_SHIFT (in word2)
    static constexpr uint32_t DATA_SIZE_MASK = 0x7Fu;   // PP_DATA_SIZE_MASK
    static inline uint32_t w0(uint32_t type, uint32_t low27) {
        return ((type & TYPE_MASK) << TYPE_SHIFT) | (low27 & LOW27_MASK);
    }
    // Full 27-bit structural id; the kind travels as its own argument, never packed into the id.
    static inline uint32_t zone_w0(uint32_t id, ZoneKind kind) {
        return w0(kind == ZoneKind::End ? T_ZONE_END : T_ZONE_START, id & LOW27_MASK);
    }
    // PP_DATA: word0 shaped exactly like a zone marker's; the payload length rides in its own word2.
    static inline uint32_t data_w0(uint32_t id) { return w0(T_DATA, id & LOW27_MASK); }
    static inline uint32_t data_w2(uint32_t size_words) { return (size_words & DATA_SIZE_MASK) << DATA_SIZE_SHIFT; }
    // PP_EVENT: 2 words, no payload, no size word.
    static inline uint32_t event_w0(uint32_t id) { return w0(T_EVENT, id & LOW27_MASK); }
};

static constexpr uint32_t SPSC_MARKER_WORDS = 2;

// Last high half emitted in a STICKY_TIMER; ~0 forces a fresh sticky on a launch's first marker.
[[maybe_unused]] static uint32_t g_prev_timer_hi = 0xFFFFFFFFu;

// Branchless latched read: reading L latches the high half, H returns it -- the ORDER is the protocol.
// Known tradeoff (tt-isa-documentation, TensixTile/DebugTimestamper.md): the latch is single-agent; a
// concurrent RISC's L read re-latching in our L->H gap across a 2^32 boundary lands a marker +2^32
// cycles (~3.2 s) in the future. ~1e-9..1e-8 per read and loud on the host (one lane's
// order-regression counter storms), so we skip the retry branch on this hot path.
inline __attribute__((always_inline)) void read_wall_clock(uint32_t& hi, uint32_t& lo) {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    lo = p_reg[WALL_CLOCK_LOW_INDEX];   // latches the high half
    hi = p_reg[WALL_CLOCK_HIGH_INDEX];  // returns the latched value
}

// The stall reserve: a naive stall scope would emit ZONE_START into the very ring whose fullness
// caused the stall -> deadlock. Ordinary markers may only fill to RING_USABLE; the reserve belongs to
// the stall zone alone. Each half = 2-word marker + the 1-word STICKY_TIMER a ~3.2 s stall can straddle.
constexpr uint32_t STALL_ZONE_HALF_WORDS = SPSC_MARKER_WORDS + 1;
constexpr uint32_t STALL_RESERVE_WORDS = 2 * STALL_ZONE_HALF_WORDS;
constexpr uint32_t RING_USABLE = RING_CAPACITY - STALL_RESERVE_WORDS;
static_assert(RING_USABLE > STALL_RESERVE_WORDS, "the ring is too small to carry a stall reserve");

// One stall-zone half, straight into the reserve with NO room check (checking would recurse).
inline __attribute__((always_inline)) void stall_mark(ZoneKind kind) {
    uint32_t hi, lo;
    read_wall_clock(hi, lo);
    if (hi != g_prev_timer_hi) {
        profiler_data_buffer[myRiscID].data[wIndex++ % RING_CAPACITY] = ppfmt::w0(ppfmt::T_STICKY_TIMER, hi);
        g_prev_timer_hi = hi;
    }
    profiler_data_buffer[myRiscID].data[wIndex++ % RING_CAPACITY] = ppfmt::zone_w0(PROFILER_STALL_ZONE_ID, kind);
    profiler_data_buffer[myRiscID].data[wIndex++ % RING_CAPACITY] = lo;
}

// RAII stall zone; distinct from profileScope only because it writes into the reserve.
struct profileScopeStall {
    inline __attribute__((always_inline)) profileScopeStall() { stall_mark(ZoneKind::Start); }
    inline __attribute__((always_inline)) ~profileScopeStall() { stall_mark(ZoneKind::End); }
};

// Full-ring path, out-of-line on purpose (one copy, not one per zone site). Bumps the L1 stall
// counter (the host's decode-free knee ground truth), opens the stall zone, then waits for the
// caller's words AND the zone's own closing half so the reserve is whole again for the next stall.
__attribute__((noinline)) void ring_ensure_room_slow(uint32_t nwords) {
    if constexpr (myRiscID < SPSC_STALL_COUNT_MAX) {
        profiler_control_buffer[SPSC_STALL_COUNT_0 + myRiscID]++;
    }
    profileScopeStall stall;
    while ((wIndex - profiler_control_buffer[HEAD_INDEX]) > (RING_USABLE - nwords - STALL_ZONE_HALF_WORDS)) {
        invalidate_l1_cache();  // re-read the drainer-updated head (and the terminate flag)
        if (profiler_control_buffer[PROFILER_TERMINATE]) {
            return;  // teardown: stop waiting on a dead ring; the destructor still closes the zone
        }
    }
}

// Fast path: one compare, against RING_USABLE (never RING_CAPACITY -- the difference is the reserve).
inline __attribute__((always_inline)) void ring_ensure_room(uint32_t nwords) {
    if ((wIndex - profiler_control_buffer[HEAD_INDEX]) <= (RING_USABLE - nwords)) {
        return;
    }
    ring_ensure_room_slow(nwords);
}

inline __attribute__((always_inline)) void ring_write_word(uint32_t v) {
    profiler_data_buffer[myRiscID].data[wIndex % RING_CAPACITY] = v;
    wIndex++;
}

inline __attribute__((always_inline)) void publish_tail() {
    if (zoneValid) {
        // Fence so the marker stores land before the tail: the drainer reads TAIL then the slots over
        // the NoC, and the stores can otherwise reach L1 SRAM out of order.
        asm volatile("fence" ::: "memory");
        profiler_control_buffer[TAIL_INDEX] = wIndex;
    }
}

// Zone marker emit. Reserve room BEFORE reading the clock: a stall must elongate the marker's
// timestamp, or the marker would carry a pre-stall time yet sit after the (later) stall zone -- a
// backwards jump on the lane. Worst case is 1-word sticky + 2-word marker, so the check runs once.
inline __attribute__((always_inline)) void mark_time(uint32_t timer_id, ZoneKind kind = ZoneKind::Start) {
    ring_ensure_room(SPSC_MARKER_WORDS + 1);
    uint32_t hi, lo;
    read_wall_clock(hi, lo);
    if (hi != g_prev_timer_hi) {
        ring_write_word(ppfmt::w0(ppfmt::T_STICKY_TIMER, hi));
        g_prev_timer_hi = hi;
    }
    ring_write_word(ppfmt::zone_w0(timer_id, kind));
    ring_write_word(lo);
    publish_tail();
}

// DeviceZoneSetCounter hook: emit the runtime host-id (ttnn's per-program runtime_id) in-band as a
// STICKY_PROG; the host forward-fills it onto this lane's following markers. Every RISC emits one at
// its launch point -- lane-granular attribution (sweep-granular misassigned zones ~2x on back-to-back
// launches). On BRISC the publish rides the zoneValid gate like everything else.
inline __attribute__((always_inline)) void set_host_counter(uint32_t counter_value) {
    if (counter_value >> 27) {
        ring_ensure_room(2);
        ring_write_word(ppfmt::w0(ppfmt::T_STICKY_PROG_EXT, 0));
        ring_write_word(counter_value);
    } else {
        ring_ensure_room(1);
        ring_write_word(ppfmt::w0(ppfmt::T_STICKY_PROG, counter_value));
    }
    publish_tail();
}

inline __attribute__((always_inline)) void set_profiler_zone_valid(bool condition) {
    zoneValid = condition;
    if (condition) {
        // Valid launch: commit what init_profiler() held back, then stream normally.
        publish_tail();
    } else {
        // Idle launch: rewind to the last committed tail; nothing from this launch is published and
        // the next launch overwrites the stale words.
        wIndex = profiler_control_buffer[TAIL_INDEX];
    }
}

__attribute__((noinline)) void init_profiler(
    uint16_t briscKernelID = 0, uint16_t ncriscKernelID = 0, uint16_t triscsKernelID = 0) {
#if defined(COMPILE_FOR_IDLE_ERISC) || (defined(COMPILE_FOR_AERISC) && (COMPILE_FOR_AERISC == 0)) || \
    defined(COMPILE_FOR_BRISC)
    // Stamp this core's identity once per FW session.
    static bool s_xy_stamped = false;
    if (!s_xy_stamped) {
        profiler_control_buffer[SPSC_CORE_XY] = (my_y[0] << 16) | (my_x[0] & 0xFFFF);
        s_xy_stamped = true;
    }
#endif
    // Seed wIndex from TAIL_INDEX ONCE per FW session, then keep it monotonic across launches. The
    // drainer drains a continuous stream and tracks its own head; re-reading the per-program-reset
    // TAIL would rewind below that head and duplicate zones.
    static bool s_windex_seeded = false;
    if (!s_windex_seeded) {
        wIndex = profiler_control_buffer[TAIL_INDEX];
        s_windex_seeded = true;
    }

    // Fresh STICKY_TIMER on this launch's first marker (guards the idle-launch rewind).
    g_prev_timer_hi = 0xFFFFFFFFu;

    // Validators defer publishing until DeviceValidateProfiler resolves the launch.
    if constexpr (PROFILER_VALIDATES_ZONE) {
        zoneValid = false;
    }
}

// Final commit point of a launch's markers.
__attribute__((noinline)) void finish_profiler() { publish_tail(); }

template <uint32_t timer_id>
struct profileScope {
    inline __attribute__((always_inline)) profileScope() { mark_time(timer_id); }
    inline __attribute__((always_inline)) ~profileScope() { mark_time(timer_id, ZoneKind::End); }
};

// FW wrapper: lifecycle only (ring init + validity gate in, final publish out); emits NO markers.
// Every kernel must be wrapped or nothing -- including plain DeviceZoneScopedN -- is published.
struct profileScopeLifecycle {
    inline __attribute__((always_inline)) profileScopeLifecycle() { init_profiler(); }
    inline __attribute__((always_inline)) ~profileScopeLifecycle() { finish_profiler(); }
};

// PP_DATA point marker: tag + timestamp + payload. The length is self-describing (word2), so trailers
// just extend the one packet; the only bound is the 7-bit length field. Same reserve-before-clock-read
// ordering as mark_time.
template <uint32_t data_id, typename... Args>
inline __attribute__((always_inline)) void time_stamped_data(uint64_t data, Args... trailers) {
    constexpr uint32_t total_data_count = 1 + sizeof...(trailers);
    static_assert(2 * total_data_count <= ppfmt::DATA_SIZE_MASK, "payload overflows PP_DATA's 7-bit length field");

    ring_ensure_room(1 + 3 + 2 * total_data_count);
    uint32_t hi, lo;
    read_wall_clock(hi, lo);
    if (hi != g_prev_timer_hi) {
        ring_write_word(ppfmt::w0(ppfmt::T_STICKY_TIMER, hi));
        g_prev_timer_hi = hi;
    }
    ring_write_word(ppfmt::data_w0(data_id));
    ring_write_word(lo);
    ring_write_word(ppfmt::data_w2(2 * total_data_count));
    ring_write_word(data >> 32);
    ring_write_word((data << 32) >> 32);
    ((ring_write_word(trailers >> 32), ring_write_word((trailers << 32) >> 32)), ...);
    publish_tail();
}

// PP_EVENT point marker: compile-time flag, 2 words, no payload.
template <uint32_t data_id>
inline __attribute__((always_inline)) void record_flag() {
    ring_ensure_room(SPSC_MARKER_WORDS + 1);
    uint32_t hi, lo;
    read_wall_clock(hi, lo);
    if (hi != g_prev_timer_hi) {
        ring_write_word(ppfmt::w0(ppfmt::T_STICKY_TIMER, hi));
        g_prev_timer_hi = hi;
    }
    ring_write_word(ppfmt::event_w0(data_id));
    ring_write_word(lo);
    publish_tail();
}

}  // namespace kernel_profiler

#include "noc_event_profiler.hpp"
#include "perf_counters.hpp"

#define DeviceZoneScopedN(name)    \
    TT_ZONE_DEFINE_ID(hash, name); \
    kernel_profiler::profileScope<hash> zone = kernel_profiler::profileScope<hash>();

// Point markers, both with a compile-time tag and an ELF-resolvable name: DeviceTimestampedData
// carries a payload (runtime values ride there); DeviceFlag is a bare 2-word flag.
#define DeviceTimestampedData(name, data)               \
    {                                                   \
        TT_ZONE_DEFINE_ID(hash, name);                  \
        kernel_profiler::time_stamped_data<hash>(data); \
    }

#define DeviceFlag(name)                      \
    {                                         \
        TT_ZONE_DEFINE_ID(hash, name);        \
        kernel_profiler::record_flag<hash>(); \
    }

#define DeviceValidateProfiler(condition) kernel_profiler::set_profiler_zone_valid(condition);

// FW wrapper: lifecycle only, no markers (no "<RISC>-FW" zone; nothing to name).
#define DeviceZoneScopedMainN(name) \
    kernel_profiler::profileScopeLifecycle zone_fw_lifecycle = kernel_profiler::profileScopeLifecycle();

// KERNEL wrapper: an ordinary zone -- a "<RISC>-KERNEL" span per kernel invocation.
#define DeviceZoneScopedMainChildN(name) DeviceZoneScopedN(name)

#define DeviceZoneSetCounter(counter) kernel_profiler::set_host_counter(counter);

// Trace hooks: the names exist because shared firmware calls them unconditionally; they are real on
// the DRAM backend (which owns trace-replay) and empty here.
#define DeviceProfilerInit()
#define DeviceTraceOnlyProfilerInit()
#define DeviceIncrementTraceCount()

#else

// No-op mirrors: keep every call site compiling with zero codegen when profiling is off.
#define DeviceValidateProfiler(condition) (void(sizeof(condition)))

#define DeviceZoneScopedMainN(name) (void(name))

#define DeviceZoneScopedMainChildN(name) (void(name))

#define DeviceZoneScopedN(name) (void(name))

#define DeviceTraceOnlyProfilerInit()

#define DeviceZoneSetCounter(counter) (void(sizeof(counter)))

#define DeviceTimestampedData(data_id, data) (void(sizeof(data_id) + sizeof(data)))

#define DeviceFlag(data_id) (void(sizeof(data_id)))

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

#endif

#endif  // ARCH_QUASAR -- see the arch dispatch at the top of this file
