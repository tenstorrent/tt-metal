// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SPSC device kernel profiler: each RISC streams markers into its own single-producer/single-consumer ring in
// L1 and the resident DRISC relay empties it; a full ring blocks the producer, so the stream is lossless. Per
// RISC r: storage profiler_data_buffer[r].data[0..PROFILER_L1_VECTOR_SIZE), tail = control[SPSC_RING_TAIL_0 + r]
// (producer), head = control[SPSC_RING_HEAD_0 + r] (relay); both are monotonic word counts, index = count %
// capacity. PP_* wire types and the SpscControlBuffer slots are owned here; ControlBuffer/PacketTypes belong
// to the DRAM backend, and sharing a control word breaks both.

#pragma once

// Quasar has no DRISC relay, so it keeps the DRAM producer; both headers define the same macro API.
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

// PROFILE_KERNEL is a global JIT define, so dispatch kernels get it too; no relay serves a dispatch core, so a
// producer there would fill its ring and wedge the next relay bring-up. The relay kernel is excluded as well:
// this producer is ~1 KB it has no code room for (its self-profiling is SpscZoneScope in profiler_common.h).
#if defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL) && !defined(STREAMING_PROFILER_RELAY_KERNEL)

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_ERISC)
// Global scope: a block-scope extern inside the namespace would look for kernel_profiler::__stack_base and
// fail to link.
extern uint32_t __stack_base[];
#endif

namespace kernel_profiler {

extern uint32_t wIndex;  // producer tail: monotonic word count, lives in FW .bss across launches

// publish_tail() advances the consumer-visible tail only while true; validator RISCs clear it in
// init_profiler() and resolve it via DeviceValidateProfiler, so an idle launch's markers are rewound, not
// published.
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

// Namespace scope only because profileScopeStall needs it.
TT_ZONE_DEFINE_ID(PROFILER_STALL_ZONE_ID, "PRODUCER-STALL");

// Wire encode, duplicated from spsc_packet.h because the JIT build lacks that include path. word0 = type(5) |
// low27. A zone ships whole at close: a 2-word ZONE_S when its end is within 2^16 cycles of the lane cursor
// and its duration fits 16 bits, else a 3-word ZONE_ATOMIC (id | end timer_low | duration) that re-anchors
// the cursor. START/END pairs serve only the stall zone and the >3.2 s fallback. Lane identity and the
// timer's high half are host-reconstructed from stickies.
struct ppfmt {
    static constexpr uint32_t TYPE_SHIFT = 27;
    static constexpr uint32_t TYPE_MASK = 0x1Fu;
    static constexpr uint32_t LOW27_MASK = 0x7FFFFFFu;
    // This wire's own type space: never pass a hostdevcommon PacketTypes value through, and never reuse
    // a retired value (11 = ZONE_TOTAL).
    static constexpr uint32_t T_ZONE_START = 0u;        // PP_ZONE_START (stall zone + long-zone fallback only)
    static constexpr uint32_t T_ZONE_END = 1u;          // PP_ZONE_END   (stall zone + long-zone fallback only)
    static constexpr uint32_t T_ZONE_ATOMIC = 2u;       // PP_ZONE_ATOMIC (3 words: id | end_lo | duration)
    static constexpr uint32_t T_ZONE_S = 3u;            // PP_ZONE_S (2 words: id | end_delta16<<16 | dur16)
    static constexpr uint32_t T_ZONE_L = 4u;            // PP_ZONE_L (5 words: id | end_lo | end_hi | dur_lo | dur_hi)
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
    static inline uint32_t zone_atomic_w0(uint32_t id) { return w0(T_ZONE_ATOMIC, id & LOW27_MASK); }
    static inline uint32_t zone_s_w0(uint32_t id) { return w0(T_ZONE_S, id & LOW27_MASK); }
    static inline uint32_t data_w0(uint32_t id) { return w0(T_DATA, id & LOW27_MASK); }
    static inline uint32_t data_w2(uint32_t size_words) { return (size_words & DATA_SIZE_MASK) << DATA_SIZE_SHIFT; }
    static inline uint32_t event_w0(uint32_t id) { return w0(T_EVENT, id & LOW27_MASK); }
};

static constexpr uint32_t SPSC_MARKER_WORDS = 2;

// Last high half emitted in a STICKY_TIMER; ~0 forces a fresh sticky on a launch's first marker.
[[maybe_unused]] static uint32_t g_prev_timer_hi = 0xFFFFFFFFu;

// Lane cursor: the end of the last S/ATOMIC zone, mirrored exactly by the decoder. hi = ~0 is invalid (no S
// can match it, so the next zone ships ATOMIC and re-anchors both sides).
[[maybe_unused]] static uint32_t g_cursor_lo = 0;
[[maybe_unused]] static uint32_t g_cursor_hi = 0xFFFFFFFFu;

// Producer-cached relay head. The head only advances, so a stale copy is conservative; the fast path compares
// against this word and reaches L1 once per drained batch. 0 is the safe floor.
[[maybe_unused]] static uint32_t g_head_cache = 0;

// Reading L latches the high half and H returns it, so the order is the protocol. The latch is single-agent
// (TensixTile/DebugTimestamper.md): another RISC's L read landing in the L->H gap across a 2^32 boundary
// puts a marker ~3.2 s in the future, at ~1e-9 per read, which is cheaper than a retry branch here.
inline __attribute__((always_inline)) void read_wall_clock(uint32_t& hi, uint32_t& lo) {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    lo = p_reg[WALL_CLOCK_LOW_INDEX];   // latches the high half
    hi = p_reg[WALL_CLOCK_HIGH_INDEX];  // returns the latched value
}

inline __attribute__((always_inline)) void publish_tail() {
    if constexpr (PROFILER_VALIDATES_ZONE) {
        if (!zoneValid) {
            return;
        }
    }
    // The relay reads TAIL then the slots over the NoC, and stores can reach L1 SRAM out of order.
    asm volatile("fence" ::: "memory");
    profiler_control_buffer[TAIL_INDEX] = wIndex;
}

// The fence pays the posted ring stores' latency, so paying it once per SPSC_PUBLISH_BATCH_WORDS is most of
// the close-side saving; the trigger is wIndex crossing a batch boundary. Visibility lags by at most one
// batch within a launch; launch boundaries and the stall path publish unconditionally, and blocking is
// head-vs-wIndex, so losslessness does not depend on the published tail.
inline __attribute__((always_inline)) void publish_tail_batched(uint32_t words_written) {
    constexpr uint32_t kBatchShift = __builtin_ctz(SPSC_PUBLISH_BATCH_WORDS);
    static_assert((1u << kBatchShift) == SPSC_PUBLISH_BATCH_WORDS, "batch must be a power of two");
    if (__builtin_expect((wIndex >> kBatchShift) != ((wIndex - words_written) >> kBatchShift), 0)) {
        publish_tail();
    }
}

inline __attribute__((always_inline)) void ring_write_word(uint32_t v) {
    profiler_data_buffer[myRiscID].data[wIndex % RING_CAPACITY] = v;
    wIndex++;
}

// The high half moves about once per 3.2 s, so test before storing; every caller's room reservation already
// covers the sticky word.
inline __attribute__((always_inline)) void ring_write_sticky_timer(uint32_t hi) {
    if (__builtin_expect(hi != g_prev_timer_hi, 0)) {
        profiler_data_buffer[myRiscID].data[wIndex % RING_CAPACITY] = ppfmt::w0(ppfmt::T_STICKY_TIMER, hi);
        wIndex++;
        g_prev_timer_hi = hi;
    }
}

// ZONE_ATOMIC packet size: word0 (type|id) + end timer_low + 32-bit duration.
static constexpr uint32_t SPSC_ATOMIC_ZONE_WORDS = 3;

// A stall zone writes into a ring that is by definition full, so its words come from a reserve ordinary
// markers may not fill into; the open writes nothing, so the reserve covers one ZONE_ATOMIC plus the
// STICKY_TIMER a stall straddling a timer_hi tick needs.
constexpr uint32_t STALL_CLOSE_WORDS = SPSC_ATOMIC_ZONE_WORDS + 1;
constexpr uint32_t STALL_RESERVE_WORDS = STALL_CLOSE_WORDS;
constexpr uint32_t RING_USABLE = RING_CAPACITY - STALL_RESERVE_WORDS;
static_assert(RING_USABLE > STALL_RESERVE_WORDS, "the ring is too small to carry a stall reserve");

// Written straight into the reserve with no room check: ring_ensure_room() from here would recurse through
// another stall scope. A stall >= 2^32 cycles saturates its duration rather than taking mark_zone_long, which
// reserves room; a 3.2 s wait is a wedged relay, not a measurement.
inline __attribute__((always_inline)) void stall_zone_close(uint32_t start_hi, uint32_t start_lo) {
    uint32_t hi, lo;
    read_wall_clock(hi, lo);
    const uint32_t lo_d = lo - start_lo;
    const uint32_t hi_d = hi - start_hi - (lo < start_lo);
    const uint32_t dur = lo_d | (0u - static_cast<uint32_t>(hi_d != 0));
    ring_write_sticky_timer(hi);
    ring_write_word(ppfmt::zone_atomic_w0(PROFILER_STALL_ZONE_ID));
    ring_write_word(lo);
    ring_write_word(dur);
    g_cursor_lo = lo;  // a ZONE_ATOMIC on the wire moves the decoder's cursor, so it must move ours
    g_cursor_hi = hi;
    // Unconditional publish: this zone is the back-pressure signal, and the path already paid a full stall.
    publish_tail();
}

// Like profileScope, but closes through stall_zone_close(), which writes into the reserve.
struct profileScopeStall {
    uint32_t start_hi, start_lo;
    inline __attribute__((always_inline)) profileScopeStall() { read_wall_clock(start_hi, start_lo); }
    inline __attribute__((always_inline)) ~profileScopeStall() { stall_zone_close(start_hi, start_lo); }
};

// Out of line so there is one copy rather than one per zone site. Waits for the caller's words and the zone's
// own closing half, so the reserve is whole again for the next stall.
__attribute__((noinline)) void ring_ensure_room_slow(uint32_t nwords) {
    if constexpr (myRiscID < SPSC_STALL_COUNT_MAX) {
        profiler_control_buffer[SPSC_STALL_COUNT_0 + myRiscID]++;
    }
    profileScopeStall stall;
    // The relay can only free words up to the published tail, so waiting on unpublished words deadlocks.
    publish_tail();
    while ((wIndex - profiler_control_buffer[HEAD_INDEX]) > (RING_USABLE - nwords - STALL_CLOSE_WORDS)) {
        invalidate_l1_cache();  // re-read the relay-updated head (and the arm flag)
        if (!profiler_control_buffer[PROFILER_ARMED]) {
            return;  // nobody drains this ring (or teardown disarmed it): overwrite rather than wait
        }
    }
    g_head_cache = profiler_control_buffer[HEAD_INDEX];
}

// One local compare against the cached head, bound RING_USABLE (the difference to capacity is the reserve).
inline __attribute__((always_inline)) void ring_ensure_room(uint32_t nwords) {
    if (__builtin_expect((wIndex - g_head_cache) > (RING_USABLE - nwords), 0)) {
        // Invalidate before the refresh: the relay's head write-back arrives over the NoC, which the core's L1 read
        // cache does not observe.
        invalidate_l1_cache();
        g_head_cache = profiler_control_buffer[HEAD_INDEX];
        if ((wIndex - g_head_cache) > (RING_USABLE - nwords)) {
            ring_ensure_room_slow(nwords);
        }
    }
}

// ZONE_L packet size: word0 (type|id) + end_lo + end_hi + dur_lo + dur_hi.
static constexpr uint32_t SPSC_ZONE_L_WORDS = 5;

// Duration >= 2^32 cycles (~3.2 s), which the 32-bit duration word cannot carry: one ZONE_L packet of two full
// 64-bit values; no sticky, cursor untouched on both sides. The decoder normalizes it to a START/END pair
// whose in-the-past START trips the order-regression diagnostic once, kept as visibility: a >3.2 s on-device
// zone is a wedge. Out of line: the always_inline zone sites can never take it.
__attribute__((noinline)) void mark_zone_long(
    uint32_t timer_id, uint32_t start_hi, uint32_t start_lo, uint32_t end_hi, uint32_t end_lo) {
    ring_ensure_room(SPSC_ZONE_L_WORDS);
    const uint32_t dur_lo = end_lo - start_lo;
    const uint32_t dur_hi = end_hi - start_hi - (end_lo < start_lo);
    ring_write_word(ppfmt::w0(ppfmt::T_ZONE_L, timer_id));
    ring_write_word(end_lo);
    ring_write_word(end_hi);
    ring_write_word(dur_lo);
    ring_write_word(dur_hi);
    publish_tail();
}

// One 3-word packet per zone with the start the scope object carried. Room is reserved before the end clock
// is read so a stall elongates the zone it happened inside; otherwise the packet would carry a pre-stall end
// yet sit after the stall zone. Worst case is a 1-word sticky plus the 3-word packet.
inline __attribute__((always_inline)) void mark_zone_close(uint32_t timer_id, uint32_t start_hi, uint32_t start_lo) {
    ring_ensure_room(SPSC_ATOMIC_ZONE_WORDS + 1);  // worst case (ATOMIC + sticky); an S zone simply uses less
    uint32_t hi, lo;
    read_wall_clock(hi, lo);
    const uint32_t lo_d = lo - start_lo;
    const uint32_t hi_d = hi - start_hi - (lo < start_lo);
    // One OR-tree into one branch: cursor delta and duration both fit 16 bits and neither subtract borrowed; an
    // invalid cursor (hi = ~0) fails via c_hi_d. Fall-through because it is the dominant case on a dense lane.
    const uint32_t c_lo_d = lo - g_cursor_lo;
    const uint32_t c_hi_d = hi - g_cursor_hi - (lo < g_cursor_lo);
    if (__builtin_expect((((c_lo_d | lo_d) >> 16) | c_hi_d | hi_d) == 0, 1)) {
        ring_write_word(ppfmt::zone_s_w0(timer_id));
        ring_write_word((c_lo_d << 16) | lo_d);
        g_cursor_lo = lo;
        g_cursor_hi = hi;
        publish_tail_batched(2);
        return;
    }
    if (__builtin_expect(hi_d != 0, 0)) {
        // The long fallback leaves the cursor alone, as the decoder's pair path does.
        mark_zone_long(timer_id, start_hi, start_lo, hi, lo);
        return;
    }
    ring_write_sticky_timer(hi);
    ring_write_word(ppfmt::zone_atomic_w0(timer_id));
    ring_write_word(lo);
    ring_write_word(lo_d);
    g_cursor_lo = lo;  // ZONE_ATOMIC re-anchors the cursor (decoder: cursor = sticky_hi<<32 | end_lo)
    g_cursor_hi = hi;
    publish_tail_batched(SPSC_ATOMIC_ZONE_WORDS + 1);
}

// DeviceZoneSetCounter hook: the runtime host-id goes in band as a STICKY_PROG the host forward-fills onto
// this lane's following markers. Every RISC emits one at its own launch point; a sweep-granular id
// misassigns about twice as many zones on back-to-back launches.
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
        publish_tail();
    } else {
        // Idle launch: rewind to the last committed tail; the rewound words may have moved our cursor past anything
        // the decoder sees, so invalidate it.
        wIndex = profiler_control_buffer[TAIL_INDEX];
        g_cursor_hi = 0xFFFFFFFFu;
    }
}

__attribute__((noinline)) void init_profiler(
    uint16_t briscKernelID = 0, uint16_t ncriscKernelID = 0, uint16_t triscsKernelID = 0) {
#if defined(COMPILE_FOR_IDLE_ERISC) || (defined(COMPILE_FOR_AERISC) && (COMPILE_FOR_AERISC == 0)) || \
    defined(COMPILE_FOR_BRISC)
    static bool s_xy_stamped = false;
    if (!s_xy_stamped) {
        profiler_control_buffer[SPSC_CORE_XY] = (my_y[0] << 16) | (my_x[0] & 0xFFFF);
        s_xy_stamped = true;
    }
#endif
    // Seeded from TAIL_INDEX once per FW session, then monotonic across launches: the relay tracks its own head,
    // so re-reading the per-program-reset TAIL would rewind below it and duplicate zones.
    static bool s_windex_seeded = false;
    if (!s_windex_seeded) {
        wIndex = profiler_control_buffer[TAIL_INDEX];
        s_windex_seeded = true;
    }

    // Fresh STICKY_TIMER on this launch's first marker (guards the idle-launch rewind).
    g_prev_timer_hi = 0xFFFFFFFFu;
    // After a rewind the decoder's cursor is not where ours is; invalidate so the first zone re-anchors.
    g_cursor_hi = 0xFFFFFFFFu;

    // Validators defer publishing until DeviceValidateProfiler resolves the launch.
    if constexpr (PROFILER_VALIDATES_ZONE) {
        zoneValid = false;
    }
}

__attribute__((noinline)) void finish_profiler() { publish_tail(); }

// Once per FW boot, before any launch: the host arms only the cores its relays drain, and it does so after the
// firmware is up, so this must not run again at the first launch or it would undo that arm.
inline void disarm_at_boot() { profiler_control_buffer[PROFILER_ARMED] = 0; }

// The constructor touches nothing but the wall clock; the whole zone ships at close with the start as member
// state, 8 B per open zone. Hold exactly these two words: anything more is register pressure across the user
// code inside the zone, and a globals-maxed kernel on the 192-256 B loader stack floor gets tight around
// 10-20 open zones (see stackCanaryScope).
template <uint32_t timer_id>
struct profileScope {
    uint32_t start_hi, start_lo;
    inline __attribute__((always_inline)) profileScope() { read_wall_clock(start_hi, start_lo); }
    inline __attribute__((always_inline)) ~profileScope() { mark_zone_close(timer_id, start_hi, start_lo); }
};

// Lifecycle only, no markers. Every kernel must be wrapped or nothing it records is published.
struct profileScopeLifecycle {
    inline __attribute__((always_inline)) profileScopeLifecycle() { init_profiler(); }
    inline __attribute__((always_inline)) ~profileScopeLifecycle() { finish_profiler(); }
};

// Tag, timestamp, payload; the length is self-describing (word2), bounded by the 7-bit length field. Same
// reserve-before-clock-read ordering as mark_zone_close.
template <uint32_t data_id, typename... Args>
inline __attribute__((always_inline)) void time_stamped_data(uint64_t data, Args... trailers) {
    constexpr uint32_t total_data_count = 1 + sizeof...(trailers);
    static_assert(2 * total_data_count <= ppfmt::DATA_SIZE_MASK, "payload overflows PP_DATA's 7-bit length field");

    ring_ensure_room(1 + 3 + 2 * total_data_count);
    uint32_t hi, lo;
    read_wall_clock(hi, lo);
    ring_write_sticky_timer(hi);
    ring_write_word(ppfmt::data_w0(data_id));
    ring_write_word(lo);
    ring_write_word(ppfmt::data_w2(2 * total_data_count));
    ring_write_word(data >> 32);
    ring_write_word((data << 32) >> 32);
    ((ring_write_word(trailers >> 32), ring_write_word((trailers << 32) >> 32)), ...);
    publish_tail_batched(1 + 3 + 2 * total_data_count);
}

// PP_EVENT point marker: compile-time flag, 2 words, no payload.
template <uint32_t data_id>
inline __attribute__((always_inline)) void record_flag() {
    ring_ensure_room(SPSC_MARKER_WORDS + 1);
    uint32_t hi, lo;
    read_wall_clock(hi, lo);
    ring_write_sticky_timer(hi);
    ring_write_word(ppfmt::event_w0(data_id));
    ring_write_word(lo);
    publish_tail_batched(SPSC_MARKER_WORDS + 1);
}

// Watcher-off net for a globals-heavy kernel whose loader-guaranteed stack floor is only MEM_*_STACK_MIN_SIZE
// (192-256 B): plant one word at __stack_base when the kernel zone opens, check it at close, emit a named
// PP_EVENT if frame data overwrote it. The pattern is the watcher's stack_usage_pattern on purpose, so an
// intact canary reads as painted-and-unused to measure_stack_usage(). Compiled out for firmware
// (__stack_base is a kernel-link symbol) and active ERISC (stack guarded by -Werror=stack-usage).
#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_ERISC)
TT_ZONE_DEFINE_ID(STACK_CANARY_DEAD_ID, "STACK-OVERFLOW");
constexpr uint32_t STACK_CANARY_PATTERN = 0xBABABABA;  // == watcher stack_usage_pattern
struct stackCanaryScope {
    inline __attribute__((always_inline)) stackCanaryScope() { ::__stack_base[0] = STACK_CANARY_PATTERN; }
    inline __attribute__((always_inline)) ~stackCanaryScope() {
        if (__builtin_expect(::__stack_base[0] != STACK_CANARY_PATTERN, 0)) {
            record_flag<STACK_CANARY_DEAD_ID>();
        }
    }
};
#else
struct stackCanaryScope {};  // FW builds and active ERISC: no kernel stack floor to watch
#endif

}  // namespace kernel_profiler

#include "noc_event_profiler.hpp"
#include "perf_counters.hpp"

#define DeviceZoneScopedN(name)    \
    TT_ZONE_DEFINE_ID(hash, name); \
    kernel_profiler::profileScope<hash> zone = kernel_profiler::profileScope<hash>();

// DeviceTimestampedData carries a payload; DeviceFlag is a bare 2-word flag. Both have a compile-time tag
// and an ELF-resolvable name.
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

#else

// Zero codegen when profiling is off.
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

#endif  // ARCH_QUASAR
