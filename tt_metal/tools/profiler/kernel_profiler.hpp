// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SPSC variant of the device kernel profiler (X280-drained).
//
// Same public macro API as the original (now kept verbatim in
// kernel_profiler_push.hpp), but a different backend: instead of filling an L1
// buffer and pushing it to DRAM on quick_push()/finish(), each RISC streams its
// markers into a per-RISC single-producer/single-consumer (SPSC) ring in L1. The
// **X280** is the consumer that continuously drains those rings. The producing
// RISC **blocks** (spins on the consumer head) when its ring is full — so the
// stream is lossless and flow-controlled, with **no DRAM traffic** on
// quick_push() or finish().
//
//   Per RISC `r` (Tensix: BRISC/NCRISC/TRISC0-2):
//     ring storage : profiler_data_buffer[r].data[0 .. PROFILER_L1_VECTOR_SIZE-1]
//     tail (prod.) : profiler_control_buffer[DEVICE_BUFFER_END_INDEX_BR_ER + r]
//     head (cons.) : profiler_control_buffer[HOST_BUFFER_END_INDEX_BR_ER  + r]
//   tail/head are MONOTONIC word counts; storage index = count % CAPACITY.
//   Append blocks while (tail - head) > CAPACITY - need, then writes + publishes
//   tail. The X280 advances head as it drains.
//
// NOTE: with this backend a profiled run REQUIRES the X280 consumer to be draining
// — if a ring fills and nothing drains it, the producing RISC blocks (by design).
// Tensix-focused; ETH cores are not a target here.

#pragma once

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
#include "internal/risc_attribs.h"

#include "hostdev/dev_msgs.h"

#include "internal/ethernet/erisc.h"

#define DO_PRAGMA(x) _Pragma(#x)

#define Stringize(L) #L
#define MakeString(M, L) M(L)
#define $Line MakeString(Stringize, __LINE__)

#define PROFILER_MSG __FILE__ "," $Line ",KERNEL_PROFILER"
#define PROFILER_MSG_NAME(name) name "," PROFILER_MSG

#define SrcLocNameToHash(name)                   \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name))); \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name));

#if defined(PROFILE_KERNEL) && \
    (!defined(DISPATCH_KERNEL) || (defined(DISPATCH_KERNEL) && (PROFILE_KERNEL & PROFILER_OPT_DO_DISPATCH_CORES)))
namespace kernel_profiler {

extern uint32_t wIndex;  // producer tail (monotonic word count for this RISC's ring)
extern uint32_t stackSize;
extern uint32_t traceCount;

// Host-side ID (run/program id) carried in the low word of the STICKY_META context packet. Fed by
// set_host_counter() (the DeviceZoneSetCounter hook) -- realizes the TODO there. The host forward-fills
// it onto following markers. 0 until set (currently ~0 on worker cores until a program_host_id is plumbed).
[[maybe_unused]] static uint32_t hostZoneId = 0;

// SPSC publish gate for the DeviceValidateProfiler filter. publish_tail() only advances the
// consumer-visible ring tail while this is true. On "validator" RISCs (those whose FW loop calls
// DeviceValidateProfiler(enables) to declare whether this launch ran a real kernel) it is set false
// at init_profiler() so the FW zone's ZONE_START is held un-published until validity is resolved:
// committed on a valid launch, rewound-and-never-published on an idle launch. This reproduces the
// old DRAM backend's "don't push invalid cores to DRAM", so an idle core's FW-only zone (e.g.
// BRISC-FW on a core that ran no kernel) never reaches the X280 drainer. Non-validator RISCs
// (TRISC/NCRISC) only ever run on valid cores, so they leave this true and publish unconditionally.
extern bool zoneValid;

// The RISCs whose FW loop calls DeviceValidateProfiler(enables) (brisc/erisc/dm own the "<RISC>-FW"
// main zone AND decide per-launch validity). Only these defer the FW ZONE_START publish.
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC) || \
    defined(COMPILE_FOR_AERISC) || defined(COMPILE_FOR_DM)
inline constexpr bool PROFILER_VALIDATES_ZONE = true;
#else
inline constexpr bool PROFILER_VALIDATES_ZONE = false;
#endif

extern uint32_t sums[SUM_COUNT];
extern uint32_t sumIDs[SUM_COUNT];

constexpr uint32_t NOC_ALIGNMENT_FACTOR = 4;

#if (PROFILE_KERNEL & PROFILER_OPT_DO_TRACE_ONLY) && !(defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC))
constexpr bool TRACE_ON_TENSIX = true;
#else
constexpr bool TRACE_ON_TENSIX = false;
#endif
#if (PROFILE_KERNEL & PROFILER_OPT_DO_SUM)
constexpr bool DO_SUM = true;
#else
constexpr bool DO_SUM = false;
#endif

// SPSC backend never drops — it blocks on a full ring — so dropping is always off.
// (Kept because noc_event_profiler.hpp references kernel_profiler::NON_DROPPING.)
constexpr bool NON_DROPPING = false;

constexpr uint32_t TRACE_MARK_FW_START = (1 << 31);
constexpr uint32_t TRACE_MARK_KERNEL_START = (1 << 30);
constexpr uint32_t TRACE_MARK_ALL_ENDS = (1 << 29);

constexpr int WALL_CLOCK_HIGH_INDEX = 1;
constexpr int WALL_CLOCK_LOW_INDEX = 0;

volatile tt_l1_ptr uint32_t* profiler_control_buffer =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.control_vector));

volatile tt_l1_ptr profiler_msg_buffer_t* profiler_data_buffer =
    reinterpret_cast<volatile tt_l1_ptr profiler_msg_buffer_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.buffer));

#if (PROFILE_KERNEL & PROFILER_OPT_DO_TRACE_ONLY)
constexpr uint32_t myRiscID = 0;
#else
constexpr uint32_t myRiscID = PROCESSOR_INDEX;
#endif

// SPSC ring geometry for this RISC.
constexpr uint32_t RING_CAPACITY = PROFILER_L1_VECTOR_SIZE;                // words (= data[] length)
constexpr uint32_t TAIL_INDEX = DEVICE_BUFFER_END_INDEX_BR_ER + myRiscID;  // producer (this RISC)
constexpr uint32_t HEAD_INDEX = HOST_BUFFER_END_INDEX_BR_ER + myRiscID;    // consumer (X280)

constexpr uint32_t Hash32_CT(const char* str, size_t n, uint32_t basis = UINT32_C(2166136261)) {
    return n == 0 ? basis : Hash32_CT(str + 1, n - 1, (basis ^ str[0]) * UINT32_C(16777619));
}

template <size_t N>
constexpr uint32_t Hash16_CT(const char (&s)[N]) {
    auto res = Hash32_CT(s, N - 1);
    return ((res & 0xFFFF) ^ ((res & 0xFFFF0000) >> 16)) & 0xFFFF;
}

enum class DoingDispatch { DISPATCH, DISPATCH_META, NOT_DISPATCH };

constexpr uint32_t get_const_id(uint32_t id, PacketTypes type) { return ((id & 0xFFFF) | ((type << 16) & 0x7FFFF)); }

inline __attribute__((always_inline)) uint32_t get_id(uint32_t id, PacketTypes type) {
    return ((id & 0xFFFF) | ((type << 16) & 0x7FFFF));
}

// ---- SPSC ring primitives -------------------------------------------------

// Reserved zone id for the X280 back-pressure stall zone. 16-bit id space; 0x7FFF won't collide with
// a real Hash16_CT zone id in practice, and the host special-cases it to the name "X280-STALL".
constexpr uint32_t PROFILER_STALL_ZONE_ID = 0x7FFF;

// ---- X280 compact wire format (2-word / 8B packets) ------------------------
// This backend emits the compact per-lane packet format the X280 drain pipeline expects:
//   word0: [31:27] type(5)  [26:0] low27       word1: [31:0] payload32
// A MARKER (ZONE_START/END/TOTAL/TS_*) carries: low27 = 16-bit zone srcloc hash (room to grow to 27),
// payload32 = timer_low. Identity is NOT in the marker anymore -- it is reconstructed on the host from
// three "sticky" packets that persist until updated:
//   STICKY_PROG  (type 8): payload32 = runtime host-id. Emitted at BRISC FW start (set_host_counter).
//   STICKY_TIMER (type 9): low27 = timer_hi. Emitted by any RISC when its wall-clock high half ticks.
//   STICKY_SRC   (type 7): (core,risc) lane -- injected by the X280 READER, never by the producer.
// So a producing RISC writes ONLY markers + (rarely) a TIMER sticky; the reader knows which ring it is
// draining, so it stamps the SRC identity. This drops the per-marker identity word (4->2 words) and the
// need for the X280 to reshape.
//
// MUST stay in sync with tools/x280_bm/include/prof_packet.h. Inlined here (not #included) because the
// kernel JIT build does not carry the tools/x280_bm/include path -- same pattern as producer_common.h.
struct ppfmt {
    static constexpr uint32_t TYPE_SHIFT = 27;
    static constexpr uint32_t TYPE_MASK = 0x1Fu;
    static constexpr uint32_t LOW27_MASK = 0x7FFFFFFu;
    static constexpr uint32_t HASH16_MASK = 0xFFFFu;
    static constexpr uint32_t T_STICKY_PROG = 8u;   // PP_STICKY_PROG
    static constexpr uint32_t T_STICKY_TIMER = 9u;  // PP_STICKY_TIMER
    static inline uint32_t w0(uint32_t type, uint32_t low27) {
        return ((type & TYPE_MASK) << TYPE_SHIFT) | (low27 & LOW27_MASK);
    }
    // marker word0 from a get_const_id/get_id timer_id: type in bits 16-18 -> the 5-bit type field,
    // 16-bit hash -> low27 (host later widens to the full 27 bits for a bigger id space).
    static inline uint32_t marker_w0(uint32_t timer_id) { return w0((timer_id >> 16) & 0x7u, timer_id & HASH16_MASK); }
};

// SPSC marker is now 2 words. The shared PROFILER_L1_MARKER_UINT32_SIZE stays 2 (L1 buffer SIZE
// unchanged), so the ring holds 256 2-word markers.
static constexpr uint32_t SPSC_MARKER_WORDS = 2;

// Last wall-clock high half this RISC emitted in a STICKY_TIMER. Init to ~0 (never a real hi) so the
// first marker forces a TIMER sticky (the "kernel start" high anchor). Static (not extern) so the
// backend definition file needn't change; constant-folds to a per-RISC .bss word.
[[maybe_unused]] static uint32_t g_prev_timer_hi = 0xFFFFFFFFu;

// Tear-free 64-bit wall-clock read: HIGH and LOW are separate registers, so a tick between them would
// pair an old high with a new (wrapped-small) low -> a timestamp ~2^32 too small = a backwards jump. Re-read
// HIGH after LOW and retry if it moved, so (hi, lo) is always one consistent snapshot.
inline __attribute__((always_inline)) void read_wall_clock(uint32_t& hi, uint32_t& lo) {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    do {
        hi = p_reg[WALL_CLOCK_HIGH_INDEX];
        lo = p_reg[WALL_CLOCK_LOW_INDEX];
    } while (hi != p_reg[WALL_CLOCK_HIGH_INDEX]);
}

// Slow path of ring_ensure_room (out-of-line: ONE copy, not inlined at every zone scope). The ring is
// FULL -> record when the stall begins, block until there's room for the caller's marker AND the
// 2-marker stall zone, then emit the {START,END} back-pressure zone (two 2-word markers) so the stall
// nests inside the caller's elongated zone.
//
// A stall CAN span a wall-clock high tick -- and under saturation EVERY lane's in-flight stall straddles
// each ~3.2 s (2^32-cycle) tick at once. So carry the FULL (hi, lo) for both start and end and emit a
// STICKY_TIMER whenever the high half advances (start vs the last emitted hi; end vs start). Without this
// the host reconstructs stall_end with the pre-tick hi -> a ~2^32 backwards jump on that lane (was a
// deterministic ~lanes*ticks batch of ts regressions). Reserve room for both stall markers + up to 2
// stickies so the whole zone is written without a mid-zone re-check.
__attribute__((noinline)) void ring_ensure_room_slow(uint32_t nwords) {
    uint32_t start_hi, start_lo;
    read_wall_clock(start_hi, start_lo);
    const uint32_t need = nwords + 2 * SPSC_MARKER_WORDS + 2;  // caller marker + {START,END} + up to 2 TIMER stickies
    while ((wIndex - profiler_control_buffer[HEAD_INDEX]) > (RING_CAPACITY - need)) {
        invalidate_l1_cache();  // re-read the X280-updated head (and the terminate flag)
        if (profiler_control_buffer[PROFILER_TERMINATE]) {
            return;  // teardown: drop the marker + skip the stall zone rather than stall on a dead ring
        }
    }
    uint32_t end_hi, end_lo;
    read_wall_clock(end_hi, end_lo);
    if (start_hi != g_prev_timer_hi) {  // hi ticked before the stall began -> anchor the START's high half
        profiler_data_buffer[myRiscID].data[wIndex++ % RING_CAPACITY] = ppfmt::w0(ppfmt::T_STICKY_TIMER, start_hi);
        g_prev_timer_hi = start_hi;
    }
    profiler_data_buffer[myRiscID].data[wIndex++ % RING_CAPACITY] = ppfmt::w0(ZONE_START, PROFILER_STALL_ZONE_ID);
    profiler_data_buffer[myRiscID].data[wIndex++ % RING_CAPACITY] = start_lo;
    if (end_hi != g_prev_timer_hi) {  // hi ticked DURING the stall -> anchor the END before its low half
        profiler_data_buffer[myRiscID].data[wIndex++ % RING_CAPACITY] = ppfmt::w0(ppfmt::T_STICKY_TIMER, end_hi);
        g_prev_timer_hi = end_hi;
    }
    profiler_data_buffer[myRiscID].data[wIndex++ % RING_CAPACITY] = ppfmt::w0(ZONE_END, PROFILER_STALL_ZONE_ID);
    profiler_data_buffer[myRiscID].data[wIndex++ % RING_CAPACITY] = end_lo;
}

// Fast path stays inline (just the room check); the full-ring path is out-of-line above.
inline __attribute__((always_inline)) void ring_ensure_room(uint32_t nwords) {
    if ((wIndex - profiler_control_buffer[HEAD_INDEX]) <= (RING_CAPACITY - nwords)) {
        return;
    }
    ring_ensure_room_slow(nwords);
}

inline __attribute__((always_inline)) void ring_write_word(uint32_t v) {
    profiler_data_buffer[myRiscID].data[wIndex % RING_CAPACITY] = v;
    wIndex++;
}

inline __attribute__((always_inline)) void publish_tail() {
    // Hold the tail while this launch is unvalidated/invalid: an idle core's FW zone is written into
    // the ring but never made visible to the X280 drainer (see zoneValid).
    if (zoneValid) {
        // Release fence: the X280 consumer reads TAIL then the marker slot over the NoC. Blackhole L1
        // is write-through, but the marker-word stores and this TAIL store can still reach L1 SRAM out
        // of order, so a remote reader could observe the bumped TAIL before the words land and read a
        // stale/empty slot. Order the marker stores BEFORE the TAIL publish so TAIL is a true commit
        // point (paired with the consumer's wait-for-valid in profzone.c -- neither is sufficient
        // alone: this fence prevents stale-but-valid reads, the consumer wait covers not-yet-visible).
        asm volatile("fence" ::: "memory");
        profiler_control_buffer[TAIL_INDEX] = wIndex;
    }
}

// Append one 2-word timing marker (type|srcloc-hash , timer_low), preceded by a STICKY_TIMER when the
// wall-clock high half ticks. Blocks if the ring is full. Identity is injected by the X280 reader.
//
// CRITICAL ordering: reserve ring room (which may BLOCK on a full ring and emit the X280-STALL zone) BEFORE
// reading the clock. If we timestamped first, a marker delayed by a stall would carry a pre-stall time yet be
// written into the ring AFTER the (later-timestamped) stall zone -> a backwards time jump on that lane. Reading
// the clock after the room is secured makes the marker's time reflect when it is actually written (>= the stall
// end), keeping every lane's stream monotonic. Reserve worst case (a TIMER sticky + the marker) so the room
// check -- and any stall -- happens once, up front, not between the two writes.
inline __attribute__((always_inline)) void mark_time(uint32_t timer_id) {
    ring_ensure_room(SPSC_MARKER_WORDS + 1);  // worst case: 1-word TIMER sticky + 2-word marker
    uint32_t hi, lo;
    read_wall_clock(hi, lo);
    if (hi != g_prev_timer_hi) {
        ring_write_word(ppfmt::w0(ppfmt::T_STICKY_TIMER, hi));  // STICKY_TIMER: 1 word (type | timer_hi)
        g_prev_timer_hi = hi;
    }
    ring_write_word(ppfmt::marker_w0(timer_id));  // word0: type | zone srcloc (16-bit hash)
    ring_write_word(lo);                          // word1: timer_low
    publish_tail();
}

// Emit the STICKY_META context packet (2 words) into ALL of this core's RISC rings, not just the
// caller's. Each per-RISC ring the X280 drains needs its own (core_x, core_y, risc) + host_id header so
// the host can forward-fill that identity onto the following raw markers -- letting the X280 reader
// bulk-copy them with NO per-marker reshape. The type sits in the same valid/type bits (w0[31], w0[30:28])
// the host reads on any marker, so a sticky is distinguished before its payload is decoded.
//
// Called from set_host_counter (BRISC's assign-ID hook). Safe to populate sibling rings because at that
// point BRISC is the only active RISC -- the others are out of reset but not yet emitting (run_triscs is
// later). init_profiler() has already zeroed the rings this launch. For sibling rings the sticky lands
// first; only BRISC's own ring has its FW ZONE_START ahead of the sticky (fine -- FW zone carries no ID).
inline __attribute__((always_inline)) void mark_sticky_meta() {
    // Retired: the combined (identity+id) context packet is gone. Identity is injected by the X280
    // reader (STICKY_SRC); the runtime host-id is emitted as STICKY_PROG from set_host_counter. Kept
    // as a no-op in case anything still references it.
    return;
}

// Fixed-index write retained only for the trace-only build mode (writes directly into the ring storage
// region; not used by the default SPSC path). 2-word marker: word0 = type|hash, word1 = timer_low.
inline __attribute__((always_inline)) void mark_time_at_index_inlined(uint32_t index, uint32_t timer_id) {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    profiler_data_buffer[myRiscID].data[index] = ppfmt::marker_w0(timer_id);
    profiler_data_buffer[myRiscID].data[index + 1] = p_reg[WALL_CLOCK_LOW_INDEX];
}

// ---- dropped-timestamp bookkeeping (no drops in blocking mode; kept for API) --

inline __attribute__((always_inline)) void mark_dropped_timestamps(uint32_t index) {
    uint32_t curr = profiler_control_buffer[DROPPED_ZONES];
    profiler_control_buffer[DROPPED_ZONES] = (1 << index) | curr;
}

inline __attribute__((always_inline)) bool get_dropped_timestamps(uint32_t index) {
    uint32_t curr = profiler_control_buffer[DROPPED_ZONES];
    return ((curr >> index) & 0x1);
}

inline __attribute__((always_inline)) void set_host_counter(uint32_t counterValue) {
    // Assign-ID hook (DeviceZoneSetCounter): emit the runtime host-id in-band as a STICKY_PROG packet.
    // counterValue is the per-program-global runtime_id (the same id ttnn assigns and the DRAM profiler
    // stamps into ID_LL). The host forward-fills it onto every following marker of this launch until the
    // next STICKY_PROG. Emitted at the ID-assign call (BRISC FW start) -- it lands just after the FW
    // ZONE_START, which carries no assigned id. Held unpublished until DeviceValidateProfiler commits the
    // launch (an idle core rewinds it), matching the FW zone's validity gate.
    hostZoneId = counterValue;
    ring_ensure_room(SPSC_MARKER_WORDS);
    ring_write_word(ppfmt::w0(ppfmt::T_STICKY_PROG, 0));  // word0: type | (low27 unused)
    ring_write_word(counterValue);                        // word1: runtime host-id
    publish_tail();
}

inline __attribute__((always_inline)) void set_profiler_zone_valid(bool condition) {
    profiler_control_buffer[PROFILER_DONE] = !condition;  // retained for host / DRAM-backend parity
    zoneValid = condition;
    if (condition) {
        // Valid launch: commit the FW ZONE_START that init_profiler() held back, then stream normally
        // for the rest of the launch (publish_tail() is now unblocked).
        publish_tail();
    } else {
        // Idle launch (this core ran no kernel): discard the un-published FW ZONE_START by rewinding
        // to the last committed tail. With zoneValid false, the matching ZONE_END and finish also
        // stay unpublished, so nothing from this launch reaches the X280 drainer. The next launch's
        // init_profiler() resets wIndex to this same tail and overwrites the stale words.
        wIndex = profiler_control_buffer[TAIL_INDEX];
    }
}

__attribute__((noinline)) void init_profiler(
    uint16_t briscKernelID = 0, uint16_t ncriscKernelID = 0, uint16_t triscsKernelID = 0) {
    stackSize = 0;
    for (int i = 0; i < SUM_COUNT; i++) {
        sumIDs[i] = 0;
        sums[i] = 0;
    }

#if defined(COMPILE_FOR_IDLE_ERISC) || (defined(COMPILE_FOR_AERISC) && (COMPILE_FOR_AERISC == 0)) || \
    defined(COMPILE_FOR_BRISC)
    uint32_t runCounter = profiler_control_buffer[RUN_COUNTER];
    profiler_control_buffer[PROFILER_DONE] = 0;
    if (runCounter == 0) {
        // First launch: empty every RISC's ring (head = tail = 0) and stamp coords. NOTE: do NOT
        // write data[ID_HH] here — ID_HH is a live ring slot in this backend, so poking it corrupts
        // a marker word (see set_host_counter). The ring is emptied via head/tail == 0 above.
        for (uint32_t riscID = 0; riscID < PROCESSOR_COUNT; riscID++) {
            profiler_control_buffer[HOST_BUFFER_END_INDEX_BR_ER + riscID] = 0;
            profiler_control_buffer[DEVICE_BUFFER_END_INDEX_BR_ER + riscID] = 0;
        }
        profiler_control_buffer[NOC_X] = my_x[0];
        profiler_control_buffer[NOC_Y] = my_y[0];
    }
#endif
    // Seed this RISC's tail from L1 ONCE per FW session, then keep wIndex monotonic across launches --
    // do NOT re-read TAIL_INDEX per launch. The X280 reader drains a CONTINUOUS stream and tracks its own
    // head; the standard device profiler resets TAIL_INDEX per program, so resuming from it would rewind
    // wIndex below the reader's head -> tail-head underflows -> the host decoder wraps the ring and emits
    // ~30x duplicate zones. wIndex lives in FW .bss (persists across kernel launches); publish_tail keeps
    // TAIL_INDEX monotonic too, overwriting any host reset. (This is the "init once, outside the per-launch
    // path" fix -- the ring is never re-initialized per kernel launch.)
    static bool s_windex_seeded = false;
    if (!s_windex_seeded) {
        wIndex = profiler_control_buffer[TAIL_INDEX];
        s_windex_seeded = true;
    }

    // Re-anchor the wall-clock high half so this launch's first marker emits a fresh STICKY_TIMER.
    // Guards the idle-launch rewind case (a discarded sticky must not leave the host with a stale hi).
    g_prev_timer_hi = 0xFFFFFFFFu;

    // On validator RISCs, defer publishing this launch until DeviceValidateProfiler() resolves it.
    // The FW zone's ZONE_START (emitted right after this returns) is written into the ring but not
    // made visible until set_profiler_zone_valid(true) commits it — or discarded if the launch is
    // idle. if constexpr keeps this a no-op on TRISC/NCRISC, which always run on valid cores.
    if constexpr (PROFILER_VALIDATES_ZONE) {
        zoneValid = false;
    }
}

// Append accumulated SUM zones (if any) and publish the tail. No DRAM.
inline __attribute__((always_inline)) void risc_finished_profiling() {
    for (int i = 0; i < SUM_COUNT; i++) {
        if (sums[i] > 0) {
            ring_ensure_room(SPSC_MARKER_WORDS);
            ring_write_word(ppfmt::marker_w0(get_id(sumIDs[i], ZONE_TOTAL)));  // word0: ZONE_TOTAL | hash
            ring_write_word(sums[i]);  // word1: accumulated sum (host reads-as-sum by type, not a timer)
        }
    }
    publish_tail();
}

__attribute__((noinline)) void finish_profiler() {
    risc_finished_profiling();
#if defined(COMPILE_FOR_IDLE_ERISC) || (defined(COMPILE_FOR_AERISC) && (COMPILE_FOR_AERISC == 0)) || \
    defined(COMPILE_FOR_BRISC)
    profiler_control_buffer[RUN_COUNTER]++;
    profiler_control_buffer[PROFILER_DONE] = 1;
#endif
}

// No DRAM push in the SPSC backend — the X280 drains the ring continuously. These
// are kept (as tail publishes / no-ops) so the existing call sites still compile.
__attribute__((noinline)) void quick_push() { publish_tail(); }

inline __attribute__((always_inline)) void quick_push_if_linked(uint32_t cmd_buf, bool linked) {
    (void)cmd_buf;
    (void)linked;
}

template <DoingDispatch dispatch = DoingDispatch::NOT_DISPATCH>
inline __attribute__((always_inline)) void flush_to_dram_if_full(uint32_t additional_slots = 0) {
    (void)additional_slots;
}

template <uint32_t timer_id, DoingDispatch dispatch = DoingDispatch::NOT_DISPATCH>
struct profileScope {
    inline __attribute__((always_inline)) profileScope() { mark_time(timer_id); }
    inline __attribute__((always_inline)) ~profileScope() { mark_time(get_const_id(timer_id, ZONE_END)); }
};

template <uint32_t timer_id, uint32_t index>
struct profileScopeGuaranteed {
    inline __attribute__((always_inline)) profileScopeGuaranteed() {
        if constexpr (TRACE_ON_TENSIX) {
            uint32_t trace_replay_status = profiler_control_buffer[TRACE_REPLAY_STATUS];
            if constexpr (index == 0) {
#if !defined(COMPILE_FOR_TRISC)
                if (trace_replay_status & TRACE_MARK_FW_START) {
                    mark_time(get_const_id(timer_id, ZONE_START));
                    profiler_control_buffer[TRACE_REPLAY_STATUS] = TRACE_MARK_KERNEL_START;
                }
#endif
            } else {
                if (trace_replay_status & TRACE_MARK_KERNEL_START) {
                    mark_time(get_const_id(timer_id, ZONE_START));
                    profiler_control_buffer[TRACE_REPLAY_STATUS] = TRACE_MARK_ALL_ENDS;
                }
            }
        } else {
            if constexpr (index == 0) {
                init_profiler();
            }
            // FW/KER wrapper zone emission DISABLED for the X280 real-profiler runs: keep the
            // init_profiler/finish_profiler lifecycle (ring init, zoneValid publish gate, RUN_COUNTER) but
            // drop the "<RISC>-FW" (index 0) / "<RISC>-KERNEL" (index 1) START/END markers, so only the user
            // DeviceZoneScopedN zones (profileScope) reach the capture. Re-enable by uncommenting.
            // mark_time(get_const_id(timer_id, ZONE_START));
        }
    }
    inline __attribute__((always_inline)) ~profileScopeGuaranteed() {
        if constexpr (TRACE_ON_TENSIX) {
            if (profiler_control_buffer[TRACE_REPLAY_STATUS] == TRACE_MARK_ALL_ENDS) {
                mark_time(get_const_id(timer_id, ZONE_END));
            }
        } else {
            // mark_time(get_const_id(timer_id, ZONE_END));  // FW/KER wrapper zone disabled -- see ctor
            if constexpr (index == 0) {
                finish_profiler();
            }
        }
    }
};

template <uint32_t timer_id, uint32_t index>
struct profileScopeAccumulate {
    uint64_t start_time = 0;
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);

    inline __attribute__((always_inline)) profileScopeAccumulate() {
        if constexpr (kernel_profiler::DO_SUM) {
            start_time = ((uint64_t)p_reg[WALL_CLOCK_HIGH_INDEX] << 32) | p_reg[WALL_CLOCK_LOW_INDEX];
        }
    }
    inline __attribute__((always_inline)) ~profileScopeAccumulate() {
        if constexpr (kernel_profiler::DO_SUM) {
            sumIDs[index] = timer_id;
            sums[index] += (((uint64_t)p_reg[WALL_CLOCK_HIGH_INDEX] << 32) | p_reg[WALL_CLOCK_LOW_INDEX]) - start_time;
        }
    }
};

template <
    uint32_t data_id,
    DoingDispatch dispatch = DoingDispatch::NOT_DISPATCH,
    PacketTypes packet_type = kernel_profiler::PacketTypes::TS_DATA,
    typename... Args>
inline __attribute__((always_inline)) void timeStampedData(uint64_t data, Args... trailers) {
    constexpr uint32_t total_data_count = 1 + sizeof...(trailers);
    constexpr uint32_t expected_size = kernel_profiler::TimestampedDataSize<packet_type>::size;

    static_assert(
        expected_size == 0 || total_data_count == expected_size,
        "Number of arguments does not match expected size for this PacketType");

    // Reserve worst case (a 1-word TIMER sticky + the timing marker + 2 words/datum) BEFORE reading the clock,
    // so a full-ring stall does not backdate the marker (see mark_time's ordering note).
    ring_ensure_room(SPSC_MARKER_WORDS + 1 + 2 * total_data_count);
    uint32_t hi, lo;
    read_wall_clock(hi, lo);
    if (hi != g_prev_timer_hi) {
        ring_write_word(ppfmt::w0(ppfmt::T_STICKY_TIMER, hi));  // STICKY_TIMER: 1 word (type | timer_hi)
        g_prev_timer_hi = hi;
    }
    uint32_t marker_id = get_const_id(data_id, packet_type);
    ring_write_word(ppfmt::marker_w0(marker_id));  // word0: packet_type | hash
    ring_write_word(lo);                           // word1: timer_low

    ring_write_word(data >> 32);
    ring_write_word((data << 32) >> 32);
    ((ring_write_word(trailers >> 32), ring_write_word((trailers << 32) >> 32)), ...);
    publish_tail();
}

template <DoingDispatch dispatch = DoingDispatch::NOT_DISPATCH>
inline __attribute__((always_inline)) void recordEvent(uint16_t event_id) {
    mark_time(get_id(event_id, TS_EVENT));
}

inline __attribute__((always_inline)) void increment_trace_count() {
    if constexpr (!TRACE_ON_TENSIX) {
        traceCount++;
    }
}

__attribute__((noinline)) void trace_only_init() {
    if constexpr (TRACE_ON_TENSIX) {
        traceCount++;
        set_host_counter(traceCount);
        profiler_control_buffer[TRACE_REPLAY_STATUS] = TRACE_MARK_FW_START;
        // Do NOT poke data[ID_HH] here: it is a live ring slot in the SPSC backend and the X280
        // drain does not consume it, so the write only corrupts a marker word (see set_host_counter).
    }
}

}  // namespace kernel_profiler

#include "noc_event_profiler.hpp"
#include "perf_counters.hpp"

// Not dispatch
#if (!defined(DISPATCH_KERNEL))

#define DeviceZoneScopedN(name)                                                \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    kernel_profiler::profileScope<hash> zone = kernel_profiler::profileScope<hash>();

#define DeviceTimestampedData(name, data)                                          \
    {                                                                              \
        DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
        auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
        kernel_profiler::timeStampedData<hash>(data);                              \
    }

#define DeviceRecordEvent(event_id) kernel_profiler::recordEvent(event_id);

// Dispatch and enabled
#elif (defined(DISPATCH_KERNEL) && (PROFILE_KERNEL & PROFILER_OPT_DO_DISPATCH_CORES))

#define DeviceZoneScopedN(name)                                                          \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                                         \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name));           \
    kernel_profiler::profileScope<hash, kernel_profiler::DoingDispatch::DISPATCH> zone = \
        kernel_profiler::profileScope<hash, kernel_profiler::DoingDispatch::DISPATCH>();

#define DeviceTimestampedData(name, data)                                                            \
    {                                                                                                \
        DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                                                 \
        auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name));                   \
        kernel_profiler::timeStampedData<hash, kernel_profiler::DoingDispatch::DISPATCH_META>(data); \
    }

#define DeviceRecordEvent(event_id) kernel_profiler::recordEvent<kernel_profiler::DoingDispatch::DISPATCH>(event_id);

// Dispatch but disabled
#else

#define DeviceZoneScopedN(name) (void(sizeof(name)))

#define DeviceTimestampedData(data_id, data) (void(sizeof(data_id) + sizeof(data)))

#define DeviceRecordEvent(event_id) (void(sizeof(event_id)))

#endif

#define DeviceValidateProfiler(condition) kernel_profiler::set_profiler_zone_valid(condition);

#define DeviceZoneScopedMainN(name)                                            \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    kernel_profiler::profileScopeGuaranteed<hash, 0> zone = kernel_profiler::profileScopeGuaranteed<hash, 0>();

#define DeviceZoneScopedMainChildN(name)                                       \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    kernel_profiler::profileScopeGuaranteed<hash, 1> zone = kernel_profiler::profileScopeGuaranteed<hash, 1>();

#define DeviceZoneScopedSumN1(name)                                            \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    kernel_profiler::profileScopeAccumulate<hash, 0> zone = kernel_profiler::profileScopeAccumulate<hash, 0>();

#define DeviceZoneScopedSumN2(name)                                            \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    kernel_profiler::profileScopeAccumulate<hash, 1> zone = kernel_profiler::profileScopeAccumulate<hash, 1>();

#define DeviceZoneSetCounter(counter)                  \
    if constexpr (!kernel_profiler::TRACE_ON_TENSIX) { \
        kernel_profiler::set_host_counter(counter);    \
    }

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_AERISC)
#define DeviceProfilerInit()                          \
    if constexpr (kernel_profiler::TRACE_ON_TENSIX) { \
        kernel_profiler::init_profiler();             \
    }                                                 \
    kernel_profiler::traceCount = 0;
#else
#define DeviceProfilerInit()                          \
    if constexpr (kernel_profiler::TRACE_ON_TENSIX) { \
        kernel_profiler::init_profiler();             \
    }
#endif

#define DeviceTraceOnlyProfilerInit() kernel_profiler::trace_only_init();

#define DeviceIncrementTraceCount() kernel_profiler::increment_trace_count();

#else

// The void(sizeof(FOO)) idiom (a) ensures FOO is syntactically and
// semantically sane and (b) means that we avoid 'var-set-but-unused'
// diagnostics, if the only use of a particular var is here.  The
// sizeof argument is processed in a non-evaluating context -- no code
// is generated.
#define DeviceValidateProfiler(condition) (void(sizeof(condition)))

#define DeviceZoneScopedMainN(name) (void(name))

#define DeviceZoneScopedMainChildN(name) (void(name))

#define DeviceZoneScopedN(name) (void(name))

#define DeviceZoneScopedSumN1(name) (void(name))

#define DeviceZoneScopedSumN2(name) (void(name))

#define DeviceTraceOnlyProfilerInit()

#define DeviceZoneSetCounter(counter) (void(sizeof(counter)))

#define DeviceTimestampedData(data_id, data) (void(sizeof(data_id) + sizeof(data)))

#define DeviceRecordEvent(event_id) (void(sizeof(event_id)))

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
