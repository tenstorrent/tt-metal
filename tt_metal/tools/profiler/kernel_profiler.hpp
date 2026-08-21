// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SPSC variant of the device kernel profiler (drainer-drained).
//
// Same public macro API as the push-to-DRAM backend (kept verbatim in
// kernel_profiler_push.hpp), but a wholly separate backend: each RISC streams its
// markers into a per-RISC single-producer/single-consumer (SPSC) ring in L1, and a
// drainer (drainer or DRISC) continuously empties those rings. The producing RISC
// **blocks** (spins on the consumer head) when its ring is full — so the stream is
// lossless and flow-controlled, and there is **no DRAM traffic** at all.
//
// The two backends never collide. This file owns SpscControlBuffer and the PP_* wire
// types; ControlBuffer and PacketTypes belong to the DRAM backend and must not appear
// here. Sharing a control word between them has silently broken both before.
//
//   Per RISC `r` (Tensix: BRISC/NCRISC/TRISC0-2):
//     ring storage : profiler_data_buffer[r].data[0 .. PROFILER_L1_VECTOR_SIZE-1]
//     tail (prod.) : profiler_control_buffer[SPSC_RING_TAIL_0 + r]
//     head (cons.) : profiler_control_buffer[SPSC_RING_HEAD_0 + r]
//     identity     : profiler_control_buffer[SPSC_CORE_XY]  = (y << 16) | x
//   tail/head are MONOTONIC word counts; storage index = count % CAPACITY.
//   Append blocks while (tail - head) > CAPACITY - need, then writes + publishes
//   tail. The drainer advances head as it drains.
//
// NOTE: with this backend a profiled run REQUIRES the drainer consumer to be draining
// — if a ring fills and nothing drains it, the producing RISC blocks (by design).
// Tensix-focused; ETH cores are not a target here.

#pragma once

// ---- Arch dispatch -------------------------------------------------------
// The SPSC backend below requires a drainer consumer to drain the rings, and the drainer is
// Blackhole hardware. Quasar has no such drainer, so an SPSC producer there would have no consumer
// and the first full ring would block the RISC forever.
//
// Quasar therefore keeps the DRAM producer (kernel_profiler_push.hpp), which is where upstream's
// Quasar profiler bringup lives (#49417 basic profiler, #50900 DeviceZoneScopedN). That code is not
// portable to this backend anyway: it needs a RUNTIME myRiscID (internal_::get_hw_thread_idx()),
// while our ring geometry needs a compile-time one for `constexpr TAIL_INDEX/HEAD_INDEX`; it reads
// the NEO_REGS_0 wall-clock registers rather than RISCV_DEBUG_REG_WALL_CLOCK_L; and it has up to 24
// processors per core against the 5 this ring layout assumes.
//
// Both headers define the same public macro API, so callers are unaffected by which one is active.
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

// The one DISPATCH_KERNEL mention this backend keeps, and it is load-bearing: PROFILE_KERNEL is a
// GLOBAL jit-build define (set for every kernel when the profiler is on, dispatch included), the
// dispatch kernels contain DeviceZoneScoped* sites, and the drainer's lane tables cover the profiled
// worker grid only. Gating them out is what keeps a dispatch core from emitting into an undrained,
// lossless-blocking ring -- measured failure: the ring fills, the dispatch core blocks, and the NEXT
// device open's drainer bring-up wedges at its write barrier (heartbeat stuck, phase=11).
#if defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL)
namespace kernel_profiler {

extern uint32_t wIndex;  // producer tail (monotonic word count for this RISC's ring)

// SPSC publish gate for the DeviceValidateProfiler filter. publish_tail() only advances the
// consumer-visible ring tail while this is true. On "validator" RISCs (those whose FW loop calls
// DeviceValidateProfiler(enables) to declare whether this launch ran a real kernel) it is set false
// at init_profiler() so the FW zone's ZONE_START is held un-published until validity is resolved:
// committed on a valid launch, rewound-and-never-published on an idle launch. This reproduces the
// old DRAM backend's "don't push invalid cores to DRAM", so an idle core's FW-only zone (e.g.
// BRISC-FW on a core that ran no kernel) never reaches the drainer drainer. Non-validator RISCs
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

// TRACE-ONLY mode (PROFILER_OPT_DO_TRACE_ONLY / TRACE_ON_TENSIX) and the SUM/accumulate zones
// (PROFILER_OPT_DO_SUM / DeviceZoneScopedSumN*) have been REMOVED from this producer; the parked DRAM
// producer (kernel_profiler_push.hpp) still has the original implementations if ever needed back.
constexpr int WALL_CLOCK_HIGH_INDEX = 1;
constexpr int WALL_CLOCK_LOW_INDEX = 0;

volatile tt_l1_ptr uint32_t* profiler_control_buffer =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.control_vector));

volatile tt_l1_ptr profiler_msg_buffer_t* profiler_data_buffer =
    reinterpret_cast<volatile tt_l1_ptr profiler_msg_buffer_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.buffer));

constexpr uint32_t myRiscID = PROCESSOR_INDEX;

// SPSC ring geometry for this RISC.
constexpr uint32_t RING_CAPACITY = PROFILER_L1_VECTOR_SIZE;  // words (= data[] length)
// SPSC layout, NOT the DRAM profiler's HOST_/DEVICE_BUFFER_END_INDEX_* slots -- see SpscControlBuffer.
constexpr uint32_t TAIL_INDEX = SPSC_RING_TAIL_0 + myRiscID;  // producer (this RISC)
constexpr uint32_t HEAD_INDEX = SPSC_RING_HEAD_0 + myRiscID;  // consumer (drainer)
static_assert(myRiscID < PROFILER_SPSC_MAX_RISC, "this processor has no slot in the SPSC control layout");

// Zone kind, this backend's own. Previously the kind was packed into bits 16-18 of the id as a
// hostdevcommon PacketTypes value and unpacked in zone_w0 -- a 3-bit channel into a 5-bit wire, whose
// correctness depended on the numeric ordinals of an enum this backend does not own. The id now carries
// the source-location hash and nothing else; the kind travels as its own argument.
enum class ZoneKind : uint32_t { Start = 0, End = 1 };

// ---- SPSC ring primitives -------------------------------------------------

// The producer's back-pressure zone. NOTHING about it is special any more: it is an ORDINARY structural
// zone id with an ORDINARY .tt_zone_meta record, so the host names it from the ELF exactly like a kernel
// zone and no host table has to know it exists. (It used to be the magic 0x7FFF, justified by "a 16-bit
// hash won't land here in practice", with the name hardcoded on the host.)
//
// It is declared here, at namespace scope, rather than at the emission site inside ring_ensure_room_slow,
// only because the scope type below needs it as a template argument.
TT_ZONE_DEFINE_ID(PROFILER_STALL_ZONE_ID, "PRODUCER-STALL");

// ---- SPSC compact wire format (2-word / 8B packets) ------------------------
// This backend emits the compact per-lane packet format the drainer drain pipeline expects:
//   word0: [31:27] type(5)  [26:0] low27       word1: [31:0] payload32
// A zone MARKER (ZONE_START/END) carries: low27 = the full 27-bit structural zone id,
// payload32 = timer_low. Identity is NOT in the marker anymore -- it is reconstructed on the host from
// three "sticky" packets that persist until updated:
//   STICKY_PROG  (type 8): low27 = runtime host-id, 1 word. Emitted per RISC at its launch point
//                (set_host_counter), so every lane's stream is self-attributing. Ids past 2^27 ship
//                as the 2-word STICKY_PROG_EXT (type 14) with the full id in word1.
//   STICKY_TIMER (type 9): low27 = timer_hi. Emitted by any RISC when its wall-clock high half ticks.
//   STICKY_SRC   (type 7): (core,risc) lane -- injected by the drainer READER, never by the producer.
// So a producing RISC writes ONLY markers + (rarely) a TIMER sticky; the reader knows which ring it is
// draining, so it stamps the SRC identity. This drops the per-marker identity word (4->2 words) and the
// need for the drainer to reshape.
//
// MUST stay in sync with tt_metal/tools/profiler/spsc_packet.h. Inlined here (not #included) because
// the kernel JIT build does not carry that include path.
struct ppfmt {
    static constexpr uint32_t TYPE_SHIFT = 27;
    static constexpr uint32_t TYPE_MASK = 0x1Fu;
    static constexpr uint32_t LOW27_MASK = 0x7FFFFFFu;
    // drainer wire type codes -- this wire's OWN space, NOT hostdevcommon PacketTypes values. The DRAM
    // readback path never co-exists with this one and shares no decode, so the two numberings are
    // independent. Passing a PacketTypes value straight through (the old 3-bit `>> 16 & 0x7`) is what
    // made ZONE_TOTAL and TS_DATA_16B collide with unrelated types on this wire.
    static constexpr uint32_t T_ZONE_START = 0u;        // PP_ZONE_START
    static constexpr uint32_t T_ZONE_END = 1u;          // PP_ZONE_END
    static constexpr uint32_t T_STICKY_PROG = 8u;       // PP_STICKY_PROG
    static constexpr uint32_t T_STICKY_PROG_EXT = 14u;  // PP_STICKY_PROG_EXT
    static constexpr uint32_t T_STICKY_TIMER = 9u;      // PP_STICKY_TIMER
    static constexpr uint32_t T_DATA = 10u;             // PP_DATA (compile-time tag, 2 + size words)
    static constexpr uint32_t T_EVENT = 12u;            // PP_EVENT (compile-time flag: 2 words, no payload)
    static constexpr uint32_t DATA_SIZE_SHIFT = 25u;    // PP_DATA_SIZE_SHIFT (word2, not word0)
    static constexpr uint32_t DATA_SIZE_MASK = 0x7Fu;   // PP_DATA_SIZE_MASK [31:25]
    static inline uint32_t w0(uint32_t type, uint32_t low27) {
        return ((type & TYPE_MASK) << TYPE_SHIFT) | (low27 & LOW27_MASK);
    }
    // Zone marker word0. The kind is passed in, not dug out of the id; data/event go through
    // data_w0/event_w0. Full 27 bits: the id shares low27 with nothing, so a zone marker carries the
    // structural id whole. (This mask used to be a 16-bit one, which is what truncated it.)
    static inline uint32_t zone_w0(uint32_t id, ZoneKind kind) {
        return w0(kind == ZoneKind::End ? T_ZONE_END : T_ZONE_START, id & LOW27_MASK);
    }
    // PP_DATA word0 -- IDENTICAL in shape to a zone marker's, type | the full 27-bit structural id. The
    // payload length moved out to its own word2 (below) precisely so a point marker's id is as wide, and
    // as nameable, as a zone's.
    static inline uint32_t data_w0(uint32_t id) { return w0(T_DATA, id & LOW27_MASK); }
    // PP_DATA word2: the payload length in 32-bit words. Its own word; [24:0] unused and zero.
    static inline uint32_t data_w2(uint32_t size_words) { return (size_words & DATA_SIZE_MASK) << DATA_SIZE_SHIFT; }
    // PP_EVENT word0: a payload-less flag, 2 words total and no size word at all. Same 27-bit
    // compile-time structural id, so it is named from the ELF like everything else on this wire.
    static inline uint32_t event_w0(uint32_t id) { return w0(T_EVENT, id & LOW27_MASK); }
};

// SPSC marker is now 2 words. The shared PROFILER_L1_MARKER_UINT32_SIZE stays 2 (L1 buffer SIZE
// unchanged), so the ring holds 256 2-word markers.
static constexpr uint32_t SPSC_MARKER_WORDS = 2;

// Last wall-clock high half this RISC emitted in a STICKY_TIMER. Init to ~0 (never a real hi) so the
// first marker forces a TIMER sticky (the "kernel start" high anchor). Static (not extern) so the
// backend definition file needn't change; constant-folds to a per-RISC .bss word.
[[maybe_unused]] static uint32_t g_prev_timer_hi = 0xFFFFFFFFu;

// Branchless 64-bit wall-clock read: reading WALL_CLOCK_L latches the high half (counter_high_at),
// which the WALL_CLOCK_H read then returns -- so L-THEN-H is one consistent snapshot with no retry.
// The ORDER is the whole protocol; read H first and you are back to torn reads.
//
// DELIBERATE tradeoff (see tt-isa-documentation, TensixTile/DebugTimestamper.md): the latch is only
// sound for a single agent. Another RISC's L read can re-latch counter_high_at between this RISC's L
// and H reads, and if the counter crossed a 2^32 boundary in that gap the marker lands +2^32 cycles
// (~3.2 s) in the FUTURE. That needs two rare events to coincide (~1e-9..1e-8 per read) and is loud
// on the host when it happens -- one lane's order-regression counter storms for the next 3.2 s of
// device time -- so we take the risk rather than pay a retry branch plus a third debug-register read
// on every marker of this hot path.
inline __attribute__((always_inline)) void read_wall_clock(uint32_t& hi, uint32_t& lo) {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    lo = p_reg[WALL_CLOCK_LOW_INDEX];   // latches the high half
    hi = p_reg[WALL_CLOCK_HIGH_INDEX];  // returns the latched value
}

// ---- The stall reserve ------------------------------------------------------------------------------
//
// The back-pressure zone is an ORDINARY RAII scope: the constructor stamps ZONE_START, the destructor
// stamps ZONE_END, both re-reading the wall clock, exactly like DeviceZoneScopedN. One thing makes that
// safe, and it is the entire reason this constant exists.
//
// A naive scope DEADLOCKS. Its constructor would emit START into the ring that is full -- being full is
// precisely why we are on this path -- so the wait that would drain it never runs. So the producer
// permanently gives up STALL_RESERVE_WORDS of its ring: ordinary markers may fill only up to RING_USABLE,
// and the reserve is available to NOTHING but the stall zone. If ordinary traffic could consume it, the
// deadlock returns under exactly the conditions that trigger a stall.
//
// Sized from the constants, never hardcoded: each half of the zone is its 2-word marker plus the 1-word
// STICKY_TIMER it may need when the wall clock's high half ticks. A stall CAN straddle a ~3.2 s (2^32
// cycle) tick, and under saturation every lane's in-flight stall straddles one at once; without the
// sticky the host reconstructs stall_end with the pre-tick high half -- a ~2^32 backwards jump on that
// lane, which used to be a deterministic batch of timestamp regressions.
//
// At PROFILER_L1_VECTOR_SIZE = 512 words per RISC the reserve costs 6 words, ~1.2% of the ring.
constexpr uint32_t STALL_ZONE_HALF_WORDS = SPSC_MARKER_WORDS + 1;
constexpr uint32_t STALL_RESERVE_WORDS = 2 * STALL_ZONE_HALF_WORDS;
constexpr uint32_t RING_USABLE = RING_CAPACITY - STALL_RESERVE_WORDS;
static_assert(RING_USABLE > STALL_RESERVE_WORDS, "the ring is too small to carry a stall reserve");

// Write one half of the stall zone straight into the reserve, with NO room check. That is what the
// reserve is for; checking would recurse into the path that emits this zone. Direct stores rather than
// ring_write_word() because that helper is declared below (and this must sit above ring_ensure_room).
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

// The drainer back-pressure zone. An ordinary scope over an ordinary structural zone id -- it is only a
// distinct type from profileScope because it writes into the reserve instead of going through
// ring_ensure_room, which on this path would be a recursion into itself.
struct profileScopeStall {
    inline __attribute__((always_inline)) profileScopeStall() { stall_mark(ZoneKind::Start); }
    inline __attribute__((always_inline)) ~profileScopeStall() { stall_mark(ZoneKind::End); }
};

// Slow path of ring_ensure_room (out-of-line: ONE copy, not inlined at every zone scope). The ring is
// FULL -> open the back-pressure zone out of the reserve, block until there is room for the caller's
// marker AND this zone's own closing half, then let the destructor close it. The zone therefore nests
// inside the caller's elongated zone exactly as before, but it is now measured the ordinary way: START is
// timestamped when the stall begins and END when it ends, each from its own clock read.
__attribute__((noinline)) void ring_ensure_room_slow(uint32_t nwords) {
    // Ground truth for the knee, straight from the producer. The stall ZONE below still goes into the ring
    // for timeline use, but this counter is what the host reads when it wants the count without decoding --
    // it cannot be lost downstream, and it costs one L1 store on a path that was already blocked.
    if constexpr (myRiscID < SPSC_STALL_COUNT_MAX) {
        profiler_control_buffer[SPSC_STALL_COUNT_0 + myRiscID]++;
    }
    profileScopeStall stall;
    // Wait for the caller's words AND this zone's own closing half. Waiting for both is what restores the
    // invariant: once the destructor and then the caller have written, occupancy is back at or under
    // RING_USABLE, so the reserve is whole again for the next stall. (nwords is a small compile-time
    // constant at every call site -- 6 words at the largest, against a 506-word RING_USABLE -- so this
    // subtraction cannot wrap.)
    while ((wIndex - profiler_control_buffer[HEAD_INDEX]) > (RING_USABLE - nwords - STALL_ZONE_HALF_WORDS)) {
        invalidate_l1_cache();  // re-read the drainer-updated head (and the terminate flag)
        if (profiler_control_buffer[PROFILER_TERMINATE]) {
            return;  // teardown: stop waiting on a dead ring; the destructor still closes the zone
        }
    }
}

// Fast path stays inline (just the room check); the full-ring path is out-of-line above. Note the bound is
// RING_USABLE, not RING_CAPACITY: the difference is the stall reserve, which ordinary markers may never
// touch.
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
    // Hold the tail while this launch is unvalidated/invalid: an idle core's FW zone is written into
    // the ring but never made visible to the drainer drainer (see zoneValid).
    if (zoneValid) {
        // Release fence: the drainer consumer reads TAIL then the marker slot over the NoC. Blackhole L1
        // is write-through, but the marker-word stores and this TAIL store can still reach L1 SRAM out
        // of order, so a remote reader could observe the bumped TAIL before the words land and read a
        // stale/empty slot. Order the marker stores BEFORE the TAIL publish so TAIL is a true commit
        // point (paired with the drain kernel's wait-for-valid -- neither is sufficient
        // alone: this fence prevents stale-but-valid reads, the consumer wait covers not-yet-visible).
        asm volatile("fence" ::: "memory");
        profiler_control_buffer[TAIL_INDEX] = wIndex;
    }
}

// Append one 2-word timing marker (type|zone-id , timer_low), preceded by a STICKY_TIMER when the
// wall-clock high half ticks. Blocks if the ring is full. Identity is injected by the drainer reader.
//
// CRITICAL ordering: reserve ring room (which may BLOCK on a full ring and emit the PRODUCER-STALL zone) BEFORE
// reading the clock. If we timestamped first, a marker delayed by a stall would carry a pre-stall time yet be
// written into the ring AFTER the (later-timestamped) stall zone -> a backwards time jump on that lane. Reading
// the clock after the room is secured makes the marker's time reflect when it is actually written (>= the stall
// end), keeping every lane's stream monotonic. Reserve worst case (a TIMER sticky + the marker) so the room
// check -- and any stall -- happens once, up front, not between the two writes.
inline __attribute__((always_inline)) void mark_time(uint32_t timer_id, ZoneKind kind = ZoneKind::Start) {
    ring_ensure_room(SPSC_MARKER_WORDS + 1);  // worst case: 1-word TIMER sticky + 2-word marker
    uint32_t hi, lo;
    read_wall_clock(hi, lo);
    if (hi != g_prev_timer_hi) {
        ring_write_word(ppfmt::w0(ppfmt::T_STICKY_TIMER, hi));  // STICKY_TIMER: 1 word (type | timer_hi)
        g_prev_timer_hi = hi;
    }
    ring_write_word(ppfmt::zone_w0(timer_id, kind));  // word0: zone type | 27-bit structural zone id
    ring_write_word(lo);                              // word1: timer_low
    publish_tail();
}

// No dropped-timestamp bookkeeping: the ring blocks rather than dropping, so the only way to lose a
// marker is after PROFILER_TERMINATE releases a producer at teardown.

inline __attribute__((always_inline)) void set_host_counter(uint32_t counterValue) {
    // Assign-ID hook (DeviceZoneSetCounter): emit the runtime host-id in-band as a STICKY_PROG packet.
    // counterValue is the per-program-global runtime_id (the same id ttnn assigns and the DRAM profiler
    // stamps into ID_LL). The host forward-fills it onto every following marker OF THIS LANE until the
    // lane's next STICKY_PROG -- so EVERY RISC emits one at its launch point, not just BRISC: a lane
    // whose ring carries no id would only be attributable at drainer-sweep granularity, which
    // misassigns zones across op boundaries on back-to-back launches (measured ~2x unions). On BRISC
    // it is held unpublished until DeviceValidateProfiler commits the launch (an idle core rewinds
    // it), matching the FW zone's validity gate; subordinates only run committed launches.
    if (counterValue >> 27) {
        ring_ensure_room(2);
        ring_write_word(ppfmt::w0(ppfmt::T_STICKY_PROG_EXT, 0));
        ring_write_word(counterValue);
    } else {
        ring_ensure_room(1);
        ring_write_word(ppfmt::w0(ppfmt::T_STICKY_PROG, counterValue));
    }
    publish_tail();
}

inline __attribute__((always_inline)) void set_profiler_zone_valid(bool condition) {
    zoneValid = condition;
    if (condition) {
        // Valid launch: commit the FW ZONE_START that init_profiler() held back, then stream normally
        // for the rest of the launch (publish_tail() is now unblocked).
        publish_tail();
    } else {
        // Idle launch (this core ran no kernel): discard the un-published FW ZONE_START by rewinding
        // to the last committed tail. With zoneValid false, the matching ZONE_END and finish also
        // stay unpublished, so nothing from this launch reaches the drainer drainer. The next launch's
        // init_profiler() resets wIndex to this same tail and overwrites the stale words.
        wIndex = profiler_control_buffer[TAIL_INDEX];
    }
}

__attribute__((noinline)) void init_profiler(
    uint16_t briscKernelID = 0, uint16_t ncriscKernelID = 0, uint16_t triscsKernelID = 0) {
#if defined(COMPILE_FOR_IDLE_ERISC) || (defined(COMPILE_FOR_AERISC) && (COMPILE_FOR_AERISC == 0)) || \
    defined(COMPILE_FOR_BRISC)
    // Stamp this core's identity once per FW session. Nothing else to initialise: the rings are
    // head==tail==0 from L1 zeroing, and this backend keeps no run counter or done flag -- those are
    // the DRAM backend's words.
    static bool s_xy_stamped = false;
    if (!s_xy_stamped) {
        profiler_control_buffer[SPSC_CORE_XY] = (my_y[0] << 16) | (my_x[0] & 0xFFFF);
        s_xy_stamped = true;
    }
#endif
    // Seed this RISC's tail from L1 ONCE per FW session, then keep wIndex monotonic across launches --
    // do NOT re-read TAIL_INDEX per launch. The drainer reader drains a CONTINUOUS stream and tracks its own
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

// Publish whatever the launch wrote; the final commit point of a kernel's markers. No DRAM.
__attribute__((noinline)) void finish_profiler() { publish_tail(); }

template <uint32_t timer_id>
struct profileScope {
    inline __attribute__((always_inline)) profileScope() { mark_time(timer_id); }
    inline __attribute__((always_inline)) ~profileScope() { mark_time(timer_id, ZoneKind::End); }
};

// FW-wrapper scope (what DeviceZoneScopedMainN used to be, via profileScopeGuaranteed<hash,0>).
// It owns the profiler LIFECYCLE ONLY -- ring init + zoneValid publish gate on the way in, finish and
// publish on the way out -- and deliberately emits NO markers, so no "<RISC>-FW" zone appears.
// This must still wrap every kernel: without init_profiler()/finish_profiler() the ring is never set up
// and NOTHING (including plain DeviceZoneScopedN) is published.
// The KERNEL wrapper (index 1) no longer needs a special type at all -- DeviceZoneScopedMainChildN now
// uses the ordinary profileScope above, so it reports exactly like any DeviceZoneScopedN zone.
// (The old profileScopeGuaranteed also implemented the TRACE_ON_TENSIX replay state machine; trace-only
// mode has since been removed from this producer entirely -- see the note near the top of the file.)
struct profileScopeLifecycle {
    inline __attribute__((always_inline)) profileScopeLifecycle() { init_profiler(); }
    inline __attribute__((always_inline)) ~profileScopeLifecycle() { finish_profiler(); }
};

// No PacketTypes template parameter: the payload length is self-describing on this wire (PP_DATA
// word2), so trailers just extend the one packet -- no per-type size enum. The only real bound is
// the 7-bit length field.
template <uint32_t data_id, typename... Args>
inline __attribute__((always_inline)) void timeStampedData(uint64_t data, Args... trailers) {
    constexpr uint32_t total_data_count = 1 + sizeof...(trailers);
    static_assert(2 * total_data_count <= ppfmt::DATA_SIZE_MASK, "payload overflows PP_DATA's 7-bit length field");

    // Reserve worst case BEFORE reading the clock, so a full-ring stall does not backdate the marker (see
    // mark_time's ordering note). A PP_DATA packet is 3 words of header (word0 | timer_low | size) plus the
    // payload, and a STICKY_TIMER may precede it.
    ring_ensure_room(1 + 3 + 2 * total_data_count);
    uint32_t hi, lo;
    read_wall_clock(hi, lo);
    if (hi != g_prev_timer_hi) {
        ring_write_word(ppfmt::w0(ppfmt::T_STICKY_TIMER, hi));  // STICKY_TIMER: 1 word (type | timer_hi)
        g_prev_timer_hi = hi;
    }
    // One PP_DATA packet. word0 is now shaped exactly like a zone marker -- type | the full 27-bit
    // structural id -- and the payload length rides in its own word2, so this marker is named from the ELF
    // like any zone. The length still travels in the packet, so the host advances over the payload without
    // a per-type length table. 2 words/datum.
    ring_write_word(ppfmt::data_w0(data_id));               // word0: PP_DATA | 27-bit structural id
    ring_write_word(lo);                                    // word1: timer_low
    ring_write_word(ppfmt::data_w2(2 * total_data_count));  // word2: payload length in words

    ring_write_word(data >> 32);
    ring_write_word((data << 32) >> 32);
    ((ring_write_word(trailers >> 32), ring_write_word((trailers << 32) >> 32)), ...);
    publish_tail();
}

// A compile-time-tagged FLAG with no payload: PP_EVENT, 2 words, and a structural source-location id that
// the host names from the ELF exactly like a zone. (Same reserve-before-clock-read ordering as mark_time.)
template <uint32_t data_id>
inline __attribute__((always_inline)) void recordFlag() {
    ring_ensure_room(SPSC_MARKER_WORDS + 1);
    uint32_t hi, lo;
    read_wall_clock(hi, lo);
    if (hi != g_prev_timer_hi) {
        ring_write_word(ppfmt::w0(ppfmt::T_STICKY_TIMER, hi));
        g_prev_timer_hi = hi;
    }
    ring_write_word(ppfmt::event_w0(data_id));
    ring_write_word(lo);  // word1: timer_low
    publish_tail();
}

}  // namespace kernel_profiler

#include "noc_event_profiler.hpp"
#include "perf_counters.hpp"

#define DeviceZoneScopedN(name)    \
    TT_ZONE_DEFINE_ID(hash, name); \
    kernel_profiler::profileScope<hash> zone = kernel_profiler::profileScope<hash>();

// The two point markers, both with a compile-time tag (string literal) and therefore a source
// location and an ELF-resolvable name: DeviceData carries a payload (runtime values ride there),
// DeviceFlag carries nothing and is 2 words. (DeviceRuntimeEvent is gone -- it was DeviceData with a
// hardcoded "RUNTIME-EVENT" tag and a u32 payload, a wire type of its own only in the era when the
// runtime value rode in the id field itself.)
#define DeviceData(name, data)                        \
    {                                                 \
        TT_ZONE_DEFINE_ID(hash, name);                \
        kernel_profiler::timeStampedData<hash>(data); \
    }

#define DeviceFlag(name)                     \
    {                                        \
        TT_ZONE_DEFINE_ID(hash, name);       \
        kernel_profiler::recordFlag<hash>(); \
    }

#define DeviceValidateProfiler(condition) kernel_profiler::set_profiler_zone_valid(condition);

// FW wrapper: DISABLED as a zone -- lifecycle only, emits no markers (so no "<RISC>-FW" in the capture).
// Keeps init_profiler()/finish_profiler(), which every kernel needs for the ring to exist at all.
// No zone id here on purpose: nothing is reported, so no source location needs registering.
#define DeviceZoneScopedMainN(name) \
    kernel_profiler::profileScopeLifecycle zone_fw_lifecycle = kernel_profiler::profileScopeLifecycle();

// KERNEL wrapper: an ordinary zone, nothing more -- a real model run shows a "<RISC>-KERNEL" span per
// kernel invocation alongside any op-level zones. (The old SPSC_KERNEL_WRAPPER_ZONE bisect toggle is
// gone; it had already cleared the ResNet teardown hang of any connection to KERNEL-zone emission.)
#define DeviceZoneScopedMainChildN(name) DeviceZoneScopedN(name)

#define DeviceZoneSetCounter(counter) kernel_profiler::set_host_counter(counter);

// Trace hooks: the NAMES must exist because shared firmware calls them unconditionally, and they are
// real on the DRAM backend (kernel_profiler_push.hpp owns trace-replay). Here they are empty -- this
// backend keeps no trace counter; nothing read it since trace-only mode was removed.
#define DeviceProfilerInit()
#define DeviceTraceOnlyProfilerInit()
#define DeviceIncrementTraceCount()

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

#define DeviceTraceOnlyProfilerInit()

#define DeviceZoneSetCounter(counter) (void(sizeof(counter)))

#define DeviceData(data_id, data) (void(sizeof(data_id) + sizeof(data)))

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

// ---- Back-compat alias ---------------------------------------------------------------------------
// DeviceData is the renamed DeviceTimestampedData; existing kernels still use the old spelling (and it
// is the one point-marker spelling the Quasar/DRAM backend also defines, so shared kernels write it).
// Defined once, outside the enabled/disabled branches, so it holds in every configuration.
#define DeviceTimestampedData(name, data) DeviceData(name, data)

#endif  // ARCH_QUASAR -- see the arch dispatch at the top of this file
