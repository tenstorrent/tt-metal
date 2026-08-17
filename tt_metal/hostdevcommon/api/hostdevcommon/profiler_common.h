// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#define PROFILER_OPT_DO_DISPATCH_CORES (1 << 1)
#define PROFILER_OPT_DO_TRACE_ONLY (1 << 2)
#define PROFILER_OPT_DO_SUM (1 << 3)
// Accumulate many invocations in L1 (main zones use growing wIndex, not fixed slots), flushing to DRAM only when nearly
// full; residual read via DRAM_AND_L1.
#define PROFILER_OPT_DO_ACCUMULATE (1 << 4)

namespace kernel_profiler {

static constexpr int SUM_COUNT = 2;
static constexpr uint32_t DRAM_PROFILER_ADDRESS_STALLED = 0xFFFFFFFF;

// Static IDs need to be unique for these features
// also must fit in 16 bits (timer_id & 0xFFFF)
static constexpr uint32_t NOC_TRACING_STATIC_ID = 12345;
static constexpr uint32_t NOC_DEBUGGING_STATIC_ID = 23456;

enum BufferIndex {
    ID_HH,
    ID_HL,
    ID_LH,
    ID_LL,
    GUARANTEED_MARKER_1_H,
    GUARANTEED_MARKER_1_L,
    GUARANTEED_MARKER_2_H,
    GUARANTEED_MARKER_2_L,
    GUARANTEED_MARKER_3_H,
    GUARANTEED_MARKER_3_L,
    GUARANTEED_MARKER_4_H,
    GUARANTEED_MARKER_4_L,
    CUSTOM_MARKERS
};

// Maximum number of RISC processors per core across all supported architectures.
// Wormhole/Blackhole Tensix have 5 (BRISC/NCRISC/TRISC0-2); Quasar Tensix has 24
// (8 DM + 4 x 4 TRISC).
static constexpr std::uint32_t PROFILER_MAX_RISC_COUNT = 24;

// ---- ID_LH marker-word bit layout (shared device packer / host decoder) ----
// Each risc's per-run ID_LH word packs three identity fields, low bits first:
//   [RISC_SHIFT , +RISC_BITS)   processor id within the core (0 .. PROFILER_MAX_RISC_COUNT-1)
//   [FLAT_SHIFT , +FLAT_BITS)   core flat id (linear index of the physical core)
//   [TRACE_SHIFT, +TRACE_BITS)  trace-replay counter
// Device (kernel_profiler.hpp) and host (profiler.cpp) MUST use these same constants so the packing
// stays in sync. RISC_BITS is sized to hold every processor so the host-side risc-id/flat-id sanity
// check applies on all archs, including Quasar's 24 processors.
static constexpr std::uint32_t PROFILER_ID_RISC_BITS = 5;
static constexpr std::uint32_t PROFILER_ID_FLAT_BITS = 8;
static constexpr std::uint32_t PROFILER_ID_TRACE_BITS = 16;
static constexpr std::uint32_t PROFILER_ID_RISC_SHIFT = 0;
static constexpr std::uint32_t PROFILER_ID_FLAT_SHIFT = PROFILER_ID_RISC_SHIFT + PROFILER_ID_RISC_BITS;
static constexpr std::uint32_t PROFILER_ID_TRACE_SHIFT = PROFILER_ID_FLAT_SHIFT + PROFILER_ID_FLAT_BITS;
static constexpr std::uint32_t PROFILER_ID_RISC_MASK = (1u << PROFILER_ID_RISC_BITS) - 1;
static constexpr std::uint32_t PROFILER_ID_FLAT_MASK = (1u << PROFILER_ID_FLAT_BITS) - 1;
static constexpr std::uint32_t PROFILER_ID_TRACE_MASK = (1u << PROFILER_ID_TRACE_BITS) - 1;
static constexpr std::uint32_t PROFILER_ID_TRACE_FIELD_MASK = PROFILER_ID_TRACE_MASK << PROFILER_ID_TRACE_SHIFT;
static constexpr std::uint32_t PROFILER_ID_RISC_FLAT_FIELD_MASK = (1u << PROFILER_ID_TRACE_SHIFT) - 1;
static_assert(
    PROFILER_MAX_RISC_COUNT <= (1u << PROFILER_ID_RISC_BITS),
    "PROFILER_ID_RISC_BITS too small to hold PROFILER_MAX_RISC_COUNT processors");
static_assert(PROFILER_ID_TRACE_SHIFT + PROFILER_ID_TRACE_BITS <= 32, "ID_LH identity fields overflow 32 bits");

// ---- Marker H-word bit layout (shared device packer / host decoder) ----
// A profiler marker is two uint32 words. The H-word packs a validity bit, the timer id, and the high
// bits of the timestamp; the following L-word holds the low 32 bits of the timestamp:
//   bit 31                       : PROFILER_MARKER_VALID  (marker-present flag)
//   [TIMER_ID_SHIFT , +TIMER_ID_BITS): timer id
//   [TS_HIGH_SHIFT , +TS_HIGH_BITS)  : high bits of the timestamp
// Device (kernel_profiler.hpp) and host (profiler.cpp) MUST use these same constants so packing stays in sync.
static constexpr std::uint32_t PROFILER_MARKER_TS_HIGH_BITS = 12;
static constexpr std::uint32_t PROFILER_MARKER_TIMER_ID_BITS = 19;
static constexpr std::uint32_t PROFILER_MARKER_TS_HIGH_SHIFT = 0;
static constexpr std::uint32_t PROFILER_MARKER_TIMER_ID_SHIFT =
    PROFILER_MARKER_TS_HIGH_SHIFT + PROFILER_MARKER_TS_HIGH_BITS;
static constexpr std::uint32_t PROFILER_MARKER_TS_HIGH_MASK = (1u << PROFILER_MARKER_TS_HIGH_BITS) - 1;
static constexpr std::uint32_t PROFILER_MARKER_TIMER_ID_MASK = (1u << PROFILER_MARKER_TIMER_ID_BITS) - 1;
static constexpr std::uint32_t PROFILER_MARKER_VALID =
    1u << (PROFILER_MARKER_TIMER_ID_SHIFT + PROFILER_MARKER_TIMER_ID_BITS);
static_assert(PROFILER_MARKER_VALID == 0x80000000u, "marker valid flag must be bit 31");
static_assert(
    PROFILER_MARKER_TIMER_ID_SHIFT + PROFILER_MARKER_TIMER_ID_BITS == 31,
    "marker H-word: timer-id field must sit just below the valid bit");

// timer id sub-layout: [packet type : 3 bits @ 16][static id : 16 bits @ 0], within the marker timer-id field.
static constexpr std::uint32_t PROFILER_TIMER_STATIC_ID_BITS = 16;
static constexpr std::uint32_t PROFILER_TIMER_STATIC_ID_MASK = (1u << PROFILER_TIMER_STATIC_ID_BITS) - 1;
static constexpr std::uint32_t PROFILER_TIMER_PACKET_TYPE_SHIFT = PROFILER_TIMER_STATIC_ID_BITS;
static constexpr std::uint32_t PROFILER_TIMER_PACKET_TYPE_MASK = 0x7;  // 3-bit packet type
static_assert(
    PROFILER_TIMER_PACKET_TYPE_SHIFT + 3 <= PROFILER_MARKER_TIMER_ID_BITS,
    "timer id: packet type must fit within the marker timer-id field");

enum ControlBuffer {
    HOST_BUFFER_END_INDEX_BR_ER = 0,
    HOST_BUFFER_END_INDEX_NC,
    HOST_BUFFER_END_INDEX_T0,
    HOST_BUFFER_END_INDEX_T1,
    HOST_BUFFER_END_INDEX_T2,
    // slots [5, PROFILER_MAX_RISC_COUNT) reserved for additional processors (e.g. Quasar DM/Neo)
    DEVICE_BUFFER_END_INDEX_BR_ER = PROFILER_MAX_RISC_COUNT,
    DEVICE_BUFFER_END_INDEX_NC,
    DEVICE_BUFFER_END_INDEX_T0,
    DEVICE_BUFFER_END_INDEX_T1,
    DEVICE_BUFFER_END_INDEX_T2,
    FW_RESET_H = 2 * PROFILER_MAX_RISC_COUNT,
    FW_RESET_L,
    DRAM_PROFILER_ADDRESS_DEFAULT,  // Used in normal profiler operation
    RUN_COUNTER,
    NOC_X,
    NOC_Y,
    FLAT_ID,
    CORE_COUNT_PER_DRAM,
    DROPPED_ZONES,
    PROFILER_DONE,
    TRACE_REPLAY_STATUS,
    // Host-set flag, non-zero on dispatch cores: in accumulate mode keeps the classic guaranteed-slot layout there so
    // their quick_push feed isn't corrupted.
    PROFILER_DISPATCH_CORE,
    // Used for device debug dump mode. Needs to come last in the control buffer
    // because we first update the host buffer end index and then the DRAM buffer address
    // Quasar device debug dump is not yet supported.
    DRAM_PROFILER_ADDRESS_BR_ER_0,
    DRAM_PROFILER_ADDRESS_NC_0,
    DRAM_PROFILER_ADDRESS_T0_0,
    DRAM_PROFILER_ADDRESS_T1_0,
    DRAM_PROFILER_ADDRESS_T2_0,
};

// ---- SPSC / drainer backend control-word layout ------------------------------------------------------
// The drainer backend overlays its OWN layout on the same profiler control vector. It deliberately does not
// reuse ControlBuffer's HOST_/DEVICE_BUFFER_END_INDEX_* slots, and deliberately derives nothing from
// PROFILER_MAX_RISC_COUNT: those are the DRAM profiler's DRAM-readout bookkeeping and its processor count,
// and this backend has no stake in either.
//
// That coupling was not hypothetical. Upstream raised PROFILER_MAX_RISC_COUNT from 5 to 24 for Quasar,
// which silently relocated this backend's ring tails from words 5..9 to 24..28 while the drainer firmware
// still read 5..9. Those became dead reserved slots reading 0, so tail always equalled head: nothing ever
// drained, the worker L1 rings filled, and every producing RISC blocked forever. A constant belonging to
// the other backend moved this one's flow-control words.
//
// Overlaying the same physical words is safe because the two backends are mutually exclusive: perf_debug
// stands the DRAM profiler down (profiler.cpp getDeviceProfilerState()) and the SPSC producer replaces the
// DRAM producer outright on every arch it compiles for.
//
// Sized for the widest processor count -- including Quasar's 24 -- so the layout is arch-uniform and the
// host indexes it identically everywhere, whatever the DRAM side later does with its own count.
static constexpr std::uint32_t PROFILER_SPSC_MAX_RISC = 24;

enum SpscControlBuffer {
    // [0, PROFILER_SPSC_MAX_RISC): ring HEAD per RISC -- consumer-written (drainer), monotonic word count.
    SPSC_RING_HEAD_0 = 0,
    // [PROFILER_SPSC_MAX_RISC, 2*): ring TAIL per RISC -- producer-written, monotonic word count.
    SPSC_RING_TAIL_0 = PROFILER_SPSC_MAX_RISC,
    // Host->kernel terminate signal: set at teardown when the drainer consumer is stopping. While clear, a
    // producing RISC BLOCKS on a full ring (lossless). While set, the producer stops blocking and
    // proceeds, so a dispatch core cannot get stuck in ring_ensure_room and wedge wait_until_cores_done()
    // during device close.
    PROFILER_TERMINATE = 2 * PROFILER_SPSC_MAX_RISC,
    // Core identity: NoC coords packed as (y << 16) | x, written once by BRISC FW at init from
    // my_x[0]/my_y[0]. Deliberately NOT the push-to-DRAM profiler's ControlBuffer::NOC_X/NOC_Y or its
    // FLAT_ID -- that enum belongs to the other backend and the two never share a word.
    //
    // Coords rather than a flat id on purpose: the flat id is a dense rank over a sorted map of
    // Tensix+Eth cores, so it has no positional formula, shifts with harvesting, and can only be
    // computed host-side. (x,y) is what the core already knows about itself, needs no host round-trip,
    // and the host can map it back however it likes. It rides along free in every bulk read, since the
    // control vector is the first 256 B of the span -- a drainer never injects or constructs identity.
    SPSC_CORE_XY = 2 * PROFILER_SPSC_MAX_RISC + 1,
    // Per-RISC count of times this producer BLOCKED on a full ring. Written by the producer itself in the
    // stall path, read by the host directly out of L1 at teardown.
    //
    // This is the knee metric, and it lives here rather than being counted from decoded PROFILER_STALL_ZONE
    // markers because a marker has to survive the whole pipeline to be counted -- DRISC frame, PCIe FIFO,
    // buffer pool, decoder, BroadcastRing -- and every one of those can drop it (measured: 1.29 M records
    // dropped at the ring, ~27 K unaccounted elsewhere). A counter in the producer's own L1 cannot be lost,
    // and reading it needs no decode at all, so the measurement stops perturbing the thing it measures.
    //
    // 8 slots, indexed by processor id: enough for Tensix (5), Eth and DRAM, without pushing
    // SPSC_CONTROL_END past the 64-word control vector.
    SPSC_STALL_COUNT_0 = 2 * PROFILER_SPSC_MAX_RISC + 2,
    SPSC_STALL_COUNT_MAX = 8,
    SPSC_CONTROL_END = SPSC_STALL_COUNT_0 + SPSC_STALL_COUNT_MAX,  // first unused word; grow the layout here
};

// Size of the drain kernel's results block, in words. Shared because the host both ZEROES and READS it and
// lays the handshake block out immediately behind it -- three places that silently disagreed would each fail
// differently (a stale counter, a short read, an overlapping handshake). Was 64 and exactly full; the DRISC
// self-profiling counters need out[64..87], and the NoC-footprint counters out[88..119].
//
// WHY 144 IS FREE, and why it must not be raised carelessly. This block lives inside the drain kernel's
// `kMiscBytes` budget in perf_debug_profiler.cpp, which is 1024 B holding done(64) + stop(64) + results +
// handshake(64). At 96 words that was 576 B of 1024, i.e. 448 B of slack, and 144 words spends 192 of it
// (768 B total). Nothing else moves: `kMiscBytes` is unchanged, so `fixed` is unchanged, so the number of
// STAGING SLOTS the same L1 can hold is unchanged -- which matters because nstage is 7 by a margin of well
// under one slot, and losing one would silently drop a mover's max batch 7 -> 6. Raising this past ~208
// words WOULD grow kMiscBytes and cost a staging slot. Check that arithmetic, not just this constant.
static constexpr std::uint32_t SPSC_DRAIN_RESULT_WORDS = 144;

// ---- Reserved zone ids for DRAINER-AUTHORED zones (DRISC self-profiling) ----------------------------
//
// The drainer emits its own zones into its own span frame, so those zones need ids the host can name. They
// cannot come from Hash16_CT: the host resolves names by harvesting `#pragma message` source locations out
// of the JIT build log, and the drain kernel's zones are not scoped by the DeviceZoneScopedN macros (they
// are stamped from timestamps the drain loop had already read -- see drisc_profiler_drain.cpp). So they take
// FIXED ids in the same reserved band PROFILER_STALL_ZONE_ID (0x7FFF) already uses, and the host registers
// their names explicitly next to PRODUCER-STALL.
//
// 0x7FF0..0x7FF8, i.e. immediately below the stall zone: a 16-bit hash landing here is possible in principle
// but has the same (accepted) probability the stall id has carried since it was introduced.
static constexpr std::uint32_t PROFILER_DRISC_ZONE_BASE = 0x7FF0;
enum DriscSelfZone : std::uint32_t {
    DRISC_ZONE_SWEEP = PROFILER_DRISC_ZONE_BASE + 0,        // one whole poll sweep (the parent)
    DRISC_ZONE_READ = PROFILER_DRISC_ZONE_BASE + 1,         // filler: span-read ISSUE. mover: the DRAM read
    DRISC_ZONE_READ_WAIT = PROFILER_DRISC_ZONE_BASE + 2,    // filler: the read-barrier wait left after proc
    DRISC_ZONE_PROC = PROFILER_DRISC_ZONE_BASE + 3,         // control-vector scan + head write-back
    DRISC_ZONE_CREDIT_WAIT = PROFILER_DRISC_ZONE_BASE + 4,  // filler: DRAM ring room. mover: socket credit
    DRISC_ZONE_WRITE = PROFILER_DRISC_ZONE_BASE + 5,        // the egress write itself
    DRISC_ZONE_WR_BARRIER = PROFILER_DRISC_ZONE_BASE + 6,   // write barrier before staging is reused
    // The inter-sweep PACING GAP -- a sibling of SWEEP at depth 0, not a child. A filler's gap is set by the
    // fill-ratio controller and measured 17,156 cycles (12.7 us) against its own 8.5 us sweep, so without a
    // zone for it a filler's Tracy row is more whitespace than zones and the whitespace looks like a gap in
    // the instrument. A MOVER is excluded from the controller (see drisc_profiler_drain.cpp) so its gap is 0
    // and this zone simply never appears there -- which is itself the answer to "is the mover being paced".
    DRISC_ZONE_PACE = PROFILER_DRISC_ZONE_BASE + 7,
    // COMMON-TRIGGER SYNC EVENT: every drainer marks the SAME physical instant (released together from a
    // rendezvous barrier), so the spread in these zones' rendered timestamps is anchor + render error only.
    // Replaces comparing zone-window OPENs, which are two independent events and cannot validate an anchor.
    DRISC_ZONE_SYNC = PROFILER_DRISC_ZONE_BASE + 8,
    DRISC_ZONE_COUNT = 9,
};

// STICKY_META (SPSC/drainer backend): an 8B context packet emitted once per RISC per launch at the main
// zone scope. High word carries (core_x, core_y, risc) + this type; low word a 32-bit host-side ID. The
// host forward-fills that identity onto the following timing markers so the drainer reader can bulk-copy raw
// markers with NO per-marker reshape. Its type sits in the same bits (28-30 of w0) as a marker's type, so
// the host distinguishes it before decoding the rest. Must stay <= 7 (3-bit type field).
enum PacketTypes { ZONE_START, ZONE_END, ZONE_TOTAL, TS_DATA, TS_EVENT, TS_DATA_16B, STICKY_META };

// Number of expected uint64_t data values for each PacketType
template <PacketTypes packet_type>
struct TimestampedDataSize {
    // No checks
    static constexpr std::uint32_t size = 0;
};

template <>
struct TimestampedDataSize<TS_DATA> {
    static constexpr std::uint32_t size = 1;
};

template <>
struct TimestampedDataSize<TS_DATA_16B> {
    static constexpr std::uint32_t size = 2;
};

// TODO: use data types in profile_msg_t rather than addresses/sizes
// NOTE: this cannot be grown. On Wormhole sizeof(mailboxes_t) is EXACTLY MEM_MAILBOX_SIZE (13296 bytes),
// so +1 word here fails the static_assert in llrt/hal/tt-1xx/wormhole/wh_hal_tensix.cpp; 8-byte padding
// makes the real cost of one extra word 16 bytes. Measured, not assumed: 64 -> 13296, 66 -> 13312,
// 72 -> 13328. Anything new must reuse a slot (see PROFILER_TERMINATE above), not append one.
//
// PRE-EXISTING OVERFLOW, not introduced here and not fixable by growing: with
// PROFILER_MAX_RISC_COUNT = 24 the two RISC-indexed blocks consume [0,48), which pushes the tail entries
// to 48..64, so DRAM_PROFILER_ADDRESS_T2_0 evaluates to 64 -- one past the last valid index of a 64-word
// vector. Host-side that is an out-of-bounds write into a 64-element std::vector (profiler.cpp resizes to
// PROFILER_L1_CONTROL_VECTOR_SIZE); device-side it writes past control_vector[] inside mailboxes_t. It is
// latent because only device debug-dump mode touches those five slots. Fixing it properly means moving
// the DRAM_PROFILER_ADDRESS_*_0 block into the reserved band too, which is an upstream layout decision.
constexpr static std::uint32_t PROFILER_L1_CONTROL_VECTOR_SIZE = 64;
constexpr static std::uint32_t PROFILER_L1_CONTROL_BUFFER_SIZE = PROFILER_L1_CONTROL_VECTOR_SIZE * sizeof(uint32_t);

// Bounds the SPSC backend's whole control block. Deliberately not asserted on
// DRAM_PROFILER_ADDRESS_T2_0: that entry is already out of bounds upstream (see above), so asserting it
// would fail the build on a defect this branch neither introduced nor can fix here.
static_assert(
    SPSC_CONTROL_END <= PROFILER_L1_CONTROL_VECTOR_SIZE,
    "SPSC/drainer control layout overflows the profiler L1 control vector");
// Governs the L1 buffer SIZING (part of mailboxes_t, which is L1-size-bounded) and the DRAM path.
// BOTH paths use 2-word markers -- SPSC_MARKER_WORDS in kernel_profiler.hpp is 2 -- so the 512-word ring
// that PROFILER_L1_VECTOR_SIZE yields below holds 256 markers per RISC.
// Do NOT size anything off a 4-word drainer marker: this comment used to claim that (and 128 markers per
// ring), which was never true of the code as committed.
constexpr static std::uint32_t PROFILER_L1_MARKER_UINT32_SIZE = 2;
constexpr static std::uint32_t PROFILER_L1_PROGRAM_ID_COUNT = 2;
constexpr static std::uint32_t PROFILER_L1_GUARANTEED_MARKER_COUNT = 4;
constexpr static std::uint32_t PROFILER_L1_OPTIONAL_MARKER_COUNT = 250;
constexpr static std::uint32_t PROFILER_L1_VECTOR_SIZE =
    (PROFILER_L1_OPTIONAL_MARKER_COUNT + PROFILER_L1_GUARANTEED_MARKER_COUNT + PROFILER_L1_PROGRAM_ID_COUNT) *
    PROFILER_L1_MARKER_UINT32_SIZE;
constexpr static std::uint32_t PROFILER_L1_BUFFER_SIZE = PROFILER_L1_VECTOR_SIZE * sizeof(uint32_t);

// ---- SPSC SPAN frame: the identity-free drain wire format ------------------------------------------
//
// A drainer should not know, or claim to know, who it is draining. Everything the host needs is already
// in the 256 B control vector the drainer had to poll anyway: identity in SPSC_CORE_XY, progress in the
// heads, extent in the tails. So the frame is that control vector, verbatim, followed by the ring bytes
// it describes -- and the drainer injects nothing.
//
//   [0]                       w0 = SPSC_SPAN_PACKET_TYPE << PP_TYPE_SHIFT; low 27 bits RESERVED, zero
//   [1]                       payload_words = PROFILER_L1_CONTROL_VECTOR_SIZE + shipped ring words
//   [2 .. PREFIX)             zero
//   [PREFIX .. +CONTROL)      the worker's profiler control vector, verbatim
//   [.. +payload)             for each RISC in ascending order, its LIVE run, packed exactly
//   [.. frame_words)          zero pad to a 64 B socket page
//
// The host recomputes the geometry from the control vector the frame carries -- run = tail - head per
// RISC -- so there is nothing on the wire to desynchronize: no lane tags, no run lengths, no core id.
//
// Runs are packed EXACTLY, with no per-lane alignment padding, because the drainer assembles the frame with
// CPU loads and stores out of a bulk snapshot rather than landing NoC reads into it. That is the payoff of
// the fused read: L1-local copies are not bound by L1_ALIGNMENT, so the alignment hazard below cannot arise
// on the payload at all, and the ring wrap is resolved device-side into a flat array.
//
// Prefix is 16 words (64 B) so the control vector starts at 64 B and the payload at 320 B -- both
// multiples of L1_ALIGNMENT (16 B on Blackhole), which keeps the WHOLE-PAGE PCIe write aligned. Alignment
// is not a nicety here: the NoC MIS-DELIVERS a misaligned transfer rather than rejecting it.
constexpr static std::uint32_t SPSC_SPAN_PREFIX_WORDS = 16;
// Wire type code. Must equal PP_BULK_SPAN in tt_metal/tools/profiler/spsc_packet.h, which is plain C and
// cannot include this header; spsc_marker_decode.hpp static_asserts that the two agree.
constexpr static std::uint32_t SPSC_SPAN_PACKET_TYPE = 13;
// Where the packet type sits in word0 of every packet in this stream (PP_TYPE_SHIFT in spsc_packet.h).
constexpr static std::uint32_t SPSC_SPAN_TYPE_SHIFT = 27;
// Socket page granularity, in words -- a frame is padded up to a whole number of these.
//
// MEASURED, do not "optimize" again without re-measuring: setting this to the whole frame (2,640 words)
// collapsed page operations 165x (2.5 M -> 8.9 K) and cut bytes shipped 41%, and bought NOTHING -- total
// busy time went 34.0 -> 34.3 ms. It also concentrated the same credit-wait into fewer, longer stalls
// (167 -> 339 us per busy sweep), pushing the worst sweep past the ring-fill deadline and turning 0
// producer stalls into 952. Socket page bookkeeping is not the host-egress wall.
constexpr static std::uint32_t SPSC_SPAN_PAGE_WORDS = 16;

// Live words one RISC contributes to a span frame. With exact packing this is the entire geometry: the
// drainer ships `run` words and the host consumes `run` words, both derived from the same control vector.
//
// Head and tail are monotonic word counters, so the subtraction is wrap-safe. A lossless producer blocks at
// capacity, so a run wider than the ring means the snapshot was read torn -- clamped here, and counted by
// the caller rather than trusted.
inline std::uint32_t spsc_span_live(std::uint32_t head, std::uint32_t tail, std::uint32_t cap) {
    const std::uint32_t run = tail - head;
    return run > cap ? cap : run;
}

inline std::uint32_t spsc_span_w0() { return SPSC_SPAN_PACKET_TYPE << SPSC_SPAN_TYPE_SHIFT; }

// ---- Packing for a DRAINER-AUTHORED marker (DRISC self-profiling) ----------------------------------
//
// The drain kernel produces its own zones, so it needs the same 2-word packing kernel_profiler.hpp's ppfmt
// does -- but it cannot use that header: kernel_profiler.hpp binds itself to the mailbox profiler region and
// to PROCESSOR_INDEX, and a DRISC drain kernel emits into a span frame it assembles in its staging area
// instead. These are the two encodings it needs, defined here where the wire's producer, its host decoder
// and the frame layout above all already live. Codes MUST match spsc_packet.h's PP_* -- asserted in
// spsc_marker_decode.hpp, which is the one place that sees both headers.
static constexpr std::uint32_t SPSC_TYPE_ZONE_START = 0;
static constexpr std::uint32_t SPSC_TYPE_ZONE_END = 1;
static constexpr std::uint32_t SPSC_TYPE_STICKY_TIMER = 9;
static constexpr std::uint32_t SPSC_TIMER_HI_MASK = 0x7FFFFFFu;  // the 27-bit low field of word0

inline std::uint32_t spsc_marker_w0(std::uint32_t type, std::uint32_t zone_id) {
    return (type << SPSC_SPAN_TYPE_SHIFT) | (zone_id & 0xFFFFu);
}
inline std::uint32_t spsc_sticky_timer_w0(std::uint32_t timer_hi) {
    return (SPSC_TYPE_STICKY_TIMER << SPSC_SPAN_TYPE_SHIFT) | (timer_hi & SPSC_TIMER_HI_MASK);
}
// PP_DATA word0: a point-in-time packet with a self-describing payload length. Mirrors spsc_packet.h's
// pp_data_w0 (low27 = size(7) << 20 | id(20)) and is asserted against it in spsc_marker_decode.hpp, the one
// translation unit that sees both headers.
static constexpr std::uint32_t SPSC_TYPE_DATA = 10;
static constexpr std::uint32_t SPSC_DATA_SIZE_SHIFT = 20;
inline std::uint32_t spsc_data_w0(std::uint32_t id, std::uint32_t size_words) {
    return (SPSC_TYPE_DATA << SPSC_SPAN_TYPE_SHIFT) | ((size_words & 0x7Fu) << SPSC_DATA_SIZE_SHIFT) | (id & 0xFFFFFu);
}

// ---- NoC-FOOTPRINT per-sweep sample: the PP_DATA payload contract ------------------------------------
//
// One PP_DATA packet per drainer sweep, carrying that sweep's NoC counter DELTAS. Defined HERE, in the one
// header both the drain kernel and the host consumer already include, because a positional payload described
// in two places is a format that drifts -- which is the whole reason PP_* lives in a shared header at all.
//
// Wire shape (see spsc_packet.h PP_DATA): [0] word0 = data_w0(id, size)  [1] timer_low  [2..] payload.
// The 7-bit size field DECLARES the payload length and the host walk advances by 2 + size, so the payload
// carries exactly the live counters for this role and is never padded to a fixed shape.
//
// ROLE-SPECIALISED, and that is where the code saving comes from: kRole is a compile-time arg and each role
// gets its own JIT ELF, so `if constexpr` compiles only that role's four counters. The other four are
// structurally zero -- measured exactly 0 on silicon (FINDINGS N+42/N+43: a filler's NoC 0 read and NoC 1
// write columns, a mover's entire NoC 1 side) -- so shipping them per sweep would be paying to transmit a
// proven constant. The out[] LIFETIME totals still carry all eight; that is where the NoC-split proof lives
// as a standing invariant, and it costs nothing because it is read once at teardown.
//
// ONE TIMESTAMP FOR THE WHOLE PACKET, and this is correct rather than a compromise. Every value is a DELTA
// OVER THE INTERVAL since the previous sample, and a delta belongs to an interval, not to an instant. The
// counters are also read back-to-back a few tens of cycles apart, so per-counter timestamps would be
// fictitious precision. Do not "fix" this by splitting the packet.
static constexpr std::uint32_t SPSC_DATA_ID_NOCFP = 0x7FF0;  // same reserved band as the DRISC zone ids
static constexpr std::uint32_t SPSC_NOCFP_WORDS = 4;         // payload words == values, in the order below
enum SpscNocFpWord {
    // A FILLER reads on kReadNoc (NoC 1) and writes on NOC_INDEX (NoC 0); a MOVER does both on NOC_INDEX.
    // So these four names are per-ROLE meanings of "the NoC this role reads on" / "the NoC it writes on",
    // NOT fixed NoC indices -- the host resolves which physical NoC from out[38]/out[39], which already
    // report NOC_INDEX and kReadNoc.
    SPSC_NOCFP_RD_WORDS = 0,  // NIU_MST_RD_DATA_WORD_RECEIVED delta, in NoC words (NOC_WORD_BYTES each)
    SPSC_NOCFP_RD_TXNS = 1,   // NIU_MST_RD_REQ_SENT delta
    SPSC_NOCFP_WR_WORDS = 2,  // NIU_MST_NONPOSTED_WR_DATA_WORD_SENT delta, in NoC words
    SPSC_NOCFP_WR_TXNS = 3,   // NIU_MST_NONPOSTED_WR_REQ_SENT delta
};

// Total words a frame occupies on the wire, including the prefix and the pad up to a socket page.
inline std::uint32_t spsc_span_frame_words(std::uint32_t payload_words) {
    const std::uint32_t n = SPSC_SPAN_PREFIX_WORDS + payload_words;
    return (n + SPSC_SPAN_PAGE_WORDS - 1u) & ~(SPSC_SPAN_PAGE_WORDS - 1u);
}

static_assert(
    (SPSC_SPAN_PREFIX_WORDS + PROFILER_L1_CONTROL_VECTOR_SIZE) % SPSC_SPAN_PAGE_WORDS == 0,
    "the payload must start on a socket page boundary");

}  // namespace kernel_profiler
