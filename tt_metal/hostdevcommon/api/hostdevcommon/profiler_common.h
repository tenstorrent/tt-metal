// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "hostdevcommon/profiler_zone_id.h"

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

// The relay backend overlays its own layout on the same control vector, reusing none of ControlBuffer's slots
// and deriving nothing from PROFILER_MAX_RISC_COUNT (a change there would relocate this backend's flow-control
// words). Safe because the two backends are mutually exclusive: getDeviceProfilerState() stands the DRAM
// profiler down. Sized for 24 processors so the layout is arch-uniform.
static constexpr std::uint32_t PROFILER_SPSC_MAX_RISC = 24;
// Tensix RISCs whose rings the relay sweeps and the host decodes (BRISC, NCRISC, TRISC0-2).
static constexpr std::uint32_t PROFILER_SPSC_TENSIX_RISC = 5;

enum SpscControlBuffer {
    // [0, PROFILER_SPSC_MAX_RISC): ring head per RISC, consumer-written (relay), monotonic word count.
    SPSC_RING_HEAD_0 = 0,
    // [PROFILER_SPSC_MAX_RISC, 2*): ring tail per RISC, producer-written, monotonic word count.
    SPSC_RING_TAIL_0 = PROFILER_SPSC_MAX_RISC,
    // Host->kernel arm: while set a producer blocks on a full ring, because a relay is draining this core; while
    // clear it proceeds and overwrites. BRISC FW clears it once per session, and the host sets it only on the cores
    // its relays serve, so a core nobody drains (dispatch cores, whose rings fill one launch at a time across
    // processes) can never park in the stall path and wedge wait_until_cores_done() at device close.
    PROFILER_ARMED = 2 * PROFILER_SPSC_MAX_RISC,
    // NoC coords packed (y << 16) | x, written once by BRISC FW at init. Coords, not the flat id: the flat id is a
    // dense rank over a sorted core map with no positional formula, computable only host-side.
    SPSC_CORE_XY = 2 * PROFILER_SPSC_MAX_RISC + 1,
    // Per-RISC count of full-ring blocks, written in the stall path and read by the host from L1 at teardown;
    // counting decoded stall markers would undercount, since a marker can be dropped between the relay frame and
    // the BroadcastRing. 8 slots so SPSC_CONTROL_END stays inside the 64-word vector.
    SPSC_STALL_COUNT_0 = 2 * PROFILER_SPSC_MAX_RISC + 2,
    SPSC_STALL_COUNT_MAX = 8,
    SPSC_CONTROL_END = SPSC_STALL_COUNT_0 + SPSC_STALL_COUNT_MAX,  // first unused word; grow the layout here
};

// Host->relay stop word: quiesce drains everything with every wait still holding, release is the kill
// switch that abandons the waits and hands the NIU back.
static constexpr std::uint32_t kRelayStopQuiesce = 1;
static constexpr std::uint32_t kRelayStopRelease = 2;
// Relay->host completion word, published only after the socket barrier; the host matches the high half.
static constexpr std::uint32_t kRelayDoneWord = 0xD09E0000u;
static constexpr std::uint32_t kRelayDoneMask = 0xFFFF0000u;
// Each relay control word owns a 64 B pad, so the words that share it (the sync rendezvous triple behind
// the stop word, the heartbeat behind done) travel in one host write.
static constexpr std::uint32_t kRelayCtrlWordStride = 64;

// STICKY_META: an 8 B context packet once per RISC per launch; high word (core_x, core_y, risc) plus this
// type, low word a host-side id the host forward-fills onto the following markers. The type shares bits
// 28-30 with a marker's type. Must stay <= 7.
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
// Cannot be grown: on Wormhole sizeof(mailboxes_t) is exactly MEM_MAILBOX_SIZE, so anything new must reuse a
// slot. ControlBuffer already overflows this vector with PROFILER_MAX_RISC_COUNT = 24 (DRAM_PROFILER_ADDRESS_T2_0
// lands at index 64), latent because only device debug-dump mode touches those slots.
constexpr static std::uint32_t PROFILER_L1_CONTROL_VECTOR_SIZE = 64;
constexpr static std::uint32_t PROFILER_L1_CONTROL_BUFFER_SIZE = PROFILER_L1_CONTROL_VECTOR_SIZE * sizeof(uint32_t);

// Not asserted on DRAM_PROFILER_ADDRESS_T2_0, which is already out of bounds.
static_assert(
    SPSC_CONTROL_END <= PROFILER_L1_CONTROL_VECTOR_SIZE,
    "SPSC/relay control layout overflows the profiler L1 control vector");
// Both paths use 2-word markers, so the 512-word ring holds 256 markers per RISC.
constexpr static std::uint32_t PROFILER_L1_MARKER_UINT32_SIZE = 2;
constexpr static std::uint32_t PROFILER_L1_PROGRAM_ID_COUNT = 2;
constexpr static std::uint32_t PROFILER_L1_GUARANTEED_MARKER_COUNT = 4;
constexpr static std::uint32_t PROFILER_L1_OPTIONAL_MARKER_COUNT = 250;
constexpr static std::uint32_t PROFILER_L1_VECTOR_SIZE =
    (PROFILER_L1_OPTIONAL_MARKER_COUNT + PROFILER_L1_GUARANTEED_MARKER_COUNT + PROFILER_L1_PROGRAM_ID_COUNT) *
    PROFILER_L1_MARKER_UINT32_SIZE;
constexpr static std::uint32_t PROFILER_L1_BUFFER_SIZE = PROFILER_L1_VECTOR_SIZE * sizeof(uint32_t);

// SPSC span frame, the relay wire format: a control block (identity from SPSC_CORE_XY, progress from the heads,
// extent from the tails) followed by the ring words it describes; the relay injects nothing of its own.
// 
//   [0]                       w0 = SPSC_SPAN_PACKET_TYPE << PP_TYPE_SHIFT; low 27 bits are layout flags
//   [1]                       payload_words = control block + pack pads + shipped ring words
//   [2 .. PREFIX)             zero
//   [PREFIX .. +CONTROL)      packed frame: the SPSC_SPAN_WIRE_CTRL_WORDS control block; raw frame: the
//                             worker's whole 64-word control vector
//   [.. +payload)             per RISC in ascending order with a live run: spsc_span_pack_pad() skipped words,
//                             then the run, ring wrap resolved into a flat array
//   [.. frame_words)          skipped words up to a 64 B socket page
// 
// The host recomputes the geometry from the control block (per RISC, run = spsc_span_live(head, tail) ending
// at word counter tail), so the wire carries no lane tags, run lengths or core id. The NIU gathers each live
// window straight into the host FIFO, and a NoC write mis-delivers a transfer whose destination is not
// congruent to the source modulo NOC_PCIE_WRITE_ALIGNMENT_BYTES (16 B), so each run is preceded by
// spsc_span_pack_pad() skipped words, never written; the host reads past them. The 16-word prefix puts the
// control block at 64 B and the payload at 320 B, both L1_ALIGNMENT multiples.
constexpr static std::uint32_t SPSC_SPAN_PREFIX_WORDS = 16;
// Wire type code. Must equal PP_BULK_SPAN in tt_metal/tools/profiler/spsc_packet.h, which is plain C and
// cannot include this header; spsc_marker_decode.hpp static_asserts that the two agree.
constexpr static std::uint32_t SPSC_SPAN_PACKET_TYPE = 13;
// Where the packet type sits in word0 of every packet in this stream (PP_TYPE_SHIFT in spsc_packet.h).
constexpr static std::uint32_t SPSC_SPAN_TYPE_SHIFT = 27;
// Socket page granularity in words; frames pad up to a whole number. Larger pages concentrate the same credit
// wait into stalls long enough to miss the ring-fill deadline.
constexpr static std::uint32_t SPSC_SPAN_PAGE_WORDS = 16;

// Live words one RISC contributes. Head and tail are monotonic, so the subtraction is wrap-safe; a run wider
// than the ring means a torn snapshot and is clamped here and counted by the caller.
constexpr std::uint32_t spsc_span_live(std::uint32_t head, std::uint32_t tail, std::uint32_t cap) {
    const std::uint32_t run = tail - head;
    return run > cap ? cap : run;
}

// NoC L1->PCIe write congruence quantum (NOC_PCIE_WRITE_ALIGNMENT_BYTES), in words.
constexpr static std::uint32_t SPSC_SPAN_PACK_ALIGN_WORDS = 4;
// Skipped words before a live run so the NIU gather lands src/dst congruent; frame start and staged span sit
// at alignment multiples, so only the run's ring phase and the wire offset decide it. Relay and host must
// agree or every later lane mis-walks.
constexpr std::uint32_t spsc_span_pack_pad(std::uint32_t start_counter, std::uint32_t frame_off_words) {
    return (start_counter - frame_off_words) & (SPSC_SPAN_PACK_ALIGN_WORDS - 1u);
}

// A wrapping run ships as its whole ring image only when the dead remainder is small, else as a two-piece
// split: the one-read image saves a NoC issue at the saturation boundary, but at sustained rates it inflates
// egress by the remainder. Both sides derive this from (start, extent) alone; the wire carries no flag.
constexpr static std::uint32_t SPSC_SPAN_WRAP_IMAGE_MAX_PAD_WORDS = 64;
constexpr bool spsc_span_wrap_image(std::uint32_t start, std::uint32_t extent, std::uint32_t ring_cap) {
    return (start & (ring_cap - 1u)) + extent > ring_cap && ring_cap - extent <= SPSC_SPAN_WRAP_IMAGE_MAX_PAD_WORDS;
}

inline std::uint32_t spsc_span_w0() { return SPSC_SPAN_PACKET_TYPE << SPSC_SPAN_TYPE_SHIFT; }

// Control block of a packed frame: just the words the decoder walks. The L1 vector is 64 words laid out for
// 24 RISCs, ~50 of them dead on the wire. Raw frames carry the true vector; the w0 raw flag picks the geometry.
constexpr static std::uint32_t SPSC_SPAN_WIRE_CTRL_WORDS = 16;
enum SpscWireCtrl : std::uint32_t {
    SPSC_WIRE_HEAD_0 = 0,  // ..4
    SPSC_WIRE_TAIL_0 = 5,  // ..9
    SPSC_WIRE_XY = 10,
};

// w0 flag: the payload is the raw span (control vector plus five whole rings, wrap unresolved) rather than
// packed live runs. Packing trades bytes for ~10 extra NoC issues per frame, so above the fill threshold raw
// wins.
constexpr static std::uint32_t SPSC_SPAN_RAW_FLAG = 1u;

// Packing for a relay-authored marker: the relay cannot use kernel_profiler.hpp (bound to the mailbox region
// and PROCESSOR_INDEX). Codes must match spsc_packet.h's PP_*, asserted in spsc_marker_decode.hpp.

// Producer tail-publish batch: the published tail may lag occupancy by this many words between fenced
// publishes. Power of two.
static constexpr std::uint32_t SPSC_PUBLISH_BATCH_WORDS = 16;

static constexpr std::uint32_t SPSC_TYPE_ZONE_START = 0;   // legacy pair (workers: stall zone, >3.2s fallback)
static constexpr std::uint32_t SPSC_TYPE_ZONE_END = 1;     // legacy pair
static constexpr std::uint32_t SPSC_TYPE_ZONE_ATOMIC = 2;  // one whole zone: id | end timer_low | duration32
static constexpr std::uint32_t SPSC_TYPE_ZONE_L = 4;       // >3.2 s zone: id | end_lo | end_hi | dur_lo | dur_hi
static constexpr std::uint32_t SPSC_TYPE_STICKY_TIMER = 9;
static constexpr std::uint32_t SPSC_TIMER_HI_MASK = 0x7FFFFFFu;  // the 27-bit low field of word0

// Truncating the id is invisible at runtime (names fail to resolve); a width change must hit every copy of
// the packer: this, ppfmt in kernel_profiler.hpp, pp_* in spsc_packet.h.
inline std::uint32_t spsc_zone_atomic_w0(std::uint32_t zone_id) {
    return (SPSC_TYPE_ZONE_ATOMIC << SPSC_SPAN_TYPE_SHIFT) | (zone_id & TT_ZONE_ID_MASK);
}

inline std::uint32_t spsc_marker_w0(std::uint32_t type, std::uint32_t zone_id) {
    return (type << SPSC_SPAN_TYPE_SHIFT) | (zone_id & TT_ZONE_ID_MASK);
}
inline std::uint32_t spsc_sticky_timer_w0(std::uint32_t timer_hi) {
    return (SPSC_TYPE_STICKY_TIMER << SPSC_SPAN_TYPE_SHIFT) | (timer_hi & SPSC_TIMER_HI_MASK);
}
// 3 + N words: word0 is shaped like a zone marker (type | 27-bit id), the length lives in word2. Mirrors
// spsc_packet.h's pp_data_w0/w2.
static constexpr std::uint32_t SPSC_TYPE_DATA = 10;
static constexpr std::uint32_t SPSC_DATA_SIZE_SHIFT = 25;
inline std::uint32_t spsc_data_w0(std::uint32_t id) {
    return (SPSC_TYPE_DATA << SPSC_SPAN_TYPE_SHIFT) | (id & TT_ZONE_ID_MASK);
}
inline std::uint32_t spsc_data_w2(std::uint32_t size_words) { return (size_words & 0x7Fu) << SPSC_DATA_SIZE_SHIFT; }

// The relay's RAII zone; identity, ELF record and naming match a worker zone's, only the transport (`mark`)
// differs. `started_`: the relay decides mid-sweep whether to instrument, and a destructor must not write an
// orphan END.
template <std::uint32_t ZoneId, typename MarkFn>
class SpscZoneScope {
public:
    inline __attribute__((always_inline)) explicit SpscZoneScope(MarkFn& mark) :
        mark_(mark), started_(mark(spsc_marker_w0(SPSC_TYPE_ZONE_START, ZoneId))) {}
    inline __attribute__((always_inline)) ~SpscZoneScope() {
        if (started_) {
            (void)mark_(spsc_marker_w0(SPSC_TYPE_ZONE_END, ZoneId));
        }
    }
    SpscZoneScope(const SpscZoneScope&) = delete;
    SpscZoneScope& operator=(const SpscZoneScope&) = delete;

private:
    MarkFn& mark_;
    bool started_;
};

// A slot holds a packed span image, which can exceed the raw span: the packed layout places extents back to
// back, so each lane may need up to SPSC_SPAN_PACK_ALIGN_WORDS-1 words of pad.
constexpr std::uint32_t spsc_span_slot_words(std::uint32_t num_risc) {
    const std::uint32_t span = PROFILER_L1_CONTROL_VECTOR_SIZE + num_risc * PROFILER_L1_VECTOR_SIZE;
    const std::uint32_t worst = SPSC_SPAN_PREFIX_WORDS + span + num_risc * (SPSC_SPAN_PACK_ALIGN_WORDS - 1u);
    return (worst + SPSC_SPAN_PAGE_WORDS - 1u) & ~(SPSC_SPAN_PAGE_WORDS - 1u);
}

constexpr std::uint32_t spsc_span_frame_words(std::uint32_t payload_words) {
    const std::uint32_t n = SPSC_SPAN_PREFIX_WORDS + payload_words;
    return (n + SPSC_SPAN_PAGE_WORDS - 1u) & ~(SPSC_SPAN_PAGE_WORDS - 1u);
}

static_assert(
    (SPSC_SPAN_PREFIX_WORDS + PROFILER_L1_CONTROL_VECTOR_SIZE) % SPSC_SPAN_PAGE_WORDS == 0,
    "the payload must start on a socket page boundary");

}  // namespace kernel_profiler
