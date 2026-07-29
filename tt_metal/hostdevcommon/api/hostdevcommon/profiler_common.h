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
    //
    // Host->kernel terminate signal (SPSC/X280 backend): set at teardown when the X280 consumer is
    // stopping. While clear, a producing RISC BLOCKS on a full ring (lossless). While set, the producer
    // stops blocking and proceeds, so a dispatch core cannot get stuck in ring_ensure_room and wedge
    // wait_until_cores_done() during device close.
    //
    // It borrows the TOP of that reserved band instead of being appended to the end of this enum,
    // because the control vector has NO room to grow: on Wormhole sizeof(mailboxes_t) is EXACTLY
    // MEM_MAILBOX_SIZE (13296 bytes, measured), so appending even one slot trips wh_hal_tensix's
    // static_assert (and 8-byte padding makes the true cost of +1 word 16 bytes). The band is reserved
    // for processors 5..23, which exist only on Quasar -- and the SPSC backend is never compiled there
    // (see the arch dispatch at the top of kernel_profiler.hpp), so on tt-1xx these slots are dead space.
    PROFILER_TERMINATE = PROFILER_MAX_RISC_COUNT - 1,
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

// STICKY_META (SPSC/X280 backend): an 8B context packet emitted once per RISC per launch at the main
// zone scope. High word carries (core_x, core_y, risc) + this type; low word a 32-bit host-side ID. The
// host forward-fills that identity onto the following timing markers so the X280 reader can bulk-copy raw
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

// Guards this backend's own slot. Deliberately not asserted on DRAM_PROFILER_ADDRESS_T2_0: that entry is
// already out of bounds upstream (see above), so asserting it would fail the build on a defect this
// branch neither introduced nor can fix here.
static_assert(
    PROFILER_TERMINATE < PROFILER_L1_CONTROL_VECTOR_SIZE &&
        PROFILER_TERMINATE >= 5,  // inside the [5, PROFILER_MAX_RISC_COUNT) reserved band, not a real RISC slot
    "PROFILER_TERMINATE must live in the reserved processor band and inside the L1 control vector");
// Governs the L1 buffer SIZING (part of mailboxes_t, which is L1-size-bounded) and the DRAM path.
// The X280 SPSC markers are 4 words (see SPSC_MARKER_WORDS in kernel_profiler.hpp) but this stays 2
// so the L1 profiler ring keeps its size (holding 128 4-word markers instead of 256 2-word ones).
constexpr static std::uint32_t PROFILER_L1_MARKER_UINT32_SIZE = 2;
constexpr static std::uint32_t PROFILER_L1_PROGRAM_ID_COUNT = 2;
constexpr static std::uint32_t PROFILER_L1_GUARANTEED_MARKER_COUNT = 4;
constexpr static std::uint32_t PROFILER_L1_OPTIONAL_MARKER_COUNT = 250;
constexpr static std::uint32_t PROFILER_L1_VECTOR_SIZE =
    (PROFILER_L1_OPTIONAL_MARKER_COUNT + PROFILER_L1_GUARANTEED_MARKER_COUNT + PROFILER_L1_PROGRAM_ID_COUNT) *
    PROFILER_L1_MARKER_UINT32_SIZE;
constexpr static std::uint32_t PROFILER_L1_BUFFER_SIZE = PROFILER_L1_VECTOR_SIZE * sizeof(uint32_t);

}  // namespace kernel_profiler
