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

// Structural zone id + profiler marker-word bit layout. Single source of truth, shared by the device
// (kernel_profiler.hpp), the JIT build (build.cpp), and the host parser (profiler.cpp).
//
// One profiler marker word is 32 bits:
//   [ valid : 1 (bit 31) | timer_id : TIMER_ID_BITS | timestamp_hi : TIMESTAMP_HI_BITS ]
//   timer_id = [ zone id : ZONE_ID_BITS | packet type : PACKET_TYPE_BITS ]
//   zone id  = [ file id : (ZONE_ID_BITS - LOCAL_BITS) | local index : LOCAL_BITS ]
//
// Zone ids are structural, not hashed, so (file, local) pairs are unique by construction. The id is
// 18 bits, widened from 16 by borrowing 2 bits from timestamp_hi (12 -> 10); with LOCAL_BITS=6 that
// is 4096 file ids x 64 zones/TU. timestamp_hi is then 10 bits, so device timestamps wrap after
// ~2^42 cycles (~54 min at 1.35 GHz / ~73 min at 1.0 GHz) -- ample for a profiling capture.
#define KERNEL_PROFILER_ZONE_ID_BITS 18
#define KERNEL_PROFILER_LOCAL_BITS 6
#define KERNEL_PROFILER_PACKET_TYPE_BITS 3
#define KERNEL_PROFILER_TIMER_ID_BITS (KERNEL_PROFILER_ZONE_ID_BITS + KERNEL_PROFILER_PACKET_TYPE_BITS)
#define KERNEL_PROFILER_TIMESTAMP_HI_BITS (31 - KERNEL_PROFILER_TIMER_ID_BITS)
#define KERNEL_PROFILER_ZONE_ID_MASK ((1u << KERNEL_PROFILER_ZONE_ID_BITS) - 1u)
#define KERNEL_PROFILER_TIMER_ID_MASK ((1u << KERNEL_PROFILER_TIMER_ID_BITS) - 1u)
#define KERNEL_PROFILER_TIMESTAMP_HI_MASK ((1u << KERNEL_PROFILER_TIMESTAMP_HI_BITS) - 1u)
#define KERNEL_PROFILER_FILE_ID_COUNT (1u << (KERNEL_PROFILER_ZONE_ID_BITS - KERNEL_PROFILER_LOCAL_BITS))

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

enum ControlBuffer {
    HOST_BUFFER_END_INDEX_BR_ER,
    HOST_BUFFER_END_INDEX_NC,
    HOST_BUFFER_END_INDEX_T0,
    HOST_BUFFER_END_INDEX_T1,
    HOST_BUFFER_END_INDEX_T2,
    DEVICE_BUFFER_END_INDEX_BR_ER,
    DEVICE_BUFFER_END_INDEX_NC,
    DEVICE_BUFFER_END_INDEX_T0,
    DEVICE_BUFFER_END_INDEX_T1,
    DEVICE_BUFFER_END_INDEX_T2,
    FW_RESET_H,
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
    DRAM_PROFILER_ADDRESS_BR_ER_0,
    DRAM_PROFILER_ADDRESS_NC_0,
    DRAM_PROFILER_ADDRESS_T0_0,
    DRAM_PROFILER_ADDRESS_T1_0,
    DRAM_PROFILER_ADDRESS_T2_0,
};

enum PacketTypes { ZONE_START, ZONE_END, ZONE_TOTAL, TS_DATA, TS_EVENT, TS_DATA_16B };

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
constexpr static std::uint32_t PROFILER_L1_CONTROL_VECTOR_SIZE = 32;
constexpr static std::uint32_t PROFILER_L1_CONTROL_BUFFER_SIZE = PROFILER_L1_CONTROL_VECTOR_SIZE * sizeof(uint32_t);
constexpr static std::uint32_t PROFILER_L1_MARKER_UINT32_SIZE = 2;
constexpr static std::uint32_t PROFILER_L1_PROGRAM_ID_COUNT = 2;
constexpr static std::uint32_t PROFILER_L1_GUARANTEED_MARKER_COUNT = 4;
constexpr static std::uint32_t PROFILER_L1_OPTIONAL_MARKER_COUNT = 250;
constexpr static std::uint32_t PROFILER_L1_VECTOR_SIZE =
    (PROFILER_L1_OPTIONAL_MARKER_COUNT + PROFILER_L1_GUARANTEED_MARKER_COUNT + PROFILER_L1_PROGRAM_ID_COUNT) *
    PROFILER_L1_MARKER_UINT32_SIZE;
constexpr static std::uint32_t PROFILER_L1_BUFFER_SIZE = PROFILER_L1_VECTOR_SIZE * sizeof(uint32_t);

}  // namespace kernel_profiler
