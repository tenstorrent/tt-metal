// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#define PROFILER_OPT_DO_DISPATCH_CORES (1 << 1)
#define PROFILER_OPT_DO_TRACE_ONLY (1 << 2)
#define PROFILER_OPT_DO_SUM (1 << 3)

namespace kernel_profiler {

static constexpr int SUM_COUNT = 2;
static constexpr uint32_t DRAM_PROFILER_ADDRESS_STALLED = 0xFFFFFFFF;

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
