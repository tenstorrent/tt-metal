// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#define PROFILER_OPT_DO_DISPATCH_CORES 2

namespace kernel_profiler {

constexpr static uint32_t PADDING_MARKER = ((1 << 16) - 1);
constexpr static uint32_t NOC_ALIGNMENT_FACTOR = 4;

static constexpr int SUM_COUNT = 2;

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
    DRAM_PROFILER_ADDRESS,
    RUN_COUNTER,
    NOC_X,
    NOC_Y,
    FLAT_ID,
    CORE_COUNT_PER_DRAM,
    DROPPED_ZONES,
    PROFILER_DONE,
};

enum PacketTypes { ZONE_START, ZONE_END, ZONE_TOTAL, TS_DATA, TS_EVENT };

// TODO: use data types in profile_msg_t rather than addresses/sizes
constexpr static std::uint32_t PROFILER_L1_CONTROL_VECTOR_SIZE = 32;
constexpr static std::uint32_t PROFILER_L1_CONTROL_BUFFER_SIZE = PROFILER_L1_CONTROL_VECTOR_SIZE * sizeof(uint32_t);
constexpr static std::uint32_t PROFILER_L1_MARKER_UINT32_SIZE = 2;
constexpr static std::uint32_t PROFILER_L1_PROGRAM_ID_COUNT = 2;
constexpr static std::uint32_t PROFILER_L1_GUARANTEED_MARKER_COUNT = 4;
constexpr static std::uint32_t PROFILER_L1_OPTIONAL_MARKER_COUNT = 250;
constexpr static std::uint32_t PROFILER_L1_OP_MIN_OPTIONAL_MARKER_COUNT = 2;
constexpr static std::uint32_t PROFILER_L1_VECTOR_SIZE =
    (PROFILER_L1_OPTIONAL_MARKER_COUNT + PROFILER_L1_GUARANTEED_MARKER_COUNT + PROFILER_L1_PROGRAM_ID_COUNT) *
    PROFILER_L1_MARKER_UINT32_SIZE;
constexpr static std::uint32_t PROFILER_L1_BUFFER_SIZE = PROFILER_L1_VECTOR_SIZE * sizeof(uint32_t);

}  // namespace kernel_profiler

constexpr static std::uint32_t PROFILER_OP_SUPPORT_COUNT = 1000;
constexpr static std::uint32_t PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC =
    kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE *
    (kernel_profiler::PROFILER_L1_PROGRAM_ID_COUNT + kernel_profiler::PROFILER_L1_GUARANTEED_MARKER_COUNT +
     kernel_profiler::PROFILER_L1_OP_MIN_OPTIONAL_MARKER_COUNT) *
    PROFILER_OP_SUPPORT_COUNT;
constexpr static std::uint32_t PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC =
    PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC * sizeof(uint32_t);

static_assert(PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC > kernel_profiler::PROFILER_L1_BUFFER_SIZE);
