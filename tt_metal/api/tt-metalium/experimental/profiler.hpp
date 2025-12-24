// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal::experimental {

static constexpr uint64_t INVALID_NUM_PROGRAM_EXECUTION_UID = (1LL << 63);

struct ProgramExecutionUID {
    uint64_t runtime_id = INVALID_NUM_PROGRAM_EXECUTION_UID;
    uint64_t trace_id = INVALID_NUM_PROGRAM_EXECUTION_UID;
    uint64_t trace_id_counter = INVALID_NUM_PROGRAM_EXECUTION_UID;

    bool operator==(const ProgramExecutionUID& other) const;

    bool operator<(const ProgramExecutionUID& other) const;
};

struct ProgramSingleAnalysisResult {
    uint64_t start_timestamp = UINT64_MAX;
    uint64_t end_timestamp = 0;
    uint64_t duration = 0;

    bool operator==(const ProgramSingleAnalysisResult& other) const;

    bool operator!=(const ProgramSingleAnalysisResult& other) const;

    bool operator<(const ProgramSingleAnalysisResult& other) const;
};

struct ProgramAnalysisData {
    ProgramExecutionUID program_execution_uid;
    std::unordered_map<std::string, ProgramSingleAnalysisResult> program_analyses_results;
    uint32_t core_count = 0;
    uint32_t num_available_cores = 0;

    bool operator==(const ProgramAnalysisData& other) const;

    bool operator<(const ProgramAnalysisData& other) const;
};

// clang-format off
/**
 * Get performance results for all programs that were read in the most recent call to `ReadMeshDeviceProfilerResults()`.
 *
 * This function only works in PROFILER builds. Please refer to the "Device Program Profiler" section for more information.
 *
 * Return value: std::map<ChipId, std::set<ProgramAnalysisData>>
 */
// clang-format on
std::map<ChipId, std::set<ProgramAnalysisData>> GetLatestProgramsPerfData();

// clang-format off
/**
 * Get performance results for all programs that have been read so far across all calls to `ReadMeshDeviceProfilerResults()`.
 *
 * This function only works in PROFILER builds. Please refer to the "Device Program Profiler" section for more information.
 *
 * Return value: std::map<ChipId, std::set<ProgramAnalysisData>>
 */
// clang-format on
std::map<ChipId, std::set<ProgramAnalysisData>> GetAllProgramsPerfData();

}  // namespace tt::tt_metal::experimental

namespace std {
template <>
struct hash<tt::tt_metal::experimental::ProgramExecutionUID> {
    std::size_t operator()(const tt::tt_metal::experimental::ProgramExecutionUID& id) const;
};
}  // namespace std
