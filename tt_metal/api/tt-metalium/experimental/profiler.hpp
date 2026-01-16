// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
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

struct DurationHistogram {
    uint64_t min_ns = 0;
    uint64_t max_ns = 0;
    uint32_t num_buckets = 0;

    // Uniform bucket edges in nanoseconds. Size is num_buckets + 1.
    std::vector<uint64_t> bucket_edges_ns;

    // Bucket counts. Size is num_buckets.
    std::vector<uint64_t> bucket_counts;

    uint64_t underflow = 0;  // samples < histogram start
    uint64_t overflow = 0;   // samples >= histogram end
};

struct KernelDurationSummary {
    // Summary for "DEVICE KERNEL DURATION [ns]" durations.
    uint64_t count = 0;
    uint64_t min_ns = 0;
    uint64_t max_ns = 0;
    double avg_ns = 0.0;
    DurationHistogram histogram;
};

// clang-format off
/**
 * Get a summary (min/max/avg + histogram) of DEVICE KERNEL duration for the latest captured program set.
 *
 * This function only works in PROFILER builds. Please refer to the "Device Program Profiler" section for more information.
 *
 * Return value: std::map<ChipId, KernelDurationSummary>
 */
// clang-format on
std::map<ChipId, KernelDurationSummary> GetLatestKernelDurationSummary(
    uint64_t histogram_min_ns = 0, uint64_t histogram_max_ns = 0, uint32_t histogram_buckets = 10);

// clang-format off
/**
 * Get a summary (min/max/avg + histogram) of DEVICE KERNEL duration across all captured program sets so far.
 *
 * This function only works in PROFILER builds. Please refer to the "Device Program Profiler" section for more information.
 *
 * Return value: std::map<ChipId, KernelDurationSummary>
 */
// clang-format on
std::map<ChipId, KernelDurationSummary> GetAllKernelDurationSummary(
    uint64_t histogram_min_ns = 0, uint64_t histogram_max_ns = 0, uint32_t histogram_buckets = 10);

}  // namespace tt::tt_metal::experimental

namespace std {
template <>
struct hash<tt::tt_metal::experimental::ProgramExecutionUID> {
    std::size_t operator()(const tt::tt_metal::experimental::ProgramExecutionUID& id) const;
};
}  // namespace std
