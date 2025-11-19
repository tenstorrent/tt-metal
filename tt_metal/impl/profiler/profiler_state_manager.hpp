// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include <map>
#include <vector>
#include <set>
#include <unordered_set>
#include <mutex>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/profiler.hpp>
#include "profiler.hpp"

namespace tt {

namespace tt_metal {

// static function should store default op support count and return the actual op support count, outside of the class

// env var should control PROFILER_OP_SUPPORT_COUNT; cli option in tracy module shouldn't be a number, it should be
// {small, medium, large} default to 1000 if env var is not set
constexpr inline static uint32_t DEFAULT_PROFILER_PROGRAM_SUPPORT_COUNT = 1000;

uint32_t get_profiler_dram_bank_size_per_risc_bytes(uint32_t profiler_program_support_count);

// these 2 vars should be passed to kernel_profiler with -D flag at jit build time
// constexpr static std::uint32_t PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC =
//     kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE *
//     (kernel_profiler::PROFILER_L1_PROGRAM_ID_COUNT + kernel_profiler::PROFILER_L1_GUARANTEED_MARKER_COUNT +
//      kernel_profiler::PROFILER_L1_OP_MIN_OPTIONAL_MARKER_COUNT) *
//     PROFILER_OP_SUPPORT_COUNT;
// constexpr static std::uint32_t PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC =
//     PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC * sizeof(uint32_t);

struct ProfilerStateManager {
public:
    ProfilerStateManager();

    ~ProfilerStateManager() = default;

    void cleanup_device_profilers();

    uint32_t calculate_optimal_num_threads_for_device_profiler_thread_pool() const;

    void mark_trace_begin(ChipId device_id, uint32_t trace_id);
    void mark_trace_end(ChipId device_id, uint32_t trace_id);
    void mark_trace_replay(ChipId device_id, uint32_t trace_id);
    void add_runtime_id_to_trace(ChipId device_id, uint32_t trace_id, uint32_t runtime_id);

    ProfilerStateManager& operator=(const ProfilerStateManager&) = delete;
    ProfilerStateManager& operator=(ProfilerStateManager&&) = delete;
    ProfilerStateManager(const ProfilerStateManager&) = delete;
    ProfilerStateManager(ProfilerStateManager&&) = delete;

    static constexpr CoreCoord SYNC_CORE = {0, 0};

    std::unordered_map<ChipId, DeviceProfiler> device_profiler_map;

    std::map<ChipId, std::vector<std::set<experimental::ProgramAnalysisData>>> device_programs_perf_analyses_map;

    std::unordered_map<ChipId, std::vector<std::pair<uint64_t, uint64_t>>> device_host_time_pair;
    std::unordered_map<ChipId, std::unordered_map<ChipId, std::vector<std::pair<uint64_t, uint64_t>>>>
        device_device_time_pair;
    std::unordered_map<ChipId, uint64_t> smallest_host_time;

    bool do_sync_on_close{};

    std::unordered_set<ChipId> sync_set_devices;

    std::mutex log_file_write_mutex;
    std::mutex programs_perf_report_write_mutex;
};

}  // namespace tt_metal

}  // namespace tt
