// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <unordered_set>

#include "core_coord.hpp"
#include "hostdevcommon/profiler_common.h"
#include "profiler.hpp"
#include "rtoptions.hpp"
#include "tt_stl/assert.hpp"
#include <tt-metalium/experimental/profiler.hpp>

namespace tt {

namespace tt_metal {

// static function should store default op support count and return the actual op support count, outside of the class

// env var should control PROFILER_OP_SUPPORT_COUNT; cli option in tracy module shouldn't be a number, it should be
// {small, medium, large} default to 1000 if env var is not set
constexpr static uint32_t DEFAULT_PROFILER_PROGRAM_SUPPORT_COUNT = 1000;

uint32_t get_profiler_dram_bank_size_per_risc_bytes(const llrt::RunTimeOptions& rtoptions) {
    const uint32_t dram_bank_size_per_risc_bytes =
        kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE *
        (kernel_profiler::PROFILER_L1_PROGRAM_ID_COUNT + kernel_profiler::PROFILER_L1_GUARANTEED_MARKER_COUNT +
         kernel_profiler::PROFILER_L1_OP_MIN_OPTIONAL_MARKER_COUNT) *
        rtoptions.get_profiler_program_support_count() * sizeof(uint32_t);
    TT_ASSERT(dram_bank_size_per_risc_bytes > kernel_profiler::PROFILER_L1_BUFFER_SIZE);
    return dram_bank_size_per_risc_bytes;
}

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
    ProfilerStateManager() : do_sync_on_close(true) {};

    ~ProfilerStateManager() = default;

    void cleanup_device_profilers() {
        std::vector<std::thread> threads(this->device_profiler_map.size());

        uint32_t i = 0;
        for (auto it = this->device_profiler_map.begin(); it != this->device_profiler_map.end(); ++it) {
            threads[i] = std::thread([it]() {
                DeviceProfiler& profiler = it->second;
                profiler.dumpDeviceResults();
                profiler.destroyTracyContexts();
            });
            i++;
        }

        for (auto& thread : threads) {
            thread.join();
        }

        this->device_profiler_map.clear();
    }

    uint32_t calculate_optimal_num_threads_for_device_profiler_thread_pool() const {
        const uint32_t num_threads_available = std::thread::hardware_concurrency();

        if (num_threads_available == 0 || this->device_profiler_map.size() > num_threads_available) {
            // If hardware_concurrency() is unable to determine the number of threads supported by the CPU, or the
            // number of device profilers is greater than the max number of threads, return 2
            return 2;
        } else {
            // Otherwise, return min(8, number of threads available / number of device profilers)
            // Empirically, 8 threads per device profiler seems to result in optimal performance
            return std::min(8U, static_cast<uint32_t>(num_threads_available / this->device_profiler_map.size()));
        }
    }

    void mark_trace_begin(ChipId device_id, uint32_t trace_id) {
        TT_ASSERT(this->device_profiler_map.find(device_id) != this->device_profiler_map.end());
        DeviceProfiler& device_profiler = this->device_profiler_map.at(device_id);
        device_profiler.markTraceBegin(trace_id);
    }

    void mark_trace_end(ChipId device_id, uint32_t trace_id) {
        TT_ASSERT(this->device_profiler_map.find(device_id) != this->device_profiler_map.end());
        DeviceProfiler& device_profiler = this->device_profiler_map.at(device_id);
        device_profiler.markTraceEnd(trace_id);
    }

    void mark_trace_replay(ChipId device_id, uint32_t trace_id) {
        TT_ASSERT(this->device_profiler_map.find(device_id) != this->device_profiler_map.end());
        DeviceProfiler& device_profiler = this->device_profiler_map.at(device_id);
        device_profiler.markTraceReplay(trace_id);
    }

    void add_runtime_id_to_trace(ChipId device_id, uint32_t trace_id, uint32_t runtime_id) {
        TT_ASSERT(this->device_profiler_map.find(device_id) != this->device_profiler_map.end());
        DeviceProfiler& device_profiler = this->device_profiler_map.at(device_id);
        device_profiler.addRuntimeIdToTrace(trace_id, runtime_id);
    }

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
