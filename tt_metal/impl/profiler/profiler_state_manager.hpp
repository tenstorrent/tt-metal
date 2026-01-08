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

namespace llrt {
class RunTimeOptions;
}

namespace tt_metal {

namespace detail {
void ReadDeviceProfilerResultsInternal(
    distributed::MeshDevice* mesh_device,
    IDevice* device,
    const std::vector<CoreCoord>& virtual_cores,
    ProfilerReadState state,
    const std::optional<ProfilerOptionalMetadata>& metadata);
}  // namespace detail

void LaunchIntervalBasedProfilerReadThread(const std::vector<IDevice*>& active_devices);
uint32_t get_profiler_dram_bank_size_per_risc_bytes(llrt::RunTimeOptions& rtoptions);
uint32_t get_profiler_dram_bank_size_per_risc_bytes();
uint32_t get_profiler_dram_bank_size_for_hal_allocation(llrt::RunTimeOptions& rtoptions);

struct ProfilerStateManager {
public:
    ProfilerStateManager();

    ~ProfilerStateManager() = default;

    void cleanup_device_profilers();
    void start_debug_dump_thread(
        std::vector<IDevice*> active_devices, std::unordered_map<ChipId, std::vector<CoreCoord>> virtual_cores_map);
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
    mutable std::recursive_mutex device_profiler_map_mutex;

    std::map<ChipId, std::vector<std::set<experimental::ProgramAnalysisData>>> device_programs_perf_analyses_map;

    std::unordered_map<ChipId, std::vector<std::pair<uint64_t, uint64_t>>> device_host_time_pair;
    std::unordered_map<ChipId, std::unordered_map<ChipId, std::vector<std::pair<uint64_t, uint64_t>>>>
        device_device_time_pair;
    std::unordered_map<ChipId, uint64_t> smallest_host_time;

    bool do_sync_on_close{};

    std::unordered_set<ChipId> sync_set_devices;

    std::mutex log_file_write_mutex;
    std::mutex programs_perf_report_write_mutex;

    std::thread debug_dump_thread;
    std::mutex debug_dump_thread_mutex;
    std::atomic<bool> stop_debug_dump_thread = false;
    std::condition_variable stop_debug_dump_thread_cv;
};

}  // namespace tt_metal

}  // namespace tt
