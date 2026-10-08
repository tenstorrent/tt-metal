// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include <map>
#include <vector>
#include <set>
#include <unordered_set>
#include <atomic>
#include <mutex>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/profiler.hpp>
#include "profiler.hpp"

namespace tt {

namespace llrt {
class RunTimeOptions;
}

namespace tt_metal {

class MetalContext;
class MetalEnvImpl;

namespace detail {
void ReadDeviceProfilerResultsInternal(
    MetalContext& ctx,
    distributed::MeshDevice* mesh_device,
    IDevice* device,
    const std::vector<CoreCoord>& virtual_cores,
    ProfilerReadState state,
    const std::optional<ProfilerOptionalMetadata>& metadata,
    bool include_l1 = false);
}  // namespace detail

void LaunchIntervalBasedProfilerReadThread(MetalContext& ctx, const std::vector<IDevice*>& active_devices);
uint32_t get_profiler_dram_bank_size_per_risc_bytes(llrt::RunTimeOptions& rtoptions);
uint32_t get_profiler_dram_bank_size_for_hal_allocation(llrt::RunTimeOptions& rtoptions);

struct ProfilerStateManager {
public:
    // A state manager whose env can be profiled (silicon, profiler enabled) attaches itself to the ProfilerRegistry
    // for its lifetime.
    explicit ProfilerStateManager(MetalContext& ctx);

    ~ProfilerStateManager();

    void cleanup_device_profilers();
    void start_debug_dump_thread(
        std::vector<IDevice*> active_devices, std::unordered_map<ChipId, std::vector<CoreCoord>> virtual_cores_map);
    void signal_debug_dump_read();
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

    MetalContext& ctx_;
    MetalEnvImpl& env_;
    std::unordered_map<ChipId, DeviceProfiler> device_profiler_map;
    mutable std::recursive_mutex device_profiler_map_mutex;

    std::map<ChipId, std::vector<std::set<experimental::ProgramAnalysisData>>> device_programs_perf_analyses_map;

    std::unordered_map<ChipId, std::vector<std::pair<uint64_t, uint64_t>>> device_host_time_pair;
    std::unordered_map<ChipId, std::unordered_map<ChipId, std::vector<std::pair<uint64_t, uint64_t>>>>
        device_device_time_pair;
    std::unordered_map<ChipId, uint64_t> smallest_host_time;

    bool do_sync_on_close{};

    std::unordered_set<ChipId> sync_set_devices;

    std::thread debug_dump_thread;
    std::mutex debug_dump_thread_mutex;
    std::atomic<bool> stop_debug_dump_thread = false;
    std::atomic<bool> force_read_debug_dump = false;
    std::atomic<bool> force_read_complete = false;
    std::condition_variable stop_debug_dump_thread_cv;
    std::condition_variable force_read_complete_cv;

    bool attached_to_registry = false;
};

// The process-wide part of the device profiler: the per-context ProfilerStateManagers that are being profiled, and
// the state shared by all of them (the profiler output files and the first-init log wipe). The handle-less public
// profiler APIs go through it. It never owns a ProfilerStateManager; each one attaches in its constructor and detaches
// in its destructor. Only one profiled (silicon) context is supported at a time.
class ProfilerRegistry {
public:
    static ProfilerRegistry& instance();

    void attach(ProfilerStateManager& profiler_state_manager);
    void detach(ProfilerStateManager& profiler_state_manager);

    // Calls f(ProfilerStateManager&) on every attached state manager, holding the registry lock.
    template <typename F>
    void for_each_attached(F&& f) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (ProfilerStateManager* profiler_state_manager : attached_) {
            f(*profiler_state_manager);
        }
    }

    // Calls f(ProfilerStateManager*) with the attached state manager, or nullptr if none, holding the registry lock.
    template <typename F>
    auto with_sole_attached(F&& f) {
        std::lock_guard<std::mutex> lock(mutex_);
        return f(attached_.empty() ? nullptr : attached_.front());
    }

    // Guard the profiler output files, which all device profilers share by default.
    std::mutex log_file_write_mutex;
    std::mutex programs_perf_report_write_mutex;

    // Set until the first device profiler of the process is created, which wipes the previous run's logs.
    std::atomic<bool> first_device_profiler_init{true};

private:
    std::mutex mutex_;
    std::vector<ProfilerStateManager*> attached_;
};

}  // namespace tt_metal

}  // namespace tt
