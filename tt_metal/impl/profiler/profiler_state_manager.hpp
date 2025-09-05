// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_set>

#include "thread_pool.hpp"
#include "core_coord.hpp"
#include "profiler.hpp"

namespace tt {

namespace tt_metal {

struct ProfilerStateManager {
public:
    ProfilerStateManager() : do_sync_on_close(true), thread_pool(create_device_bound_thread_pool(64)) {};

    ~ProfilerStateManager() = default;

    void cleanup_device_profilers() {
        std::vector<std::thread> threads;
        for (auto& [device_id, profiler] : this->device_profiler_map) {
            threads.emplace_back([&profiler]() { profiler.cleanup(); });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        this->device_profiler_map.clear();
    }

    ProfilerStateManager& operator=(const ProfilerStateManager&) = delete;
    ProfilerStateManager& operator=(ProfilerStateManager&&) = delete;
    ProfilerStateManager(const ProfilerStateManager&) = delete;
    ProfilerStateManager(ProfilerStateManager&&) = delete;

    static constexpr CoreCoord SYNC_CORE = {0, 0};

    std::unordered_map<chip_id_t, DeviceProfiler> device_profiler_map{};

    std::unordered_map<chip_id_t, std::vector<std::pair<uint64_t, uint64_t>>> device_host_time_pair{};
    std::unordered_map<chip_id_t, std::unordered_map<chip_id_t, std::vector<std::pair<uint64_t, uint64_t>>>>
        device_device_time_pair{};
    std::unordered_map<chip_id_t, uint64_t> smallest_host_time{};

    bool do_sync_on_close{};

    std::unordered_set<chip_id_t> sync_set_devices{};

    std::shared_ptr<ThreadPool> thread_pool{};

    std::mutex mid_run_dump_mutex{};
};

}  // namespace tt_metal

}  // namespace tt
