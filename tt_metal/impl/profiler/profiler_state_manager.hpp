// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <unordered_set>

#include "core_coord.hpp"
#include "profiler.hpp"

namespace tt {

namespace tt_metal {

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

    std::mutex file_write_mutex{};
};

}  // namespace tt_metal

}  // namespace tt
