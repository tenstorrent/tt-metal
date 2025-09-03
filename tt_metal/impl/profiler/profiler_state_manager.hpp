// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_set>

#include "core_coord.hpp"
#include "profiler.hpp"

namespace tt {

namespace tt_metal {

struct ProfilerStateManager {
public:
    ProfilerStateManager() {
        this->device_profiler_map = {};
        this->device_host_time_pair = {};
        this->device_device_time_pair = {};
        this->smallest_host_time = {};
        this->do_sync_on_close = true;
        this->sync_set_devices = {};
    };
    ~ProfilerStateManager() {
        this->device_profiler_map.clear();
        this->device_host_time_pair.clear();
        this->device_device_time_pair.clear();
        this->smallest_host_time.clear();
        this->do_sync_on_close = false;
        this->sync_set_devices.clear();
    };

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

    bool do_sync_on_close = false;

    std::unordered_set<chip_id_t> sync_set_devices{};
};

}  // namespace tt_metal

}  // namespace tt
