// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/span.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <hostdevcommon/common_values.hpp>
#include "umd/device/types/cluster_descriptor_types.hpp"

namespace tt::tt_metal {
class IDevice;
class DeviceManager {
public:
    ~DeviceManager();
    DeviceManager();

    bool is_initialized() { return is_initialized_; }

    void initialize(
        const std::vector<ChipId>& device_ids,
        uint8_t num_hw_cqs,
        size_t l1_small_size,
        size_t trace_region_size,
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE,
        bool init_profiler = true,
        bool initialize_fabric_and_dispatch_fw = true);

    IDevice* get_active_device(ChipId device_id) const;
    std::vector<IDevice*> get_all_active_devices() const;
    bool close_device(ChipId device_id);
    std::vector<ChipId> get_all_active_device_ids() const;
    std::unordered_map<ChipId, std::vector<uint32_t>> get_all_command_queue_event_infos() const;
    bool close_devices(const std::vector<IDevice*>& devices, bool skip_synchronize = false);
    bool is_device_active(ChipId id) const;
    // True if dispatch firmware is active on this device pool
    bool is_dispatch_firmware_active() const;
    void init_profiler() const;
    void initialize_fabric_and_dispatch_fw();
    // API needed due to Issue #19729
    std::size_t get_max_num_eth_cores_across_all_devices() const;

private:
    uint8_t num_hw_cqs_{};
    size_t l1_small_size_{};
    size_t trace_region_size_{};
    size_t worker_l1_size_{};
    std::vector<uint32_t> l1_bank_remap_;
    bool using_fast_dispatch_ = false;
    bool init_profiler_ = true;
    bool initialize_fabric_and_dispatch_fw_ = false;
    // This variable tracks the state of dispatch firmware on device.
    // It is set to true when dispatch firmware is launched, and reset
    // after the terminate command is sent.
    bool dispatch_firmware_active_ = false;
    bool is_initialized_ = false;

    mutable std::mutex lock_;
    std::vector<std::unique_ptr<tt_metal::IDevice>> devices_;

    bool skip_remote_devices_{};

    // Determine which CPU cores the worker threads need to be placed on for each device
    std::unordered_map<uint32_t, uint32_t> worker_thread_to_cpu_core_map_;
    std::unordered_map<uint32_t, uint32_t> completion_queue_reader_to_cpu_core_map_;
    void init_firmware_on_active_devices();
    void activate_device(ChipId id);

    // Initialize DeviceManager
    void initialize_devices(const std::vector<ChipId>& device_ids);

    // Initialize state on the host for this device
    void initialize_host(IDevice* dev) const;

    // Initialize state for activated devices
    static void init_fabric(const std::vector<IDevice*>& active_devices);
    void initialize_active_devices();
    void add_devices_to_pool(const std::vector<ChipId>& device_ids);
    void wait_for_fabric_router_sync(uint32_t timeout_ms = 5000) const;
    IDevice* get_device(ChipId id) const;
    void teardown_fd(const std::unordered_set<ChipId>& devices_to_close);

    // Retrieves the fabric router sync timeout value from configuration or returns a default
    static uint32_t get_fabric_router_sync_timeout_ms();
};

}  // namespace tt::tt_metal
