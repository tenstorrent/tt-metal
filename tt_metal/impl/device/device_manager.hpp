// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <tt_stl/span.hpp>
#include <unordered_map>
#include <vector>
#include <hostdevcommon/common_values.hpp>
#include "umd/device/types/cluster_descriptor_types.hpp"
#include "device_impl.hpp"

namespace tt::tt_metal {

namespace experimental {
class DispatchContext;
}  // namespace experimental

class ContextDescriptor;
class IDevice;
class FirmwareInitializer;
enum class InitializerKey;

class DeviceManager {
public:
    ~DeviceManager();
    DeviceManager();

    bool is_initialized() const { return is_initialized_; }

    void initialize(
        const std::vector<ChipId>& device_ids,
        bool init_profiler,
        bool initialize_fabric_and_dispatch_fw,
        std::shared_ptr<ContextDescriptor> descriptor);

    IDevice* get_active_device(ChipId device_id) const;
    std::vector<IDevice*> get_all_active_devices() const;
    bool close_device(ChipId device_id);
    std::vector<ChipId> get_all_active_device_ids() const;
    std::unordered_map<ChipId, std::vector<uint32_t>> get_all_command_queue_event_infos() const;
    bool close_devices(const std::vector<IDevice*>& devices, bool skip_synchronize = false);
    bool is_device_active(ChipId id) const;
    // True if dispatch firmware is active on this device pool
    bool is_dispatch_firmware_active() const;
    // Called by the mesh device
    void initialize_profiler();
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
    bool is_initialized_ = false;

    mutable std::mutex lock_;
    std::vector<std::unique_ptr<Device>> devices_;

    bool skip_remote_devices_{};

    std::shared_ptr<ContextDescriptor> descriptor_;
    std::map<InitializerKey, std::unique_ptr<FirmwareInitializer>> initializers_;
    std::unordered_set<InitializerKey> init_done_;
    // Determine which CPU cores the worker threads need to be placed on for each device
    std::unordered_map<uint32_t, uint32_t> worker_thread_to_cpu_core_map_;
    std::unordered_map<uint32_t, uint32_t> completion_queue_reader_to_cpu_core_map_;
    void init_firmware_on_active_devices();
    void activate_device(ChipId id);
    Device* get_active_device_internal(ChipId device_id) const;

    // Open requested devices, configure fabric, and initialize firmware.
    void open_devices(const std::vector<ChipId>& device_ids);

    void add_devices_to_pool(const std::vector<ChipId>& device_ids);
    Device* get_device(ChipId id) const;
    std::vector<Device*> get_all_active_devices_impl() const;

    // Initialize dispatch firmware (compile + configure device CQs).
    void initialize_dispatch_firmware();

    friend class experimental::DispatchContext;
};

}  // namespace tt::tt_metal
