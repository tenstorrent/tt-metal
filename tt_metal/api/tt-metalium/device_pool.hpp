// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/span.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <umd/device/types/cluster_descriptor_types.h>

namespace tt {

class DevicePool {
    friend void tt_metal::detail::CloseDevices(const std::map<chip_id_t, tt_metal::IDevice*>& devices);

public:
    DevicePool& operator=(const DevicePool&) = delete;
    DevicePool& operator=(DevicePool&& other) noexcept = delete;
    DevicePool(const DevicePool&) = delete;
    DevicePool(DevicePool&& other) noexcept = delete;

    static bool is_initialized() { return _inst != nullptr; }

    static DevicePool& instance() noexcept {
        TT_ASSERT(DevicePool::is_initialized(), "Trying to get DevicePool without initializing it");
        return *_inst;
    }

    static void initialize(
        const std::vector<chip_id_t>& device_ids,
        const uint8_t num_hw_cqs,
        size_t l1_small_size,
        size_t trace_region_size,
        const tt_metal::DispatchCoreConfig& dispatch_core_config,
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE,
        bool init_profiler = true,
        bool use_max_eth_core_count_on_all_devices = false,
        bool initialize_fabric_and_dispatch_fw = true);

    tt_metal::IDevice* get_active_device(chip_id_t device_id) const;
    std::vector<tt_metal::IDevice*> get_all_active_devices() const;
    bool close_device(chip_id_t device_id);
    bool close_devices(const std::vector<tt_metal::IDevice*>& devices, bool skip_synchronize = false);
    bool is_device_active(chip_id_t id) const;
    // True if dispatch firmware is active on this device pool
    bool is_dispatch_firmware_active() const;
    void init_profiler() const;
    void initialize_fabric_and_dispatch_fw() const;
    // API needed due to Issue #19729
    std::size_t get_max_num_eth_cores_across_all_devices() const;

private:
    ~DevicePool();
    DevicePool();
    uint8_t num_hw_cqs;
    size_t l1_small_size;
    size_t trace_region_size;
    size_t worker_l1_size;
    std::vector<uint32_t> l1_bank_remap;
    bool using_fast_dispatch;
    bool init_profiler_ = true;
    bool initialize_fabric_and_dispatch_fw_ = false;
    // This variable tracks the state of dispatch firmware on device.
    // It is set to true when dispatch firmware is launched, and reset
    // after the terimnate command is sent.
    bool dispatch_firmware_active_ = false;

    std::mutex lock;
    std::vector<std::unique_ptr<tt_metal::IDevice>> devices;

    bool skip_remote_devices;
    // Issue #19729: use_max_eth_core_count_on_all_devices_ is a workaround
    // to allow TT-Mesh Workload dispatch to target active ethernet cores.
    bool use_max_eth_core_count_on_all_devices_;
    std::unordered_set<uint32_t> firmware_built_keys;

    // Determine which CPU cores the worker threads need to be placed on for each device
    std::unordered_map<uint32_t, uint32_t> worker_thread_to_cpu_core_map;
    std::unordered_map<uint32_t, uint32_t> completion_queue_reader_to_cpu_core_map;
    void init_firmware_on_active_devices() const;
    void activate_device(chip_id_t id);

    // Initialize state on the host for this device
    void initialize_host(tt_metal::IDevice* dev) const;

    // Initialize state for activated devices
    void init_fabric(const std::vector<tt_metal::IDevice*>& active_devices) const;
    void initialize_active_devices() const;
    void add_devices_to_pool(const std::vector<chip_id_t>& device_ids);
    void wait_for_fabric_router_sync() const;
    tt_metal::IDevice* get_device(chip_id_t id) const;
    void teardown_fd(const std::unordered_set<chip_id_t>& devices_to_close);

    static DevicePool* _inst;
};

}  // namespace tt
