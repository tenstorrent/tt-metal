// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "impl/debug/dprint_server.hpp"
#include "impl/debug/watcher_server.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/third_party/umd/device/tt_cluster_descriptor.h"
namespace tt {

using Device = tt_metal::Device;
class DevicePool {
   public:
    DevicePool &operator=(const DevicePool &) = delete;
    DevicePool &operator=(DevicePool &&other) noexcept = delete;
    DevicePool(const DevicePool &) = delete;
    DevicePool(DevicePool &&other) noexcept = delete;

    static DevicePool &instance() noexcept {
        TT_ASSERT(_inst != nullptr, "Trying to get DevicePool without initializing it");
        return *_inst;
    }

    static void initialize(
        std::vector<chip_id_t> device_ids,
        const uint8_t num_hw_cqs,
        size_t l1_small_size,
        size_t trace_region_size,
        DispatchCoreType dispatch_core_type,
        const std::vector<uint32_t> &l1_bank_remap = {}) noexcept;

    Device *get_active_device(chip_id_t device_id) const;
    std::vector<Device *> get_all_active_devices() const;
    bool close_device(chip_id_t device_id) const;
    bool is_device_active(chip_id_t id) const;

   private:
    ~DevicePool();
    DevicePool(
        std::vector<chip_id_t> device_ids,
        const uint8_t num_hw_cqs,
        size_t l1_small_size,
        size_t trace_region_size,
        const std::vector<uint32_t> &l1_bank_remap);
    uint8_t num_hw_cqs;
    size_t l1_small_size;
    size_t trace_region_size;
    std::vector<uint32_t> l1_bank_remap;
    std::mutex lock;
    std::vector<std::unique_ptr<Device>> devices;
    bool skip_remote_devices;
    std::unordered_set<uint32_t> firmware_built_keys;

    // Determine which CPU cores the worker threads need to be placed on for each device
    std::unordered_map<uint32_t, uint32_t> device_to_core_map;

    void init_firmware_on_active_devices() const;
    void activate_device(chip_id_t id);
    void initialize_device(Device *dev) const;
    void deactivate_device(chip_id_t id);
    void add_devices_to_pool(std::vector<chip_id_t> device_ids);
    static DevicePool *_inst;
};

}  // namespace tt
