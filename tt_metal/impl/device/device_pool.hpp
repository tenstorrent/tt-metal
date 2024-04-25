// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/third_party/umd/device/tt_cluster_descriptor.h"
namespace tt {
class DevicePool {
   public:
    DevicePool &operator=(const DevicePool &) = delete;
    DevicePool &operator=(DevicePool &&other) noexcept = delete;
    DevicePool(const DevicePool &) = delete;
    DevicePool(DevicePool &&other) noexcept = delete;

    static const DevicePool &instance(
        std::vector<chip_id_t> device_ids,
        const uint8_t num_hw_cqs = 1,
        const std::vector<uint32_t> &l1_bank_remap = {});
    Device *get_active_device(chip_id_t device_id) const;
    Device *get_associated_mmio_device(chip_id_t device_id) const;

    std::map<chip_id_t, Device *> get_all_active_devices() const;

    bool close_device(chip_id_t device_id) const;

   private:
    ~DevicePool();
    DevicePool(std::vector<chip_id_t> device_ids, const uint8_t num_hw_cqs, const std::vector<uint32_t> &l1_bank_remap);
    std::vector<uint32_t> l1_bank_remap;
    std::map<chip_id_t, Device *> active_devices;
};

}  // namespace tt
