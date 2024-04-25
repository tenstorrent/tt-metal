// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/device/device_pool.hpp"

#include "tt_metal/detail/tt_metal.hpp"

namespace tt {

const DevicePool& DevicePool::instance(
    std::vector<chip_id_t> device_ids, const uint8_t num_hw_cqs, const std::vector<uint32_t>& l1_bank_remap) {
    static DevicePool device_pool(device_ids, num_hw_cqs, l1_bank_remap);
    device_pool.l1_bank_remap = l1_bank_remap;
    return device_pool;
}

DevicePool::DevicePool(
    std::vector<chip_id_t> device_ids, const uint8_t num_hw_cqs, const std::vector<uint32_t>& l1_bank_remap) {
    ZoneScoped;
    for (const auto& device_id : device_ids) {
        const auto& mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        for (const auto& mmio_controlled_device_id :
             tt::Cluster::instance().get_devices_controlled_by_mmio_device(mmio_device_id)) {
            if (this->active_devices.find(mmio_controlled_device_id) == this->active_devices.end()) {
                auto dev = new Device(mmio_controlled_device_id, num_hw_cqs, l1_bank_remap);
                this->active_devices.insert({mmio_controlled_device_id, dev});

                detail::InitDeviceProfiler(dev);
            }
        }
    }
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);
}

Device* DevicePool::get_active_device(chip_id_t device_id) const {
    TT_ASSERT(
        this->active_devices.find(device_id) != this->active_devices.end(),
        "DevicePool does not contain active device {}",
        device_id);
    auto dev = this->active_devices.at(device_id);
    if (not dev->is_initialized()) {
        dev->initialize(this->l1_bank_remap);
    }
    return dev;
}

std::map<chip_id_t, Device*> DevicePool::get_all_active_devices() const {
    for (const auto& [id, dev] : this->active_devices) {
        if (not dev->is_initialized()) {
            dev->initialize(this->l1_bank_remap);
        }
    }
    return this->active_devices;
}

Device* DevicePool::get_associated_mmio_device(chip_id_t device_id) const {
    const auto& mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    TT_ASSERT(
        this->active_devices.at(mmio_device_id)->is_initialized(),
        "DevicePool trying to access uninitialized mmio device");
    return this->get_active_device(mmio_device_id);
}

bool DevicePool::close_device(chip_id_t device_id) const {
    auto device = this->get_active_device(device_id);
    return device->close();
}

DevicePool::~DevicePool() {
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
    // TODO: should this be done explicitly here?
    for (const auto& [device_id, dev] : this->active_devices) {
        if (dev->is_initialized()) {
            dev->close();
        }
    }
    this->active_devices.clear();
}

}  // namespace tt
