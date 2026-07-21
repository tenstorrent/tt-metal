// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/pcie_core_writer.hpp"

#include <umd/device/cluster.hpp>
#include <umd/device/pcie/pci_device.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::distributed {

std::mutex PCIeCoreWriter::cluster_mutex_;
std::unique_ptr<tt::umd::Cluster> PCIeCoreWriter::shared_cluster_;
// The descriptor carries the *physical* PCIe device number (/dev/tenstorrent/N), which is
// stable across processes regardless of TT_VISIBLE_DEVICES. UMD's Cluster, however, addresses
// chips by *logical* id (the index into this process's device enumeration). Translate the
// physical device number into this process's local logical chip id.
static uint32_t physical_to_local_logical_id(uint32_t physical_device_num) {
    const auto enumerated = tt::umd::PCIDevice::enumerate_devices();
    for (size_t i = 0; i < enumerated.size(); ++i) {
        if (enumerated[i] == static_cast<int>(physical_device_num)) {
            return static_cast<uint32_t>(i);
        }
    }
    TT_THROW(
        "H2D socket connector cannot find physical PCIe device /dev/tenstorrent/{} in this process. "
        "Ensure the device is visible to the connector (check TT_VISIBLE_DEVICES).",
        physical_device_num);
}


tt::umd::Cluster* PCIeCoreWriter::get_or_create_cluster() {
    std::lock_guard lock(cluster_mutex_);
    if (!shared_cluster_) {
        // No target_devices: it's a no-op for SILICON, so the ctor discovers all visible PCIe chips.
        shared_cluster_ = std::make_unique<tt::umd::Cluster>(tt::umd::ClusterOptions{});
    }
    return shared_cluster_.get();
}

PCIeCoreWriter::PCIeCoreWriter(uint32_t device_id, uint32_t virtual_core_x, uint32_t virtual_core_y) :
    device_id_(physical_to_local_logical_id(device_id)), virtual_core_x_(virtual_core_x), virtual_core_y_(virtual_core_y) {
    // Init eagerly so topology-discovery cost is paid here rather than lazily during the first send.
    get_or_create_cluster();
}

std::function<void(void*, uint32_t, uint64_t)> PCIeCoreWriter::get_pcie_writer() const {
    auto* cluster = get_or_create_cluster();
    auto chip_id = static_cast<int>(device_id_);
    auto vx = virtual_core_x_;
    auto vy = virtual_core_y_;
    tt::umd::CoreCoord core(vx, vy, CoreType::TENSIX, CoordSystem::TRANSLATED);
    return [cluster, chip_id, core](void* data, uint32_t num_bytes, uint64_t device_addr) {
        cluster->write_to_device(data, num_bytes, chip_id, core, device_addr);
    };
}

}  // namespace tt::tt_metal::distributed
