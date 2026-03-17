// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/umd_device_access.hpp"

#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal::distributed {

UmdDeviceAccess::UmdDeviceAccess(uint32_t device_id, uint32_t virtual_core_x, uint32_t virtual_core_y) :
    device_id_(device_id), virtual_core_x_(virtual_core_x), virtual_core_y_(virtual_core_y) {
    umd_cluster_ = std::make_unique<tt::umd::Cluster>(tt::umd::ClusterOptions{
        .target_devices = {static_cast<int>(device_id)},
    });
}

UmdDeviceAccess::~UmdDeviceAccess() = default;
UmdDeviceAccess::UmdDeviceAccess(UmdDeviceAccess&&) noexcept = default;
UmdDeviceAccess& UmdDeviceAccess::operator=(UmdDeviceAccess&&) noexcept = default;

std::function<void(void*, uint32_t, uint64_t)> UmdDeviceAccess::get_pcie_writer() const {
    auto* cluster = umd_cluster_.get();
    auto chip_id = static_cast<int>(device_id_);
    auto vx = virtual_core_x_;
    auto vy = virtual_core_y_;
    return [cluster, chip_id, vx, vy](void* data, uint32_t num_bytes, uint64_t device_addr) {
        tt::umd::CoreCoord core(vx, vy, CoreType::TENSIX, CoordSystem::TRANSLATED);
        cluster->write_to_device(data, num_bytes, chip_id, core, device_addr);
    };
}

}  // namespace tt::tt_metal::distributed
