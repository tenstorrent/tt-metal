// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/mesh_handle.hpp"
#include "tt_metal/distributed/mesh_config.hpp"
#include "tt_metal/distributed/system_mesh.hpp"
#include "tt_metal/detail/tt_metal.hpp"

namespace tt::tt_metal::distributed {

MeshHandle::MeshHandle(
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const DispatchCoreConfig& dispatch_core_config,
    const MeshDeviceConfig& config) {
    auto& system_mesh = SystemMesh::instance();
    auto physical_device_ids = system_mesh.request_available_devices(config);

    opened_devices_ = tt::tt_metal::detail::CreateDevices(
        physical_device_ids, num_command_queues, l1_small_size, trace_region_size, dispatch_core_config);

    for (auto physical_device_id : physical_device_ids) {
        devices_.push_back(opened_devices_.at(physical_device_id));
    }
}

MeshHandle::~MeshHandle() { close(); }
void MeshHandle::close() {
    if (opened_devices_.size() > 0) {
        tt::tt_metal::detail::CloseDevices(opened_devices_);
        opened_devices_.clear();
        devices_.clear();
    }
}

const std::vector<IDevice*>& MeshHandle::get_devices() const { return devices_; }

}  // namespace tt::tt_metal::distributed
