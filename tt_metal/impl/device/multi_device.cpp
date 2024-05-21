// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "tt_metal/impl/device/multi_device.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"


namespace ttnn {

namespace multi_device {


DeviceMesh::DeviceMesh(const DeviceGrid& device_grid, const DeviceIds &device_ids, size_t l1_small_size)
    : device_grid(device_grid)
{
    auto [num_rows, num_cols] = device_grid;
    auto num_requested_devices = num_rows * num_cols;
    auto num_available_devices = tt::tt_metal::GetNumAvailableDevices();
    TT_ASSERT(num_requested_devices <= num_available_devices, "Requested more devices than available");
    TT_ASSERT(num_requested_devices <= device_ids.size(), "User provided insufficient number of device_ids for DeviceMesh");


    //TODO: for DevicePool feature delete CreateDevices and merge with this function
    //TODO: should there be an explicit CloseDevices call somewhere?
    managed_devices = tt::tt_metal::detail::CreateDevices(device_ids, 1, l1_small_size);
    for (int i = 0; i < num_requested_devices; i++) {
        mesh_devices.emplace_back(device_ids[i], std::unique_ptr<Device>(managed_devices.at(device_ids[i])));
    }
}


DeviceMesh::~DeviceMesh() {
    if (not managed_devices.empty()) {
        close_devices();
    }
}


Device* DeviceMesh::get_device(int queried_device_id)
{
    for (const auto& [device_id, device] : mesh_devices) {
        if (device_id == queried_device_id) {
            return device.get();
        }
    }
    TT_THROW("User has provided an invalid device index");
}


std::vector<Device*> DeviceMesh::get_devices() const
{
    std::vector<Device*> devices;
    for (const auto& [device_id, device] : mesh_devices) {
        devices.push_back(device.get());
    }
    return devices;
}


const DeviceIds DeviceMesh::get_device_ids() const
{
    DeviceIds device_ids;
    for (const auto& [device_id, device] : mesh_devices) {
        device_ids.push_back(device_id);
    }
    return device_ids;
}

int DeviceMesh::num_devices() const
{
    return mesh_devices.size();
}

void DeviceMesh::close_devices() {
    // TODO: change api to a yield, shouldn't allow closing sub grids in a pool of devices
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
    for (const auto &[device_id, device] : managed_devices) {
        if (device->is_initialized()) {
            tt::tt_metal::detail::DeallocateBuffers(device);
            device->close();
        }
    }
    mesh_devices.clear();
    managed_devices.clear();
}

}  // namespace multi_device

}  // namespace ttnn

namespace tt {

namespace tt_metal {

bool validate_worker_modes(const std::vector<Device*>& workers) {
    bool worker_modes_match = true;
    auto first_worker_mode = workers.at(0)->get_worker_mode();
    for (auto worker : workers) {
        worker_modes_match &= (worker->get_worker_mode() == first_worker_mode);
    }
    return worker_modes_match;
}

}

}
