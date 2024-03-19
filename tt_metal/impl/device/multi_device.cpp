// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "tt_metal/impl/device/multi_device.hpp"
#include "tt_metal/host_api.hpp"


namespace ttnn {

namespace multi_device {


DeviceMesh::DeviceMesh(const DeviceGrid& device_grid, const DeviceIds &device_ids)
    : device_grid(device_grid)
{
    auto num_requested_devices = device_ids.size();
    auto num_available_devices = tt::tt_metal::GetNumAvailableDevices();
    TT_ASSERT(num_requested_devices <= num_available_devices, "Requested more devices than available");

    for (auto device_id : device_ids) {
        managed_devices.emplace_back(device_id, std::unique_ptr<Device>(CreateDevice(device_id, 1)));
    }
}


Device &DeviceMesh::get_device(int queried_device_id)
{
    for (const auto& [device_id, device] : managed_devices) {
        if (device_id == queried_device_id) {
            return *device;
        }
    }
    TT_THROW("User has provided an invalid device index");
}


std::vector<Device*> DeviceMesh::get_devices() const
{
    std::vector<Device*> devices;
    for (const auto& [device_id, device] : managed_devices) {
        devices.push_back(device.get());
    }
    return devices;
}


const DeviceIds DeviceMesh::get_device_ids() const
{
    DeviceIds device_ids;
    for (const auto& [device_id, device] : managed_devices) {
        device_ids.push_back(device_id);
    }
    return device_ids;
}

int DeviceMesh::num_devices() const
{
    return managed_devices.size();
}


}  // namespace multi_device

}  // namespace ttnn
