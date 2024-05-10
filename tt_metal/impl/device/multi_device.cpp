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
    bool is_galaxy = tt::Cluster::instance().is_galaxy_cluster();
    std::vector<chip_id_t> mmio_device_ids = {};
    if (is_galaxy) {
        mmio_device_ids.push_back(0);
        if (num_requested_devices > 8) {
            mmio_device_ids.push_back(1);
        }
        if (num_requested_devices > 16) {
            mmio_device_ids.push_back(2);
        }
        if (num_requested_devices > 24) {
            mmio_device_ids.push_back(3);
        }
    } else {
        mmio_device_ids = device_ids;
    }
    managed_devices = tt::tt_metal::detail::CreateDevices(mmio_device_ids, 1, l1_small_size);
    if (is_galaxy) {
        DeviceIds galaxy_device_ids;
        for (const auto &[dev_id, dev]: managed_devices) {
            galaxy_device_ids.emplace_back(dev_id);
        }
        for (int i = 0; i < num_requested_devices; i++) {
            mesh_devices.emplace_back(device_ids[i], std::unique_ptr<Device>(managed_devices.at(galaxy_device_ids[i])));
        }
    } else {
      for (int i = 0; i < num_requested_devices; i++) {
            mesh_devices.emplace_back(device_ids[i], std::unique_ptr<Device>(managed_devices.at(device_ids[i])));
      }
    }
    for (const auto& [dev_id, dev]: mesh_devices) {
        std::cout << "dev_id " << dev_id << " dev " << dev->id() << std::endl;
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
    tt::tt_metal::detail::CloseDevices(managed_devices);
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
