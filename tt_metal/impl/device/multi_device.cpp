// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "tt_metal/impl/device/multi_device.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"


namespace ttnn {

namespace multi_device {

DeviceMesh::DeviceMesh(const DeviceGrid& device_grid, const DeviceIds &device_ids, size_t l1_small_size, size_t trace_region_size, size_t num_command_queues)
    : device_grid(device_grid)
{
    auto [num_rows, num_cols] = device_grid;
    auto num_requested_devices = num_rows * num_cols;
    auto num_available_devices = tt::tt_metal::GetNumAvailableDevices();
    TT_ASSERT(num_requested_devices <= num_available_devices, "Requested more devices than available");
    TT_ASSERT(num_requested_devices <= device_ids.size(), "User provided insufficient number of device_ids for DeviceMesh");

    bool is_galaxy = tt::Cluster::instance().is_galaxy_cluster();
    if (is_galaxy) {
        // Temp solution until we add algorithmic way to determine chip connectivity
        // Map col to tunnel depth and row to tunnel count
        int cluster_tunnel_depth = tt::Cluster::instance().get_mmio_device_max_tunnel_depth(0);
        int cluster_tunnel_count = tt::Cluster::instance().get_mmio_device_tunnel_count(0);
        int num_mmio_devices = tt::Cluster::instance().number_of_pci_devices();
        TT_ASSERT(num_cols <= cluster_tunnel_depth and num_rows <= cluster_tunnel_count * num_mmio_devices, "Unsupported Galaxy mesh shape");

        DeviceIds galaxy_device_ids;
        for (int mmio_device_id = 0; mmio_device_id < num_mmio_devices; mmio_device_id++) {
            auto tunnels_from_mmio = tt::Cluster::instance().get_tunnels_from_mmio_device(mmio_device_id);
            for (uint32_t t = 0; t < tunnels_from_mmio.size(); t++) {
                if (galaxy_device_ids.size() == num_requested_devices) {
                    break;
                }
                int col_idx = 0;
                for (uint32_t ts = 1; ts < tunnels_from_mmio[t].size(); ts++) {
                    galaxy_device_ids.push_back(tunnels_from_mmio[t][ts]);
                    col_idx ++;
                    if (col_idx == num_cols) {
                        break;
                    }
                }
            }
        }
        managed_devices = tt::tt_metal::detail::CreateDevices(galaxy_device_ids, num_command_queues, l1_small_size, trace_region_size);
        for (int i = 0; i < num_requested_devices; i++) {
            mesh_devices.emplace_back(device_ids[i], managed_devices.at(galaxy_device_ids[i]));
        }
    } else {
        managed_devices = tt::tt_metal::detail::CreateDevices(device_ids, num_command_queues, l1_small_size, trace_region_size);
        for (int i = 0; i < num_requested_devices; i++) {
            mesh_devices.emplace_back(device_ids[i], managed_devices.at(device_ids[i]));
        }
    }

    for (const auto& [dev_id, dev]: mesh_devices) {
        log_debug(tt::LogMetal, "TTNN Dev {}: Metal Dev {}", dev_id, dev->id());
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
            return device;
        }
    }
    TT_THROW("User has provided an invalid device index");
}


std::vector<Device*> DeviceMesh::get_devices() const
{
    std::vector<Device*> devices;
    for (const auto& [device_id, device] : mesh_devices) {
        devices.push_back(device);
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
