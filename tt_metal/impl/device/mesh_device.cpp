// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "tt_metal/impl/device/mesh_device.hpp"
#include "tt_metal/impl/device/mesh_device_view.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"


namespace tt::tt_metal {

MeshDevice::MeshDevice(const MeshShape& mesh_shape, const DeviceIds &device_ids, size_t l1_small_size, size_t trace_region_size, size_t num_command_queues, DispatchCoreType dispatch_core_type)
    : mesh_shape(mesh_shape)
{
    auto [num_rows, num_cols] = mesh_shape;
    auto num_requested_devices = num_rows * num_cols;
    auto num_available_devices = tt::tt_metal::GetNumAvailableDevices();
    TT_ASSERT(num_requested_devices <= num_available_devices, "Requested more devices than available");
    TT_ASSERT(num_requested_devices <= device_ids.size(), "User provided insufficient number of device_ids for MeshDevice");

    this->is_galaxy_ = tt::Cluster::instance().is_galaxy_cluster();
    if (this->is_galaxy_) {
        // Temp solution until we add algorithmic way to determine chip connectivity
        // Map col to tunnel depth and row to tunnel count
        int cluster_tunnel_depth = tt::Cluster::instance().get_mmio_device_max_tunnel_depth(0);
        int cluster_tunnel_count = tt::Cluster::instance().get_mmio_device_tunnel_count(0);
        int num_mmio_devices = tt::Cluster::instance().number_of_pci_devices();
        TT_FATAL(num_cols <= cluster_tunnel_depth and num_rows <= cluster_tunnel_count * num_mmio_devices, "Unsupported Galaxy mesh shape");

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
        managed_devices = tt::tt_metal::detail::CreateDevices(galaxy_device_ids, num_command_queues, l1_small_size, trace_region_size, dispatch_core_type);
        for (int i = 0; i < num_requested_devices; i++) {
            mesh_devices.emplace_back(device_ids[i], managed_devices.at(galaxy_device_ids[i]));
        }
        this->view = std::make_unique<tt::tt_metal::MeshDeviceView>(*this);
    } else {
        managed_devices = tt::tt_metal::detail::CreateDevices(device_ids, num_command_queues, l1_small_size, trace_region_size, dispatch_core_type);
        for (int i = 0; i < num_requested_devices; i++) {
            mesh_devices.emplace_back(device_ids[i], managed_devices.at(device_ids[i]));
        }
    }

    for (const auto& [dev_id, dev]: mesh_devices) {
        log_debug(tt::LogMetal, "TTNN Dev {}: Metal Dev {}", dev_id, dev->id());
    }
}


MeshDevice::~MeshDevice() {
    if (not managed_devices.empty()) {
        close_devices();
    }
}

Device* MeshDevice::get_device(int logical_device_id) const {
    for (const auto& [device_id, device] : mesh_devices) {
        if (device_id == logical_device_id) {
            return device;
        }
    }
    TT_THROW("User has provided an invalid device index");
}

std::vector<Device*> MeshDevice::get_devices() const
{
    std::vector<Device*> devices;
    for (const auto& [device_id, device] : mesh_devices) {
        devices.push_back(device);
    }
    return devices;
}

Device* MeshDevice::get_device(int row_idx, int col_idx) const {
    if (not is_galaxy_) {
        TT_THROW("Non-galaxy device mesh does not currently support indexing over rows and columns of a logical 2D mesh.");
    }

    TT_FATAL(
        this->num_rows() != 0 and this->num_cols() != 0,
        "#10419, Current device mesh does not support indexing by row or col indices.");
    TT_FATAL(row_idx >= 0 and row_idx < this->num_rows(), "Invalid row index.");
    TT_FATAL(col_idx >= 0 and col_idx < this->num_cols(), "Invalid col index.");
    int idx = row_idx * this->num_cols() + col_idx;
    return this->mesh_devices[idx].second;
}

std::vector<Device*> MeshDevice::get_devices_on_row(int row_idx) const {
    if (not is_galaxy_) {
        TT_THROW("Non-galaxy device mesh does not currently support indexing over rows and columns of a logical 2D mesh.");
    }
    return this->view->get_devices_on_row(row_idx);
}

std::vector<Device*> MeshDevice::get_devices_on_column(int col_idx) const {
    if (not is_galaxy_) {
        TT_THROW("Non-galaxy device mesh does not currently support indexing over rows and columns of a logical 2D mesh.");
    }
    return this->view->get_devices_on_column(col_idx);
}

const DeviceIds MeshDevice::get_device_ids() const
{
    DeviceIds device_ids;
    for (const auto& [device_id, device] : mesh_devices) {
        device_ids.push_back(device_id);
    }
    return device_ids;
}

int MeshDevice::num_devices() const
{
    return mesh_devices.size();
}

CoreCoord MeshDevice::compute_with_storage_grid_size() const {
    return mesh_devices.at(0).second->compute_with_storage_grid_size();
}

CoreCoord MeshDevice::dram_grid_size() const {
    return mesh_devices.at(0).second->dram_grid_size();
}

tt::ARCH MeshDevice::arch() const {
    return mesh_devices.at(0).second->arch();
}

int MeshDevice::num_rows() const
{
    return this->mesh_shape.first;
}

int MeshDevice::num_cols() const
{
    return this->mesh_shape.second;
}

MeshShape MeshDevice::shape() const
{
    return this->mesh_shape;
}

void MeshDevice::close_devices() {
    tt::tt_metal::detail::CloseDevices(managed_devices);
    mesh_devices.clear();
    managed_devices.clear();
}

std::string MeshDevice::to_string() const {
    return fmt::format("MeshDevice({}x{} grid, {} devices)",
                       this->num_rows(),
                       this->num_cols(),
                       this->num_devices());
}

std::shared_ptr<const MeshDeviceView> MeshDevice::get_view() const {
    return this->view;
}

std::shared_ptr<MeshDeviceView> MeshDevice::get_view() {
    return this->view;
}

std::ostream& operator<<(std::ostream& os, const MeshDevice& mesh_device) {
    return os << mesh_device.to_string();
}

bool validate_worker_modes(const std::vector<Device*>& workers) {
    bool worker_modes_match = true;
    auto first_worker_mode = workers.at(0)->get_worker_mode();
    for (auto worker : workers) {
        worker_modes_match &= (worker->get_worker_mode() == first_worker_mode);
    }
    return worker_modes_match;
}

} // namespace tt::tt_metal
