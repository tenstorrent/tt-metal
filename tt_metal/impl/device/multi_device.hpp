// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>
#include <map>
#include <optional>

#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/device/device_mesh_view.hpp"

using Device = tt::tt_metal::Device;
using DeviceMeshView = tt::tt_metal::DeviceMeshView;
using Coordinate = tt::tt_metal::Coordinate;


namespace ttnn {

namespace multi_device {
using DeviceGrid = std::pair<int, int>;
using DeviceIds = std::vector<int>;

class DeviceMesh
{
public:
    DeviceGrid device_grid;
    std::map<chip_id_t, Device *> managed_devices;
    std::vector<std::pair<int, Device *>> mesh_devices;
    std::unique_ptr<DeviceMeshView> view;

    DeviceMesh(const DeviceGrid &device_grid, const DeviceIds &device_ids, size_t l1_small_size, size_t trace_region_size, size_t num_command_queues);
    ~DeviceMesh();

    DeviceMesh(const DeviceMesh &) = delete;
    DeviceMesh &operator=(const DeviceMesh &) = delete;

    DeviceMesh(DeviceMesh &&) = delete;
    DeviceMesh &operator=(DeviceMesh &&) = delete;

    std::vector<Device*> get_devices() const;
    Device *get_device(int logical_device_id) const;
    Device *get_device(int row_idx, int col_idx) const;
    std::vector<Device *> get_devices_on_row(int row_idx) const;
    std::vector<Device *> get_devices_on_column(int col_idx) const;

    std::optional<Coordinate> find_device(int device_id) const;

    const DeviceIds get_device_ids() const;

    int num_devices() const;
    int num_rows() const;
    int num_cols() const;
    DeviceGrid shape() const;

    CoreCoord compute_with_storage_grid_size() const;

    CoreCoord dram_grid_size() const;

    tt::ARCH arch() const;

    void close_devices();

   private:
    bool is_galaxy_;
};


}  // namespace multi_device

}  // namespace ttnn

namespace tt {
namespace tt_metal {
    using DeviceMesh = ttnn::multi_device::DeviceMesh;
    bool validate_worker_modes(const std::vector<Device*>& workers);
} // namespace tt_metal
} // namespace tt
