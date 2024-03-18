// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "tt_metal/impl/device/device.hpp"

using Device = tt::tt_metal::Device;


namespace ttnn {

namespace multi_device {
using DeviceGrid = std::pair<int, int>;
using DeviceIds = std::vector<int>;

class DeviceMesh
{
public:
    DeviceGrid device_grid;
    std::vector<std::pair<int, std::unique_ptr<Device>>> managed_devices;

    DeviceMesh(const DeviceGrid& device_grid, const DeviceIds &device_ids);
    ~DeviceMesh() = default;

    DeviceMesh(const DeviceMesh &) = delete;
    DeviceMesh &operator=(const DeviceMesh &) = delete;

    DeviceMesh(DeviceMesh &&) = delete;
    DeviceMesh &operator=(DeviceMesh &&) = delete;

    Device &get_device(int queried_device_id);

    const DeviceIds get_device_ids() const;

    int num_devices() const;
};


}  // namespace multi_device

}  // namespace ttnn
