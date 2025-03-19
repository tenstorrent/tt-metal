// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

#include "mesh_coord.hpp"

namespace tt::tt_metal::distributed {

using DeviceIds = std::vector<int>;
using MeshDeviceID = int;
using chip_id_t = int;

/**
 * @brief Defines the organization of physical devices in a user-defined MeshDevice.
 *
 * The mesh type imposes properties on the physical connectivity of devices:
 *
 * - RowMajor: Devices are arranged in a 2D grid and accessed in row-major order.
 *             This is the default configuration for most multi-device setups.
 *
 * - Ring: Devices are arranged in a circular topology where each device is connected
 *         to its neighbors, forming a ring structure.
 *
 * - Line: Devices are arranged linearly in a single dimension.
 */

struct MeshDeviceConfig {
    MeshShape mesh_shape{0, 0};
    std::optional<MeshCoordinate> offset;
    std::vector<chip_id_t> physical_device_ids{};
};

}  // namespace tt::tt_metal::distributed
