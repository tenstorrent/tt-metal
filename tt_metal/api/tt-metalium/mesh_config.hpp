// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

namespace tt::tt_metal::distributed {

using DeviceIds = std::vector<int>;
using MeshDeviceID = int;
using chip_id_t = int;

struct MeshOffset {
    size_t row = 0;
    size_t col = 0;
};

struct MeshShape {
    size_t num_rows = 0;
    size_t num_cols = 0;
};

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
    MeshOffset offset{0, 0};
    std::vector<chip_id_t> physical_device_ids{};
};

}  // namespace tt::tt_metal::distributed
