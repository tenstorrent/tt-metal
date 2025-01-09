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

enum class MeshType { RowMajor, Ring, Line };

struct MeshDeviceConfig {
    MeshShape mesh_shape{0, 0};
    MeshOffset offset{0, 0};
    std::vector<chip_id_t> physical_device_ids{};
    MeshType mesh_type{MeshType::RowMajor};
};

}  // namespace tt::tt_metal::distributed
