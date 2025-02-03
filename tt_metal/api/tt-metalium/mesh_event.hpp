// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mesh_device.hpp"

namespace tt::tt_metal::distributed {
using LogicalDeviceRange = CoreRange;

struct MeshEvent {
    MeshDevice* device = nullptr;
    LogicalDeviceRange device_range = LogicalDeviceRange({0, 0});
    uint32_t cq_id = -1;
    uint32_t event_id = -1;
};

}  // namespace tt::tt_metal::distributed
