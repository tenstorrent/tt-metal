// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mesh_device.hpp"

namespace tt::tt_metal::distributed {

class MeshEvent {
public:
    MeshDevice* device = nullptr;
    LogicalDeviceRange device_range = LogicalDeviceRange({0, 0});
    uint32_t cq_id = 0;
    uint32_t event_id = 0;
};

}  // namespace tt::tt_metal::distributed
