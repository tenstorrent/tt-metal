// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mesh_device.hpp"

namespace tt::tt_metal::distributed {

class MeshEvent {
private:
    MeshDevice* device_ = nullptr;
    MeshCoordinateRange device_range_;
    uint32_t mesh_cq_id_ = 0;
    uint32_t event_id_ = 0;

public:
    MeshEvent(MeshDevice* device, const MeshCoordinateRange& device_range, uint32_t mesh_cq_id, uint32_t event_id);

    // Returns references to the event data.
    MeshDevice* device() const;
    const MeshCoordinateRange& device_range() const;
    uint32_t mesh_cq_id() const;
    uint32_t event_id() const;

    friend std::ostream& operator<<(std::ostream& os, const MeshEvent& event);
};

}  // namespace tt::tt_metal::distributed
