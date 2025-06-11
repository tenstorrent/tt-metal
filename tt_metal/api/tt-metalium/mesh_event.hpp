// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <ostream>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt {
namespace tt_metal {
namespace distributed {
class MeshDevice;
}  // namespace distributed
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal::distributed {

class MeshEvent {
public:
    MeshEvent(uint32_t id, MeshDevice* device, uint32_t mesh_cq_id, const MeshCoordinateRange& device_range);

    // Returns references to the event data.
    uint32_t id() const;
    MeshDevice* device() const;
    uint32_t mesh_cq_id() const;
    const MeshCoordinateRange& device_range() const;

    friend std::ostream& operator<<(std::ostream& os, const MeshEvent& event);

private:
    uint32_t id_ = 0;
    MeshDevice* device_ = nullptr;
    uint32_t mesh_cq_id_ = 0;
    MeshCoordinateRange device_range_;
};

}  // namespace tt::tt_metal::distributed
