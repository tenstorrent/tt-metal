// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <ostream>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::distributed {

class MeshEvent {
public:
    MeshEvent(
        uint32_t id,
        MeshDevice* device,
        uint32_t mesh_cq_id,
        const MeshCoordinateRange& device_range,
        uint32_t quiesce_epoch = 0);

    // Returns references to the event data.
    uint32_t id() const;
    MeshDevice* device() const;
    uint32_t mesh_cq_id() const;
    const MeshCoordinateRange& device_range() const;
    // CQ quiesce generation captured when the event was recorded.
    uint32_t quiesce_epoch() const;

    friend std::ostream& operator<<(std::ostream& os, const MeshEvent& event);

private:
    uint32_t id_ = 0;
    MeshDevice* device_ = nullptr;
    uint32_t mesh_cq_id_ = 0;
    MeshCoordinateRange device_range_;
    uint32_t quiesce_epoch_ = 0;
};

}  // namespace tt::tt_metal::distributed
