// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_event.hpp>

namespace tt::tt_metal::distributed {

class MeshDevice;

class MeshEventImpl {
public:
    MeshEventImpl(uint32_t id, MeshDevice* device, uint32_t mesh_cq_id, const MeshCoordinateRange& device_range);

    uint32_t id() const { return id_; }
    MeshDevice* device() const { return device_; }
    uint32_t mesh_cq_id() const { return mesh_cq_id_; }
    const MeshCoordinateRange& device_range() const { return device_range_; }

private:
    uint32_t id_ = 0;
    MeshDevice* device_ = nullptr;
    uint32_t mesh_cq_id_ = 0;
    MeshCoordinateRange device_range_;
};

// Internal factory used by mesh command queues / tests.
MeshEvent make_mesh_event(
    uint32_t id, MeshDevice* device, uint32_t mesh_cq_id, const MeshCoordinateRange& device_range);

}  // namespace tt::tt_metal::distributed
