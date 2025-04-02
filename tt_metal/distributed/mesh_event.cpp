// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <mesh_event.hpp>

#include "mesh_device.hpp"

namespace tt::tt_metal::distributed {

MeshEvent::MeshEvent(uint32_t id, MeshDevice* device, uint32_t mesh_cq_id, const MeshCoordinateRange& device_range) :
    id_(id), device_(device), mesh_cq_id_(mesh_cq_id), device_range_(device_range) {}

uint32_t MeshEvent::id() const { return id_; }
MeshDevice* MeshEvent::device() const { return device_; }
uint32_t MeshEvent::mesh_cq_id() const { return mesh_cq_id_; }
const MeshCoordinateRange& MeshEvent::device_range() const { return device_range_; }

std::ostream& operator<<(std::ostream& os, const MeshEvent& event) {
    os << "MeshEvent(id=" << event.id() << ", device_id=" << event.device()->id()
       << ", mesh_cq_id=" << event.mesh_cq_id() << ", device_range=" << event.device_range() << ")";
    return os;
}

}  // namespace tt::tt_metal::distributed
