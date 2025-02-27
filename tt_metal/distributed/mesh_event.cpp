// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <mesh_event.hpp>

namespace tt::tt_metal::distributed {

MeshEvent::MeshEvent(
    MeshDevice* device, const MeshCoordinateRange& device_range, uint32_t mesh_cq_id, uint32_t event_id) :
    device_(device), device_range_(device_range), mesh_cq_id_(mesh_cq_id), event_id_(event_id) {}

MeshDevice* MeshEvent::device() const { return device_; }
const MeshCoordinateRange& MeshEvent::device_range() const { return device_range_; }
uint32_t MeshEvent::mesh_cq_id() const { return mesh_cq_id_; }
uint32_t MeshEvent::event_id() const { return event_id_; }

std::ostream& operator<<(std::ostream& os, const MeshEvent& event) {
    os << "MeshEvent(device_id=" << event.device()->id() << ", device_range=" << event.device_range()
       << ", mesh_cq_id=" << event.mesh_cq_id() << ", event_id=" << event.event_id() << ")";
    return os;
}

}  // namespace tt::tt_metal::distributed
